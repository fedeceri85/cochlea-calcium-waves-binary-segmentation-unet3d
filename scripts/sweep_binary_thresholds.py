"""Sweep binary prediction thresholds on the validation set.

Inference is performed once per movie. Metrics for every threshold are then
calculated from the same logits, so adding thresholds is much cheaper than
running validation repeatedly.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from monai.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, transforms  # noqa: E402
from src.config import load_yaml_config  # noqa: E402
from src.model import build_binary_model  # noqa: E402
from src.validate import (  # noqa: E402
    calculate_binary_metrics,
    predict_binary_logits,
    prediction_threshold_to_cutoff,
)


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(configured_device):
    if configured_device != "auto":
        return torch.device(configured_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_validation_loader(config):
    data_config = config["data"]
    augmentation_config = config["augmentation"]
    training_config = config["training"]

    data_dir = resolve_project_path(data_config["data_dir"])
    dataframe = pd.read_csv(data_dir / "dfdata.csv")

    # Match the Windows path remapping used by the training script.
    if os.name == "nt":
        for column in ("Folder", "image", "label", "waves"):
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].str.replace(
                    r"^/Volumes/Data/Users(?=/|$)",
                    "D:/Users/Federico",
                    regex=True,
                )

    _, validation_records = data.train_test_split(
        dataframe,
        column=data_config["column"],
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        split_json_path=resolve_project_path(data_config["split_file"]),
        save_split_json=False,
    )

    _, validation_transforms = transforms.get_transforms_binary_classification(
        patch_size=data_config["patch_size"],
        foreground_probability=data_config["foreground_probability"],
        min_waves=data_config["min_waves"],
        min_foreground_fraction=data_config["min_foreground_fraction"],
        background_max_fraction=data_config["background_max_fraction"],
        max_attempts=data_config["max_attempts"],
        flip_vertical=augmentation_config["flip_vertical"],
        flip_vertical_prob=augmentation_config["flip_vertical_prob"],
        flip_horizontal=augmentation_config["flip_horizontal"],
        flip_horizontal_prob=augmentation_config["flip_horizontal_prob"],
        rotate90=augmentation_config["rotate90"],
        rotate90_prob=augmentation_config["rotate90_prob"],
        rotate90_max_k=augmentation_config["rotate90_max_k"],
        intensity_scale=augmentation_config["intensity_scale"],
        intensity_scale_factors=augmentation_config["intensity_scale_factors"],
        intensity_scale_prob=augmentation_config["intensity_scale_prob"],
        intensity_shift=augmentation_config["intensity_shift"],
        intensity_shift_offsets=augmentation_config["intensity_shift_offsets"],
        intensity_shift_prob=augmentation_config["intensity_shift_prob"],
    )

    dataset = Dataset(validation_records, transform=validation_transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training_config.get("val_num_workers", 0),
    )
    return loader


def load_model(config, checkpoint_path, device):
    model_config = config["model"]
    model = build_binary_model(
        architecture=model_config["architecture"],
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def sweep_thresholds(
    model,
    loader,
    device,
    thresholds,
    validation_config,
    normalization,
    overlap,
    blend_mode,
    buffer_steps,
    chunk_size,
):
    sigmoid = validation_config["prediction_sigmoid"]
    cutoffs = [
        prediction_threshold_to_cutoff(value, sigmoid)
        for value in thresholds
    ]

    totals = [
        {"intersection": 0, "pred": 0, "gt": 0, "movie_metrics": []}
        for _ in thresholds
    ]

    with torch.inference_mode():
        for movie_number, batch in enumerate(loader, start=1):
            print(f"Movie {movie_number}/{len(loader)}")
            image = batch["image"].to("cpu", dtype=torch.float32)
            label = batch["label"].to("cpu")
            del batch

            started = time.perf_counter()
            logits = predict_binary_logits(
                model=model,
                image=image,
                device=device,
                roi_size=tuple(validation_config["roi_size"]),
                sw_batch_size=validation_config["sw_batch_size"],
                normalization=normalization,
                overlap=overlap,
                blend_mode=blend_mode,
                buffer_steps=buffer_steps,
                progress=True,
            )
            print(f"  inference: {time.perf_counter() - started:.1f}s")
            del image

            movie_counts = [
                calculate_binary_metrics(
                    logits=logits,
                    label=label,
                    prediction_cutoff=cutoff,
                    label_threshold=validation_config["label_threshold"],
                    chunk_size=chunk_size,
                )
                for cutoff in cutoffs
            ]

            for index, counts in enumerate(movie_counts):
                intersection = counts["intersection"]
                pred_count = counts["pred"]
                gt_count = counts["gt"]
                totals[index]["intersection"] += intersection
                totals[index]["pred"] += pred_count
                totals[index]["gt"] += gt_count
                totals[index]["movie_metrics"].append(
                    (counts["dice"], counts["precision"], counts["recall"])
                )

            del logits, label

    rows = []
    for threshold, total in zip(thresholds, totals):
        metrics = total["movie_metrics"]
        if not metrics:
            raise ValueError("Validation loader is empty.")
        intersection = total["intersection"]
        pred_count = total["pred"]
        gt_count = total["gt"]
        denominator = pred_count + gt_count
        rows.append({
            "threshold": threshold,
            "mean_dice": sum(value[0] for value in metrics) / len(metrics),
            "mean_precision": sum(value[1] for value in metrics) / len(metrics),
            "mean_recall": sum(value[2] for value in metrics) / len(metrics),
            "pooled_dice": 1.0 if denominator == 0 else 2 * intersection / denominator,
            "pooled_precision": intersection / pred_count if pred_count else 0.0,
            "pooled_recall": intersection / gt_count if gt_count else 0.0,
            "predicted_voxels": pred_count,
            "ground_truth_voxels": gt_count,
        })
    return pd.DataFrame(rows)


def main(args):
    config_path = resolve_project_path(args.config)
    config = load_yaml_config(config_path)
    device = select_device(config["training"]["device"])
    checkpoint_path = resolve_project_path(
        args.checkpoint or config["model"]["best_model_path"]
    )
    output_path = resolve_project_path(args.output)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Normalization: {args.normalization}")
    print(f"Thresholds: {args.thresholds}")

    loader = make_validation_loader(config)
    model = load_model(config, checkpoint_path, device)
    results = sweep_thresholds(
        model=model,
        loader=loader,
        device=device,
        thresholds=args.thresholds,
        validation_config=config["validation"],
        normalization=args.normalization,
        overlap=args.overlap,
        blend_mode=args.blend_mode,
        buffer_steps=args.buffer_steps,
        chunk_size=args.chunk_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print("\nThreshold sweep:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    best = results.loc[results["mean_dice"].idxmax()]
    print(
        f"\nBest mean Dice: threshold={best['threshold']:.4f}, "
        f"Dice={best['mean_dice']:.4f}, "
        f"precision={best['mean_precision']:.4f}, "
        f"recall={best['mean_recall']:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the training YAML configuration.")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path (defaults to model.best_model_path in the config).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[value / 20 for value in range(1, 20)],
        help="Thresholds to evaluate (default: 0.05 through 0.95).",
    )
    parser.add_argument(
        "--output",
        default="results/binary_threshold_sweep.csv",
        help="Output CSV path relative to the project root.",
    )
    parser.add_argument(
        "--normalization",
        choices=("whole_movie", "per_window"),
        default="per_window",
        help="Default matches src/validate.py.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Sliding-window overlap (default matches src/validate.py).",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("constant", "gaussian"),
        default="constant",
        help="Window blending mode (default matches src/validate.py).",
    )
    parser.add_argument(
        "--buffer-steps",
        type=int,
        default=4,
        help="Buffered temporal slabs (default matches src/validate.py).",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    main(parser.parse_args())
