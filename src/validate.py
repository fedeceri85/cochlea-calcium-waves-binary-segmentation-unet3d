from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensity
import math
import torch
import time


def prediction_threshold_to_cutoff(prediction_threshold, prediction_sigmoid):
    """Convert a probability threshold to the equivalent logit cutoff."""
    if not prediction_sigmoid:
        return prediction_threshold
    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be in [0, 1] when sigmoid is enabled.")
    if prediction_threshold == 0.0:
        return float("-inf")
    if prediction_threshold == 1.0:
        return float("inf")
    return math.log(prediction_threshold / (1.0 - prediction_threshold))


def predict_binary_logits(
    model,
    image,
    device,
    roi_size,
    sw_batch_size,
    normalization="per_window",
    overlap=0.25,
    blend_mode="constant",
    buffer_steps=4,
    progress=True,
):
    """Run the shared sliding-window inference pipeline and return CPU logits."""
    if normalization not in ("per_window", "whole_movie"):
        raise ValueError("normalization must be 'per_window' or 'whole_movie'.")

    scale_intensity = ScaleIntensity(minv=0.0, maxv=1.0)
    image = image.to(device="cpu", dtype=torch.float32)
    if normalization == "whole_movie":
        image = scale_intensity(image)

    def predict_windows(windows):
        if normalization == "per_window":
            windows = torch.stack([
                scale_intensity(window) for window in windows
            ])
        return model(windows)

    return sliding_window_inference(
        inputs=image,
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=predict_windows,
        sw_device=device,
        device="cpu",
        progress=progress,
        buffer_steps=buffer_steps,
        buffer_dim=0,
        overlap=overlap,
        mode=blend_mode,
        sigma_scale=0.125,
    )


def calculate_binary_metrics(
    logits,
    label,
    prediction_cutoff,
    label_threshold,
    chunk_size=128,
):
    """Calculate voxel counts and metrics without allocating full boolean volumes."""
    intersection = 0
    pred_count = 0
    gt_count = 0

    for chunk_start in range(0, logits.shape[2], chunk_size):
        chunk_stop = min(chunk_start + chunk_size, logits.shape[2])
        pred_chunk = logits[:, :, chunk_start:chunk_stop] > prediction_cutoff
        gt_chunk = label[:, :, chunk_start:chunk_stop] > label_threshold
        intersection += torch.count_nonzero(pred_chunk & gt_chunk).item()
        pred_count += torch.count_nonzero(pred_chunk).item()
        gt_count += torch.count_nonzero(gt_chunk).item()

    denominator = pred_count + gt_count
    return {
        "intersection": intersection,
        "pred": pred_count,
        "gt": gt_count,
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "precision": intersection / pred_count if pred_count else 0.0,
        "recall": intersection / gt_count if gt_count else 0.0,
    }


def validate_binary(
    model,
    loader,
    dice_metric,
    device,
    prediction_sigmoid,
    prediction_threshold,
    label_threshold,
    roi_size,
    sw_batch_size,
):
    prediction_cutoff = prediction_threshold_to_cutoff(
        prediction_threshold, prediction_sigmoid
    )

    model.eval()

    with torch.no_grad():
        scores = []
        for movie_number, batch in enumerate(loader, start=1):
            print(f"Starting movie {movie_number}/{len(loader)}")
            # Keep the complete movie and its label on the CPU.
            image = batch["image"].to("cpu")
            label = batch["label"].to("cpu")
            del batch

            start = time.perf_counter()
            output = predict_binary_logits(
                model=model,
                image=image,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
            )
            inference_seconds = time.perf_counter() - start
            print(f"Inference and assembly: {inference_seconds:.1f}s")

            # The input movie is no longer needed after inference. Release it
            # before allocating temporary tensors for the Dice calculation.
            del image

            start_dice_time = time.perf_counter()
            metrics = calculate_binary_metrics(
                logits=output,
                label=label,
                prediction_cutoff=prediction_cutoff,
                label_threshold=label_threshold,
            )
            intersection = metrics["intersection"]
            pred_count = metrics["pred"]
            gt_count = metrics["gt"]
            movie_dice = metrics["dice"]
            scores.append(movie_dice)

            dice_seconds = time.perf_counter() - start_dice_time
            print(f"Dice calculation: {dice_seconds:.1f}s")
            precision = metrics["precision"]
            recall = metrics["recall"]

            print(
                f"Movie {movie_number}: "
                f"Dice={movie_dice:.4f}, "
                f"pred={pred_count}, "
                f"gt={gt_count}, "
                f"intersection={intersection}, "
                f"precision={precision:.4f}, "
                f"recall={recall:.4f}"
            )
            del output, label

    if not scores:
        raise ValueError("Validation loader is empty.")
    score = sum(scores) / len(scores)

    return score, precision, recall
