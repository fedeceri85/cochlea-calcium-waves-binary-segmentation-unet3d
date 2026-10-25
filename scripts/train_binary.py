import argparse
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_yaml_config
import torch
import pandas as pd
from src import data, transforms
from src.model import build_binary_model
from src.train import train_one_epoch_binary
from src.validate import validate_binary

from monai.losses import DiceLoss
from monai.metrics import DiceMetric

DATA_DIR = './data'

def main(config_path):

    # Load configs
    config = load_yaml_config(config_path)
    data_config = config["data"]
    augmentation_config = config["augmentation"]
    training_config = config["training"]
    model_config = config["model"]
    loss_config = config["loss"]
    optimizer_config = config["optimizer"]
    metric_config = config["metric"]
    validation_config = config["validation"]


    data_dir = data_config["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)


    split_json_path = data_config["split_file"]
    if not os.path.isabs(split_json_path):
        split_json_path = os.path.join(PROJECT_ROOT, split_json_path)

    checkpoint_path = model_config["checkpoint_path"]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)

    best_model_path = model_config["best_model_path"]
    if not os.path.isabs(best_model_path):
        best_model_path = os.path.join(PROJECT_ROOT, best_model_path)

    configured_device = training_config["device"]
    if configured_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(configured_device)


    print("Using device:", device)
    print("Using config:", config_path)
    
    print("Loading data...")
    dfdata = pd.read_csv(os.path.join(data_dir, 'dfdata.csv'))
    if os.name == "nt":  # Windows
        for column in ("Folder", "image", "label", "waves"):
            if column in dfdata.columns:
                dfdata[column] = dfdata[column].str.replace(
                    r"^/Volumes/Data/Users(?=/|$)",
                    "Z:/Users",
                    regex=True,
                )

    print("Splitting data into training and validation sets...")
    train_data, val_data = data.train_test_split(
        dfdata,
        column=data_config["column"],
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        split_json_path=split_json_path,
        save_split_json=data_config["save_split_json"],
    )

    train_transforms, val_transforms = transforms.get_transforms_binary_classification(
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
        intensity_scale_prob=augmentation_config["intensity_scale_prob"],
        intensity_scale_factors=augmentation_config["intensity_scale_factors"],
        intensity_shift=augmentation_config["intensity_shift"],
        intensity_shift_prob=augmentation_config["intensity_shift_prob"],
        intensity_shift_offsets=augmentation_config["intensity_shift_offsets"],
    )

    print("Creating datasets and loaders...")
    train_ds, train_loader, val_ds, val_loader = data.create_datasets_and_loaders(
        train_data,
        val_data,
        train_transforms,
        val_transforms,
        batch_size=training_config["batch_size"],
        val_batch_size=training_config["val_batch_size"],
        num_workers=training_config["num_workers"],
        val_num_workers=training_config.get("val_num_workers", training_config["num_workers"]),
        shuffle_train=training_config["shuffle_train"],
        shuffle_val=training_config["shuffle_val"],
    )

    print("Building model...")
    model = build_binary_model(
        architecture=model_config["architecture"],
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"],
    ).to(device)
    print("Model architecture:", model_config["architecture"])


    print("Setting up optimizer and loss function")
    if loss_config["name"]!= 'DiceLoss':
        raise NotImplementedError("Only Dice Loss is supported at the moment.")
    loss_function = DiceLoss(sigmoid = loss_config["sigmoid"])

    if optimizer_config["name"]!="Adam":
        raise NotImplementedError("Only Adam optimizer is supported at the moment")
    optimizer = torch.optim.Adam(model.parameters(),lr=optimizer_config['learning_rate'])
    
    print("Loading checkpoint if available")
    #########
    #Add checkpoint loading here
    #########
    start_epoch = 0
    best_dice = float("-inf")
    best_epoch = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint["best_dice"]
        best_epoch = checkpoint["best_epoch"]

        if os.path.exists(best_model_path):
            best_model = torch.load(best_model_path)
            best_dice = best_model["best_dice"]
            best_epoch = checkpoint["best_epoch"]

        print(f"Loaded checkpoint from epoch {start_epoch} with best dice {best_dice} at epoch {best_epoch}")
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    else:
        print("Training from scratch.")

    
    if metric_config["name"] != "DiceMetric":
        raise ValueError(f"Unsupported metric.name: {metric_config['name']}")
    dice_metric = DiceMetric(
        include_background=metric_config["include_background"],
        reduction=metric_config["reduction"],
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    best_model_dir = os.path.dirname(best_model_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    num_epochs = training_config["epochs"]
    if start_epoch >= num_epochs:
        print(f"Checkpoint is already at epoch {start_epoch}; configured epochs is {num_epochs}. Nothing to train.")
        return

    print("Starting training...")
    for epoch in range(start_epoch,num_epochs):
        train_loss = train_one_epoch_binary(model,
            train_loader,
            optimizer,
            loss_function,
            device
        )
        
        val_dice = validate_binary(
            model,
            val_loader,
            dice_metric,
            device,
            validation_config["prediction_sigmoid"],
            validation_config["prediction_threshold"],
            validation_config["label_threshold"],
            validation_config["roi_size"],
            validation_config["sw_batch_size"]
        )

        print(
            f"Epoch {epoch + 1:3d} "
            f"Loss={train_loss:.4f} "
            f"Dice={val_dice:.4f}"
        )

        print(f"Validation on epoch {epoch}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config_path": config_path,
            }, best_model_path)
            print("Saved new best model")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "config_path": config_path,
        }, checkpoint_path)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
