import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_yaml_config
import torch
import os
import pandas as pd

from src import data, transforms
from src.model import build_binary_model

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


    excel_folder = data_config["excel_folder"]
    if not os.path.isabs(excel_folder):
        excel_folder = os.path.join(PROJECT_ROOT, excel_folder)

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