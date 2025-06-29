import torch
import torch.nn as nn
from utils.trainer import validate
from utils.utils import get_device
from utils.logger import get_logger
from data.dataloader import get_dataloaders
from models.resnet_arcface import ResNetArcModel
import yaml
import argparse
import os


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path, checkpoint_path):
    config = load_config(config_path)
    device = get_device()
    logger = get_logger()

    logger.info("Loading data...")
    _, val_loader = get_dataloaders(config)

    logger.info("Building model...")
    model = ResNetArcModel(
        num_classes=config["num_classes"],
        embedding_size=config["embedding_size"],
        backbone=config["backbone"],
        arcface_margin=config["arcface_margin"],
        arcface_scale=config["arcface_scale"],
        arcface_easy_margin=config["arcface_easy_margin"],
    ).to(device)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    logger.info("Starting evaluation...")
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    logger.info(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint)
