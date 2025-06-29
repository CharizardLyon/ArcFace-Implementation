# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.trainer import train_one_epoch, validate
from utils.utils import set_seed, get_device
from utils.logger import get_logger
from data.dataloader import get_dataloaders
from models.resnet_arcface import ResNetArcModel

import argparse
import yaml
import os


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device()
    logger = get_logger()

    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    logger.info("Building model...")
    model = ResNetArcModel(
        num_classes=config["num_classes"],
        embedding_size=config["embedding_size"],
        backbone=config["backbone"],
        arcface_margin=config["arcface_margin"],
        arcface_scale=config["arcface_scale"],
        arcface_easy_margin=config["arcface_easy_margin"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    logger.info(" Starting training...")
    for epoch in range(1, config["epochs"] + 1):
        logger.info(f"\n Epoch {epoch}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f" Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        checkpoint_path = os.path.join(
            config["checkpoint_dir"], f"model_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f" Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
