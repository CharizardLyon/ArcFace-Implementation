import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(config):
    """
    Creates PyTorch dataloaders for training and validation sets.

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        train_loader, val_loader
    """

    # Image transformations
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        config["train_data_path"], transform=transform_train
    )
    val_dataset = datasets.ImageFolder(config["val_data_path"], transform=transform_val)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    return train_loader, val_loader
