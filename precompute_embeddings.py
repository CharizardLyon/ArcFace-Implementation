import torch
from torch.utils.data import DataLoader
from models.resnet_arcface import ResNetArcModel
from utils.utils import get_device
import yaml
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ImageOnlyDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.image_files = sorted(
            [
                f
                for f in os.listdir(images_folder)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # return path to keep track of identity


def main(config_path, checkpoint_path, image_dir, save_dir):
    config = load_config(config_path)
    device = get_device()

    model = ResNetArcModel(
        num_classes=config["num_classes"],
        embedding_size=config["embedding_size"],
        backbone=config["backbone"],
        arcface_margin=config["arcface_margin"],
        arcface_scale=config["arcface_scale"],
        arcface_easy_margin=config["arcface_easy_margin"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageOnlyDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    os.makedirs(save_dir, exist_ok=True)
    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            embeddings = model(images)  # shape [B, embedding_size]
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            all_paths.extend(paths)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(os.path.join(save_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(save_dir, "paths.npy"), np.array(all_paths))
    print(f"Saved embeddings and paths to {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to directory of images"
    )
    parser.add_argument("--save_dir", type=str, default="embeddings")
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.image_dir, args.save_dir)
