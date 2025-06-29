import torch
from torchvision import transforms
from torch import Tensor
from PIL import Image
from models.resnet_arcface import ResNetArcModel
from utils.utils import get_device
from utils.logger import get_logger
from typing import cast
import yaml
import argparse


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image_path: str) -> Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    # Tell the type checker this is definitely a Tensor
    image_tensor = cast(Tensor, image_tensor)

    # Now add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def main(config_path, checkpoint_path, image_path):
    config = load_config(config_path)
    device = get_device()
    logger = get_logger()

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
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        embeddings = model(image_tensor)  # No labels: returns embeddings

    logger.info(f"Embedding vector shape: {embeddings.shape}")
    print(embeddings.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.image)
