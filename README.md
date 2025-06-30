# ArcFace-Implementation

This project implements a modular face recognition system using a ResNet backbone and ArcFace loss for improved classification in angular space for Pytorch. It includes training, evaluation, inference, and precomputed embedding support for real-time recognition.

# Project Structure

├── train.py # Train model
├── eval.py # Evaluate model
├── infer.py # Run inference on a new image
├── precompute_embeddings.py # Precompute embeddings from a gallery
├── config/
│ └── default.yaml # All config settings
├── data/
│ └── dataloader.py # Dataset and DataLoader utilities
├── models/
│ ├── arcface.py # ArcFace implementation
│ └── resnet_arcface.py # ResNet + ArcFace model
├── utils/
│ ├── logger.py # Logging setup
│ ├── trainer.py # Training/validation functions
│ ├── metrics.py # Accuracy metrics
│ └── utils.py # Device, seed, helpers
├── experiment/
│ └── checkpoints/ # Trained model weights
└── embeddings/ # Precomputed feature vectors

# Training

"""
python train.py --config config/default.yaml
"""
