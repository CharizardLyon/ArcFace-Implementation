# ArcFace-Implementation

This project implements a modular face recognition system using a ResNet backbone and ArcFace loss for improved classification in angular space for Pytorch. It includes training, evaluation, inference, and precomputed embedding support for real-time recognition.

# Project Structure

├── train.py # Train model <br>
├── eval.py # Evaluate model <br>
├── infer.py # Run inference on a new image <br>
├── precompute_embeddings.py # Precompute embeddings from a gallery <br>
├── config/ <br>
│ └── default.yaml # All config settings <br>
├── data/ <br>
│ └── dataloader.py # Dataset and DataLoader utilities <br>
├── models/ <br>
│ ├── arcface.py # ArcFace implementation <br>
│ └── resnet_arcface.py # ResNet + ArcFace model <br>
├── utils/ <br>
│ ├── logger.py # Logging setup <br>
│ ├── trainer.py # Training/validation functions <br>
│ ├── metrics.py # Accuracy metrics <br>
│ └── utils.py # Device, seed, helpers <br>
├── experiment/ <br>
│ └── checkpoints/ # Trained model weights <br>
└── embeddings/ # Precomputed feature vectors <br>

# Training

python train.py --config config/default.yaml
