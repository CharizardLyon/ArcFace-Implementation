# ArcFace-Implementation

This project implements a modular face recognition system using a ResNet backbone and ArcFace loss for improved classification in angular space for Pytorch. It includes training, evaluation, inference, and precomputed embedding support for real-time recognition.

# Project Structure

├── train.py # Train model <br><br>
├── eval.py # Evaluate model <br><br>
├── infer.py # Run inference on a new image <br><br>
├── precompute_embeddings.py # Precompute embeddings from a gallery <br><br>
├── config/ <br><br>
│ └── default.yaml # All config settings <br><br>
├── data/ <br><br>
│ └── dataloader.py # Dataset and DataLoader utilities <br><br>
├── models/ <br><br>
│ ├── arcface.py # ArcFace implementation <br><br>
│ └── resnet_arcface.py # ResNet + ArcFace model <br><br>
├── utils/ <br><br>
│ ├── logger.py # Logging setup <br><br>
│ ├── trainer.py # Training/validation functions <br><br>
│ ├── metrics.py # Accuracy metrics <br><br>
│ └── utils.py # Device, seed, helpers <br><br>
├── experiment/ <br><br>
│ └── checkpoints/ # Trained model weights <br><br>
└── embeddings/ # Precomputed feature vectors <br><br>

# Training

"""
python train.py --config config/default.yaml
"""
