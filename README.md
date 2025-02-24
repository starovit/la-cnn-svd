# Reduced VGG Project

This repository implements a Reduced VGG model on the CIFAR100 dataset with PyTorch. It includes:
- A custom VGG-like model (`ReducedVGG`)
- Training and validation loops with early stopping and logging (integrated with Weights & Biases)
- SVD-based compression routines to approximate convolutional layers
- Plotting functions to analyze parameter count and accuracy vs. compression rank

## Repository Structure

- **src/models.py**: Model definition and evaluation functions.
- **src/train.py**: Training and validation routines.
- **src/train.py**: Fine-tuning (fixed-rank).
- **src/compress.py**: Functions to apply SVD and compress the model.
- **weights/best_model.pth**: Weights for trained full model.
- **main.py**: Example of usage: import weights; apply SVD and show results.


## Requirements

See [requirements.txt](requirements.txt) for the required packages.