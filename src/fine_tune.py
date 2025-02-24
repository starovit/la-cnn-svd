import torch
import copy
import torch.nn as nn
import torch.optim as optim
import wandb

from src.models import ReducedVGG, count_parameters, test_model
from src.train import train_validate
from src.compress import compress_model_with_svd

def finetune_after_svd(model_path, trainloader, validloader, rank=60, num_epochs=100, patience=5, lr=0.001, batch_size=128):
    """
    Loads a pre-trained model from model_path, applies SVD compression using a fixed rank,
    and fine-tunes the compressed model.
    
    Args:
        model_path (str): Path to the pre-trained model state dict.
        trainloader (DataLoader): Training DataLoader.
        validloader (DataLoader): Validation DataLoader.
        rank (int): Fixed SVD rank for compression.
        num_epochs (int): Number of fine-tuning epochs.
        patience (int): Early stopping patience.
        lr (float): Learning rate for fine-tuning.
        batch_size (int): Batch size.
    
    Returns:
        best_model: The fine-tuned model.
        history: Dictionary of training history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the original model state
    state_dict = torch.load(model_path, map_location=device)
    model = ReducedVGG(num_classes=100).to(device)
    model.load_state_dict(state_dict)
    
    print("Original Model Evaluation:")
    test_model(model)
    print(f"Total trainable parameters: {count_parameters(model)}")
    
    # Create a copy and compress using fixed-rank SVD
    reduced_model = copy.deepcopy(model)
    reduced_model = compress_model_with_svd(reduced_model, rank=rank)
    
    print("\nCompressed Model Evaluation (Before Fine-Tuning):")
    test_model(reduced_model)
    print(f"Total trainable parameters: {count_parameters(reduced_model)}")
    
    # Prepare Weights & Biases configuration
    wandb_name = f"svd{rank}_finetune_lr{str(lr).replace('.', '')}"
    wandb_config = {
        "epochs": num_epochs,
        "patience": patience,
        "learning_rate": lr,
        "batch_size": batch_size,
        "model": "ReducedVGG"
    }
    
    # Define Loss and Optimizer for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(reduced_model.parameters(), lr=lr)
    
    # Fine-tune the compressed model
    best_model, history = train_validate(reduced_model, trainloader, validloader, criterion, optimizer,
                                         num_epochs=num_epochs, patience=patience,
                                         wandb_name=wandb_name, wandb_config=wandb_config)
    
    return best_model, history