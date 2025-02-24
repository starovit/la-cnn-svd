import torch
import copy
import wandb

def train_validate(model, trainloader, validloader, criterion, optimizer, num_epochs, patience, wandb_name, wandb_config=None):
    """
    Train the model with validation, early stopping, and logging to Weights & Biases.
    
    Returns:
        model: The best model based on validation loss.
        history: Dictionary of training and validation metrics.
    """
    if wandb_config is None:
        wandb_config = {
            "epochs": num_epochs,
            "patience": patience,
            "learning_rate": optimizer.defaults.get("lr", 0.001)
        }

    wandb.init(project="reduced_vgg_project", name=wandb_name, config=wandb_config)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in trainloader:
            images, labels = images.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_train_acc = train_correct / train_total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_train_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        epoch_val_loss = val_loss / len(validloader.dataset)
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        print(f"Validation Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "Loss/Train": epoch_loss,
            "Loss/Validation": epoch_val_loss,
            "Accuracy/Train": epoch_train_acc,
            "Accuracy/Validation": epoch_val_acc
        })

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_wts, 'best_model.pth')
            print("Saving best model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    model.load_state_dict(best_model_wts)
    wandb.finish()
    return model, history