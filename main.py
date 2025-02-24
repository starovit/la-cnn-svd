import torch
import torchvision
import torchvision.transforms as transforms
from src.models import ReducedVGG, count_parameters, test_model
from src.fine_tune import finetune_after_svd

def main():
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms for training and testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR100 dataset for training and validation
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    valid_size = 0.1
    num_train = len(trainset)
    num_valid = int(valid_size * num_train)
    num_train_new = num_train - num_valid
    train_dataset, valid_dataset = torch.utils.data.random_split(trainset, [num_train_new, num_valid])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Load CIFAR100 test set
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Path to the pre-trained model (make sure this file exists)
    model_path = "weights/best_model.pth"

    # Fine-tune after SVD compression (fixed-rank example with rank=60)
    print("Fine-tuning the compressed model (fixed-rank SVD with rank=60)")
    best_model, history = finetune_after_svd(
        model_path=model_path,
        trainloader=trainloader,
        validloader=validloader,
        rank=60,
        num_epochs=100,
        patience=5,
        lr=0.001,
        batch_size=128
    )

    # Evaluate the fine-tuned compressed model on the test set
    print("Evaluating the fine-tuned compressed model on the test set:")
    test_model(best_model, testloader)

if __name__ == '__main__':
    main()