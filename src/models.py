import torch
import torch.nn as nn

# Define the ReducedVGG model
class ReducedVGG(nn.Module):
    def __init__(self, num_classes=100):
        super(ReducedVGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> after pool 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16 -> after pool 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8 -> after pool 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_topk_accuracy(output, target, topk=(1, 5)):
    """Compute the top-k accuracy for model outputs."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracies.append((correct_k / batch_size).item() * 100)
        return accuracies

def test_model(model, testloader=None, device=None):
    """Evaluate the model on the test set and print Top-1 and Top-5 accuracy."""
    # If device or testloader are not provided, use defaults
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if testloader is None:
        import torchvision
        import torchvision.transforms as transforms
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model.eval()
    total_samples = 0
    top1_correct, top5_correct = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            top1_acc, top5_acc = compute_topk_accuracy(outputs, labels, topk=(1, 5))
            total_samples += labels.size(0)
            top1_correct += (top1_acc / 100) * labels.size(0)
            top5_correct += (top5_acc / 100) * labels.size(0)
    final_top1_acc = (top1_correct / total_samples) * 100
    final_top5_acc = (top5_correct / total_samples) * 100
    print(f"Top-1 Accuracy: {final_top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {final_top5_acc:.2f}%")
    return final_top1_acc, final_top5_acc