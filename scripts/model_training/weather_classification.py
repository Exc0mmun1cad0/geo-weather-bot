import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from torchvision.models import ConvNeXt_Tiny_Weights
from tqdm import tqdm


class Config:
    data_dir = "classification dataset"
    classes = ["-10 - 0", "-30 - -10", "0 - 10", "10 - 30"]
    batch_size = 8
    num_epochs = 5
    lr = 0.001  # learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations (according to the ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # convert to pytorch tensor and normalise pixel values
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # normalization of the image using RGB channels
    ])


def print_dataset_structure():
    print("Dataset structure:")
    for cls in Config.classes:
        path = Path(Config.data_dir) / cls
        count = sum(1 for _ in path.glob('*/*') if _.is_file())
        print(f"{cls}: {count} images")


# Data load
def load_data(transform):
    try:
        full_dataset = datasets.ImageFolder(root=Config.data_dir, transform=transform)
        print("\nDict: class -> num_images:", full_dataset.class_to_idx)
    except Exception as e:
        print(f"\nData load err: {e}")
        exit()

    return full_dataset


def split_data(full_dataset):
    # Data split (train/val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset


def train(model, train_loader, val_loader, optimizer, criterion, scheduler):
    best_acc = 0.0
    train_losses = []
    train_accuracies = []

    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()  # set grad to zero
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # back propagation
            optimizer.step()  # renew model weights

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Train Los.s: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Train Loss: {train_loss / len(train_loader):.3f} | Acc: {epoch_acc:.2f}%")
        print(f"Val Loss: {val_loss / len(val_loader):.3f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "classification_model.pth")

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

    return train_losses, train_accuracies


def plot_results(train_losses, train_accuracies):
    # plot graphs
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Loss", color="red")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Accuracy", color="green")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("figures\\convnext_tiny.png")
    plt.show()


def main():
    print_dataset_structure()

    # Transformations (according to the ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # convert to pytorch tensor and normalise pixel values
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # normalization of the image using RGB channels
    ])

    full_dataset = load_data(Config.transform)

    train_dataset, val_dataset = split_data(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    # Model
    # model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT)
    # model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    # model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # in_features - number of features to the last layer's neuron
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(Config.classes))
    model = model.to(Config.device)

    # Optimizer and Criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    # Scheduler controls learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, train_accuracies = train(model, train_loader, val_loader, optimizer, criterion, scheduler)
    plot_results(train_losses, train_accuracies)

if __name__ == "__main__":
    main()
