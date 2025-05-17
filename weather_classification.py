import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets  # Добавлен импорт datasets
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm


# Конфигурация
class Config:
    data_dir = "classification dataset"
    classes = ["-10 - 0", "-30 - -10", "0 - 10", "10 - 30", "30 - 50"]
    batch_size = 32
    num_epochs = 15
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Проверка датасета
print("Проверка структуры датасета:")
for cls in Config.classes:
    path = Path(Config.data_dir) / cls
    count = sum(1 for _ in path.glob('*/*') if _.is_file())
    print(f"{cls}: {count} изображений")

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка данных
try:
    full_dataset = datasets.ImageFolder(root=Config.data_dir, transform=transform)
    print("\nСоответствие классов:", full_dataset.class_to_idx)
except Exception as e:
    print(f"\nОшибка загрузки данных: {e}")
    exit()

# Разделение данных
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

# Модель
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(Config.classes))
model = model.to(Config.device)

# Оптимизатор и критерий
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# Обучение
def train():
    best_acc = 0.0
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")

        # Тренировка
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total

        # Валидация
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

        print(f"Train Loss: {train_loss / len(train_loader):.3f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss / len(val_loader):.3f} | Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "classification_model.pth")

        scheduler.step()

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()