import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os



# Класс датасета для работы с CSV
class CSVTemperatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Путь к CSV-файлу
            root_dir (string): Директория с изображениями
            transform (callable, optional): Трансформации для изображений
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        temperature = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(temperature, dtype=torch.float32)


# Параметры
CSV_FILE = "dataset/temperatures.csv"
ROOT_DIR = "dataset/images"
BATCH_SIZE = 32
EPOCHS = 10

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Создание датасета и загрузчика
full_dataset = CSVTemperatureDataset(
    csv_file=CSV_FILE,
    root_dir=ROOT_DIR,
    transform=transform
)

# Разделение на train/val
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# Модель
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Тренировочная эпоха
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    # Статистика
    epoch_loss = running_loss / len(train_loader.dataset)
    val_epoch_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
    print("-" * 50)

# Сохранение модели
torch.save(model.state_dict(), "temperature_model.pth")