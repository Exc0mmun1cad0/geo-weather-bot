import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Disable warnings
warnings.filterwarnings('ignore')


class EnhancedTempModel(nn.Module):
    """Model representation layer by layer"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    # Pass input through feature, then regression, and remove extra dimensions
    def forward(self, x):
        x = self.features(x)
        return self.regressor(x).squeeze()


class CSVTemperatureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Dataset for uploading images and temperatures from CSV
        Args:
            csv_file (str): CSV file path
            root_dir (str): Img dir
            transform (callable, optional): Img transformations
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.valid_indices = self._validate_files()

        # Temperature statistics
        self.min_temp = float('inf')
        self.max_temp = float('-inf')
        self._calculate_temp_stats()

    def _validate_files(self):
        """Проверка доступности файлов изображений"""
        valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.root_dir, str(row.iloc[0]).strip())
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found - {img_path}")
        return valid_indices

    def _calculate_temp_stats(self):
        """Вычисление статистики по температуре"""
        for idx in self.valid_indices:
            temp = float(self.df.iloc[idx, 1])
            if temp < self.min_temp:
                self.min_temp = temp
            if temp > self.max_temp:
                self.max_temp = temp

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_name = str(self.df.iloc[real_idx, 0]).strip()
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            temperature = float(self.df.iloc[real_idx, 1])

            # Уберите нормализацию температуры - оставьте исходные значения
            # temperature = (temperature - self.min_temp) / (self.max_temp - self.min_temp)

            if self.transform:
                image = self.transform(image)

            if not (10.0 <= temperature <= 30.0):  # Ваш диапазон
                print(f"Invalid temperature {temperature} in {img_name}")
                temperature = 40.0  # Замените на среднее

                # Визуализация для отладки
                if idx == 0:  # Показать первое изображение
                    plt.imshow(image)
                    plt.title(f"Temp: {temperature}")
                    plt.show()

            return image, torch.tensor([temperature], dtype=torch.float32)

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            return torch.zeros(3, 224, 224), torch.tensor([15.0])


def get_dataloaders(csv_file, root_dir, transform, batch_size=32, val_ratio=0.2):
    """
    Creates DataLoaders for training and validation
    Args:
        csv_file (str): The path to the CSV file
        root_dir (str): Image directory
        transform (callable): Image transformations
        batch_size (int): Batch size
        val_ratio (float): Validation ratio
    Returns:
        train_loader, val_loader: Dataloaders for training and validation
    """
    full_dataset = CSVTemperatureDataset(csv_file, root_dir, transform)

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty! Check your paths and CSV file.")

    # Data split
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        # for fast loading with GPU
        pin_memory=True,
        # workers' life matter
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        # don't shuffle validation data for stability
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Debug info print
    print("\n" + "=" * 50)
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature range: {full_dataset.min_temp:.2f} to {full_dataset.max_temp:.2f}")
    print("=" * 50 + "\n")

    return train_loader, val_loader


def initialize_model(device, model_name='simple'):
    """
    Initializing a model with support for different architectures
    Args:
        device: 'cuda' or 'cpu'
        model_name: 'simple', 'resnet18' or 'efficientnet'
    """
    if model_name == 'simple':
        model = EnhancedTempModel()
    elif model_name == 'resnet18':
        # Load ResNet18
        model = models.resnet18(pretrained=True)
        # Replace the last layer. 1 for regressor
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'efficientnet':
        # No comments
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                         f"Available options: 'simple', 'resnet18', 'efficientnet'")

    # Weights for last layer
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) # helps to avoid problems with disappearing/exploding gradients
            nn.init.constant_(m.bias, 0.0)

    if model_name in ['resnet18', 'efficientnet']:
        model.fc.apply(init_weights) if model_name == 'resnet18' else model.classifier[1].apply(init_weights)

    return model.to(device)


def evaluate(model, val_loader, criterion, device):
    """Loss calculation on validation set"""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            val_loss += criterion(outputs, labels).item()

    return val_loss  # Avg loss by all batches


def train_model(model, train_loader, val_loader, device, config):
    """Train function with configuration verification"""
    # Params check
    required_keys = ['lr', 'weight_decay', 'patience', 'min_delta', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'] # regularisation coef
    )

    best_val_loss = float('inf')
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': []} # statistics

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience']
    )

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # renew weights

            train_loss += loss.item()

        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)

        # History save
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "10-30_test_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # get lr from optimiser's hyperparams
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    return model, history


def plot_results(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures\\training_history_efficientnet.png')
    plt.close()


def main():
    config = {
        'csv_file': 'reg_data\\10 - 30\\images.csv',
        'root_dir': 'reg_data\\10 - 30\\images',
        'batch_size': 4,
        'epochs': 10,
        'lr': 1e-5,
        'weight_decay': 1e-4, # L2 regularisation
        'patience': 10, # little data at first
        'min_delta': 0.001,  # early stop
        'model_name': 'efficientnet',
        'scheduler': {
            'name': 'plateau',
            'mode': 'min',
            'factor': 0.1,
            'patience': 5
        }
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Img transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data load
    train_loader, val_loader = get_dataloaders(
        config['csv_file'],
        config['root_dir'],
        train_transform,
        batch_size=config['batch_size']
    )

    model = initialize_model(device, config['model_name'])
    print(f"\nModel architecture:\n{model}\n")

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        config
    )

    plot_results(history)
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
