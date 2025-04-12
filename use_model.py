import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# Загружаем сохранённую модель
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 1)  # Такая же архитектура как при обучении
model.load_state_dict(torch.load('temperature_model.pth'))
model.eval()  # Переводим модель в режим оценки


def preprocess_image(image_path):
    # Те же трансформации, что использовались при обучении
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.513, 0.529, 0.509],
                             std=[0.211, 0.209, 0.266])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Добавляем batch dimension


def predict_temperature(model, image_path):
    # Предобработка
    input_tensor = preprocess_image(image_path)

    # Предсказание
    with torch.no_grad():
        output = model(input_tensor)

    return output.item()  # Возвращаем значение температуры

# Путь к тестовому изображению
test_image = "St.-Louis-Arch-Cam_1744476714438.jpg"

# Получаем предсказание
predicted_temp = predict_temperature(model, test_image)
print(f"Предсказанная температура: {predicted_temp:.2f}°C")