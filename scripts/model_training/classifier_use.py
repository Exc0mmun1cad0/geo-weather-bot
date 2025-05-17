import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


class TemperatureClassifier:
    def __init__(self, model_path="classification_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["-10 - 0", "-30 - -10", "0 - 10", "10 - 30", "30 - 50"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Загрузка предобученной модели"""
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        """Предсказание температурного диапазона для одного изображения"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        return self.classes[predicted.item()]

    def predict_batch(self, images_dir, output_file="predictions.csv"):
        """Пакетная обработка изображений в директории"""
        image_paths = [p for p in Path(images_dir).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        with open(output_file, "w") as f:
            f.write("image,temperature_range\n")
            for img_path in image_paths:
                try:
                    pred = self.predict(str(img_path))
                    f.write(f"{img_path.name},{pred}\n")
                    print(f"{img_path.name}: {pred}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")


if __name__ == "__main__":
    # Пример использования
    classifier = TemperatureClassifier()

    # 1. Предсказание для одного изображения
    single_pred = classifier.predict("0668.jpg")
    print(f"\nPredicted temperature range: {single_pred}")

    # 2. Пакетная обработка директории
    print("\nProcessing directory...")
    classifier.predict_batch("test_images")