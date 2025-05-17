import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class TemperaturePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["-30 - -10", "-10 - 0", "0 - 10", "10 - 30", "30 - 50"]
        self.transform = self._get_transforms()
        self.classifier = self._load_classifier("classification_model.pth")
        self.regressors = self._load_regressors()

    def _load_classifier(self, model_path):
        """Загрузка классификатора диапазонов"""
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.classes))

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Удаляем несовместимые веса последнего слоя
            for key in ['classifier.1.weight', 'classifier.1.bias']:
                if key in state_dict and state_dict[key].shape[0] != len(self.classes):
                    del state_dict[key]

            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Ошибка загрузки классификатора: {e}")
            exit()

        model = model.to(self.device)
        model.eval()
        return model

    def _load_regressors(self):
        """Загрузка регрессионных моделей для каждого диапазона"""
        regressors = {}
        range_files = {
            "0 - 10": "0-10_model.pth",
            "10 - 30": "10-30_model.pth",
            "30 - 50": "30-50_model.pth",
            "-10 - 0": "-10-0_model.pth",
            "-30 - -10": "-30--10_model.pth"
        }

        for range_name, file_name in range_files.items():
            try:
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)  # Регрессия на 1 значение

                state_dict = torch.load(file_name, map_location=self.device)
                model.load_state_dict(state_dict)

                model = model.to(self.device)
                model.eval()
                regressors[range_name] = model
                print(f"Успешно загружен регрессор для {range_name}")
            except Exception as e:
                print(f"Не удалось загрузить {file_name}: {e}")

        return regressors

    def _get_transforms(self):
        """Общие трансформации для изображений"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """Полный процесс предсказания"""
        try:
            # Загрузка и преобразование изображения
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 1. Классификация диапазона
            with torch.no_grad():
                class_output = self.classifier(img_tensor)
                _, class_idx = torch.max(class_output, 1)
                predicted_range = self.classes[class_idx.item()]

            # 2. Регрессия для точной температуры
            temperature = None
            if predicted_range in self.regressors:
                with torch.no_grad():
                    temp_pred = self.regressors[predicted_range](img_tensor)
                    temperature = temp_pred.item()

                    # Ограничиваем температуру границами диапазона
                    min_temp, max_temp = map(int, predicted_range.split(" - "))
                    temperature = max(min_temp, min(max_temp, temperature))
                    temperature = round(temperature, 1)

            return {
                "range": predicted_range,
                "temperature": temperature,
                "image": image_path
            }

        except Exception as e:
            print(f"Ошибка обработки {image_path}: {e}")
            return None


if __name__ == "__main__":
    # Инициализация предсказателя
    predictor = TemperaturePredictor()

    # Пример использования
    result = predictor.predict("punk1.jpg")

    if result:
        if result["temperature"] is not None:
            print(f"Результат: {result['range']}, Температура: {result['temperature']}°C")
        else:
            print(f"Результат: {result['range']} (регрессор не доступен)")
    else:
        print("Не удалось выполнить предсказание")