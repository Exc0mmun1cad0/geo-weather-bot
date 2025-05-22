import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class TemperaturePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["-30 - -10", "-10 - 0", "0 - 10", "10 - 30"]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classifier = self._load_classifier("classification_model.pth")
        self.regressors = self._load_regressors()

    def _load_classifier(self, model_path):
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Remove incompatible final layer weights if class count differs
            for key in ['classifier.3.weight', 'classifier.3.bias']:
                if key in state_dict and state_dict[key].shape[0] != len(self.classes):
                    del state_dict[key]

            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error during classifier loading: {e}")
            exit()

        model = model.to(self.device)
        model.eval()
        return model

    def _load_regressors(self):
        """Load regression models for each range"""
        regressors = {}
        range_files = {
            "0 - 10": "0-10_model.pth",
            "10 - 30": "10-30_model.pth",
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

    def predict(self, image_path):
        """Full prediction process"""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 1. Range classification
            with torch.no_grad():
                class_output = self.classifier(img_tensor)
                _, class_idx = torch.max(class_output, 1)
                predicted_range = self.classes[class_idx.item()]

            # 2. Regression of the exact temperature
            temperature = None
            if predicted_range in self.regressors:
                with torch.no_grad():
                    temp_pred = self.regressors[predicted_range](img_tensor)
                    temperature = temp_pred.item()

                    # Limit the temperature to the limits of the range
                    min_temp, max_temp = map(int, predicted_range.split(" - "))
                    temperature = max(min_temp, min(max_temp, temperature))
                    temperature = round(temperature, 1)

            return {
                "range": predicted_range,
                "temperature": temperature,
                "image": image_path
            }

        except Exception as e:
            print(f"Error handling {image_path}: {e}")
            return None


if __name__ == "__main__":
    # predictor init
    predictor = TemperaturePredictor()

    # usage
    result = predictor.predict("punk.jpg")

    if result:
        if result["temperature"] is not None:
            print(f"Range: {result['range']}, temperature: {result['temperature']}°C")
        else:
            print(f"Range: {result['range']}, regression failed")
    else:
        print("Failed to predict")
