import torch
from torchvision import models, transforms
from PIL import Image

# Class names — must match the order used during training
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Use CPU
device = torch.device("cpu")

# Create the same MobileNetV3 Small architecture
model = models.mobilenet_v3_small(weights=None)

# Change final layer for 6 classes
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    6
)

# Load our trained model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "wastewise_mobilenetv3_best.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# Image preprocessing — same normalization used during testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_class = class_names[predicted_index.item()]
    confidence = confidence.item()

    return predicted_class, confidence


if __name__ == "__main__":

    # Change this later to an actual image path
    image_path = "test.jpg"

    predicted_class, confidence = predict_image(image_path)

    print("Predicted Waste Type:", predicted_class)
    print("Confidence:", round(confidence * 100, 2), "%")