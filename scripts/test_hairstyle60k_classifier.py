import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt


def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, transform, device, class_names):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()], image


def show_prediction_with_example(input_image, pred_class, example_image_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(input_image)
    axes[0].set_title("input image")
    axes[0].axis('off')

    example_img = Image.open(example_image_path).convert('RGB')
    axes[1].imshow(example_img)
    axes[1].set_title(f"Recommended Style: {pred_class}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'data/checkpoints/hairstyle60k_cnn.pth'
    data_dir = 'datasets/hairstyle60k/processed'
    class_names = sorted(os.listdir(os.path.join(data_dir, 'train')))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    model = load_model(model_path, len(class_names), device)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python test_hairstyle60k_classifier.py [image_path]")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"Error: 이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    print("Class names:", class_names)

    pred_class, input_image = predict_image(model, image_path, transform, device, class_names)
    print(f"Predicted hairstyle class: {pred_class}")

    example_dir = os.path.join(data_dir, 'train', pred_class)
    example_images = [f for f in os.listdir(example_dir) if f.lower().endswith(('.jpg', '.png'))]
    if example_images:
        example_path = os.path.join(example_dir, example_images[0])
        show_prediction_with_example(input_image, pred_class, example_path)
    else:
        print(f"예시 이미지를 찾을 수 없습니다: {example_dir}")
