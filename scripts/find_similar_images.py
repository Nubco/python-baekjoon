import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

def extract_features(model, x):
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        features = torch.flatten(x, 1)
    return features

def find_similar_images(model, input_image_path, features_data_path, device, topk=5):
    print("Loading features data...")
    data = torch.load(features_data_path)
    dataset_features = data['features'].to(device)  # tensor (N,512)
    img_paths = data['img_paths']
    class_names = data['classes']

    print("Preprocessing input image and extracting features...")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        input_feature = extract_features(model, input_tensor)  # (1,512)

    # Normalize features for cosine similarity
    input_feature = nn.functional.normalize(input_feature, dim=1)
    dataset_features = nn.functional.normalize(dataset_features, dim=1)

    print("Calculating similarities...")
    similarities = torch.matmul(dataset_features, input_feature.T).squeeze(1)  # (N,)
    topk_vals, topk_idxs = torch.topk(similarities, k=topk, largest=True)

    print(f"Top {topk} similar images:")
    for rank, idx in enumerate(topk_idxs):
        img_path = img_paths[idx]
        class_name = os.path.basename(os.path.dirname(img_path))
        print(f"{rank+1}. {img_path} - 클래스: {class_name} - 유사도: {topk_vals[rank]:.4f}")

    print("Visualizing results...")
    plt.figure(figsize=(15,5))
    plt.subplot(1, topk+1, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    for i, idx in enumerate(topk_idxs):
        sim_img = Image.open(img_paths[idx])
        plt.subplot(1, topk+1, i+2)
        plt.imshow(sim_img)
        plt.title(f"Top {i+1}")
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 (features_extract.py에서 사용한 동일한 모델 구조여야 함)
    model_path = 'data/checkpoints/hairstyle60k_cnn.pth'
    from torchvision import models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 61)  # 클래스 수 맞게 수정 필요
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 특징 데이터 경로
    features_data_path = 'data/features/features_data.pt'

    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
    else:
        print("Usage: python scripts/find_similar_images.py [input_image_path]")
        sys.exit(1)

    if not os.path.isfile(input_image_path):
        print(f"Error: 이미지 파일을 찾을 수 없습니다: {input_image_path}")
        sys.exit(1)

    find_similar_images(model, input_image_path, features_data_path, device, topk=5)
