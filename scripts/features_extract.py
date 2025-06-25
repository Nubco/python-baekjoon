import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFile
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = 'datasets/hairstyle60k/processed/train'  # 학습용 데이터 경로 (클래스별 폴더 구조)
    batch_size = 64
    num_workers = 4
    save_path = 'data/features/features_data.pt'

    # 이미지 전처리 (ResNet18 기준)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ResNet18 모델 로드 및 fc 이전 feature 추출 준비
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device)
    model.eval()

    all_features = []
    all_img_paths = []
    class_names = dataset.classes  # 클래스명 리스트

    print(f"총 이미지 수: {len(dataset)}")
    print(f"클래스 수: {len(class_names)}")

    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm.tqdm(dataloader, desc='Extracting Features'):
            batch_imgs = batch_imgs.to(device)
            features = extract_features(model, batch_imgs)  # (batch, 512)
            features = nn.functional.normalize(features, dim=1)  # 정규화
            all_features.append(features.cpu())

            # 이미지 경로 수집
            start_idx = len(all_img_paths)
            for i in range(len(batch_labels)):
                # ImageFolder의 samples 리스트에서 경로 가져오기
                img_path, _ = dataset.samples[start_idx + i]
                all_img_paths.append(img_path)

    all_features = torch.cat(all_features, dim=0)  # (N, 512)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'features': all_features,
        'img_paths': all_img_paths,
        'classes': class_names
    }, save_path)

    print(f"✅ 특징 데이터 저장 완료: {save_path}")

if __name__ == '__main__':
    main()
