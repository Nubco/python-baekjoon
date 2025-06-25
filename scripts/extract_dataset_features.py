import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch

def save_dataset_features(model, dataset_path, save_path, device):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    features_list = []
    img_paths = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = extract_features(model, imgs)  # 배치 단위 특징 추출
            features_list.append(feats.cpu())
    
    features = torch.cat(features_list, dim=0)  # (N,512)
    # 이미지 경로 목록도 저장
    img_paths = [dataset.samples[i][0] for i in range(len(dataset))]

    torch.save({'features': features, 'img_paths': img_paths, 'classes': dataset.classes}, save_path)
    print(f"✅ 데이터셋 feature 저장 완료: {save_path}")
