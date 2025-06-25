# 파일명: scripts/train_face_shape_classifier.py

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse

def train_face_shape_classifier(train_dir, val_dir, save_path, batch_size=32, epochs=15, lr=1e-4, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 데이터셋 및 로더
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 모델 초기화
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_dataset)
        train_acc = correct / len(train_dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # 가장 좋은 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ 모델 저장: {save_path}")

    print(f"최종 검증 정확도: {best_val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train face shape classifier')
    parser.add_argument('--train_dir', type=str, default='datasets/face_shape/train', help='Train data folder')
    parser.add_argument('--val_dir', type=str, default='datasets/face_shape/val', help='Validation data folder')
    parser.add_argument('--save_path', type=str, default='data/checkpoints/face_shape_classifier.pth', help='Model save path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    train_face_shape_classifier(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
    )
