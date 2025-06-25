# train_hairstyle_gan.py

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from hairstyle_transfer_gan import Generator, Discriminator

# ✅ 1. 하이퍼파라미터
batch_size = 16
lr = 0.0002
epochs = 50
image_size = 64
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 데이터셋 로딩 (여기선 ./data/ 폴더 기준)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = torchvision.datasets.ImageFolder(root='./data', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ 3. 모델, 손실함수, 옵티마이저
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ✅ 4. 학습 루프
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size_curr = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = torch.ones(batch_size_curr, 1).to(device)
        fake_labels = torch.zeros(batch_size_curr, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        z = torch.randn(batch_size_curr, latent_dim, 1, 1).to(device)
        fake_images = G(z)

        real_loss = criterion(D(real_images), real_labels)
        fake_loss = criterion(D(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        g_loss = criterion(D(fake_images), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"[Epoch {epoch+1}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # 예시 이미지 저장
    torchvision.utils.save_image(fake_images[:16], f'outputs/fake_{epoch+1}.png', normalize=True)

# 최종 모델 저장
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
