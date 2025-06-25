"""
generate_hairstyle.py

학습 완료 후 저장된 Generator(`generator.pth`)를 불러와서
랜덤 노이즈(또는 샘플 이미지)를 입력으로 새로운 이미지를 생성하고 파일로 저장합니다.

- Generator 모델 정의는 hairstyle_transfer_gan.py 에서 가져옵니다.
- 저장된 가중치 파일은 프로젝트 루트 또는 지정 경로에 있어야 합니다.
"""

import os
import torch
from torchvision.utils import save_image
from hairstyle_transfer_gan import Generator

# 1. 설정
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.getcwd(), "backend", "model_training", "generator.pth")
OUTPUT_DIR   = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 모델 로드
G = Generator().to(DEVICE)
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(f"Generator 가중치가 없습니다: {WEIGHTS_PATH}")

checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
G.load_state_dict(checkpoint)
G.eval()

# 3. 이미지 생성 함수
def generate_and_save(num_samples=1, save_prefix="generated"):
    """
    랜덤 노이즈를 입력으로 num_samples개의 이미지를 생성하고 저장합니다.
    (Generator 구조가 3×64×64 크기의 노이즈를 받는 경우 가정)
    """
    with torch.no_grad():
        for i in range(num_samples):
            # 랜덤 노이즈 생성 (3채널 64×64)
            z = torch.randn(1, 3, 64, 64).to(DEVICE)
            fake_img = G(z)  # 출력: [1, 3, 64, 64], 픽셀 범위는 -1 ~ +1

            # save_image를 통해 [0,1] 범위로 정규화해서 이미지 저장
            out_name = f"{save_prefix}_{i+1}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            save_image(fake_img, out_path, normalize=True)
            print(f"저장됨: {out_path}")

if __name__ == "__main__":
    """
    사용 예)
    프로젝트 루트에서:
        python backend/image_generation/generate_hairstyle.py
    """
    generate_and_save(num_samples=4, save_prefix="hairstyle")
