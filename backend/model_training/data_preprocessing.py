"""
data_preprocessing.py

Raw 이미지(얼굴, 헤어스타일 등)를 불러와서 모델 학습용으로 전처리 후 저장합니다.

- 원본 이미지 폴더: data/raw/faces/, data/raw/hairstyles/
- 전처리 후 이미지 폴더: data/processed/faces/, data/processed/hairstyles/
"""

import os
from PIL import Image

# 1. 경로 설정
RAW_DIR       = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
SUBDIRS = ["faces", "hairstyles"]

# 2. 전처리 함수
def preprocess_image(img_path, save_path, size=(64, 64)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    img.save(save_path)

# 3. 메인 루프: 원본 → 전처리
def main():
    for sub in SUBDIRS:
        raw_subdir       = os.path.join(RAW_DIR, sub)
        processed_subdir = os.path.join(PROCESSED_DIR, sub)
        os.makedirs(processed_subdir, exist_ok=True)

        if not os.path.isdir(raw_subdir):
            print(f"[Warning] Raw 폴더가 없습니다: {raw_subdir}")
            continue

        for fname in os.listdir(raw_subdir):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            raw_path = os.path.join(raw_subdir, fname)
            save_name = os.path.splitext(fname)[0] + ".png"
            save_path = os.path.join(processed_subdir, save_name)
            try:
                preprocess_image(raw_path, save_path)
                print(f"Processed: {raw_path} → {save_path}")
            except Exception as e:
                print(f"[Error] {raw_path}: {e}")

if __name__ == "__main__":
    main()
