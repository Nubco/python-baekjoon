# scripts/preprocess_sorted_hairstyle60k.py
import os, shutil, random, json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
SRC_DIR = "datasets/hairstyle60k/sorted"
TRAIN_DIR = "datasets/hairstyle60k/processed/train"
VAL_DIR = "datasets/hairstyle60k/processed/val"
META_PATH = "datasets/hairstyle60k/metadata/label_map.json"
IMG_SIZE = (224, 224)
VAL_RATIO = 0.2

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

label_map = {}
for idx, label_name in enumerate(sorted(os.listdir(SRC_DIR))):
    label_map[label_name] = idx
    class_dir = os.path.join(SRC_DIR, label_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('jpg', 'png'))]
    random.shuffle(images)
    
    split = int(len(images) * (1 - VAL_RATIO))
    train_imgs, val_imgs = images[:split], images[split:]
    
    for target, img_list in zip([TRAIN_DIR, VAL_DIR], [train_imgs, val_imgs]):
        label_idx_dir = os.path.join(target, str(idx))
        os.makedirs(label_idx_dir, exist_ok=True)
        for img in img_list:
            try:
                img_path = os.path.join(class_dir, img)
                im = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                im.save(os.path.join(label_idx_dir, img))
            except Exception as e:
                print(f"[ERR] {img_path}: {e}")

with open(META_PATH, "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ 분류된 폴더 기반 전처리 완료.")
