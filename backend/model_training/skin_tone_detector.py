"""
skin_tone_detector.py

This script detects the skin tone of a face from an image by analyzing average skin region color.
"""

import cv2
import numpy as np

def detect_skin_tone(image_path):
    """
    Detects the average skin tone from the image.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        str: Skin tone category (e.g., 'Fair', 'Medium', 'Dark')
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to YCrCb color space
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Skin color range for YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    # Mask skin area
    mask = cv2.inRange(img_YCrCb, lower, upper)
    skin = cv2.bitwise_and(image, image, mask=mask)

    # Calculate average color of the skin region
    skin_pixels = skin[np.where(mask != 0)]
    if skin_pixels.size == 0:
        raise RuntimeError("No skin region detected.")

    avg_color = np.mean(skin_pixels, axis=0)

    # Simple rule-based skin tone classification using brightness
    brightness = np.mean(cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[...,2])

    if brightness > 170:
        return "Fair"
    elif brightness > 100:
        return "Medium"
    else:
        return "Dark"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect skin tone from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input face image.")
    args = parser.parse_args()

    try:
        skin_tone = detect_skin_tone(args.image_path)
        print(f"Detected skin tone: {skin_tone}")
    except Exception as e:
        print(f"Error: {e}")
