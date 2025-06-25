"""
face_shape_classifier.py

This script handles face shape classification using MediaPipe for landmark detection
and a simple rule-based heuristic or a machine learning model for classification.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                  refine_landmarks=True, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    """
    Extracts facial landmarks from an image using MediaPipe Face Mesh.

    Args:
        image_path (str): Path to the input image.

    Returns:
        landmarks (np.ndarray): Array of (x, y, z) coordinates for detected landmarks.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected in the image.")

    # Extract landmarks for the first detected face
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    landmarks = []

    for landmark in face_landmarks.landmark:
        landmarks.append((landmark.x * w, landmark.y * h, landmark.z * w))

    return np.array(landmarks)

def classify_face_shape(landmarks):
    """
    Classify the face shape based on landmark ratios/heuristics.

    Args:
        landmarks (np.ndarray): Array of (x, y, z) facial landmarks.

    Returns:
        str: Detected face shape (e.g., 'Oval', 'Round', 'Square', 'Heart', 'Diamond').
    """
    # Placeholder heuristic: width to height ratio
    # TODO: Replace with a trained ML model or a more robust rule-based approach
    left_cheek = landmarks[234]  # Example landmark index
    right_cheek = landmarks[454]  # Example landmark index
    top_forehead = landmarks[10]  # Example landmark index
    chin = landmarks[152]  # Example landmark index

    width = np.linalg.norm(np.array(left_cheek[:2]) - np.array(right_cheek[:2]))
    height = np.linalg.norm(np.array(top_forehead[:2]) - np.array(chin[:2]))

    ratio = width / height

    if 0.9 < ratio < 1.1:
        return 'Round'
    elif ratio >= 1.1:
        return 'Square'
    else:
        return 'Oval'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify face shape from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input face image.")
    args = parser.parse_args()

    try:
        landmarks = extract_landmarks(args.image_path)
        shape = classify_face_shape(landmarks)
        print(f"Detected face shape: {shape}")
    except Exception as e:
        print(f"Error: {e}")
