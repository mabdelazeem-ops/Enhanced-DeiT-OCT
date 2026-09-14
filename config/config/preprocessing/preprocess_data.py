import cv2
import numpy as np
import tensorflow as tf

def apply_clahe(image_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to OCT B-scans."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_gray)

def crop_retinal_roi(image_gray, threshold_val=20):
    """Crop non-informative background regions from retinal B-scans."""
    _, thresh = cv2.threshold(image_gray, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = image_gray[y:y+h, x:x+w]
        return cropped
    return image_gray

def preprocess_oct_scan(image_path, target_size=(224, 224)):
    """Full preprocessing pipeline for a single OCT B-scan."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    cropped = crop_retinal_roi(image)
    enhanced = apply_clahe(cropped)
    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert grayscale to 3-channel RGB for DeiT/CNN compatibility
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0
    return normalized_image
