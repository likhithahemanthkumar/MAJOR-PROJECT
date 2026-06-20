import cv2
import numpy as np

def clean_fundus_image(img_array):
    """Isolates the green channel and applies CLAHE for structural vessel visibility."""
    # Convert back to uint8 if scaled
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
        
    # Split channels and isolate Green channel
    b, g, r = cv2.split(img_array)
    
    # Apply Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(g)
    
    # Convert back to 3 channels so ResNet can process it
    processed_img = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    processed_img = cv2.resize(processed_img, (224, 224))
    return processed_img.astype(np.float32) / 255.0

def clean_oct_image(img_array):
    """Applies noise reduction filtering to clean cross-sectional speckle artifact noise."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
        
    # Apply bilateral filter to remove noise while keeping tissue layer lines sharp
    denoised = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    processed_img = cv2.resize(denoised, (224, 224))
    return processed_img.astype(np.float32) / 255.0