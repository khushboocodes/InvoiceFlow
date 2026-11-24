"""
InvoiceFlow Contrast Enhancement & Normalization
Applies CLAHE on Lab L-channel and illumination flattening.
"""
import cv2
import numpy as np

def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """
    Normalizes low-contrast scans and carbon copies using CLAHE in Lab space,
    edge-preserving bilateral denoising, and unsharp mask sharpening.
    Significantly boosts OCR and vision token resolution across all invoices.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    # Recombine and convert back to BGR
    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    contrast_enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Edge-preserving denoising + unsharp mask sharpening
    denoised = cv2.bilateralFilter(contrast_enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(denoised, 1.35, blur, -0.35, 0)

    return sharpened

def flatten_background_illumination(image: np.ndarray) -> np.ndarray:
    """
    Removes mobile camera shadow gradients via morphological background division.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Approximate background using large morphological dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = cv2.medianBlur(background, 21)

    # Divide foreground by background to normalize illumination
    diff = 255 - cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return norm
