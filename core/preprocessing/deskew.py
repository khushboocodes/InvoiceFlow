"""
InvoiceFlow Image Deskewing Engine
Hough Transform baseline detection and affine deskewing.
"""
import cv2
import numpy as np

def detect_and_deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detects horizontal text line skew angle using Canny + Hough Transform
    and applies affine rotation to rectify text baselines.
    Clamps correction within [-max_angle, max_angle].
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=100, 
        minLineLength=80, 
        maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return image  # No strong lines detected, return original

    angles = []
    for line in lines:
        coords = line.flatten() if hasattr(line, "flatten") else line[0]
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))

    # If skew is negligible (<0.3 degrees), skip interpolation
    if abs(median_angle) < 0.3:
        return image

    # Rotate around center
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new bounding box to prevent cropping corners
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, 
        rot_mat, 
        (new_w, new_h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    return rotated
