"""
Integral Image & Haar-Like Feature Calculation.
Core 4: Visual Perception Daemon (Eyes - v1.2.0)
Architecture §4.1:
Enables constant-time rectangle sums in 4 array lookups:
Sum(rect) = I(x2, y2) - I(x1, y2) - I(x2, y1) + I(x1, y1)
"""
import numpy as np
from typing import Tuple

def compute_integral_image(img: np.ndarray) -> np.ndarray:
    """
    Computes cumulative sum integral image with leading zero pad for O(1) bounds lookups.
    Returns array of shape (H + 1, W + 1).
    """
    if img.ndim == 3:
        # Convert RGB to grayscale: 0.299R + 0.587G + 0.114B
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)

    H, W = gray.shape
    integral = np.zeros((H + 1, W + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(gray, axis=0), axis=1)
    return integral

def rectangle_sum(
    integral: np.ndarray,
    r1: int,
    c1: int,
    r2: int,
    c2: int
) -> float:
    """
    Computes sum of pixel intensities in rectangle [r1, r2] x [c1, c2] in 4 lookups.
    Coordinates are 0-indexed inclusive.
    """
    # Offset by +1 for padded integral image
    bottom = r2 + 1
    right = c2 + 1
    top = r1
    left = c1

    return float(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )

def haar_two_rectangle_feature(
    integral: np.ndarray,
    r: int,
    c: int,
    w: int,
    h: int,
    horizontal: bool = True
) -> float:
    """
    Computes Haar-like 2-rectangle feature (white region - black region).
    """
    if horizontal:
        # Left white, right black
        mid = c + w // 2
        sum_white = rectangle_sum(integral, r, c, r + h - 1, mid - 1)
        sum_black = rectangle_sum(integral, r, mid, r + h - 1, c + w - 1)
    else:
        # Top white, bottom black
        mid = r + h // 2
        sum_white = rectangle_sum(integral, r, c, mid - 1, c + w - 1)
        sum_black = rectangle_sum(integral, mid, c, r + h - 1, c + w - 1)

    return sum_white - sum_black
