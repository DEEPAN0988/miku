"""
HSV Thresholding and Connected Components Segmentation.
Core 4: Visual Perception Daemon (Eyes - v1.2.0)
Architecture §4.4:
Enables targeted clicking on UI elements/colored buttons:
mask(x,y) = 1 if H_lo <= Hue(x,y) <= H_hi AND S_lo <= Sat(x,y) <= S_hi
"""
import numpy as np
from typing import Dict, Any, List, Tuple

# Pre-calibrated HSV bounds (Hue in [0, 180], Sat in [0, 255], Val in [0, 255])
COLOR_BOUNDS = {
    "red": ((0, 100, 100), (10, 255, 255)),
    "red2": ((170, 100, 100), (180, 255, 255)),
    "blue": ((100, 100, 100), (130, 255, 255)),
    "green": ((40, 100, 100), (80, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255))
}

class HSVSegmentation:
    def __init__(self):
        pass

    def rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """
        Converts RGB [0, 255] uint8 array to HSV (H in [0, 180], S in [0, 255], V in [0, 255]).
        """
        arr = rgb.astype(np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        diff = cmax - cmin

        h = np.zeros_like(cmax)
        # Red max
        mask_r = (cmax == r) & (diff > 0)
        h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360.0) % 360.0
        # Green max
        mask_g = (cmax == g) & (diff > 0)
        h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120.0) % 360.0
        # Blue max
        mask_b = (cmax == b) & (diff > 0)
        h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240.0) % 360.0

        # Scale to standard 8-bit HSV
        h_scaled = (h / 2.0).astype(np.uint8)
        s_scaled = np.where(cmax == 0, 0, (diff / (cmax + 1e-8)) * 255.0).astype(np.uint8)
        v_scaled = (cmax * 255.0).astype(np.uint8)

        return np.stack([h_scaled, s_scaled, v_scaled], axis=2)

    def segment_color(self, rgb_image: np.ndarray, color_name: str) -> List[Dict[str, Any]]:
        """
        Segments image for specified color and groups into contiguous blobs.
        Returns list of regions: [{"centroid": (x, y), "bbox": (x, y, w, h), "area": int}, ...]
        """
        hsv = self.rgb_to_hsv(rgb_image)
        color_lower = color_name.lower()
        if color_lower not in COLOR_BOUNDS:
            return []

        lower, upper = COLOR_BOUNDS[color_lower]
        mask = (
            (hsv[:, :, 0] >= lower[0]) & (hsv[:, :, 0] <= upper[0]) &
            (hsv[:, :, 1] >= lower[1]) & (hsv[:, :, 1] <= upper[1]) &
            (hsv[:, :, 2] >= lower[2]) & (hsv[:, :, 2] <= upper[2])
        )

        if color_lower == "red":
            l2, u2 = COLOR_BOUNDS["red2"]
            mask2 = (
                (hsv[:, :, 0] >= l2[0]) & (hsv[:, :, 0] <= u2[0]) &
                (hsv[:, :, 1] >= l2[1]) & (hsv[:, :, 1] <= u2[1]) &
                (hsv[:, :, 2] >= l2[2]) & (hsv[:, :, 2] <= u2[2])
            )
            mask = mask | mask2

        # Connected components via scipy.ndimage or fallback
        from scipy.ndimage import label, find_objects
        labeled, num_features = label(mask)

        blobs = []
        slices = find_objects(labeled)
        for i, sl in enumerate(slices):
            if sl is None:
                continue
            r_slice, c_slice = sl
            blob_mask = (labeled[r_slice, c_slice] == (i + 1))
            area = int(np.sum(blob_mask))
            if area < 15:  # Ignore tiny noise
                continue

            ys, xs = np.nonzero(blob_mask)
            cy = int(r_slice.start + np.mean(ys))
            cx = int(c_slice.start + np.mean(xs))
            w = c_slice.stop - c_slice.start
            h = r_slice.stop - r_slice.start

            blobs.append({
                "label": f"{color_lower}_region",
                "centroid": (cx, cy),
                "bbox": (c_slice.start, r_slice.start, w, h),
                "area": area,
                "confidence": min(1.0, area / 200.0)
            })

        blobs.sort(key=lambda b: b["area"], reverse=True)
        return blobs
