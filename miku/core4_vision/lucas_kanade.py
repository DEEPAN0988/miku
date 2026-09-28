"""
Frame Differencing and Lucas-Kanade Optical Flow.
Core 4: Visual Perception Daemon (Eyes - v1.2.0)
Architecture §4.2:
Solves 2x2 system:
[ sum(Ix^2)   sum(Ix*Iy) ] [ u ] = [ -sum(Ix*It) ]
[ sum(Ix*Iy)  sum(Iy^2)  ] [ v ]   [ -sum(Iy*It) ]
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional

class LucasKanadeMotionDetector:
    def __init__(self, tau: float = 25.0, window_size: int = 5):
        self.tau = tau
        self.win = window_size
        self.prev_frame: Optional[np.ndarray] = None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        return frame.astype(np.float64)

    def solve_optical_flow_at_point(
        self,
        Ix: np.ndarray,
        Iy: np.ndarray,
        It: np.ndarray,
        r: int,
        c: int
    ) -> Tuple[float, float]:
        """
        Solves 2x2 least-squares system in window around (r, c).
        """
        half = self.win // 2
        r_start = max(0, r - half)
        r_end = min(Ix.shape[0], r + half + 1)
        c_start = max(0, c - half)
        c_end = min(Ix.shape[1], c + half + 1)

        win_Ix = Ix[r_start:r_end, c_start:c_end].flatten()
        win_Iy = Iy[r_start:r_end, c_start:c_end].flatten()
        win_It = It[r_start:r_end, c_start:c_end].flatten()

        A11 = float(np.sum(win_Ix ** 2))
        A12 = float(np.sum(win_Ix * win_Iy))
        A22 = float(np.sum(win_Iy ** 2))

        b1 = float(-np.sum(win_Ix * win_It))
        b2 = float(-np.sum(win_Iy * win_It))

        det = A11 * A22 - A12 * A12
        if abs(det) < 1e-6:
            return 0.0, 0.0  # Singular or aperture problem

        u = (b1 * A22 - b2 * A12) / det
        v = (A11 * b2 - A12 * b1) / det
        return float(u), float(v)

    def process_frame(self, cur_frame: np.ndarray) -> Dict[str, Any]:
        """
        Performs frame differencing + Lucas-Kanade optical flow on motion regions.
        """
        gray = self._to_gray(cur_frame)
        if self.prev_frame is None or self.prev_frame.shape != gray.shape:
            self.prev_frame = gray
            return {
                "motion_detected": False,
                "motion_area": 0,
                "velocity": (0.0, 0.0),
                "centroid": (0, 0)
            }

        # 1. Frame differencing: D = |I(t) - I(t-1)| > tau
        diff = np.abs(gray - self.prev_frame)
        motion_mask = diff > self.tau
        motion_count = int(np.sum(motion_mask))

        if motion_count < 20:
            self.prev_frame = gray
            return {
                "motion_detected": False,
                "motion_area": motion_count,
                "velocity": (0.0, 0.0),
                "centroid": (0, 0)
            }

        # Compute spatio-temporal gradients
        # Ix, Iy via Sobel / central difference
        Ix = np.zeros_like(gray)
        Iy = np.zeros_like(gray)
        Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
        Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
        It = gray - self.prev_frame

        # Centroid of motion
        ys, xs = np.nonzero(motion_mask)
        mean_y = int(np.mean(ys))
        mean_x = int(np.mean(xs))

        # Solve Lucas-Kanade at the motion centroid
        u, v = self.solve_optical_flow_at_point(Ix, Iy, It, mean_y, mean_x)

        self.prev_frame = gray
        return {
            "motion_detected": True,
            "motion_area": motion_count,
            "velocity": (round(u, 2), round(v, 2)),
            "centroid": (mean_x, mean_y)
        }
