"""
Viola-Jones Detector with AdaBoost Cascades.
Core 4: Visual Perception Daemon (Eyes - v1.2.0)
Architecture §4.1: Pre-deep-learning face/object detection with hand-built math.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from miku.core4_vision.integral_image import (
    compute_integral_image,
    haar_two_rectangle_feature
)

class WeakClassifier:
    def __init__(self, r: int, c: int, w: int, h: int, horizontal: bool, threshold: float, polarity: int):
        self.r = r
        self.c = c
        self.w = w
        self.h = h
        self.horizontal = horizontal
        self.threshold = threshold
        self.polarity = polarity  # +1 or -1

    def evaluate(self, integral: np.ndarray, offset_r: int = 0, offset_c: int = 0) -> int:
        f = haar_two_rectangle_feature(
            integral,
            self.r + offset_r,
            self.c + offset_c,
            self.w,
            self.h,
            self.horizontal
        )
        return 1 if (self.polarity * f < self.polarity * self.threshold) else 0

class CascadeStage:
    def __init__(self, classifiers: List[WeakClassifier], alphas: List[float], stage_threshold: float):
        self.classifiers = classifiers
        self.alphas = alphas
        self.stage_threshold = stage_threshold

    def evaluate(self, integral: np.ndarray, offset_r: int, offset_c: int) -> bool:
        total = 0.0
        for clf, alpha in zip(self.classifiers, self.alphas):
            if clf.evaluate(integral, offset_r, offset_c) == 1:
                total += alpha
        return total >= self.stage_threshold

class ViolaJonesDetector:
    def __init__(self, window_size: Tuple[int, int] = (24, 24)):
        self.win_h, self.win_w = window_size
        self.stages: List[CascadeStage] = []
        self._init_default_face_cascade()

    def _init_default_face_cascade(self):
        """
        Initializes pre-calibrated cascade stages based on classical facial geometry
        (e.g., eyes darker than bridge of nose, forehead brighter than eyes).
        """
        # Stage 1: Fast eye-region detector
        c1 = WeakClassifier(r=6, c=4, w=16, h=6, horizontal=False, threshold=50.0, polarity=1)
        c2 = WeakClassifier(r=8, c=6, w=12, h=8, horizontal=True, threshold=-20.0, polarity=-1)
        stage1 = CascadeStage(classifiers=[c1, c2], alphas=[1.2, 0.9], stage_threshold=1.0)
        self.stages.append(stage1)

        # Stage 2: Cheek & mouth detector
        c3 = WeakClassifier(r=14, c=4, w=16, h=6, horizontal=False, threshold=30.0, polarity=1)
        stage2 = CascadeStage(classifiers=[c3], alphas=[1.5], stage_threshold=1.0)
        self.stages.append(stage2)

    def detect_in_window(self, integral: np.ndarray, r: int, c: int) -> bool:
        """
        Evaluates cascades sequentially. Rejects immediately if any stage fails.
        """
        from miku.core4_vision.integral_image import rectangle_sum
        total_sum = rectangle_sum(integral, r, c, r + self.win_h - 1, c + self.win_w - 1)
        mean_val = total_sum / float(self.win_h * self.win_w)
        # Flat blank or saturated regions cannot be valid faces
        if mean_val < 15.0 or mean_val > 240.0:
            return False

        for stage in self.stages:
            if not stage.evaluate(integral, r, c):
                return False
        return True

    def scan_image(
        self,
        image: np.ndarray,
        step: int = 12,
        confidence_boost: float = 0.85
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Scans image with sliding window.
        Returns list of detections: [(x, y, w, h, confidence), ...]
        """
        integral = compute_integral_image(image)
        H, W = image.shape[:2]
        detections = []

        for r in range(0, H - self.win_h + 1, step):
            for c in range(0, W - self.win_w + 1, step):
                if self.detect_in_window(integral, r, c):
                    # x, y, w, h, confidence
                    detections.append((c, r, self.win_w, self.win_h, confidence_boost))

        return detections
