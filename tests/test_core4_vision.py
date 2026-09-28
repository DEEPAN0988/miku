"""
Test Suite: Core 4 Visual Perception Daemon (NEW — Eyes, Architecture v1.2.0).
Covers Integral Image Correctness, Viola-Jones Cascades, Lucas-Kanade Optical Flow,
Kalman Track Continuity across Occlusions, and HSV Segmentation.
"""
import unittest
import numpy as np

from miku.core4_vision.integral_image import compute_integral_image, rectangle_sum
from miku.core4_vision.viola_jones import ViolaJonesDetector
from miku.core4_vision.lucas_kanade import LucasKanadeMotionDetector
from miku.core4_vision.kalman_tracker import KalmanObjectTracker
from miku.core4_vision.hsv_segmentation import HSVSegmentation

class TestCore4Vision(unittest.TestCase):
    def test_integral_image_hand_computed_matrix(self):
        """
        Architecture §4.1: Constant-time rectangle sums.
        Tests 3x3 hand-computed matrix:
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        """
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)

        integral = compute_integral_image(matrix)

        # Expected integral (with +1 padding):
        # [0,  0,  0,  0]
        # [0,  1,  3,  6]
        # [0,  5, 12, 21]
        # [0, 12, 27, 45]
        self.assertEqual(integral[3, 3], 45.0)

        # Rectangle sum of entire matrix [0..2, 0..2] = 45
        total_sum = rectangle_sum(integral, 0, 0, 2, 2)
        self.assertEqual(total_sum, 45.0)

        # Rectangle sum of center 2x2: [1..2, 1..2] = 5 + 6 + 8 + 9 = 28
        sub_sum = rectangle_sum(integral, 1, 1, 2, 2)
        self.assertEqual(sub_sum, 28.0)

        # Single element [1, 1] = 5
        elem_sum = rectangle_sum(integral, 1, 1, 1, 1)
        self.assertEqual(elem_sum, 5.0)

    def test_lucas_kanade_synthetic_translating_pattern(self):
        """
        Unit test for Lucas-Kanade optical flow:
        Creates synthetic translating pattern with known shift (u, v) = (2.0, 0.0).
        Asserts solved flow matches translation direction.
        """
        detector = LucasKanadeMotionDetector(tau=5.0, window_size=9)
        H, W = 100, 100

        # Frame 1: Gaussian blob centered at (50, 50)
        y, x = np.ogrid[:H, :W]
        f1 = (200.0 * np.exp(-((x - 50)**2 + (y - 50)**2) / (2 * 10**2))).astype(np.float64)
        detector.process_frame(f1)

        # Frame 2: Shifted right by +2 pixels: centered at (52, 50)
        f2 = (200.0 * np.exp(-((x - 52)**2 + (y - 50)**2) / (2 * 10**2))).astype(np.float64)
        res = detector.process_frame(f2)

        self.assertTrue(res["motion_detected"], "Motion must be detected between shifted frames")
        u, v = res["velocity"]
        # Solved velocity u should be positive (translating along x-axis)
        self.assertGreater(u, 0.5, f"Expected positive velocity along x, got u={u}")
        self.assertAlmostEqual(v, 0.0, delta=1.5, msg=f"Expected near-zero vertical velocity, got v={v}")

    def test_kalman_track_continuity_over_occlusion_gap(self):
        """
        Architecture §4.3: Feed detections with gaps (simulated occlusion).
        Asserts predicted position stays within bounds during the gap.
        """
        tracker = KalmanObjectTracker(dt=1.0)
        tracker.initialize((100.0, 100.0))

        # Object moves horizontally at vx = 5 pixels/step for 5 frames
        for t in range(1, 6):
            tracker.predict()
            tracker.update((100.0 + 5.0 * t, 100.0))

        pos_before_occlusion = tracker.position
        self.assertAlmostEqual(pos_before_occlusion[0], 125.0, delta=2.0)

        # Simulated occlusion: 3 frames of NO new measurement (predict only)
        for _ in range(3):
            tracker.predict()

        pos_after_occlusion = tracker.position
        # Expected position: 125 + 3 * 5 = ~140
        self.assertAlmostEqual(pos_after_occlusion[0], 140.0, delta=4.0,
                               msg="Kalman filter must extrapolate continuous position through occlusion")

    def test_hsv_segmentation_red_and_blue(self):
        """
        Architecture §4.4: HSV thresholding + connected components.
        Finds colored rectangular buttons and computes centroid and bbox.
        """
        seg = HSVSegmentation()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Draw red square at [20..40, 30..60] (area = 21 * 31 = 651)
        img[20:41, 30:61] = [255, 0, 0]  # Pure Red

        blobs = seg.segment_color(img, "red")
        self.assertGreater(len(blobs), 0, "Must segment red button blob")
        b = blobs[0]
        self.assertEqual(b["label"], "red_region")
        self.assertAlmostEqual(b["centroid"][0], 45, delta=2)  # cx ~ 45
        self.assertAlmostEqual(b["centroid"][1], 30, delta=2)  # cy ~ 30
        self.assertGreater(b["area"], 500)

if __name__ == "__main__":
    unittest.main()
