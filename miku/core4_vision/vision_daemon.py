"""
Core 4: Visual Perception Daemon (NEW — Eyes, Architecture v1.2.0).
Coordinates Viola-Jones, Lucas-Kanade Optical Flow, Kalman Tracking, and HSV Segmentation.
Pushes detections (never raw frames) to Core 2 over IPC Queue.
"""
import time
import numpy as np
from multiprocessing import Queue
from typing import Optional, Dict, Any, List

from miku.ipc.messages import VisionDetectionMsg
from miku.ipc.shared_ring_buffer import SharedRingBuffer
from miku.core4_vision.integral_image import compute_integral_image
from miku.core4_vision.viola_jones import ViolaJonesDetector
from miku.core4_vision.lucas_kanade import LucasKanadeMotionDetector
from miku.core4_vision.kalman_tracker import KalmanObjectTracker
from miku.core4_vision.hsv_segmentation import HSVSegmentation
from miku.core4_vision.scene_understanding import CustomSceneUnderstanding

class VisionDaemon:
    def __init__(
        self,
        detection_queue: Queue,
        shm_ring_buffer: Optional[SharedRingBuffer] = None
    ):
        self.detection_queue = detection_queue
        self.shm_ring_buffer = shm_ring_buffer

        self.viola_jones = ViolaJonesDetector()
        self.optical_flow = LucasKanadeMotionDetector()
        self.kalman = KalmanObjectTracker()
        self.hsv = HSVSegmentation()
        self.scene_engine = CustomSceneUnderstanding()

        self.slot_counter = 0

    def process_frame(self, rgb_frame: np.ndarray) -> List[VisionDetectionMsg]:
        """
        Processes single camera or screen frame:
        1. Write raw frame into zero-copy SharedRingBuffer.
        2. Motion & Optical Flow detection.
        3. HSV Color / Region segmentation (e.g., red/blue button detection).
        4. Viola-Jones Face/Object presence detection.
        5. Kalman tracking update.
        6. Dispatches detection messages (metadata only) to Core 2.
        """
        now = time.time()
        # 1. Zero-copy shared memory write
        if self.shm_ring_buffer:
            raw_bytes = rgb_frame.tobytes()
            self.shm_ring_buffer.write(self.slot_counter, raw_bytes)
            self.slot_counter += 1

        detections: List[VisionDetectionMsg] = []

        # 2. Motion Detection
        motion_res = self.optical_flow.process_frame(rgb_frame)
        if motion_res["motion_detected"]:
            mx, my = motion_res["centroid"]
            u, v = motion_res["velocity"]
            # Track motion centroid with Kalman
            tx, ty = self.kalman.update((float(mx), float(my)))
            det = VisionDetectionMsg(
                label="motion",
                bbox=(int(tx) - 15, int(ty) - 15, 30, 30),
                confidence=min(1.0, motion_res["motion_area"] / 500.0),
                motion_vector=(u, v),
                timestamp=now
            )
            detections.append(det)

        # 3. HSV Color Targets (Red, Blue)
        for color in ("red", "blue"):
            blobs = self.hsv.segment_color(rgb_frame, color)
            for b in blobs[:1]:  # top blob
                det = VisionDetectionMsg(
                    label=b["label"],
                    bbox=b["bbox"],
                    confidence=b["confidence"],
                    motion_vector=(0.0, 0.0),
                    timestamp=now
                )
                detections.append(det)

        # 4. Viola-Jones scan
        vj_hits = self.viola_jones.scan_image(rgb_frame, step=24)
        for x, y, w, h, conf in vj_hits[:1]:
            det = VisionDetectionMsg(
                label="face_presence",
                bbox=(x, y, w, h),
                confidence=conf,
                motion_vector=(0.0, 0.0),
                timestamp=now
            )
            detections.append(det)

        # 5. Custom Scene Understanding
        scene_info = self.scene_engine.analyze_scene(rgb_frame)
        det_scene = VisionDetectionMsg(
            label=f"scene_{scene_info['scene_class']}",
            bbox=(0, 0, rgb_frame.shape[1], rgb_frame.shape[0]),
            confidence=scene_info["confidence"],
            motion_vector=(0.0, 0.0),
            timestamp=now
        )
        detections.append(det_scene)

        # Dispatch detections over IPC Queue (Zero raw frames sent across queue)
        for d in detections:
            self.detection_queue.put(d)

        return detections
