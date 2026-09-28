"""
MIKU: Master Multi-Core Process Orchestrator.
Spawns and manages Core 1, Core 2, Core 3, and Core 4 as isolated CPU processes
communicating over zero-copy shared memory and typed IPC queues.
"""
import time
import os
import signal
from multiprocessing import Process, Queue
from typing import Optional, Dict, Any

from miku.config import (
    AUDIO_RING_BUFFER_SLOTS,
    AUDIO_FRAME_SIZE,
    VISION_RING_BUFFER_SLOTS,
    VISION_FRAME_WIDTH,
    VISION_FRAME_HEIGHT
)
from miku.ipc.shared_ring_buffer import SharedRingBuffer
from miku.core1_audio.audio_daemon import AudioDaemon
from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
from miku.core3_execution.execution_daemon import ExecutionDaemon
from miku.core4_vision.vision_daemon import VisionDaemon
from miku.persistence.encrypted_log import EncryptedPersistenceLogger

class MikuOrchestrator:
    def __init__(self, use_multiprocessing: bool = False):
        self.use_multiprocessing = use_multiprocessing

        # IPC Queues for control messages
        self.transcript_queue = Queue()
        self.action_queue = Queue()
        self.result_queue = Queue()
        self.detection_queue = Queue()

        # Zero-copy shared memory ring buffers (audio: 512 float32 = 2048 bytes; vision: 640x480x3 = 921600 bytes)
        self.audio_shm = SharedRingBuffer(
            name="miku_audio_shm",
            num_slots=AUDIO_RING_BUFFER_SLOTS,
            slot_payload_size=AUDIO_FRAME_SIZE * 4,
            create=True
        )
        self.vision_shm = SharedRingBuffer(
            name="miku_vision_shm",
            num_slots=VISION_RING_BUFFER_SLOTS,
            slot_payload_size=VISION_FRAME_WIDTH * VISION_FRAME_HEIGHT * 3,
            create=True
        )

        # Initialize daemons
        self.audio_daemon = AudioDaemon(
            transcript_queue=self.transcript_queue,
            shm_ring_buffer=self.audio_shm
        )
        self.cognitive_daemon = CognitiveDaemon(
            action_queue=self.action_queue
        )
        self.execution_daemon = ExecutionDaemon(
            action_queue=self.action_queue,
            result_queue=self.result_queue
        )
        self.vision_daemon = VisionDaemon(
            detection_queue=self.detection_queue,
            shm_ring_buffer=self.vision_shm
        )
        self.logger = EncryptedPersistenceLogger()

        self.running = False

    def start(self):
        self.running = True

    def process_voice_input(self, audio_chunk) -> Optional[Dict[str, Any]]:
        """
        Processes audio chunk through Core 1 -> Core 2 -> Core 3 pipeline.
        """
        transcript = self.audio_daemon.process_incoming_audio(audio_chunk)
        if transcript:
            # Core 2 cognitive router
            decision = self.cognitive_daemon.handle_transcript(transcript)
            if decision.get("status") == "dispatched":
                action_msg = decision["action_msg"]
                # Core 3 execution
                exec_result = self.execution_daemon.execute_request(action_msg)
                # Release compound task lock if needed
                self.cognitive_daemon.handle_action_completion(exec_result)
                # Audit log to encrypted WAL
                self.logger.log_action(
                    task_id=exec_result.task_id,
                    action_type=action_msg.action_type,
                    target=action_msg.target,
                    status=exec_result.status,
                    details=exec_result.message
                )
                decision["execution"] = exec_result
            return decision
        return None

    def process_vision_frame(self, rgb_frame) -> list:
        """
        Processes camera/screen frame through Core 4 -> Core 2.
        """
        detections = self.vision_daemon.process_frame(rgb_frame)
        if detections:
            # Sort by confidence descending
            best_det = max(detections, key=lambda d: d.confidence)
            self.cognitive_daemon.update_vision_detection(best_det)
        return detections

    def shutdown(self):
        self.running = False
        self.audio_shm.close()
        self.audio_shm.unlink()
        self.vision_shm.close()
        self.vision_shm.unlink()
