"""
Core 1: Audio Daemon (Sensory & Voice).
Runs in isolated process, filters echo via NLMS, recognizes speech with GMM-HMM,
and sends transcripts to Core 2. Never writes raw PCM to disk.
"""
import time
import numpy as np
from multiprocessing import Queue
from typing import Optional

from miku.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_FRAME_SIZE,
    NLMS_FILTER_LENGTH,
    NLMS_STEP_SIZE,
    NLMS_EPSILON
)
from miku.core1_audio.nlms_filter import NLMSFilter
from miku.core1_audio.gmm_hmm_stt import GMMHMMModel
from miku.core1_audio.td_psola_tts import TDPSOLASynthesizer
from miku.ipc.messages import STTTranscriptMsg, AudioMetadataMsg
from miku.ipc.shared_ring_buffer import SharedRingBuffer

class AudioDaemon:
    def __init__(
        self,
        transcript_queue: Queue,
        shm_ring_buffer: Optional[SharedRingBuffer] = None,
        sample_rate: int = AUDIO_SAMPLE_RATE
    ):
        self.transcript_queue = transcript_queue
        self.shm_ring_buffer = shm_ring_buffer
        self.sample_rate = sample_rate
        self.running = False

        self.nlms = NLMSFilter(
            filter_length=NLMS_FILTER_LENGTH,
            step_size=NLMS_STEP_SIZE,
            epsilon=NLMS_EPSILON
        )
        self.stt = GMMHMMModel(sample_rate=self.sample_rate)
        self.tts = TDPSOLASynthesizer(sample_rate=self.sample_rate)
        self.slot_counter = 0

    def process_incoming_audio(self, mic_chunk: np.ndarray, ref_chunk: Optional[np.ndarray] = None):
        """
        Process a chunk of microphone audio:
        1. Apply NLMS echo cancellation if reference audio exists.
        2. Write PCM to SharedRingBuffer (in-memory zero-copy, never to disk).
        3. Decode with GMM-HMM.
        4. If confidence > threshold, dispatch STTTranscriptMsg to Core 2.
        """
        if ref_chunk is not None:
            clean_chunk = self.nlms.process_block(mic_chunk, ref_chunk)
        else:
            clean_chunk = mic_chunk.astype(np.float32)

        # Store in shared memory ring buffer if available
        if self.shm_ring_buffer:
            max_bytes = self.shm_ring_buffer.slot_payload_size
            raw_bytes = clean_chunk.tobytes()
            for offset in range(0, len(raw_bytes), max_bytes):
                chunk = raw_bytes[offset:offset + max_bytes]
                self.shm_ring_buffer.write(self.slot_counter, chunk)
                self.slot_counter += 1

        # Run GMM-HMM STT
        phrase, confidence = self.stt.decode(clean_chunk)
        if phrase and confidence > 0.35:
            msg = STTTranscriptMsg(
                text=phrase,
                confidence=confidence,
                timestamp=time.time()
            )
            self.transcript_queue.put(msg)
            return msg
        return None

    def speak(self, text: str) -> np.ndarray:
        """
        Synthesizes spoken audio using TD-PSOLA concatenative synthesis.
        """
        return self.tts.synthesize(text)
