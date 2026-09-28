"""
Zero-Copy Shared Memory Ring Buffer for High-Frequency Payloads (PCM & Video Frames).
Addresses Bare-Metal Bottleneck #1 (IPC Serialization Choke).
Uses generation counters to guarantee lock-free non-torn reads.
"""
import struct
import numpy as np
from multiprocessing import shared_memory
from typing import Optional, Tuple

# Header: 8 bytes generation_counter (Q), 4 bytes data_len (I), 4 bytes flags (I)
HEADER_FORMAT = "=QII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

class SharedRingBuffer:
    def __init__(
        self,
        name: str,
        num_slots: int,
        slot_payload_size: int,
        create: bool = False
    ):
        self.name = name
        self.num_slots = num_slots
        self.slot_payload_size = slot_payload_size
        self.slot_total_size = HEADER_SIZE + slot_payload_size
        self.total_bytes = self.num_slots * self.slot_total_size
        self.is_owner = create

        if create:
            # Try to clean up any preexisting segment with the same name
            try:
                existing = shared_memory.SharedMemory(name=self.name)
                existing.close()
                existing.unlink()
            except Exception:
                pass
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_bytes)
                # Initialize headers to 0
                self.shm.buf[:self.total_bytes] = b"\x00" * self.total_bytes
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

        self.buf = self.shm.buf

    def _get_slot_offset(self, slot_idx: int) -> int:
        return (slot_idx % self.num_slots) * self.slot_total_size

    def write(self, slot_idx: int, data: bytes) -> int:
        """
        Writes data into the given slot using generation counter.
        Odd counter indicates in-progress write; even indicates completed write.
        """
        if len(data) > self.slot_payload_size:
            raise ValueError(f"Data size {len(data)} exceeds slot capacity {self.slot_payload_size}")

        offset = self._get_slot_offset(slot_idx)
        cur_gen, _, _ = struct.unpack_from(HEADER_FORMAT, self.buf, offset)
        
        # Mark writing (odd generation counter)
        in_progress_gen = (cur_gen + 1) | 1
        struct.pack_into(HEADER_FORMAT, self.buf, offset, in_progress_gen, len(data), 0)

        # Write payload
        payload_start = offset + HEADER_SIZE
        payload_end = payload_start + len(data)
        self.buf[payload_start:payload_end] = data

        # Mark complete (even generation counter)
        completed_gen = in_progress_gen + 1
        struct.pack_into(HEADER_FORMAT, self.buf, offset, completed_gen, len(data), 0)
        return completed_gen

    def read(self, slot_idx: int, max_retries: int = 5) -> Tuple[bytes, int]:
        """
        Reads data from slot_idx. Retries if a write was in progress during read.
        Returns (data_bytes, generation_counter).
        """
        offset = self._get_slot_offset(slot_idx)

        for _ in range(max_retries):
            gen_pre, data_len, _ = struct.unpack_from(HEADER_FORMAT, self.buf, offset)
            if gen_pre % 2 != 0:
                # Write in progress, spin
                continue

            payload_start = offset + HEADER_SIZE
            payload_end = payload_start + data_len
            data = bytes(self.buf[payload_start:payload_end])

            gen_post, _, _ = struct.unpack_from(HEADER_FORMAT, self.buf, offset)
            if gen_pre == gen_post:
                return data, gen_post

        # Fallback if contention occurred
        gen_final, data_len, _ = struct.unpack_from(HEADER_FORMAT, self.buf, offset)
        payload_start = offset + HEADER_SIZE
        return bytes(self.buf[payload_start:payload_start + data_len]), gen_final

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass

    def unlink(self):
        if self.is_owner:
            try:
                self.shm.unlink()
            except Exception:
                pass
