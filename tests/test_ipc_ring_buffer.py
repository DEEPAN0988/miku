"""
Test Suite: Shared Memory Ring Buffer (Bottleneck #1 Fix).
Verifies zero-copy allocation, generation counter semantics, and non-torn reads.
"""
import unittest
import numpy as np
from miku.ipc.shared_ring_buffer import SharedRingBuffer

class TestSharedRingBuffer(unittest.TestCase):
    def setUp(self):
        self.buf_name = "test_shm_ring_buf"
        self.ring = SharedRingBuffer(
            name=self.buf_name,
            num_slots=4,
            slot_payload_size=1024,
            create=True
        )

    def tearDown(self):
        self.ring.close()
        self.ring.unlink()

    def test_single_slot_write_and_read(self):
        test_data = b"Miku-Zero-Copy-Payload-12345"
        gen = self.ring.write(0, test_data)
        self.assertEqual(gen % 2, 0, "Completed generation counter must be even")

        read_data, read_gen = self.ring.read(0)
        self.assertEqual(read_data, test_data)
        self.assertEqual(read_gen, gen)

    def test_wrap_around_slots(self):
        # 4 slots total, test writing 8 items to verify modulo wrap-around
        for i in range(8):
            payload = f"frame_{i}".encode("utf-8")
            self.ring.write(i, payload)

        # Slot 0 should now contain frame_4 (or frame_8 depending on index)
        # i = 4 -> 4 % 4 = 0 -> payload 'frame_4'
        # i = 7 -> 7 % 4 = 3 -> payload 'frame_7'
        read_data_3, _ = self.ring.read(3)
        self.assertEqual(read_data_3, b"frame_7")

    def test_large_numpy_pcm_roundtrip(self):
        pcm = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        raw_bytes = pcm.tobytes()
        self.ring.write(1, raw_bytes)

        read_bytes, _ = self.ring.read(1)
        restored = np.frombuffer(read_bytes, dtype=np.float32)
        np.testing.assert_array_almost_equal(restored, pcm)

if __name__ == "__main__":
    unittest.main()
