"""
Normalized Least Mean Squares (NLMS) Acoustic Echo Cancellation Filter.
Core 1: Audio Daemon
"""
import numpy as np

class NLMSFilter:
    def __init__(self, filter_length: int = 128, step_size: float = 0.1, epsilon: float = 1e-6):
        """
        filter_length: number of filter taps (L)
        step_size: adaptation parameter mu (0 < mu < 2)
        epsilon: regularization constant to prevent division by zero
        """
        self.L = filter_length
        self.mu = step_size
        self.epsilon = epsilon
        self.weights = np.zeros(self.L, dtype=np.float32)
        self.history = np.zeros(self.L, dtype=np.float32)

    def process_sample(self, mic_sample: float, ref_sample: float) -> float:
        """
        Processes a single sample:
        mic_sample (d): microphone signal containing desired speech + echo
        ref_sample (x): reference signal played to system speakers
        returns: error signal (e) with echo canceled
        """
        # Shift history
        self.history = np.roll(self.history, 1)
        self.history[0] = ref_sample

        # Predict echo: y = w^T * x
        echo_est = float(np.dot(self.weights, self.history))

        # Error signal (clean speech estimate): e = d - y
        error = mic_sample - echo_est

        # Normalization factor: ||x||^2 + epsilon
        norm = float(np.dot(self.history, self.history)) + self.epsilon

        # Update weights: w = w + (mu / norm) * e * x
        self.weights += (self.mu / norm) * error * self.history

        return error

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """
        Processes a block of samples.
        """
        mic_block = np.asarray(mic_block, dtype=np.float32)
        ref_block = np.asarray(ref_block, dtype=np.float32)
        n = min(len(mic_block), len(ref_block))
        output = np.empty(n, dtype=np.float32)

        for i in range(n):
            output[i] = self.process_sample(mic_block[i], ref_block[i])

        return output

    def reset(self):
        self.weights.fill(0.0)
        self.history.fill(0.0)
