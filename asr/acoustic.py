"""
Local Acoustic Distance & Dynamic Time Warping (DTW) Matcher for Offline ASR.
Performs command recognition by matching input acoustic MFCC sequences against
user-calibrated command templates.
Zero external pretrained models, zero API keys.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from wakeword.audio_features import extract_mfcc


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance between two MFCC feature sequences.
    seq1: (T1, D), seq2: (T2, D)
    """
    n, m = seq1.shape[0], seq2.shape[0]
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean_distance(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    # Normalize by path length
    return float(dtw_matrix[n, m] / (n + m))


class AcousticTemplateMatcher:
    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {}

    def register_template(self, command_label: str, mfcc_sequence: np.ndarray):
        """Register an acoustic MFCC template for a command phrase."""
        if command_label not in self.templates:
            self.templates[command_label] = []
        self.templates[command_label].append(mfcc_sequence)

    def match(self, input_mfcc: np.ndarray) -> Tuple[str, float]:
        """
        Find best matching command template for input MFCC sequence.
        Returns (best_command, score) where lower distance = higher similarity.
        """
        if not self.templates:
            return "unknown", float("inf")

        best_cmd = "unknown"
        min_dist = float("inf")

        for cmd, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                dist = compute_dtw_distance(input_mfcc, tmpl)
                if dist < min_dist:
                    min_dist = dist
                    best_cmd = cmd

        # Convert distance to a confidence percentage (heuristic based on normalized DTW cost)
        confidence = max(0.0, min(1.0, 1.0 - (min_dist / 10.0)))
        return best_cmd, confidence
