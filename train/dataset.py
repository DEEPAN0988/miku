"""
train/dataset.py — MIKU Token Dataset
STATUS: IMPLEMENTED

Loads the uint16 binary token files produced by prepare_corpus.py
using memory-mapped numpy arrays for efficient random access without
loading the full corpus into RAM.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """
    Memory-mapped dataset over a flat uint16 binary token file.

    File format: produced by data/prepare_corpus.py — a flat uint16
    numpy array where tokens from all documents are concatenated
    (with BOS/EOS between documents).

    Each sample is (input_ids, target_ids) of length seq_len.
    input_ids[i] and target_ids[i] = input_ids[i+1] (next-token prediction).
    """

    def __init__(self, bin_path: str, seq_len: int) -> None:
        """
        Args:
            bin_path: path to the .bin file (uint16 flat token array)
            seq_len:  context length — returned sequences have this length
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Token file not found: {bin_path}\n"
                f"Run: python data/prepare_corpus.py --stage tokenize"
            )

        # Memory-map the binary file — does not load into RAM
        self._data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self._data)

        # Number of valid starting positions
        # We need seq_len tokens for input and 1 more for the last target
        self.n_samples = max(0, self.n_tokens - seq_len)

        if self.n_samples == 0:
            raise ValueError(
                f"Token file {bin_path} has {self.n_tokens} tokens, "
                f"but seq_len={seq_len} requires at least {seq_len + 1} tokens."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [seq_len] int64 tensor — input tokens
            y: [seq_len] int64 tensor — target tokens (shifted right by 1)
        """
        chunk = self._data[idx : idx + self.seq_len + 1]
        # Convert uint16 → int64 for nn.Embedding compatibility
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1]   # [seq_len]
        y = chunk[1:]    # [seq_len]
        return x, y

    def token_count(self) -> int:
        """Total number of tokens in this dataset file."""
        return self.n_tokens

    def __repr__(self) -> str:
        return (
            f"TokenDataset("
            f"n_tokens={self.n_tokens:,}, "
            f"seq_len={self.seq_len}, "
            f"n_samples={self.n_samples:,})"
        )


def create_dataloader(
    bin_path: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for a binary token file.

    Args:
        bin_path:    path to .bin token file
        seq_len:     context length per sample
        batch_size:  mini-batch size
        shuffle:     shuffle samples each epoch (True for train, False for val)
        num_workers: DataLoader worker processes (0 = main process)
        pin_memory:  pin memory for faster GPU transfer (True if CUDA available)

    Returns:
        torch DataLoader
    """
    dataset = TokenDataset(bin_path, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,    # Drop incomplete final batch for consistent shapes
    )
    return loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    bin_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.bin"
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    print(f"Testing TokenDataset on {bin_path} with seq_len={seq_len}")
    ds = TokenDataset(bin_path, seq_len)
    print(ds)

    x, y = ds[0]
    print(f"Sample 0: x.shape={x.shape}, y.shape={y.shape}")
    print(f"x[:10]: {x[:10].tolist()}")
    print(f"y[:10]: {y[:10].tolist()}")
    print(f"(y[i] == x[i+1]): {(y[:-1] == x[1:]).all().item()}")
