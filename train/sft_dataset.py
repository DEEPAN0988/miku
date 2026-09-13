"""
train/sft_dataset.py — SFT Dataset with Response-Only Target Loss Masking

In Supervised Fine-Tuning (SFT), the model is trained to predict RESPONSE tokens
conditioned on the PROMPT (Context/Instruction/Response tag).
Loss is computed ONLY on response tokens (targets = -1 for prompt tokens).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


class SFTDataset(Dataset):
    """
    SFT Dataset that returns input_ids and target_ids with prompt tokens masked to -1.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: MikuTokenizer,
        max_seq_len: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict[str, str]] = []

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"SFT jsonl file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        # Pre-tokenize all examples to maximize training throughput
        self.tokenized_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._tokenize_all()

    def _tokenize_all(self) -> None:
        tok = self.tokenizer
        bos_id = tok.bos_id
        eos_id = tok.eos_id

        for ex in self.examples:
            prompt_text = ex["prompt"]
            response_text = ex["response"]

            prompt_ids = tok.encode(prompt_text, add_bos=False, add_eos=False)
            response_ids = tok.encode(response_text, add_bos=False, add_eos=False)

            # Full sequence: [BOS] + prompt + response + [EOS]
            full_input = [bos_id] + prompt_ids + response_ids + [eos_id]

            # In causal LM next-token prediction, input is full_input[:-1] and target is full_input[1:]
            # We want to mask out (set to -1) all targets that correspond to predicting PROMPT tokens.
            # prompt boundary in input:
            # input: [BOS, p_1, p_2, ..., p_k, r_1, r_2, ..., r_m]
            # target:[p_1, p_2, ..., p_k, r_1, r_2, ..., r_m, EOS]
            # The prompt tokens to predict are p_1 ... p_k.
            # We mask targets from index 0 up to len(prompt_ids) (inclusive of p_k).
            # The first unmasked target will be r_1 (which is at index len(prompt_ids)).
            
            prompt_len = len(prompt_ids)
            input_ids = full_input[:-1]
            target_ids = full_input[1:]

            # Truncate to max_seq_len
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                target_ids = target_ids[:self.max_seq_len]

            # Build masked target: set -1 for all prompt positions
            masked_targets = []
            for idx, tgt in enumerate(target_ids):
                if idx < prompt_len:
                    masked_targets.append(-1)
                else:
                    masked_targets.append(tgt)

            # Convert to tensors
            x = torch.tensor(input_ids, dtype=torch.long)
            y = torch.tensor(masked_targets, dtype=torch.long)

            # Only keep sample if there is at least one active (unmasked) target token
            if (y != -1).any():
                self.tokenized_samples.append((x, y))

    def __len__(self) -> int:
        return len(self.tokenized_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokenized_samples[idx]


def sft_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate batch by padding sequences dynamically to the maximum length in the batch.
    Pads input_ids with pad_id (0) and target_ids with -1 (ignore_index).
    """
    batch_max_len = max(x.size(0) for x, _ in batch)

    batch_x = []
    batch_y = []

    for x, y in batch:
        pad_len = batch_max_len - x.size(0)
        if pad_len > 0:
            x_pad = torch.cat([x, torch.full((pad_len,), 0, dtype=torch.long)])
            y_pad = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])
        else:
            x_pad = x
            y_pad = y

        batch_x.append(x_pad)
        batch_y.append(y_pad)

    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)


def create_sft_dataloader(
    jsonl_path: str,
    tokenizer: MikuTokenizer,
    batch_size: int,
    max_seq_len: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SFTDataset(jsonl_path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=sft_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Running SFT target masking self-test ...")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    test_ds = SFTDataset("data/processed/sft/val_sft.jsonl", tok, max_seq_len=256)
    print(f"Loaded {len(test_ds)} validation samples.")

    # Inspect sample 0
    x, y = test_ds[0]
    print(f"\nSample 0 length: input={x.size(0)}, target={y.size(0)}")
    print(f"Number of masked tokens (-1): {(y == -1).sum().item()}")
    print(f"Number of active target tokens: {(y != -1).sum().item()}")

    print("\nToken alignment inspection (first 25 tokens):")
    for i in range(min(25, x.size(0))):
        in_tok = tok._sp.id_to_piece(x[i].item())
        tgt_val = y[i].item()
        tgt_str = "-1 [MASKED]" if tgt_val == -1 else f"{tgt_val} ('{tok._sp.id_to_piece(tgt_val)}')"
        print(f"  pos {i:02d}: input={in_tok:<15s} -> target={tgt_str}")

    print("\n[OK] SFT target masking verified successfully.")
