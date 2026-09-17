"""
Local Training Script for Miku Causal Transformer LLM.
Trained 100% from scratch on device in PyTorch.
Zero cloud API keys, zero external weights. Fast CPU training.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from .tokenizer import MikuTokenizer
from .model import MikuTransformerLM
from .dataset import CONVERSATION_CORPUS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "miku_llm.pth")
VOCAB_PATH = os.path.join(DATA_DIR, "miku_vocab.json")


def train_miku_llm(
    epochs: int = 70,
    batch_size: int = 8,
    lr: float = 3e-3,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 128,
    save_path: str = MODEL_PATH,
    vocab_path: str = VOCAB_PATH
) -> Dict[str, Any]:
    print("=" * 60)
    print("[*] TRAINING MIKU CAUSAL TRANSFORMER LLM FROM SCRATCH")
    print("=" * 60)
    start_time = time.time()

    # Step 1: Train Tokenizer
    print("[1/4] Building local tokenizer from dialogue corpus...")
    tokenizer = MikuTokenizer()
    tokenizer.train_from_corpus(CONVERSATION_CORPUS, max_vocab=400)
    tokenizer.save(vocab_path)
    vocab_size = tokenizer.vocab_size
    print(f"       Vocabulary built: {vocab_size} unique tokens (subwords + byte-fallback).")

    # Step 2: Prepare Training Tensors
    print("[2/4] Tokenizing and padding conversation batches...")
    all_token_seqs = [tokenizer.encode(item) for item in CONVERSATION_CORPUS]

    # Create padded inputs and targets (shifted by 1 for autoregressive next-token prediction)
    input_tensors = []
    target_tensors = []

    for seq in all_token_seqs:
        if len(seq) < 2:
            continue
        seq = seq[:max_seq_len]
        inp = seq[:-1]
        tgt = seq[1:]
        
        # Pad to max_len
        pad_len = max_seq_len - 1 - len(inp)
        if pad_len > 0:
            inp = inp + [0] * pad_len
            tgt = tgt + [0] * pad_len
            
        input_tensors.append(inp)
        target_tensors.append(tgt)

    X = torch.tensor(input_tensors, dtype=torch.long)
    Y = torch.tensor(target_tensors, dtype=torch.long)
    print(f"       Prepared {len(X)} training sequences, sequence length: {X.shape[1]}")

    # Step 3: Instantiate Transformer
    print("[3/4] Initializing Causal Transformer Architecture...")
    model = MikuTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Model initialized with {total_params:,} parameters (~{total_params * 4 / 1024 / 1024:.2f} MB).")

    # Step 4: Training Loop
    print(f"[4/4] Optimizing over {epochs} epochs on CPU (AdamW + Cosine Schedule)...")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    model.train()
    num_samples = len(X)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i:i + batch_size]
            b_x, b_y = X[batch_idx], Y[batch_idx]

            optimizer.zero_grad()
            _, loss = model(b_x, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"       Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save Model Checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len
        }
    }, save_path)

    elapsed = time.time() - start_time
    print(f"\n[OK] TRAINING COMPLETE in {elapsed:.2f}s! Model saved to: {save_path}\n")

    return {
        "success": True,
        "parameters": total_params,
        "final_loss": avg_loss,
        "model_path": save_path,
        "vocab_path": vocab_path,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    train_miku_llm()
