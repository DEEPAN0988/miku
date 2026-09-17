"""
Local Training Script for Miku Causal Transformer LLM.
Supports fixed epoch training or continuous duration-based training (e.g. 30 minutes).
Trained 100% from scratch on device in PyTorch.
Zero cloud API keys, zero external weights.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import functools
from typing import Dict, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure unbuffered output for live log monitoring
print = functools.partial(print, flush=True)

from .tokenizer import MikuTokenizer
from .model import MikuTransformerLM
from .dataset import CONVERSATION_CORPUS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "miku_llm.pth")
VOCAB_PATH = os.path.join(DATA_DIR, "miku_vocab.json")


def train_miku_llm(
    epochs: Optional[int] = None,
    duration_minutes: Optional[float] = None,
    batch_size: int = 8,
    lr: float = 2.5e-3,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 128,
    save_path: str = MODEL_PATH,
    vocab_path: str = VOCAB_PATH
) -> Dict[str, Any]:
    print("=" * 60)
    mode_str = f"DURATION: {duration_minutes:.1f} minutes" if duration_minutes else f"EPOCHS: {epochs or 70}"
    print(f"[*] TRAINING MIKU CAUSAL TRANSFORMER LLM FROM SCRATCH ({mode_str})")
    print("=" * 60)
    start_time = time.time()

    # Step 1: Train Tokenizer
    print("[1/4] Building local tokenizer from dialogue corpus...")
    tokenizer = MikuTokenizer()
    tokenizer.train_from_corpus(CONVERSATION_CORPUS, max_vocab=600)
    vocab_size = tokenizer.vocab_size
    tokenizer.save(vocab_path)
    print(f"       Vocabulary built: {vocab_size} unique tokens (subwords + byte-fallback).")

    # Step 2: Prepare Training Tensors
    print("[2/4] Tokenizing conversation sequences...")
    all_token_seqs = [tokenizer.encode(item) for item in CONVERSATION_CORPUS]

    input_tensors = []
    target_tensors = []

    for seq in all_token_seqs:
        if len(seq) < 2:
            continue
        seq = seq[:max_seq_len]
        inp = seq[:-1]
        tgt = seq[1:]
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

    # If previous weights exist, resume training
    if os.path.exists(save_path):
        try:
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt and ckpt.get("config", {}).get("vocab_size") == vocab_size:
                model.load_state_dict(ckpt["model_state_dict"])
                print("       Resumed weights from existing checkpoint.")
        except Exception:
            pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Model architecture: {total_params:,} parameters (~{total_params * 4 / 1024 / 1024:.2f} MB).")

    # Step 4: Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    target_seconds = (duration_minutes * 60.0) if duration_minutes else None
    effective_epochs = epochs or (999999 if target_seconds else 70)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs if not target_seconds else 500, eta_min=1e-4)

    model.train()
    num_samples = len(X)
    last_print_time = time.time()
    last_save_time = time.time()
    epoch = 0
    avg_loss = 0.0

    print(f"[4/4] Starting training loop (saving checkpoints to {save_path})...")

    try:
        while True:
            epoch += 1
            if not target_seconds and epoch > effective_epochs:
                break
            if target_seconds and (time.time() - start_time) >= target_seconds:
                break

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

            curr_time = time.time()
            elapsed = curr_time - start_time

            # Periodic log printing
            if target_seconds:
                if curr_time - last_print_time >= 30.0 or epoch == 1:
                    pct = min(100.0, (elapsed / target_seconds) * 100.0)
                    rem = max(0.0, (target_seconds - elapsed) / 60.0)
                    print(f"       [{pct:5.1f}% | {elapsed/60:.1f}m / {duration_minutes:.1f}m] Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Remaining: {rem:.1f}m")
                    last_print_time = curr_time
            else:
                if epoch % 10 == 0 or epoch == 1 or epoch == effective_epochs:
                    print(f"       Epoch {epoch:2d}/{effective_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint every 60 seconds or at completion
            if curr_time - last_save_time >= 60.0:
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
                last_save_time = curr_time

    except KeyboardInterrupt:
        print("\n[!] Training paused by user. Saving current checkpoint...")

    # Final Save
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

    total_time = time.time() - start_time
    print(f"\n[OK] TRAINING FINISHED: {epoch} epochs completed in {total_time/60:.2f} minutes!")
    print(f"     Final Loss: {avg_loss:.4f}")
    print(f"     Model Checkpoint: {save_path}\n")

    return {
        "success": True,
        "epochs": epoch,
        "parameters": total_params,
        "final_loss": avg_loss,
        "elapsed_minutes": total_time / 60.0,
        "model_path": save_path
    }


def main():
    parser = argparse.ArgumentParser(description="Miku LLM Trainer")
    parser.add_argument("--duration-minutes", type=float, default=None, help="Train continuously for N minutes")
    parser.add_argument("--epochs", type=int, default=70, help="Train for N epochs (ignored if duration is set)")
    args = parser.parse_args()

    train_miku_llm(epochs=args.epochs, duration_minutes=args.duration_minutes)


if __name__ == "__main__":
    main()
