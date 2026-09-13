"""
train/train.py — MIKU Training Loop
STATUS: IMPLEMENTED

Trains MikuLM from scratch on the tokenized binary corpus.

Key behaviors (per project constitution):
  - Checkpoints every N steps (configurable)
  - Generation SAMPLES saved alongside EVERY checkpoint — not just loss
    This is the mechanism for catching quality/loss divergence early.
  - Loss logged to CSV after every checkpoint interval
  - Val loss computed alongside train checkpoints
  - Mixed precision (bfloat16) on CUDA / float32 on CPU
  - Hardware auto-detection at startup (prefers CUDA → MPS → CPU)
  - torch.compile() support for RTX 40-series Ada Lovelace speedup
  - num_workers config key for async DataLoader prefetch on GPU

Usage:
    python train/train.py --config configs/stage_a.yaml

    # Resume from latest checkpoint:
    python train/train.py --config configs/stage_a.yaml --resume

    # Dry run (verifies data pipeline + one forward pass, then exits):
    python train/train.py --config configs/stage_a.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Imports (project-local)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from train.dataset import create_dataloader


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_device() -> torch.device:
    """
    Detect the best available device and print hardware info.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"[HW] CUDA device: {props.name}")
        print(f"     VRAM: {vram_gb:.1f} GB")
        print(f"     Compute capability: {props.major}.{props.minor}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[HW] Apple MPS device detected")
    else:
        device = torch.device("cpu")
        print("[HW] No GPU detected - training on CPU (slow)")

    ram_gb = _get_system_ram_gb()
    if ram_gb:
        print(f"     System RAM: {ram_gb:.1f} GB")

    return device


def _get_system_ram_gb() -> Optional[float]:
    """Return system RAM in GB, or None if unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        return None


def select_dtype(device: torch.device, config_dtype: str) -> torch.dtype:
    """
    Select training precision.

    auto → bfloat16 on CUDA (Ampere+) or MPS, float32 otherwise.
    """
    if config_dtype == "float32":
        return torch.float32
    elif config_dtype == "float16":
        return torch.float16
    elif config_dtype == "bfloat16":
        return torch.bfloat16
    else:  # auto
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:  # Ampere or newer supports bfloat16 well
                return torch.bfloat16
            else:
                return torch.float16
        elif device.type == "mps":
            return torch.float32   # MPS has limited bfloat16 support
        else:
            return torch.float32


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Linear warmup followed by cosine decay.

    - Steps 0..warmup_steps:  linear ramp from 0 → base_lr
    - Steps warmup_steps..max_steps: cosine decay from base_lr → min_lr
    - After max_steps: constant min_lr
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)

    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_factor * (base_lr - min_lr)


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_loss(
    model: MikuLM,
    val_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> float:
    """
    Compute average cross-entropy loss on the validation set.

    Caps at max_batches to keep eval time bounded.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Generation samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: MikuLM,
    tokenizer: MikuTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: Optional[float] = None,
) -> List[Dict[str, str]]:
    """
    Generate verbatim text samples from a list of prompt strings.

    These are saved alongside every checkpoint so quality can be
    tracked independent of the loss curve.

    Returns:
        List of {"prompt": str, "generated": str} dicts.
    """
    model.eval()
    samples = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        try:
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_p is None else None,
                top_p=top_p,
                eos_token_id=tokenizer.eos_id,
            )
            # Decode only the newly generated tokens (not the prompt)
            new_ids = output[0, len(ids):].tolist()
            generated_text = tokenizer.decode(new_ids)
        except Exception as e:
            generated_text = f"[GENERATION ERROR: {e}]"

        samples.append({"prompt": prompt, "generated": generated_text})

    model.train()
    return samples


# ---------------------------------------------------------------------------
# Loss CSV logging
# ---------------------------------------------------------------------------

def append_loss_log(
    log_path: str,
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    lr: float,
    tokens_per_sec: float,
) -> None:
    """Append one row to the loss CSV log. Creates file + header if missing."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])
        writer.writerow([
            step,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.8f}",
            f"{tokens_per_sec:.1f}",
        ])


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_training_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg = load_training_config(args.config)

    # ── Hardware ──────────────────────────────────────────────────────
    device = detect_device()
    dtype = select_dtype(
        device,
        cfg.get("training", {}).get("dtype", "auto"),
    )
    print(f"[SETUP] Training dtype: {dtype}")

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = ModelConfig.from_yaml(args.config)
    print(f"[MODEL] {model_cfg.summary()}")

    model = MikuLM(model_cfg).to(device)
    n_params = model.count_parameters()
    print(f"[MODEL] Trainable parameters: {n_params:,}")

    # ── torch.compile (RTX 40-series / Ada Lovelace) ──────────────────
    train_cfg_raw = cfg.get("training", {})
    use_compile = bool(train_cfg_raw.get("compile", False))
    if use_compile:
        if device.type == "cuda" and hasattr(torch, "compile"):
            print("[COMPILE] Compiling model with torch.compile (mode='reduce-overhead') ...")
            model = torch.compile(model, mode="reduce-overhead")
            print("[COMPILE] Done.")
        else:
            print("[COMPILE] torch.compile skipped (not on CUDA or torch too old).")

    # Print VRAM used so far (after model load + optional compile)
    if device.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated(0) / 1024**2
        reserved_mb  = torch.cuda.memory_reserved(0)  / 1024**2
        total_gb     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[VRAM]  allocated={allocated_mb:.0f} MB  reserved={reserved_mb:.0f} MB  "
              f"total={total_gb:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tok_prefix = cfg["tokenizer"]["model_prefix"]
    tokenizer = MikuTokenizer.load(tok_prefix)
    print(f"[TOKENIZER] {tokenizer}")

    # ── Data ──────────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    seq_len = model_cfg.max_seq_len

    # num_workers: 0 is safe everywhere; >0 enables async prefetch on GPU
    num_workers = int(train_cfg.get("num_workers", 0))
    # On Windows, num_workers > 0 requires spawn — only safe with CUDA
    if num_workers > 0 and device.type != "cuda":
        print(f"[DATA] num_workers={num_workers} ignored on {device.type} (set to 0)")
        num_workers = 0
    pin_memory = device.type == "cuda"

    train_loader = create_dataloader(
        bin_path=data_cfg["train_bin"],
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        bin_path=data_cfg["val_bin"],
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"[DATA] Train: {train_loader.dataset}")
    print(f"[DATA] Val:   {val_loader.dataset}")

    if args.dry_run:
        print("\n[DRY RUN] Attempting one forward pass ...")
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits, loss = model(x, y)
        print(f"[DRY RUN] Forward pass OK: logits={logits.shape}, loss={loss.item():.4f}")
        print("[DRY RUN] Complete. Exiting.")
        return

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
    )

    # ── Resume ────────────────────────────────────────────────────────
    ckpt_dir = train_cfg["checkpoint_dir"]
    log_dir = train_cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    start_step = 0
    if args.resume or args.resume_from:
        if args.resume_from:
            latest = args.resume_from
            print(f"[RESUME] Loading specific checkpoint: {latest}")
        else:
            latest = find_latest_checkpoint(ckpt_dir)
            if latest:
                print(f"[RESUME] Loading checkpoint: {latest}")
            else:
                print("[RESUME] No checkpoint found — starting from scratch")
                latest = None
        if latest:
            warm_restart = getattr(args, 'warm_restart', False)
            if warm_restart:
                # Warm restart: load model weights only.
                # Optimizer state and step counter are NOT restored so the
                # new LR schedule runs a full cosine decay from step 0.
                ckpt = load_checkpoint(latest, model, optimizer=None, device=device)
                start_step = 0
                print(f"[WARM RESTART] Loaded weights from step {ckpt['step']} "
                      f"(train_loss={ckpt.get('train_loss', '?'):.4f}). "
                      f"Step counter and optimizer reset to 0.")
            else:
                ckpt = load_checkpoint(latest, model, optimizer, device)
                start_step = ckpt["step"]
                print(f"[RESUME] Resuming from step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────
    max_steps = train_cfg["max_steps"]
    ckpt_every = train_cfg["checkpoint_every_n_steps"]
    log_every = train_cfg["log_every_n_steps"]
    sample_every = train_cfg["sample_every_n_steps"]
    grad_clip = train_cfg["grad_clip"]
    base_lr = train_cfg["learning_rate"]
    warmup_steps = train_cfg["lr_warmup_steps"]
    min_lr = base_lr * train_cfg["lr_min_factor"]
    eval_max_batches = cfg["eval"]["eval_max_batches"]

    sample_prompts = train_cfg["sample_prompts"]
    sample_max_tokens = train_cfg["sample_max_new_tokens"]
    sample_temperature = train_cfg["sample_temperature"]
    sample_top_k = train_cfg["sample_top_k"]

    log_path = os.path.join(log_dir, "loss.csv")

    print(f"\n[TRAIN] Starting at step {start_step}, max={max_steps}")
    print(f"        Effective batch: {batch_size * grad_accum} sequences x {seq_len} tokens")
    print(f"        Checkpoint every {ckpt_every} steps")
    print(f"        Samples saved every {sample_every} steps (separate from loss)\n")

    model.train()
    optimizer.zero_grad()

    step = start_step
    accum_loss = 0.0
    accum_count = 0
    t_start = time.time()
    tokens_seen = 0
    best_val_loss = float("inf")
    best_val_step = 0
    consecutive_val_rises = 0

    # Infinite epoch loop — step limit controls termination
    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break

            x, y = x.to(device), y.to(device)

            # ── Forward pass ──────────────────────────────────────────
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                _, loss = model(x, y)
                loss_scaled = loss / grad_accum

            loss_scaled.backward()

            accum_loss += loss.item()
            accum_count += 1
            tokens_seen += x.numel()

            if accum_count < grad_accum:
                continue  # Accumulate more gradients before stepping

            # ── Optimizer step ────────────────────────────────────────
            lr = get_lr(step, warmup_steps, max_steps, base_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            mean_loss = accum_loss / accum_count
            accum_loss = 0.0
            accum_count = 0
            step += 1

            # -- Logging -----------------------------------------------
            if step % log_every == 0:
                elapsed = time.time() - t_start
                tps = tokens_seen / max(elapsed, 1e-9)
                print(
                    f"step {step:>7,} | loss {mean_loss:.4f} | "
                    f"lr {lr:.2e} | {tps:.0f} tok/s",
                    flush=True,
                )

            # -- Checkpoint + samples -----------------------------------
            if step % ckpt_every == 0:
                val_loss = compute_val_loss(
                    model, val_loader, device, dtype, max_batches=eval_max_batches
                )

                # Samples ALWAYS saved with checkpoint - not just loss
                samples = generate_samples(
                    model, tokenizer, sample_prompts, device,
                    max_new_tokens=sample_max_tokens,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                )

                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=mean_loss,
                    config_dict=model_cfg.to_dict(),
                    checkpoint_dir=ckpt_dir,
                    val_loss=val_loss,
                    samples=samples,
                )

                # Log to CSV
                elapsed = time.time() - t_start
                tps = tokens_seen / max(elapsed, 1e-9)
                append_loss_log(log_path, step, mean_loss, val_loss, lr, tps)

                print(
                    f"  [OK] CHECKPOINT step={step} | "
                    f"train_loss={mean_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"-> {ckpt_path}",
                    flush=True,
                )

                # Print fixed prompt sample to console for monitoring
                if samples:
                    s = samples[0]
                    try:
                        print(f"  [MONITOR PROMPT]: {s['prompt']!r}", flush=True)
                        print(f"  [MONITOR GENERATED]: {s['generated']!r}", flush=True)
                    except Exception:
                        safe_gen = s['generated'].encode('ascii', errors='backslashreplace').decode('ascii')
                        print(f"  [MONITOR GENERATED (ASCII)]: {safe_gen!r}", flush=True)

                # Overfitting knee check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    consecutive_val_rises = 0
                else:
                    consecutive_val_rises += 1
                    print(
                        f"  [OVERFITTING WATCH] val_loss ({val_loss:.4f}) >= best ({best_val_loss:.4f} at step {best_val_step}). "
                        f"Consecutive rises: {consecutive_val_rises}/{getattr(args, 'early_stop_rises', 0)}",
                        flush=True,
                    )
                    if getattr(args, "early_stop_rises", 0) > 0 and consecutive_val_rises >= args.early_stop_rises:
                        print(
                            f"\n[EARLY STOP] Overfitting knee reached: val loss rose {consecutive_val_rises} consecutive "
                            f"checkpoints past min ({best_val_loss:.4f} at step {best_val_step}). Halting training.\n",
                            flush=True,
                        )
                        break

        if getattr(args, "early_stop_rises", 0) > 0 and consecutive_val_rises >= args.early_stop_rises:
            break

    print(f"\n[TRAIN] Finished at step {step}.")
    print(f"        Loss log: {log_path}")
    print(f"        Checkpoints: {ckpt_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MIKU from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML training config (e.g. configs/stage_a.yaml)"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint in checkpoint_dir"
    )
    p.add_argument(
        "--resume-from", dest="resume_from", default=None, metavar="PATH",
        help="Resume from a specific checkpoint file (overrides --resume)"
    )
    p.add_argument(
        "--warm-restart", dest="warm_restart", action="store_true",
        help=(
            "With --resume-from: load model weights only. "
            "Optimizer state and step counter are reset to 0. "
            "Use for LR warm restart diagnostics."
        )
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Verify data pipeline + one forward pass, then exit"
    )
    p.add_argument(
        "--early-stop-rises", type=int, default=0,
        help="Halt training when val loss rises consecutively for N checkpoints above the min"
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
