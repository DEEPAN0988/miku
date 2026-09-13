"""
train/train_sft.py — Supervised Instruction Tuning Loop for MIKU

Fine-tunes the canonical Stage A checkpoint on Dolly-15k using response-only target masking.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint, save_checkpoint
from train.sft_dataset import create_sft_dataloader


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[HW] CUDA device: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
        return device
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


@torch.no_grad()
def compute_val_loss(
    model: MikuLM,
    val_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 25,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def generate_samples(
    model: MikuLM,
    tokenizer: MikuTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    model.eval()
    results = []
    for p in prompts:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_id,
            )
        gen_ids = out[0, len(ids):].tolist()
        gen = tokenizer.decode(gen_ids).strip()
        results.append({"prompt": p, "generated": gen})
    model.train()
    return results


def train_sft(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("MIKU PHASE 4: SUPERVISED INSTRUCTION TUNING (SFT)")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = detect_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load tokenizer
    tok = MikuTokenizer.load(cfg["tokenizer"]["model_prefix"])
    print(f"[TOKENIZER] Vocab size: {tok.vocab_size} | EOS ID: {tok.eos_id}")

    # Build model
    model_cfg = ModelConfig.from_yaml(args.config)
    model = MikuLM(model_cfg).to(device)
    print(f"[MODEL] {model_cfg.summary()} | params={model.count_parameters():,}")

    # Warm start: load Stage A pretrained weights
    warm_start_path = args.warm_start or "checkpoints/stage_a_tiny_25pct/canonical_step_0042000.pt"
    if not os.path.exists(warm_start_path):
        raise FileNotFoundError(f"Base checkpoint not found at: {warm_start_path}")
    print(f"[WARM START] Loading Stage A pretrained weights from: {warm_start_path}")
    ckpt = load_checkpoint(warm_start_path, model, optimizer=None, device=device)
    print(f"             Stage A checkpoint step={ckpt.get('step')} | train_loss={ckpt.get('train_loss'):.4f} | val_loss={ckpt.get('val_loss')}")

    # Data loaders
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    max_seq_len = data_cfg.get("max_seq_len", 256)

    print(f"[DATA] Loading Dolly-15k SFT dataset from {data_cfg['train_jsonl']} ...")
    train_loader = create_sft_dataloader(data_cfg["train_jsonl"], tok, batch_size=batch_size, max_seq_len=max_seq_len, shuffle=True)
    val_loader = create_sft_dataloader(data_cfg["val_jsonl"], tok, batch_size=batch_size, max_seq_len=max_seq_len, shuffle=False)
    print(f"       Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    if args.dry_run:
        print("\n[DRY RUN] Performing one forward pass with masked loss ...")
        model.eval()
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits, loss = model(x, y)
        print(f"[DRY RUN] Forward pass OK: logits={logits.shape}, loss={loss.item():.4f}")
        print("[DRY RUN] Complete. Exiting.")
        return

    # Optimizer (fresh optimizer with lower SFT learning rate)
    base_lr = train_cfg["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
    )

    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["lr_warmup_steps"]
    min_lr = base_lr * train_cfg.get("lr_min_factor", 0.1)
    ckpt_every = train_cfg["checkpoint_every_n_steps"]
    log_every = train_cfg["log_every_n_steps"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    sample_prompts = train_cfg.get("sample_prompts", [])

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    log_dir = Path(train_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "loss.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

    print(f"\n[SFT] Starting training: max_steps={max_steps} | effective_batch={batch_size * grad_accum}")
    print(f"      Base LR={base_lr:.2e} | Warmup={warmup_steps} | Min LR={min_lr:.2e}")
    print(f"      Checkpoints every {ckpt_every} steps | Early stop threshold: {args.early_stop_rises} rises\n")

    model.train()
    optimizer.zero_grad()

    step = 0
    accum_loss = 0.0
    accum_count = 0
    tokens_seen = 0
    t_start = time.time()

    best_val_loss = float("inf")
    best_step = 0
    consecutive_rises = 0

    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            tokens_seen += (y != -1).sum().item()

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                _, loss = model(x, y)
                loss_scaled = loss / grad_accum

            loss_scaled.backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count < grad_accum:
                continue

            # Optimizer step
            lr = get_lr(step, warmup_steps, max_steps, base_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            mean_train_loss = accum_loss / accum_count
            accum_loss = 0.0
            accum_count = 0
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t_start
                tps = tokens_seen / max(elapsed, 1e-9)
                print(f"step {step:>6,} | loss {mean_train_loss:.4f} | lr {lr:.2e} | {tps:.0f} tok/s", flush=True)

            if step % ckpt_every == 0:
                val_loss = compute_val_loss(model, val_loader, device, dtype, max_batches=30)
                samples = generate_samples(model, tok, sample_prompts, device)

                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=mean_train_loss,
                    config_dict=model_cfg.to_dict(),
                    checkpoint_dir=str(ckpt_dir),
                    val_loss=val_loss,
                    samples=samples,
                )

                elapsed = time.time() - t_start
                tps = tokens_seen / max(elapsed, 1e-9)
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([step, f"{mean_train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}", f"{tps:.1f}"])

                print(
                    f"  [CHECKPOINT] step={step} | train_loss={mean_train_loss:.4f} | val_loss={val_loss:.4f} | -> {ckpt_path}",
                    flush=True,
                )
                if samples:
                    print(f"  [SAMPLE]: {samples[0]['prompt'].replace(chr(10), ' ')!r}", flush=True)
                    print(f"  [GEN]   : {samples[0]['generated']!r}", flush=True)

                # Overfitting knee check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    consecutive_rises = 0
                else:
                    consecutive_rises += 1
                    print(
                        f"  [KNEE WATCH] val_loss ({val_loss:.4f}) >= best ({best_val_loss:.4f} at step {best_step}). "
                        f"Consecutive rises: {consecutive_rises}/{args.early_stop_rises}",
                        flush=True,
                    )
                    if args.early_stop_rises > 0 and consecutive_rises >= args.early_stop_rises:
                        print(
                            f"\n[EARLY STOP] Overfitting knee reached: val loss rose {consecutive_rises} consecutive checkpoints "
                            f"past min ({best_val_loss:.4f} at step {best_step}). Halting SFT training.\n",
                            flush=True,
                        )
                        break

            if step >= max_steps:
                break

        if args.early_stop_rises > 0 and consecutive_rises >= args.early_stop_rises:
            break

    print(f"\n[SFT] Training finished at step {step}.")
    print(f"      Best checkpoint: step {best_step} (val_loss={best_val_loss:.4f})")
    print(f"      Loss log: {log_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MIKU Supervised Instruction Tuning (SFT)")
    p.add_argument("--config", default="configs/phase4_sft.yaml", help="Path to SFT YAML config")
    p.add_argument("--warm-start", default="checkpoints/stage_a_tiny_25pct/canonical_step_0042000.pt", help="Base Stage A checkpoint")
    p.add_argument("--early-stop-rises", type=int, default=2, help="Halt on N consecutive val loss rises")
    p.add_argument("--dry-run", action="store_true", help="Perform one forward pass and exit")
    return p


if __name__ == "__main__":
    train_sft(build_parser().parse_args())
