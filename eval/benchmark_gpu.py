"""
eval/benchmark_gpu.py — Explicit GPU Benchmark for MIKU Architecture
STATUS: IMPLEMENTED

Measures real forward+backward pass throughput on the requested device (cuda or cpu).
Prints exact tokens/second and allocated/reserved VRAM.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig


def run_benchmark(device_str: str = "cuda", preset: str = "tiny", batch_size: int = 16, seq_len: int = 256, n_steps: int = 20):
    print("=" * 65)
    print(f"MIKU HARDWARE BENCHMARK: {device_str.upper()} | Preset: {preset}")
    print("=" * 65)

    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("\n[ERROR] CUDA is not available. Device count = 0.")
            print("Cannot benchmark GPU until hardware MUX switch is enabled in Lenovo Vantage.")
            sys.exit(1)
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {props.name}")
        print(f"Total VRAM: {props.total_memory / 1e9:.2f} GB ({props.total_memory / 1024**3:.2f} GiB)")
        dtype = torch.bfloat16 if props.major >= 8 else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("Device: CPU (Intel Core i5-HX)")

    cfg_dict = {
        "tiny": {"n_layers": 4, "n_heads": 4, "d_model": 256, "d_ff": 1024, "max_seq_len": seq_len},
        "small": {"n_layers": 12, "n_heads": 8, "d_model": 512, "d_ff": 2048, "max_seq_len": seq_len},
    }[preset]

    model_cfg = ModelConfig(**cfg_dict)
    model = MikuLM(model_cfg).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params:,} (~{params/1e6:.1f}M)")
    print(f"Precision: {dtype}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len} ({batch_size * seq_len:,} tokens/step)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    # Warmup steps
    print(f"\nWarming up (3 steps)...")
    for _ in range(3):
        x = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed benchmark
    print(f"Running {n_steps} timed benchmark steps...")
    t0 = time.perf_counter()
    total_tokens = 0

    for step in range(n_steps):
        x = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch_size * seq_len

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    tps = total_tokens / elapsed

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Device:               {device_str.upper()}")
    print(f"  Total tokens timed:   {total_tokens:,}")
    print(f"  Elapsed time:         {elapsed:.3f} s")
    print(f"  Throughput:           {tps:,.1f} tokens/second")

    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(0) / 1024**2
        res_mb = torch.cuda.memory_reserved(0) / 1024**2
        print(f"  VRAM allocated:       {alloc_mb:.1f} MB")
        print(f"  VRAM reserved:        {res_mb:.1f} MB")
    print("=" * 65)


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    preset = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    run_benchmark(dev, preset)
