"""
eval/eval_sft_prompts.py — Comparative Evaluation on Instruction Prompts

Compares completion behavior of:
  1. Base Pretrained Model (Stage A Canonical: canonical_step_0042000.pt)
  2. Supervised Instruction Tuned Model (Phase 4 SFT Canonical)

Evaluates on 8 instruction-style prompts testing directness, termination (<eos>),
and simple format/brevity adherence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint

INSTRUCTION_PROMPTS: List[Tuple[str, str, str]] = [
    (
        "Brevity / Summarization",
        "Summarize the water cycle in one sentence.",
        "Expected: 1 sentence explaining evaporation, condensation, precipitation.",
    ),
    (
        "List Formatting",
        "List three facts about France.",
        "Expected: numbered or bulleted list of 3 facts.",
    ),
    (
        "Persona / Dialogue",
        "Write a short greeting to a friend.",
        "Expected: friendly, concise greeting without narrative babble.",
    ),
    (
        "Explanation",
        "Explain what a solar system is in simple terms.",
        "Expected: direct explanation of sun and orbiting bodies.",
    ),
    (
        "Brevity Constraint",
        "Write a one-sentence description of autumn leaves.",
        "Expected: single sentence about falling/changing leaves.",
    ),
    (
        "Direct Factual QA",
        "What is the capital of France?",
        "Expected: direct answer naming Paris.",
    ),
    (
        "Direct Science QA",
        "What do birds have that allow them to fly?",
        "Expected: direct answer naming wings/feathers.",
    ),
    (
        "Deductive Logic QA",
        "If all roses are flowers and all flowers need water, do roses need water?",
        "Expected: 'Yes' with brief explanation.",
    ),
]


def format_instruction_prompt(instruction: str) -> str:
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def generate_completion(
    model: MikuLM,
    tokenizer: MikuTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 90,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> Tuple[str, bool, int]:
    """
    Returns (decoded_response, stopped_on_eos, tokens_generated).
    """
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_id,
        )
    gen_ids = out[0, len(ids):].tolist()
    stopped_on_eos = (len(gen_ids) > 0 and gen_ids[-1] == tokenizer.eos_id)
    if stopped_on_eos:
        # Strip trailing EOS token ID before decoding
        gen_ids = gen_ids[:-1]
    decoded = tokenizer.decode(gen_ids).strip()
    return decoded, stopped_on_eos, len(gen_ids)


def run_comparative_eval(
    base_ckpt: str = "checkpoints/stage_a_tiny_25pct/canonical_step_0042000.pt",
    sft_ckpt: str = "checkpoints/phase4_sft/canonical_sft.pt",
    config_path: str = "configs/phase4_sft.yaml",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MIKU PHASE 4: SFT BEHAVIORAL EVALUATION ON INSTRUCTION PROMPTS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Base Pretrained Model : {base_ckpt}")
    print(f"Phase 4 SFT Model     : {sft_ckpt}")
    print("=" * 80)

    cfg = ModelConfig.from_yaml(config_path)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    # Load base model
    print(f"\n[LOAD] Loading Base Pretrained Checkpoint: {base_ckpt} ...")
    base_model = MikuLM(cfg).to(device)
    load_checkpoint(base_ckpt, base_model, device=device)
    base_model.eval()

    # Load SFT model
    print(f"[LOAD] Loading SFT Fine-Tuned Checkpoint: {sft_ckpt} ...")
    sft_model = MikuLM(cfg).to(device)
    load_checkpoint(sft_ckpt, sft_model, device=device)
    sft_model.eval()

    results = []

    for i, (task_type, instruction, criteria) in enumerate(INSTRUCTION_PROMPTS, 1):
        formatted_prompt = format_instruction_prompt(instruction)

        base_gen, base_eos, base_len = generate_completion(base_model, tok, formatted_prompt, device)
        sft_gen, sft_eos, sft_len = generate_completion(sft_model, tok, formatted_prompt, device)

        print(f"\n[{i:02d}] Task: {task_type.upper()}")
        print(f"     Instruction: {instruction!r}")
        print(f"     Criteria   : {criteria}")
        print("-" * 80)
        print(f"  [BASE PRETRAINED] (EOS: {'YES' if base_eos else 'NO'}, Tokens: {base_len}):")
        print(f"    {base_gen!r}")
        print()
        print(f"  [PHASE 4 SFT]     (EOS: {'YES' if sft_eos else 'NO'}, Tokens: {sft_len}):")
        print(f"    {sft_gen!r}")
        print("=" * 80)

        results.append({
            "num": i,
            "task": task_type,
            "instruction": instruction,
            "base_gen": base_gen,
            "base_eos": base_eos,
            "base_len": base_len,
            "sft_gen": sft_gen,
            "sft_eos": sft_eos,
            "sft_len": sft_len,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF BEHAVIORAL CHANGES (BASE vs SFT)")
    print("=" * 80)
    print(f"{'#':<3} | {'Task':<22} | {'Base Stopped (<eos>)':<22} | {'SFT Stopped (<eos>)':<20} | {'SFT Direct Ans'}")
    print("-" * 80)
    for r in results:
        base_stop_str = f"{'YES' if r['base_eos'] else 'NO'} ({r['base_len']} tok)"
        sft_stop_str = f"{'YES' if r['sft_eos'] else 'NO'} ({r['sft_len']} tok)"
        print(f"{r['num']:<3} | {r['task']:<22} | {base_stop_str:<22} | {sft_stop_str:<20} | ")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT instruction prompts")
    parser.add_argument("--base-checkpoint", default="checkpoints/stage_a_tiny_25pct/canonical_step_0042000.pt")
    parser.add_argument("--sft-checkpoint", default="checkpoints/phase4_sft/canonical_sft.pt")
    parser.add_argument("--config", default="configs/phase4_sft.yaml")
    args = parser.parse_args()

    run_comparative_eval(
        base_ckpt=args.base_checkpoint,
        sft_ckpt=args.sft_checkpoint,
        config_path=args.config,
    )
