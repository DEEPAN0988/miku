"""
eval/eval_multiturn_sft.py — Evaluation of Multi-Turn SFT Model

Executes:
  Task 3: Re-runs the exact 3 multi-turn reference-resolution dialogues:
          1. Name Recall ("Alice")
          2. Pet Identity ("cat" / "Whiskers")
          3. Pronoun Resolution ("Eiffel Tower" / construction date)
  Task 4: Re-runs the 4 single-turn QA prompts (with vs without history)
          to measure distractor-interference regression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint


def load_model(ckpt_path: str, cfg_path: str, device: torch.device) -> Tuple[MikuLM, MikuTokenizer]:
    cfg = ModelConfig.from_yaml(cfg_path)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    model = MikuLM(cfg).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, tok


def run_reference_dialogues(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("=" * 80)
    print("TASK 3: MULTI-TURN REFERENCE RESOLUTION EVALUATION")
    print("Evaluating whether multi-turn SFT resolves prior-turn entities/pronouns")
    print("=" * 80)

    dialogues = [
        {
            "id": 1,
            "name": "Dialog 1: Name Recall",
            "prompt": (
                "Context:\n"
                "Conversation:\n"
                "User: Hello! My name is Alice.\n"
                "Assistant: Hello Alice! How can I help you today?\n\n"
                "Instruction:\n"
                "What is my name?\n\n"
                "Response:\n"
            ),
            "expected": "Expected: 'Alice' (Previous zero-shot output was 1 token: 'lic')",
            "target_keywords": ["alice"],
        },
        {
            "id": 2,
            "name": "Dialog 2: Pet Identity",
            "prompt": (
                "Context:\n"
                "Conversation:\n"
                "User: I have a ginger cat named Whiskers.\n"
                "Assistant: Whiskers sounds like a wonderful cat!\n\n"
                "Instruction:\n"
                "What kind of pet do I have?\n\n"
                "Response:\n"
            ),
            "expected": "Expected: 'cat' or 'Whiskers' (Previous zero-shot output: drifted to dogs)",
            "target_keywords": ["cat", "whiskers"],
        },
        {
            "id": 3,
            "name": "Dialog 3: Contextual Pronoun / Topic Follow-up",
            "prompt": (
                "Context:\n"
                "Conversation:\n"
                "User: The Eiffel Tower is located in Paris.\n"
                "Assistant: Yes, it is one of the most famous landmarks in Paris.\n\n"
                "Instruction:\n"
                "When was it built?\n\n"
                "Response:\n"
            ),
            "expected": "Expected: date/year (e.g. 1889) (Previous zero-shot output: '1480 years old... 2760')",
            "target_keywords": ["1889", "built"],
        },
    ]

    for d in dialogues:
        print(f"\n[{d['name']}]")
        print(f"Goal: {d['expected']}")
        inp_ids = tok.encode(d["prompt"], add_bos=True, add_eos=False)
        inp_t = torch.tensor([inp_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out_t = model.generate(
                prompt_tokens=inp_t,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tok.eos_id,
            )

        gen_ids = out_t[0, len(inp_ids) :].tolist()
        hit_eos = tok.eos_id in gen_ids
        if hit_eos:
            gen_ids = gen_ids[: gen_ids.index(tok.eos_id)]
        output_text = tok.decode(gen_ids).strip()

        # Check resolution
        lower_out = output_text.lower()
        matched = any(kw in lower_out for kw in d["target_keywords"])
        status = "PASS (Resolved)" if matched else "FAIL (Did not resolve)"

        print(f"Prompt ({len(inp_ids)} tokens):\n{d['prompt'].strip()}")
        print(f"Output ({len(gen_ids)} tokens, EOS={hit_eos}):\n  '{output_text}'")
        print(f"Verdict: {status}")


def run_qa_regression(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 4: SINGLE-TURN QA REGRESSION CHECK (WITH VS WITHOUT HISTORY)")
    print("Testing if conversational context degrades single-turn QA on multi-turn SFT")
    print("=" * 80)

    test_prompts = [
        ("QA - Geography", "What country is the city of Tokyo located in?"),
        ("QA - Science", "What planet is closest to the Sun?"),
        ("List Formatting", "List three primary colors."),
        ("Reasoning QA", "Which is larger: an elephant or an ant?"),
    ]

    sample_distractor_history = (
        "Context:\n"
        "Conversation:\n"
        "User: Hello, how are you today?\n"
        "Assistant: I am doing well, thank you for asking!\n"
        "User: What can you do?\n"
        "Assistant: I can help answer questions and summarize text.\n\n"
    )

    for cat, query in test_prompts:
        print(f"\n[{cat}] Prompt: '{query}'")

        # 1. Baseline (Zero History)
        prompt_clean = f"Instruction:\n{query}\n\nResponse:\n"
        inp_ids_clean = tok.encode(prompt_clean, add_bos=True, add_eos=False)
        with torch.no_grad():
            out_clean = model.generate(
                prompt_tokens=torch.tensor([inp_ids_clean], device=device),
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tok.eos_id,
            )
        clean_gen = out_clean[0, len(inp_ids_clean) :].tolist()
        clean_eos = tok.eos_id in clean_gen
        if clean_eos:
            clean_gen = clean_gen[: clean_gen.index(tok.eos_id)]
        clean_resp = tok.decode(clean_gen).strip()

        # 2. With Injected Conversation History
        prompt_with_hist = f"{sample_distractor_history}Instruction:\n{query}\n\nResponse:\n"
        inp_ids_hist = tok.encode(prompt_with_hist, add_bos=True, add_eos=False)
        with torch.no_grad():
            out_hist = model.generate(
                prompt_tokens=torch.tensor([inp_ids_hist], device=device),
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tok.eos_id,
            )
        hist_gen = out_hist[0, len(inp_ids_hist) :].tolist()
        hist_eos = tok.eos_id in hist_gen
        if hist_eos:
            hist_gen = hist_gen[: hist_gen.index(tok.eos_id)]
        hist_resp = tok.decode(hist_gen).strip()

        print(f"  BASELINE (0 history) [{len(clean_gen)} tok, EOS={clean_eos}]:")
        print(f"    '{clean_resp}'")
        print(f"  WITH HISTORY [{len(hist_gen)} tok, EOS={hist_eos}]:")
        print(f"    '{hist_resp}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/phase5_multiturn_sft/canonical_multiturn_sft.pt")
    parser.add_argument("--config", default="configs/phase5_multiturn_sft.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, tok = load_model(args.checkpoint, args.config, device)

    run_reference_dialogues(model, tok, device)
    run_qa_regression(model, tok, device)


if __name__ == "__main__":
    main()
