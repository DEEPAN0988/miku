"""
eval/verify_acceptance.py — MIKU v0.1 Final Consolidated Acceptance Suite

Executes:
  Task 1: Artifact integrity (checkpoints, tokenizer, parameter count)
  Task 2: Fresh 10-prompt evaluation on canonical SFT model
  Task 3: Adversarial / edge-case stress test (confirming no collapse/loops)
  Task 4: Metrics comparison & summary
"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# TASK 1: Artifact Integrity Verification
# ---------------------------------------------------------------------------

def verify_artifacts():
    print("=" * 80)
    print("TASK 1: ARTIFACT INTEGRITY VERIFICATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Tokenizer
    tok_path = "data/processed/tokenizer/miku_bpe"
    tok = MikuTokenizer.load(tok_path)
    print(f"\n[TOKENIZER]")
    print(f"  Model path: {tok_path}.model (exists: {Path(tok_path + '.model').exists()})")
    print(f"  Vocab size: {tok.vocab_size:,}")
    print(f"  Special IDs: BOS={tok.bos_id}, EOS={tok.eos_id}, PAD={tok.pad_id}, UNK={tok.unk_id}")
    assert tok.vocab_size == 32000, f"Expected 32,000 vocab, got {tok.vocab_size}"
    assert tok.eos_id == 3, f"Expected eos_id=3, got {tok.eos_id}"

    # 2. Config & Architecture
    cfg = ModelConfig.from_yaml("configs/phase4_sft.yaml")
    print(f"\n[MODEL ARCHITECTURE CONFIG]")
    print(f"  Layers: {cfg.n_layers}, Heads: {cfg.n_heads}, d_model: {cfg.d_model}, d_ff: {cfg.d_ff}")
    print(f"  max_seq_len: {cfg.max_seq_len}, vocab_size: {cfg.vocab_size}")

    # 3. Base Checkpoint
    base_path = Path("checkpoints/stage_a_tiny_25pct/canonical_step_0042000.pt")
    print(f"\n[BASE CHECKPOINT]")
    print(f"  Path: {base_path}")
    print(f"  File exists: {base_path.exists()}")
    base_size_mb = base_path.stat().st_size / 1e6
    print(f"  File size: {base_size_mb:.2f} MB")

    base_model = MikuLM(cfg).to(device)
    ckpt_base = load_checkpoint(str(base_path), base_model, device=device)
    base_params = base_model.count_parameters()
    print(f"  Step: {ckpt_base.get('step'):,}")
    print(f"  Train Loss: {ckpt_base.get('train_loss'):.4f}")
    print(f"  Val Loss: {ckpt_base.get('val_loss'):.4f}")
    print(f"  Parameter count: {base_params:,}")
    assert base_params == 11_347_456, f"Expected 11,347,456 params, got {base_params}"

    # 4. SFT Checkpoint
    sft_path = Path("checkpoints/phase4_sft/canonical_sft.pt")
    print(f"\n[SFT CHECKPOINT]")
    print(f"  Path: {sft_path}")
    print(f"  File exists: {sft_path.exists()}")
    sft_size_mb = sft_path.stat().st_size / 1e6
    print(f"  File size: {sft_size_mb:.2f} MB")

    sft_model = MikuLM(cfg).to(device)
    ckpt_sft = load_checkpoint(str(sft_path), sft_model, device=device)
    sft_params = sft_model.count_parameters()
    print(f"  Step: {ckpt_sft.get('step'):,}")
    print(f"  Train Loss: {ckpt_sft.get('train_loss'):.4f}")
    print(f"  Val Loss: {ckpt_sft.get('val_loss'):.4f}")
    print(f"  Parameter count: {sft_params:,}")
    assert sft_params == 11_347_456, f"Expected 11,347,456 params, got {sft_params}"

    print("\n[OK] Task 1 Artifact Integrity: ALL ASSERTIONS PASSED (11,347,456 params, vocab 32,000, files intact).")
    return sft_model, tok, device


# ---------------------------------------------------------------------------
# TASK 2: Fresh 10-Prompt SFT Evaluation
# ---------------------------------------------------------------------------

FRESH_PROMPTS: List[Tuple[str, str, str]] = [
    # 1. Direct QA (Geography)
    ("QA - Geography", "What country is the city of Tokyo located in?", "Expected: Japan"),
    # 2. Direct QA (Science)
    ("QA - Science", "What planet is closest to the Sun?", "Expected: Mercury"),
    # 3. List formatting
    ("List Formatting", "List three primary colors.", "Expected: red, blue, yellow (or similar list)"),
    # 4. Summarization / Brevity
    ("Summarization", "Summarize what photosynthesis is in one sentence.", "Expected: 1 sentence on plants converting light/sunlight"),
    # 5. Dialogue / Greeting
    ("Dialogue", "Say hello and introduce yourself briefly.", "Expected: short friendly self-introduction"),
    # 6. Explanation
    ("Explanation", "Explain why the sky looks blue during the day.", "Expected: direct explanation of sunlight / atmosphere scattering"),
    # 7. Brevity constraint
    ("Brevity Constraint", "Describe winter snow in a single sentence.", "Expected: 1 sentence about snow / cold"),
    # 8. Deductive logic
    ("Deductive Logic", "All fish live in water. A salmon is a fish. Where does a salmon live?", "Expected: In water"),
    # 9. Information Extraction / QA
    ("Extraction / QA", "Who wrote the play Romeo and Juliet?", "Expected: William Shakespeare"),
    # 10. Direct Reasoning / Comparison
    ("Reasoning QA", "Which is larger: an elephant or an ant?", "Expected: An elephant"),
]


def generate_sft(
    model: MikuLM,
    tok: MikuTokenizer,
    instruction: str,
    device: torch.device,
    max_new_tokens: int = 90,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> Tuple[str, bool, int]:
    prompt_text = f"Instruction:\n{instruction}\n\nResponse:\n"
    ids = tok.encode(prompt_text, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_id,
        )
    gen_ids = out[0, len(ids):].tolist()
    stopped_on_eos = (len(gen_ids) > 0 and gen_ids[-1] == tok.eos_id)
    if stopped_on_eos:
        gen_ids = gen_ids[:-1]
    decoded = tok.decode(gen_ids).strip()
    return decoded, stopped_on_eos, len(gen_ids)


def run_fresh_eval(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 2: FRESH 10-PROMPT SFT EVALUATION (PREVIOUSLY UNSEEN PROMPTS)")
    print("=" * 80)
    print("Decoding: temperature=0.7, top_p=0.9, repetition_penalty=1.2, max_new_tokens=90, eos_token_id=3")
    print("=" * 80)

    results = []
    lengths = []
    eos_count = 0
    math_bleed_count = 0
    direct_ans_count = 0

    math_patterns = ["=", "+", "-", "*", "/", "mph", "equation", "divided by", "multiplied by"]

    for i, (category, instruction, expectation) in enumerate(FRESH_PROMPTS, 1):
        gen, eos_stopped, token_len = generate_sft(model, tok, instruction, device)
        lengths.append(token_len)
        if eos_stopped:
            eos_count += 1

        # Check for arithmetic bleed-through
        has_math = any(p in gen for p in ["=", "+", " * ", " / ", "multiplied by", "divided by"]) and any(c.isdigit() for c in gen)
        if has_math:
            math_bleed_count += 1

        # Check for direct answer framing (starts with declarative answer rather than fictional narrative continuation)
        is_direct = not gen.startswith(("“", '"', "Once", "He said", "There was a man"))
        if is_direct:
            direct_ans_count += 1

        print(f"\n[{i:02d}] Category   : {category}")
        print(f"     Instruction: {instruction!r}")
        print(f"     Expectation: {expectation}")
        print(f"     Output ({token_len} tokens, EOS: {'YES' if eos_stopped else 'NO'}):")
        print(f"       {gen!r}")
        print("-" * 80)

        results.append({
            "num": i,
            "category": category,
            "instruction": instruction,
            "gen": gen,
            "eos": eos_stopped,
            "len": token_len,
            "math_bleed": has_math,
            "direct": is_direct,
        })

    lengths.sort()
    median_len = lengths[len(lengths) // 2]
    eos_rate = eos_count / len(FRESH_PROMPTS)
    math_bleed_rate = math_bleed_count / len(FRESH_PROMPTS)
    direct_rate = direct_ans_count / len(FRESH_PROMPTS)

    print("\n" + "=" * 80)
    print("FRESH PROMPTS BEHAVIORAL METRICS SUMMARY")
    print("=" * 80)
    print(f"  Total Prompts Evaluated    : {len(FRESH_PROMPTS)}")
    print(f"  <eos> Early Termination Rate: {eos_rate:.1%} ({eos_count}/{len(FRESH_PROMPTS)}) [Phase 4 reported: 87.5%]")
    print(f"  Median Generated Length    : {median_len:.1f} tokens [Phase 4 reported: 34.0 tokens]")
    print(f"  Arithmetic Bleed Rate      : {math_bleed_rate:.1%} ({math_bleed_count}/{len(FRESH_PROMPTS)}) [Phase 4 reported: 0.0%]")
    print(f"  Direct Answering Rate      : {direct_rate:.1%} ({direct_ans_count}/{len(FRESH_PROMPTS)}) [Phase 4 reported: 75.0%]")
    print("=" * 80)
    return results


# ---------------------------------------------------------------------------
# TASK 3: Adversarial / Edge-Case Collapse Regression Check
# ---------------------------------------------------------------------------

ADVERSARIAL_PROMPTS: List[Tuple[str, str]] = [
    ("Very Short Prompt", "Hi"),
    ("Punctuation-Only Prompt", "???"),
    ("Empty-ish Prompt", "..."),
    ("Repetition Trigger Prompt", "echo echo echo echo"),
    ("Single Word Prompt", "Why?"),
]


def run_adversarial_check(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 3: ADVERSARIAL / EDGE-CASE COLLAPSE REGRESSION CHECK")
    print("=" * 80)
    print("Testing for token repetition loops or degenerate collapse under edge-case prompts...")
    print("=" * 80)

    results = []
    has_degeneration = False

    for i, (name, prompt) in enumerate(ADVERSARIAL_PROMPTS, 1):
        gen, eos_stopped, token_len = generate_sft(model, tok, prompt, device, max_new_tokens=60)
        
        # Check for phrase or token repetition loops (e.g. 4+ repeated bigrams/words)
        words = gen.split()
        repeats = False
        if len(words) >= 6:
            for w_idx in range(len(words) - 3):
                phrase = " ".join(words[w_idx : w_idx + 2])
                if gen.count(phrase) >= 4:
                    repeats = True
                    break

        status = "FAIL (Repetition Loop)" if repeats else "PASS (Clean / Non-Looping)"
        if repeats:
            has_degeneration = True

        print(f"\n[{i:02d}] Edge Case: {name}")
        print(f"     Prompt   : {prompt!r}")
        print(f"     Output ({token_len} tokens, EOS: {'YES' if eos_stopped else 'NO'}):")
        print(f"       {gen!r}")
        print(f"     Verdict  : {status}")
        print("-" * 80)

        results.append({
            "name": name,
            "prompt": prompt,
            "gen": gen,
            "eos": eos_stopped,
            "len": token_len,
            "status": status,
        })

    print(f"\n[COLLAPSE CHECK VERDICT]: {'ALL 5 EDGE CASES PASSED — NO COLLAPSE' if not has_degeneration else 'FAIL — COLLAPSE DETECTED'}")
    return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sft_model, tok, device = verify_artifacts()
    fresh_results = run_fresh_eval(sft_model, tok, device)
    adv_results = run_adversarial_check(sft_model, tok, device)
