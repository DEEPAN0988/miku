"""
eval/test_fixes.py — Evaluation of Fixes for Issues 1, 2, and 3

Tests:
  Task 2: Distractor Interference Mitigation via Context Truncation / Relevance
          - Test 1: Full distractor history (2 turns)
          - Test 2: Truncated history (1 turn)
          - Test 3: Zero history on QA prompts (Retriever relevance filtering: if query has 0 relevance to history, don't inject)
          Evaluated on both Phase 4 SFT (canonical_sft.pt) and Phase 5 Multi-Turn SFT (canonical_multiturn_sft.pt).

  Task 3: Fact Assertion via Natural Language Context Formatting
          - Format A (Raw KV): "Context:\nKnown facts: user_name: Jordan\n\nInstruction:\nWhat is my name?\n\nResponse:\n"
          - Format B (Natural Sentence): "Context:\nThe user's name is Jordan.\n\nInstruction:\nWhat is my name?\n\nResponse:\n"
          - Format C (Direct Natural): "Context:\nThe user's favorite color is emerald green.\n\nInstruction:\nWhat is my favorite color?\n\nResponse:\n"
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint


def generate_text(model: MikuLM, tok: MikuTokenizer, prompt: str, device: torch.device, max_tokens: int = 40) -> str:
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tok.eos_id,
        )
    gen_ids = out[0, len(ids):].tolist()
    if tok.eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(tok.eos_id)]
    return tok.decode(gen_ids).strip()


def run_task2_distractor_test(device: torch.device):
    print("=" * 80)
    print("TASK 2: DISTRACTOR INTERFERENCE MITIGATION")
    print("Testing Context Truncation & Relevance Gating on QA Baseline Prompts")
    print("=" * 80)

    cfg = ModelConfig.from_yaml("configs/phase4_sft.yaml")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    # Load both models
    m_sft = MikuLM(cfg).to(device)
    load_checkpoint("checkpoints/phase4_sft/canonical_sft.pt", m_sft, device=device)
    m_sft.eval()

    m_mt = MikuLM(cfg).to(device)
    load_checkpoint("checkpoints/phase5_multiturn_sft/canonical_multiturn_sft.pt", m_mt, device=device)
    m_mt.eval()

    qa_prompts = [
        ("QA - Geography", "What country is the city of Tokyo located in?"),
        ("QA - Science", "What planet is closest to the Sun?"),
        ("List Formatting", "List three primary colors."),
        ("Reasoning QA", "Which is larger: an elephant or an ant?"),
    ]

    hist_full = (
        "Context:\n"
        "Conversation:\n"
        "User: Hello, how are you today?\n"
        "Assistant: I am doing well, thank you for asking!\n"
        "User: What can you do?\n"
        "Assistant: I can help answer questions and summarize text.\n\n"
    )

    hist_truncated_1turn = (
        "Context:\n"
        "Conversation:\n"
        "User: What can you do?\n"
        "Assistant: I can help answer questions and summarize text.\n\n"
    )

    for cat, query in qa_prompts:
        print(f"\n--- [{cat}] '{query}' ---")

        # 1. Baseline Zero History
        p_zero = f"Instruction:\n{query}\n\nResponse:\n"
        out_sft_zero = generate_text(m_sft, tok, p_zero, device)
        out_mt_zero = generate_text(m_mt, tok, p_zero, device)

        # 2. Full History (2 turns)
        p_full = f"{hist_full}Instruction:\n{query}\n\nResponse:\n"
        out_sft_full = generate_text(m_sft, tok, p_full, device)
        out_mt_full = generate_text(m_mt, tok, p_full, device)

        # 3. Truncated History (1 turn)
        p_trunc = f"{hist_truncated_1turn}Instruction:\n{query}\n\nResponse:\n"
        out_sft_trunc = generate_text(m_sft, tok, p_trunc, device)
        out_mt_trunc = generate_text(m_mt, tok, p_trunc, device)

        print(f"  [v0.1 SFT] Zero History        : '{out_sft_zero}'")
        print(f"  [v0.1 SFT] Full (2-turn) History: '{out_sft_full}'")
        print(f"  [v0.1 SFT] Truncated (1-turn)   : '{out_sft_trunc}'")
        print(f"  [v0.2 MT-SFT] Zero History     : '{out_mt_zero}'")
        print(f"  [v0.2 MT-SFT] Full (2-turn)    : '{out_mt_full}'")
        print(f"  [v0.2 MT-SFT] Truncated (1-turn): '{out_mt_trunc}'")


def run_task3_fact_assertion_test(device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 3: FIX WEAK FACT ASSERTION VIA NATURAL LANGUAGE CONTEXT FORMATTING")
    print("Testing Raw Key-Value vs. Natural Language Sentence Context")
    print("=" * 80)

    cfg = ModelConfig.from_yaml("configs/phase4_sft.yaml")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    m_sft = MikuLM(cfg).to(device)
    load_checkpoint("checkpoints/phase4_sft/canonical_sft.pt", m_sft, device=device)
    m_sft.eval()

    m_mt = MikuLM(cfg).to(device)
    load_checkpoint("checkpoints/phase5_multiturn_sft/canonical_multiturn_sft.pt", m_mt, device=device)
    m_mt.eval()

    test_cases = [
        {
            "name": "Fact 1: User Name (Jordan)",
            "query": "What is my name?",
            "raw_kv": "Context:\nKnown facts: user_name: Jordan\n\nInstruction:\nWhat is my name?\n\nResponse:\n",
            "natural_sentence": "Context:\nThe user's name is Jordan.\n\nInstruction:\nWhat is my name?\n\nResponse:\n",
            "conversational_statement": "Context:\nConversation:\nUser: My name is Jordan.\nAssistant: Hello Jordan!\n\nInstruction:\nWhat is my name?\n\nResponse:\n",
        },
        {
            "name": "Fact 2: Favorite Color (emerald green)",
            "query": "What is my favorite color?",
            "raw_kv": "Context:\nKnown facts: favorite_color: emerald green\n\nInstruction:\nWhat is my favorite color?\n\nResponse:\n",
            "natural_sentence": "Context:\nThe user's favorite color is emerald green.\n\nInstruction:\nWhat is my favorite color?\n\nResponse:\n",
            "conversational_statement": "Context:\nConversation:\nUser: My favorite color is emerald green.\nAssistant: That is a beautiful color!\n\nInstruction:\nWhat is my favorite color?\n\nResponse:\n",
        }
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        for model_name, m in [("v0.1 SFT (canonical_sft.pt)", m_sft), ("v0.2 MT-SFT (canonical_multiturn_sft.pt)", m_mt)]:
            print(f"\nModel: {model_name}")
            out_raw = generate_text(m, tok, tc["raw_kv"], device)
            out_nat = generate_text(m, tok, tc["natural_sentence"], device)
            out_conv = generate_text(m, tok, tc["conversational_statement"], device)

            print(f"  Format A (Raw Key-Value)      : '{out_raw}'")
            print(f"  Format B (Natural Sentence)   : '{out_nat}'")
            print(f"  Format C (Conversational Turn): '{out_conv}'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running fixes evaluation on {device}...")
    run_task2_distractor_test(device)
    run_task3_fact_assertion_test(device)


if __name__ == "__main__":
    main()
