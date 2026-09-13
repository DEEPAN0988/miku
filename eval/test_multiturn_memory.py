"""
eval/test_multiturn_memory.py — MIKU v0.2 Verification Suite

Covers:
  Task 1: Multi-turn context within a single session
          - Tokenizer fragmentation check for conversation delimiters
          - Context window budget calculation (256 tokens)
          - Untrained in-context multi-turn reference test (sequential & context-prefix)
  Task 2: Local persistent memory (cross-session)
          - Cross-session storage & keyword retrieval prototype (Session 1 -> Session 2)
  Task 3: Capability check & single-turn QA regression test
          - Direct comparison of QA outputs with and without conversation history
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.retriever import MemoryRetriever
from memory.session import MikuSession
from memory.storage import MikuMemoryStore
from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint


def load_canonical_sft(device: torch.device) -> Tuple[MikuLM, MikuTokenizer, ModelConfig]:
    cfg = ModelConfig.from_yaml("configs/phase4_sft.yaml")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    model = MikuLM(cfg).to(device)
    ckpt_path = "checkpoints/phase4_sft/canonical_sft.pt"
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, tok, cfg


# ==============================================================================
# TASK 1: MULTI-TURN CONTEXT WITHIN A SINGLE SESSION
# ==============================================================================

def run_task1_tokenization_and_budget(tok: MikuTokenizer, cfg: ModelConfig):
    print("=" * 80)
    print("TASK 1.1: CONVERSATION DELIMITER TOKENIZER FRAGMENTATION CHECK")
    print("=" * 80)

    tags_to_test = [
        "Instruction:",
        "Response:",
        "Context:",
        "User:",
        "Assistant:",
        "[INSTRUCTION]",
        "[RESPONSE]",
        "[USER]",
        "[ASSISTANT]",
        "<|user|>",
        "<|assistant|>",
    ]

    print(f"{'Delimiter':<18} | {'Token IDs':<40} | {'Pieces':<35} | {'Count'}")
    print("-" * 105)
    for tag in tags_to_test:
        ids = tok.encode(tag, add_bos=False, add_eos=False)
        pieces = [tok._sp.id_to_piece(i) for i in ids]
        print(f"{tag:<18} | {str(ids):<40} | {str(pieces):<35} | {len(ids)}")

    print("\n" + "=" * 80)
    print("TASK 1.2: CONTEXT WINDOW & TURN BUDGET CALCULATION")
    print("=" * 80)
    print(f"Model Max Sequence Length (max_seq_len): {cfg.max_seq_len} tokens")
    print(f"Reserved Generation Space              : 64 tokens (allows 1-3 sentences)")
    max_prompt_budget = cfg.max_seq_len - 64
    print(f"Maximum Prompt Context Budget          : {max_prompt_budget} tokens")

    avg_user_len = 15    # e.g. "What is my name?" or "What pet do I have?"
    avg_asst_len = 35    # e.g. median SFT answer length ~34-37 tokens
    tags_overhead = 8    # Instruction:\n\nResponse:\n
    turn_cost = avg_user_len + avg_asst_len + tags_overhead  # ~58 tokens per round-trip

    max_turns = max_prompt_budget // turn_cost
    print(f"\nRealistic Turn Budget Analysis:")
    print(f"  - Average user query       : ~{avg_user_len} tokens")
    print(f"  - Average assistant reply  : ~{avg_asst_len} tokens")
    print(f"  - Template tags overhead   : ~{tags_overhead} tokens")
    print(f"  - Cost per full turn pair  : ~{turn_cost} tokens")
    print(f"  - Maximum full turns before sliding-window truncation: {max_turns} turns")
    print(f"  -> Explicit Conclusion: Multi-turn at 256 tokens strictly means 2-3 short turns.")


def run_task1_incontext_multiturn(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 1.3: ZERO-SHOT MULTI-TURN IN-CONTEXT REFERENCE TEST")
    print("Testing whether untrained SFT model can reference prior turns in-context")
    print("=" * 80)

    test_conversations = [
        {
            "name": "Dialog 1: Name Recall",
            "turns": [
                ("Hello! My name is Alice.", "Hello Alice! How can I help you today?"),
                ("What is my name?", None),
            ],
            "expected": "Expected reference to 'Alice'",
        },
        {
            "name": "Dialog 2: Pet Identity",
            "turns": [
                ("I have a ginger cat named Whiskers.", "Whiskers sounds like a wonderful cat!"),
                ("What kind of pet do I have?", None),
            ],
            "expected": "Expected reference to 'cat' or 'Whiskers'",
        },
        {
            "name": "Dialog 3: Contextual Pronoun / Topic Follow-up",
            "turns": [
                ("The Eiffel Tower is located in Paris.", "Yes, it is one of the most famous landmarks in Paris."),
                ("When was it built?", None),
            ],
            "expected": "Expected date / understanding 'it' = Eiffel Tower",
        },
    ]

    formats = ["sequential", "context_prefix"]

    for fmt in formats:
        print(f"\n--- FORMAT MODE: {fmt.upper()} ---")
        for diag in test_conversations:
            print(f"\nScenario: {diag['name']} ({diag['expected']})")
            # Build prompt
            history = diag["turns"][:-1]
            last_user_query = diag["turns"][-1][0]

            if fmt == "sequential":
                prompt_parts = []
                for u, a in history:
                    prompt_parts.append(f"Instruction:\n{u}\n\nResponse:\n{a}\n\n")
                prompt_parts.append(f"Instruction:\n{last_user_query}\n\nResponse:\n")
                prompt = "".join(prompt_parts)
            else:
                conv_lines = []
                for u, a in history:
                    conv_lines.append(f"User: {u}\nAssistant: {a}")
                prompt = f"Context:\nConversation:\n" + "\n".join(conv_lines) + f"\n\nInstruction:\n{last_user_query}\n\nResponse:\n"

            inp_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp_tensor = torch.tensor([inp_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                out_tensor = model.generate(
                    prompt_tokens=inp_tensor,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    eos_token_id=tok.eos_id,
                )

            gen_ids = out_tensor[0, len(inp_ids) :].tolist()
            hit_eos = tok.eos_id in gen_ids
            if hit_eos:
                gen_ids = gen_ids[: gen_ids.index(tok.eos_id)]
            response = tok.decode(gen_ids).strip()

            print(f"Prompt ({len(inp_ids)} tokens):\n{prompt.strip()}")
            print(f"Generated ({len(gen_ids)} tokens, EOS={hit_eos}):\n  '{response}'")


# ==============================================================================
# TASK 2: LOCAL PERSISTENT MEMORY (CROSS-SESSION)
# ==============================================================================

def run_task2_persistent_memory(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 2: LOCAL PERSISTENT MEMORY (CROSS-SESSION PROTOTYPE)")
    print("=" * 80)

    db_path = "data/memory/test_miku_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = MikuMemoryStore(db_path=db_path)

    # SESSION 1
    print("\n--- SESSION 1 (User establishes preferences/facts) ---")
    session_1 = MikuSession(model, tok, store=store, session_id="sess_001", device=device)

    turn1_input = "Hi Miku! My name is Jordan and my favorite color is emerald green."
    res1 = session_1.chat(turn1_input)
    print(f"User     : {turn1_input}")
    print(f"Stored Facts in DB: {store.get_all_facts()}")
    print(f"Assistant: {res1['response']} (tokens={res1['num_tokens']}, EOS={res1['hit_eos']})")

    # SIMULATE SESSION CLOSE & RE-OPEN
    print("\n--- [SESSION 1 CLOSED] ---")
    del session_1

    print("\n--- [SESSION 2 OPENED (New Session ID, Fresh Context Buffer)] ---")
    store_reopened = MikuMemoryStore(db_path=db_path)
    session_2 = MikuSession(model, tok, store=store_reopened, session_id="sess_002", device=device)

    turn2_input = "What is my favorite color?"
    res2 = session_2.chat(turn2_input)
    print(f"User     : {turn2_input}")
    print(f"Injected Prompt ({res2['prompt_tokens_len']} tokens):\n{res2['prompt'].strip()}")
    print(f"Assistant: {res2['response']} (tokens={res2['num_tokens']}, EOS={res2['hit_eos']})")

    turn3_input = "What is my name?"
    res3 = session_2.chat(turn3_input)
    print(f"\nUser     : {turn3_input}")
    print(f"Injected Prompt ({res3['prompt_tokens_len']} tokens):\n{res3['prompt'].strip()}")
    print(f"Assistant: {res3['response']} (tokens={res3['num_tokens']}, EOS={res3['hit_eos']})")


# ==============================================================================
# TASK 3: HONEST CAPABILITY CHECK & SINGLE-TURN REGRESSION EVALUATION
# ==============================================================================

def run_task3_regression_check(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 3: REGRESSION CHECK — WITH VS WITHOUT INJECTED HISTORY")
    print("Testing if conversational context degrades single-turn QA baseline")
    print("=" * 80)

    test_prompts = [
        ("QA - Geography", "What country is the city of Tokyo located in?"),
        ("QA - Science", "What planet is closest to the Sun?"),
        ("List Formatting", "List three primary colors."),
        ("Reasoning QA", "Which is larger: an elephant or an ant?"),
    ]

    sample_distractor_history = [
        ("Hello, how are you today?", "I am doing well, thank you for asking!"),
        ("What can you do?", "I can help answer questions and summarize text."),
    ]

    for cat, query in test_prompts:
        print(f"\n[{cat}] Prompt: '{query}'")

        # 1. Baseline (Zero History - v0.1 Style)
        prompt_clean = f"Instruction:\n{query}\n\nResponse:\n"
        inp_ids_clean = tok.encode(prompt_clean, add_bos=True, add_eos=False)
        with torch.no_grad():
            out_clean = model.generate(
                prompt_tokens=torch.tensor([inp_ids_clean], device=device),
                max_new_tokens=60,
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
        conv_parts = []
        for u, a in sample_distractor_history:
            conv_parts.append(f"Instruction:\n{u}\n\nResponse:\n{a}\n\n")
        conv_parts.append(f"Instruction:\n{query}\n\nResponse:\n")
        prompt_with_hist = "".join(conv_parts)
        inp_ids_hist = tok.encode(prompt_with_hist, add_bos=True, add_eos=False)
        with torch.no_grad():
            out_hist = model.generate(
                prompt_tokens=torch.tensor([inp_ids_hist], device=device),
                max_new_tokens=60,
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

        print(f"  BASELINE (v0.1 SFT, 0 history) [{len(clean_gen)} tok, EOS={clean_eos}]:")
        print(f"    '{clean_resp}'")
        print(f"  WITH HISTORY (v0.2 multi-turn) [{len(hist_gen)} tok, EOS={hist_eos}]:")
        print(f"    '{hist_resp}'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    model, tok, cfg = load_canonical_sft(device)

    run_task1_tokenization_and_budget(tok, cfg)
    run_task1_incontext_multiturn(model, tok, device)
    run_task2_persistent_memory(model, tok, device)
    run_task3_regression_check(model, tok, device)


if __name__ == "__main__":
    main()
