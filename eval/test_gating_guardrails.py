"""
eval/test_gating_guardrails.py — Verification of Anaphora Gating,
Fact Hard-Blocking, and Realistic 3-Turn Integration.

Covers:
  Task 1: Anaphora / relevance gating unit tests & baseline QA verification.
  Task 2: Hard-block assertion verification & fact recall test.
  Task 3: Full 3-turn integration session test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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


def load_environment(device: torch.device):
    cfg = ModelConfig.from_yaml("configs/phase5_multiturn_sft.yaml")
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    model = MikuLM(cfg).to(device)
    load_checkpoint("checkpoints/phase5_multiturn_sft/canonical_multiturn_sft.pt", model, device=device)
    model.eval()
    return model, tok, cfg


# ==============================================================================
# TASK 1: GATING UNIT TESTS & BASELINE QA RUN
# ==============================================================================

def run_task1_gating_tests(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("=" * 80)
    print("TASK 1: ANAPHORA & RELEVANCE GATING UNIT TESTS")
    print("=" * 80)

    store = MikuMemoryStore("data/memory/test_gating.db")
    store.clear_all()
    sess_id = "test_gate_sess"

    # Pre-populate some history
    store.add_message(sess_id, "user", "Hello, I am learning French.")
    store.add_message(sess_id, "assistant", "French is a wonderful language to learn!")
    store.add_message(sess_id, "user", "I want to visit the Louvre.")
    store.add_message(sess_id, "assistant", "The Louvre in Paris has many incredible artworks.")

    retriever = MemoryRetriever(store, tok)
    history = store.get_history(sess_id)

    # 1. Clean Factual Queries (Should be FALSE -> ZERO history)
    clean_queries = [
        "What country is the city of Tokyo located in?",
        "What planet is closest to the Sun?",
        "List three primary colors.",
        "Which is larger: an elephant or an ant?",
    ]

    print("\n--- (A) Clean Factual Queries (Expected: Anaphora=False, History=0 turns) ---")
    for q in clean_queries:
        is_anaphoric = retriever.is_anaphoric_query(q, history)
        prompt, _ = retriever.build_prompt(q, session_id=sess_id, include_facts=False, enforce_gating=True)
        has_conv = "Conversation:" in prompt
        print(f"Query: '{q}'")
        print(f"  -> Anaphoric: {is_anaphoric} | Injected Conversation: {has_conv}")
        assert not is_anaphoric, f"Expected non-anaphoric for '{q}'"
        assert not has_conv, f"Expected no conversation block for '{q}'"

    # 2. Clear Anaphoric Queries (Should be TRUE -> 1-turn history)
    anaphoric_queries = [
        "When was it built?",               # pronoun 'it'
        "Where is that located?",           # referential 'that'
        "Can you repeat what I said?",      # conversational marker
        "Tell me more about the Louvre.",   # keyword overlap with recent turn
    ]

    print("\n--- (B) Clear Anaphoric Queries (Expected: Anaphora=True, History=1 turn) ---")
    for q in anaphoric_queries:
        is_anaphoric = retriever.is_anaphoric_query(q, history)
        prompt, _ = retriever.build_prompt(q, session_id=sess_id, include_facts=False, enforce_gating=True)
        has_conv = "Conversation:" in prompt
        # Count user turns in conversation block
        conv_user_turns = prompt.count("User:")
        print(f"Query: '{q}'")
        print(f"  -> Anaphoric: {is_anaphoric} | Injected Conversation: {has_conv} | Injected Turns: {conv_user_turns}")
        assert is_anaphoric, f"Expected anaphoric for '{q}'"
        assert has_conv, f"Expected conversation block for '{q}'"
        assert conv_user_turns == 1, f"Expected strictly 1 turn injected, got {conv_user_turns}"

    # 3. Ambiguous Cases (Documented Fallback Behavior)
    ambiguous_queries = [
        ("Why?", "Single word, no explicit pronoun/overlap -> Non-anaphoric, zero history"),
        ("What about that?", "Contains 'that' -> Conservative fallback: Anaphoric, 1 turn"),
        ("What is the weather?", "No prior mention of weather -> Non-anaphoric, zero history"),
    ]

    print("\n--- (C) Ambiguous Queries & Fallback Behavior ---")
    for q, rationale in ambiguous_queries:
        is_anaphoric = retriever.is_anaphoric_query(q, history)
        prompt, _ = retriever.build_prompt(q, session_id=sess_id, include_facts=False, enforce_gating=True)
        print(f"Query: '{q}' -> Anaphoric: {is_anaphoric}")
        print(f"  Rationale & Fallback: {rationale}")

    # 4. Re-Run 4 Baseline QA Prompts Through Gated Retriever
    print("\n--- (D) Baseline QA Prompts Executed Through Gated Retriever ---")
    session = MikuSession(model, tok, store=store, session_id=sess_id, device=device)

    for q in clean_queries:
        res = session.chat(q)
        print(f"\nPrompt: '{q}'")
        print(f"  Injected Context: {'None (Zero History)' if 'Conversation:' not in res['prompt'] else 'LEAKED'}")
        print(f"  Generated ({res['num_tokens']} tok, EOS={res['hit_eos']}):\n    '{res['response']}'")


# ==============================================================================
# TASK 2: HARD-BLOCK RAW KEY-VALUE FACTS & RE-EVALUATION
# ==============================================================================

def run_task2_fact_hardblock_tests(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 2: HARD-BLOCK RAW KEY-VALUE FACTS VERIFICATION")
    print("=" * 80)

    store = MikuMemoryStore("data/memory/test_hardblock.db")
    store.clear_all()
    store.store_fact("user_name", "Jordan")
    store.store_fact("favorite_color", "emerald green")

    retriever = MemoryRetriever(store, tok)

    # 1. Verify Retriever Emits Natural Sentences Only
    facts = retriever.retrieve_facts_for_query("What is my name?")
    print(f"Retrieved Facts for 'What is my name?':\n  {facts}")
    for f in facts:
        assert not f.startswith("user_name:"), f"Found raw key-value format: {f}"
        assert f.startswith("The user's name is"), f"Expected natural sentence: {f}"

    # 2. Verify Prompt Builder Hard-Blocks Any Injected Raw Key-Values
    print("Testing hard-block assertion on raw KV injection...")
    try:
        # Simulate a bug attempting raw KV
        bad_facts = ["user_name: Jordan"]
        retriever._build_context_prefix_prompt("Who am I?", [], bad_facts, 200)
    except AssertionError as e:
        print(f"  -> Successfully caught illegal raw key-value in prompt: {e}")

    # 3. Re-Run Fact Recall on Canonical Multi-Turn SFT Checkpoint
    session = MikuSession(model, tok, store=store, session_id="fact_sess", device=device)

    print("\n--- Testing Fact Recall on Model ---")
    res_name = session.chat("What is my name?")
    print(f"Query: 'What is my name?'")
    print(f"  Prompt:\n{res_name['prompt'].strip()}")
    print(f"  Response: '{res_name['response']}'")
    assert "Jordan" in res_name["response"], f"Expected Jordan in response, got: {res_name['response']}"

    res_color = session.chat("What is my favorite color?")
    print(f"\nQuery: 'What is my favorite color?'")
    print(f"  Prompt:\n{res_color['prompt'].strip()}")
    print(f"  Response: '{res_color['response']}'")
    assert "green" in res_color["response"].lower(), f"Expected green in response, got: {res_color['response']}"


# ==============================================================================
# TASK 3: FULL INTEGRATION TEST (REALISTIC 3-TURN SESSION)
# ==============================================================================

def run_task3_integration_session(model: MikuLM, tok: MikuTokenizer, device: torch.device):
    print("\n" + "=" * 80)
    print("TASK 3: REALISTIC 3-TURN INTEGRATION SESSION TEST")
    print("Simulating Fact Assertion -> Anaphoric Recall -> Unrelated Factual QA")
    print("=" * 80)

    store = MikuMemoryStore("data/memory/test_integration.db")
    store.clear_all()
    session = MikuSession(model, tok, store=store, session_id="integration_sess_01", device=device)

    transcript = []

    # TURN 1: User states fact
    turn1_user = "My name is Jordan."
    res1 = session.chat(turn1_user)
    transcript.append(("Turn 1", turn1_user, res1))

    # TURN 2: Anaphoric follow-up (Should get 1-turn history + natural fact via neural retriever)
    turn2_user = "What is my name?"
    res2 = session.chat(turn2_user, use_deterministic_fallback=False)
    transcript.append(("Turn 2", turn2_user, res2))

    # TURN 3: Unrelated factual QA (Should get ZERO history via gating)
    turn3_user = "What is the capital of Japan?"
    res3 = session.chat(turn3_user)
    transcript.append(("Turn 3", turn3_user, res3))

    print("\nFULL VERBATIM TRANSCRIPT:")
    for turn_name, u_text, res in transcript:
        has_conv = "Conversation:" in res["prompt"]
        conv_turns = res["prompt"].count("User:") - 1 if has_conv else 0
        facts_in_prompt = [line for line in res["prompt"].splitlines() if "The user's" in line]
        
        print(f"\n[{turn_name}]")
        print(f"User     : {u_text}")
        print(f"Retriever: Injected Turns={conv_turns} | Injected Facts={facts_in_prompt}")
        print(f"Prompt Sent to Model:\n---\n{res['prompt'].strip()}\n---")
        print(f"Assistant: '{res['response']}' (tok={res['num_tokens']}, EOS={res['hit_eos']})")

    # Assertions on integration behavior
    assert "The user's name is Jordan." in res2["prompt"], "Turn 2 should include natural name fact"
    assert "Your name is Jordan." in res2["response"], "Turn 2 should accurately state Jordan"
    assert "Conversation:" not in res3["prompt"], "Turn 3 should have ZERO conversation history (gated)"
    assert "The user's" not in res3["prompt"], "Turn 3 should have ZERO facts (unrelated)"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Gating & Hard-Block Verification on {device}...")
    model, tok, cfg = load_environment(device)

    run_task1_gating_tests(model, tok, device)
    run_task2_fact_hardblock_tests(model, tok, device)
    run_task3_integration_session(model, tok, device)


if __name__ == "__main__":
    main()
