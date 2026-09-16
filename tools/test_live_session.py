"""
tools/test_live_session.py — Live End-to-End System Test for MIKU.
Tests Conversation, Real System Tools, and Safety Guardrails together.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, ".")

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from tools.dispatcher import (
    dispatch_tool,
    extract_deterministic_slot,
    parse_tool_call,
    resolve_intent_anchor,
)

CHECKPOINT = "checkpoints/phase8_tool_sft/canonical_media_tool_sft.pt"
CONFIG = "configs/phase8_media_tool_sft.yaml"
TOKENIZER = "data/processed/tokenizer/miku_bpe"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST] Loading MIKU on device: {device} ...")

    tok = MikuTokenizer.load(TOKENIZER)
    cfg = ModelConfig.from_yaml(CONFIG)
    model = MikuLM(cfg).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    print("[TEST] MIKU v0.2 Phase 8 Loaded Successfully.")
    print("=" * 70)

    test_queries = [
        # 1. Real System Query
        "What time is it right now?",
        # 2. Real Battery Query
        "What's my battery percentage?",
        # 3. Real Media Query
        "Play the music.",
        # 4. Real Media Pause
        "Pause playback.",
        # 5. Guardrailed Medium Risk Query (Unconfirmed)
        "Search the web for python async tutorial.",
        # 6. Conversational General QA
        "What is water made of?",
    ]

    for q in test_queries:
        print(f"\nUser: {q}")
        prompt = f"Instruction:\n{q}\n\nResponse:\n"
        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=35,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tok.eos_id,
            )
        raw_gen = tok.decode(out[0, len(ids):].tolist()).strip()

        # Check if output represents a tool call
        if "Action:" in raw_gen:
            tc = parse_tool_call(raw_gen)
            anchored_action = resolve_intent_anchor(q, tc.action)
            tc.action = anchored_action
            
            # Deterministic slot extraction for argument if needed
            det_arg = extract_deterministic_slot(q, tc.action)
            if det_arg:
                tc.argument = det_arg

            res = dispatch_tool(tc, hitl_confirmed=False)
            print(f"  [ROUTED TOOL] Action='{tc.action}' | Arg='{tc.argument}' | Risk={tc.risk.value}")
            print(f"  [EXECUTION] Status: {res.status} | Executed: {res.executed}")
            print(f"  [OUTPUT]: {res.output}")
        else:
            print(f"  [DIRECT QA]: {raw_gen}")

    print("\n" + "=" * 70)
    print("[TEST] All live test queries completed cleanly.")


if __name__ == "__main__":
    main()
