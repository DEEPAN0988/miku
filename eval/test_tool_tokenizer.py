"""
eval/test_tool_tokenizer.py — Tokenizer compatibility analysis for tool-calling templates.

Compares candidate syntax formats (JSON, Python call, snake_case, natural title words)
and audits token piece fragmentation, token count, and byte fallbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


def audit_format(tok: MikuTokenizer, name: str, text: str):
    ids = tok.encode(text, add_bos=False, add_eos=False)
    pieces = [tok._sp.id_to_piece(i) for i in ids]
    byte_fallbacks = [p for p in pieces if "<0x" in p]
    has_byte = len(byte_fallbacks) > 0

    print(f"Format: {name}")
    print(f"  Text: {repr(text)}")
    print(f"  Tokens ({len(ids)}): {ids}")
    print(f"  Pieces: {pieces}")
    print(f"  Byte Fallbacks: {len(byte_fallbacks)} ({byte_fallbacks})")
    print(f"  Clean Merges: {'NO (Byte fallback detected)' if has_byte else 'YES (Clean subword/word tokens)'}")
    print("-" * 60)


def main():
    tok_path = "data/processed/tokenizer/miku_bpe"
    print(f"Loading tokenizer from {tok_path} ...")
    tok = MikuTokenizer.load(tok_path)
    print(f"Vocab size: {tok.vocab_size} | BOS: {tok.bos_id} | EOS: {tok.eos_id}\n")

    print("=" * 60)
    print("1. CANDIDATE SYNTAX FORMAT COMPARISON (for 'open calculator')")
    print("=" * 60)

    candidates = [
        ("JSON format", '{"action": "open_app", "argument": "calculator"}'),
        ("Python call syntax", "open_app(calculator)"),
        ("Bracketed tag format", "[ACTION] open_app [ARGUMENT] calculator"),
        ("Snake-case key-value", "Action: open_app\nArgument: calculator"),
        ("Natural Title-Case", "Action: Open App\nArgument: calculator"),
        ("Natural Lower-Case", "Action: open app\nArgument: calculator"),
    ]

    for name, text in candidates:
        audit_format(tok, name, text)

    print("\n" + "=" * 60)
    print("2. TOOL-NAME VOCABULARY AUDIT")
    print("=" * 60)

    tool_names = [
        ("get_current_time (snake_case)", "get_current_time"),
        ("Get Time (Natural Title)", "Get Time"),
        ("get time (Natural Lower)", "get time"),
        ("open_app (snake_case)", "open_app"),
        ("Open App (Natural Title)", "Open App"),
        ("open app (Natural Lower)", "open app"),
        ("search_web (snake_case)", "search_web"),
        ("Search Web (Natural Title)", "Search Web"),
        ("search web (Natural Lower)", "search web"),
    ]

    for name, text in tool_names:
        ids = tok.encode(text, add_bos=False, add_eos=False)
        pieces = [tok._sp.id_to_piece(i) for i in ids]
        byte_fallbacks = [p for p in pieces if "<0x" in p]
        print(f"{name:32} -> {len(ids)} tokens | Byte fallbacks: {len(byte_fallbacks)} | Pieces: {pieces}")

    print("\n" + "=" * 60)
    print("3. COMMON ARGUMENT VOCABULARY AUDIT")
    print("=" * 60)
    common_args = [
        "None", "calculator", "terminal", "browser", "notepad", "settings",
        "clock", "calendar", "camera", "files", "weather today", "python tutorial",
        "world news", "tokyo time"
    ]
    for arg in common_args:
        ids = tok.encode(arg, add_bos=False, add_eos=False)
        pieces = [tok._sp.id_to_piece(i) for i in ids]
        byte_fallbacks = [p for p in pieces if "<0x" in p]
        print(f"'{arg}': {len(ids)} tokens -> {pieces}")


if __name__ == "__main__":
    main()
