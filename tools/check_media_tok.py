import sys
from pathlib import Path
sys.path.insert(0, ".")
from model.tokenizer import MikuTokenizer

tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
candidates = ["Play Media", "Pause Media", "Next Track", "Previous Track", "Play Query"]
for c in candidates:
    text = f"Action: {c}\nArgument: None"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    pieces = [tok.id_to_piece(i) for i in ids]
    has_byte = any(p.startswith("<0x") for p in pieces)
    clean_pieces = [p.replace("\u2581", "_") for p in pieces]
    print(f"{c}: match={text.strip() == decoded.strip()}, byte_fallback={has_byte}, tokens={len(ids)}")
    print(f"  pieces={clean_pieces}")
