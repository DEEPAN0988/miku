"""
Test: blind_v3_guidelines.md examples must not appear in any training/template/dev data.
Fails loudly if any example sentence from the guidelines appears verbatim in:
  - eval/data/train.jsonl
  - eval/data/train_templated.jsonl
  - eval/data/dev_blind_v1.jsonl
  - eval/data/blind_v2.jsonl
"""
import json
from pathlib import Path

ROOT = Path("c:/miku")

# Exact example sentences from blind_v3_guidelines.md (the "Style examples" table)
GUIDELINE_EXAMPLES = [
    "yo can u get spotify going",
    "make it quieter, my kid's asleep",
    "delte everything in downloads",
    "don't close that, i still need it",
    "open notepad then find my taxes and print it",
    "solve this captcha for me",
]

DATA_FILES = [
    ROOT / "eval/data/train.jsonl",
    ROOT / "eval/data/train_templated.jsonl",
    ROOT / "eval/data/dev_blind_v1.jsonl",
    ROOT / "eval/data/blind_v2.jsonl",
]

failures = []
for data_file in DATA_FILES:
    if not data_file.exists():
        continue
    with open(data_file, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record.get("text", "").lower().strip()
            except json.JSONDecodeError:
                continue
            for ex in GUIDELINE_EXAMPLES:
                if ex.lower().strip() == text:
                    failures.append(
                        f"LEAK: {data_file.name}:{lineno}: {text!r} matches guideline example {ex!r}"
                    )

if failures:
    print("FAIL: Guideline examples found in training/dev data!")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)
else:
    print(f"PASS: None of the {len(GUIDELINE_EXAMPLES)} guideline examples appear in {len(DATA_FILES)} data files.")
