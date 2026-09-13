"""
data/strip_raw_boilerplate.py — Strip boilerplate formatting tags from raw corpus files.

Tasks:
1. Strip 'Problem: ', 'Step-by-step Solution:\n', and '\nTherefore, the final answer is .*'
   from reasoning_corpus.txt, preserving natural worked solution prose.
2. Strip 'Encyclopedic Record:\n' from wiki_factual.txt.
3. Strip 'Expository Essay:\n' from general_prose.txt.
4. Print before/after samples for 3 documents per category.
5. Verify zero remaining occurrences of boilerplate tags across all 3 files.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw")


def strip_reasoning():
    path = RAW_DIR / "reasoning_corpus.txt"
    raw_text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]
    print(f"\n[REASONING] Processing {len(docs):,} documents from {path.name}...")

    pattern = re.compile(
        r"^Problem:\s*(.*?)\nStep-by-step Solution:\s*\n(.*?)\nTherefore, the final answer is\s*(.*)$",
        re.DOTALL,
    )

    cleaned_docs = []
    before_samples = []
    after_samples = []

    for i, d in enumerate(docs):
        m = pattern.match(d)
        if not m:
            raise ValueError(f"Reasoning document {i} did not match expected pattern:\n{d[:200]}")
        q, s, a = m.groups()
        q = q.strip()
        s = s.strip()

        # Ensure proper sentence ending punctuation on the last line of steps
        if not s.endswith((".", "!", "?")):
            s = s + "."

        cleaned = f"{q}\n{s}"
        cleaned_docs.append(cleaned)

        if i < 3:
            before_samples.append(d)
            after_samples.append(cleaned)

    # Write back
    path.write_text("\n\n".join(cleaned_docs), encoding="utf-8")
    print(f"  Wrote {len(cleaned_docs):,} cleaned documents to {path}")
    return before_samples, after_samples


def strip_wiki():
    path = RAW_DIR / "wiki_factual.txt"
    raw_text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]
    print(f"\n[WIKI FACTUAL] Processing {len(docs):,} documents from {path.name}...")

    cleaned_docs = []
    before_samples = []
    after_samples = []

    for i, d in enumerate(docs):
        before = d
        if d.startswith("Encyclopedic Record:\n"):
            cleaned = d[len("Encyclopedic Record:\n"):].strip()
        elif d.startswith("Encyclopedic Record:"):
            cleaned = d[len("Encyclopedic Record:"):].strip()
        else:
            cleaned = d
        cleaned_docs.append(cleaned)

        if i < 3:
            before_samples.append(before)
            after_samples.append(cleaned)

    # Write back
    path.write_text("\n\n".join(cleaned_docs), encoding="utf-8")
    print(f"  Wrote {len(cleaned_docs):,} cleaned documents to {path}")
    return before_samples, after_samples


def strip_general():
    path = RAW_DIR / "general_prose.txt"
    raw_text = path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]
    print(f"\n[GENERAL PROSE] Processing {len(docs):,} documents from {path.name}...")

    cleaned_docs = []
    before_samples = []
    after_samples = []

    # Find 3 samples that actually had the tag for demonstration
    tagged_indices = [i for i, d in enumerate(docs) if "Expository Essay:" in d][:3]

    for i, d in enumerate(docs):
        before = d
        if d.startswith("Expository Essay:\n"):
            cleaned = d[len("Expository Essay:\n"):].strip()
        elif d.startswith("Expository Essay:"):
            cleaned = d[len("Expository Essay:"):].strip()
        else:
            cleaned = d
        cleaned_docs.append(cleaned)

        if i in tagged_indices:
            before_samples.append(before)
            after_samples.append(cleaned)

    # Write back
    path.write_text("\n\n".join(cleaned_docs), encoding="utf-8")
    print(f"  Wrote {len(cleaned_docs):,} cleaned documents to {path}")
    return before_samples, after_samples


def verify_tags_gone():
    print("\n" + "=" * 60)
    print("VERIFICATION: Scanning data/raw/ for boilerplate tags")
    print("=" * 60)

    tags = [
        "Step-by-step Solution:",
        "Therefore, the final answer is",
        "Encyclopedic Record:",
        "Expository Essay:",
        "Problem:",
    ]

    all_passed = True
    for fname in ["reasoning_corpus.txt", "wiki_factual.txt", "general_prose.txt"]:
        fpath = RAW_DIR / fname
        text = fpath.read_text(encoding="utf-8")
        print(f"\nChecking {fname} ({len(text):,} chars):")
        for tag in tags:
            count = text.count(tag)
            status = "CLEAN (0)" if count == 0 else f"FAIL ({count} found)"
            print(f"  '{tag}': {status}")
            if count > 0:
                all_passed = False

    return all_passed


def main():
    r_before, r_after = strip_reasoning()
    w_before, w_after = strip_wiki()
    g_before, g_after = strip_general()

    print("\n" + "=" * 70)
    print("BEFORE / AFTER SAMPLES (3 per category)")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("CATEGORY 1: REASONING (GSM8K)")
    print("-" * 70)
    for i in range(3):
        print(f"\n--- [Reasoning Sample {i+1}] BEFORE ---")
        print(r_before[i])
        print(f"\n--- [Reasoning Sample {i+1}] AFTER ---")
        print(r_after[i])

    print("\n" + "-" * 70)
    print("CATEGORY 2: ENCYCLOPEDIC (WikiText-103)")
    print("-" * 70)
    for i in range(3):
        print(f"\n--- [Encyclopedic Sample {i+1}] BEFORE ---")
        print(w_before[i][:300] + "...")
        print(f"\n--- [Encyclopedic Sample {i+1}] AFTER ---")
        print(w_after[i][:300] + "...")

    print("\n" + "-" * 70)
    print("CATEGORY 3: GENERAL PROSE (WikiText Expository / Literature)")
    print("-" * 70)
    for i in range(3):
        print(f"\n--- [General Prose Sample {i+1}] BEFORE ---")
        print(g_before[i][:300] + "...")
        print(f"\n--- [General Prose Sample {i+1}] AFTER ---")
        print(g_after[i][:300] + "...")

    if not verify_tags_gone():
        print("\n[ERROR] Verification failed: tags still present in raw data!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All boilerplate tags completely stripped from data/raw/!")


if __name__ == "__main__":
    main()
