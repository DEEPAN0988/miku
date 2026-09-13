"""
data/assemble_rebalanced_corpus.py — Assemble Rebalanced 25/40/35 Corpus (6-9M tokens)
STATUS: IMPLEMENTED

Assembles data/raw/ corpus strictly achieving:
  - 25.0% Reasoning: ALL 7,473 GSM8K problems (reasoning_corpus.txt)
  - 40.0% General Prose: 11,957 documents (general_prose.txt)
      * Mix of WikiText-103 expository essays + Project Gutenberg public domain literature
  - 35.0% Encyclopedic: 10,462 documents (wiki_factual.txt)
      * WikiText-103 authentic factual articles
Total: 29,892 documents.
Target token count: 6-9M tokens (~2.5x current 2.9M).

Features:
  - Preserves all 7,473 GSM8K problems without loss.
  - Substantive document lengths (averaging 200-300 words) targeting ~7M tokens.
  - Applies 8-gram near-duplicate pre-filtering across extracted documents.
  - Filters out Gutenberg TOC/boilerplate.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pyarrow.parquet as pq

# Source Parquet path in HuggingFace cache
PARQUET_PATH = Path(
    r"C:\Users\deepa\.cache\huggingface\hub\datasets--Salesforce--wikitext\snapshots\b08601e04326c79dfdd32d625aee71d232d685c3\wikitext-103-raw-v1\train-00000-of-00002.parquet"
)


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9']+\b", text.lower())


def build_8grams(tokens: List[str]) -> Set[Tuple[str, ...]]:
    if len(tokens) < 8:
        return set()
    return {tuple(tokens[i : i + 8]) for i in range(len(tokens) - 7)}


def detokenize_wikitext(text: str) -> str:
    text = re.sub(r"\s@-@\s", "-", text)
    text = re.sub(r"\s@\.@\s", ".", text)
    text = re.sub(r"\s@,@\s", ",", text)
    return text


def load_reasoning_docs(raw_dir: Path) -> List[str]:
    path = raw_dir / "reasoning_corpus.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    print(f"[REASONING] Loaded {len(docs):,} GSM8K documents from {path.name}")
    return docs


def extract_wikitext_docs(
    table: pq.Table,
    n_needed: int,
    start_row: int,
    seen_8grams: Set[Tuple[str, ...]],
    prefix: str,
    target_words: int = 180,
) -> Tuple[List[str], int]:
    print(f"Extracting {n_needed:,} substantive documents from WikiText-103 (from row {start_row:,})...")
    docs: List[str] = []
    curr_row = start_row
    num_rows = table.num_rows

    current_chunk: List[str] = []
    current_word_count = 0

    while curr_row < num_rows and len(docs) < n_needed:
        row_text = table["text"][curr_row].as_py().strip()
        curr_row += 1

        if row_text.startswith("="):
            # Section header — boundary for document chunking
            if current_chunk and current_word_count >= 100:
                body = " ".join(current_chunk)
                detok = detokenize_wikitext(body)
                tokens = tokenize_words(detok)
                ngs = build_8grams(tokens)
                if not ngs or (len(ngs & seen_8grams) / len(ngs) <= 0.15):
                    seen_8grams.update(ngs)
                    docs.append(f"{prefix}\n{detok}")
            current_chunk.clear()
            current_word_count = 0
            continue

        if len(row_text) < 100:
            continue

        detok = detokenize_wikitext(row_text)
        words = detok.split()
        current_chunk.append(detok)
        current_word_count += len(words)

        if current_word_count >= target_words:
            body = " ".join(current_chunk)
            tokens = tokenize_words(body)
            ngs = build_8grams(tokens)
            if not ngs or (len(ngs & seen_8grams) / len(ngs) <= 0.15):
                seen_8grams.update(ngs)
                docs.append(f"{prefix}\n{body}")
            current_chunk.clear()
            current_word_count = 0

    if current_chunk and len(docs) < n_needed and current_word_count >= 80:
        body = " ".join(current_chunk)
        docs.append(f"{prefix}\n{body}")

    print(f"  Extracted {len(docs):,} documents (scanned to row {curr_row:,}).")
    return docs, curr_row


def load_gutenberg_paragraphs(raw_dir: Path, seen_8grams: Set[Tuple[str, ...]], target_count: int) -> List[str]:
    path = raw_dir / "general_prose.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Filter for non-essay paragraphs (Gutenberg text)
    gut_raw = [p for p in paras if not p.startswith("Expository Essay:")]
    print(f"[GUTENBERG] Inspecting {len(gut_raw):,} raw Gutenberg paragraphs...")

    clean_gut = []
    chunk = []
    chunk_words = 0

    for p in gut_raw:
        # Filter TOC / chapter lists
        if p.count("Chapter") > 3 or p.count("Letter") > 3 or p.count("CHAPTER") > 3:
            continue
        if len(p) < 80 or len(p) > 2500:
            continue
        upper_ratio = sum(1 for c in p if c.isupper()) / max(len(p), 1)
        if upper_ratio > 0.25:
            continue

        words = p.split()
        chunk.append(p)
        chunk_words += len(words)

        if chunk_words >= 180:
            full_p = " ".join(chunk)
            tokens = tokenize_words(full_p)
            ngs = build_8grams(tokens)
            if not ngs or (len(ngs & seen_8grams) / len(ngs) <= 0.15):
                seen_8grams.update(ngs)
                clean_gut.append(full_p)
                if len(clean_gut) >= target_count:
                    break
            chunk.clear()
            chunk_words = 0

    print(f"[GUTENBERG] Retained {len(clean_gut):,} substantive literature documents.")
    return clean_gut


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MIKU CORPUS REBALANCER (25% Reasoning / 40% General / 35% Encyclopedic)")
    print("=" * 65)

    # 1. Reasoning
    reasoning_docs = load_reasoning_docs(raw_dir)
    n_reasoning = len(reasoning_docs)  # 7,473

    # Target counts:
    total_target = int(round(n_reasoning / 0.25))  # 29,892
    n_encyclopedic = int(round(total_target * 0.35))  # 10,462
    n_general = total_target - n_reasoning - n_encyclopedic  # 11,957

    print(f"\n[TARGET COUNTS]")
    print(f"  Reasoning:     {n_reasoning:>6,} ({n_reasoning / total_target:.1%}) [Target: 25%]")
    print(f"  General Prose: {n_general:>6,} ({n_general / total_target:.1%}) [Target: 40%]")
    print(f"  Encyclopedic:  {n_encyclopedic:>6,} ({n_encyclopedic / total_target:.1%}) [Target: 35%]")
    print(f"  TOTAL:         {total_target:>6,} (100.0%)\n")

    # Global 8-gram tracking for near-duplicate pre-filtering
    seen_8grams: Set[Tuple[str, ...]] = set()

    # Index reasoning 8-grams
    print("Indexing reasoning 8-grams...")
    for doc in reasoning_docs:
        tokens = tokenize_words(doc)
        seen_8grams.update(build_8grams(tokens))
    print(f"  Indexed {len(seen_8grams):,} reasoning 8-grams.")

    # 2. Encyclopedic (WikiText-103)
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing WikiText-103 parquet at {PARQUET_PATH}")

    table = pq.read_table(PARQUET_PATH)
    print(f"\nLoaded WikiText-103 parquet shard: {table.num_rows:,} rows.")

    wiki_docs, next_row = extract_wikitext_docs(
        table=table,
        n_needed=n_encyclopedic,
        start_row=0,
        seen_8grams=seen_8grams,
        prefix="",
        target_words=200,
    )

    # 3. General Prose (WikiText expository + Gutenberg literature)
    gut_target = n_general // 2  # ~5,978
    gut_selected = load_gutenberg_paragraphs(raw_dir, seen_8grams, gut_target)

    n_wiki_general = n_general - len(gut_selected)
    print(f"\nGeneral prose breakdown: {len(gut_selected):,} Gutenberg + {n_wiki_general:,} WikiText expository")

    wiki_general_docs, next_row = extract_wikitext_docs(
        table=table,
        n_needed=n_wiki_general,
        start_row=next_row,
        seen_8grams=seen_8grams,
        prefix="",
        target_words=200,
    )

    general_docs = gut_selected + wiki_general_docs
    assert len(general_docs) == n_general, f"Expected {n_general}, got {len(general_docs)}"
    assert len(wiki_docs) == n_encyclopedic, f"Expected {n_encyclopedic}, got {len(wiki_docs)}"

    # 4. Write back raw files
    print("\nWriting updated raw text files...")
    (raw_dir / "wiki_factual.txt").write_text("\n\n".join(wiki_docs), encoding="utf-8")
    print(f"  Wrote {len(wiki_docs):,} docs -> data/raw/wiki_factual.txt")

    (raw_dir / "general_prose.txt").write_text("\n\n".join(general_docs), encoding="utf-8")
    print(f"  Wrote {len(general_docs):,} docs -> data/raw/general_prose.txt")

    print("\n[VERIFICATION]")
    total_words = 0
    for fname in ["reasoning_corpus.txt", "general_prose.txt", "wiki_factual.txt"]:
        p = raw_dir / fname
        text = p.read_text(encoding="utf-8")
        docs = [d for d in text.split("\n\n") if d.strip()]
        words = sum(len(d.split()) for d in docs)
        total_words += words
        print(f"  {fname:<22s}: {len(docs):>6,} docs ({len(docs)/total_target:6.1%}) | ~{words:>9,} words")

    print(f"  TOTAL WORDS: ~{total_words:,} words (estimated ~{int(total_words * 1.35):,} tokens)")
    print("\n[OK] Assembly complete. Ready for prepare_corpus.py.\n")


if __name__ == "__main__":
    main()
