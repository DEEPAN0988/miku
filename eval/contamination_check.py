"""
eval/contamination_check.py — Train/Val Exact-Match + N-gram Overlap Check
STATUS: IMPLEMENTED

Detects two types of contamination between train and val splits:
  1. Exact-match (SHA-256 of normalized document) — primary check
  2. N-gram overlap (8-grams at word level) — catches near-duplicates

Per project constitution Rule 2: reports raw numbers only.
Never reports "PASS" without attaching the actual overlap count and fraction.

Usage:
    python eval/contamination_check.py \
        --train data/processed/corpus_train.txt \
        --val   data/processed/corpus_val.txt

    # With n-gram analysis:
    python eval/contamination_check.py \
        --train data/processed/corpus_train.txt \
        --val   data/processed/corpus_val.txt \
        --ngram 8

    # Output as JSON (for programmatic use):
    python eval/contamination_check.py \
        --train data/processed/corpus_train.txt \
        --val   data/processed/corpus_val.txt \
        --json-out contamination_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(path: str) -> List[str]:
    """
    Load documents from a text file (double-newline separated).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    return docs


# ---------------------------------------------------------------------------
# Exact-match check
# ---------------------------------------------------------------------------

def normalize_for_hash(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def document_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exact_match_check(
    train_docs: List[str],
    val_docs: List[str],
) -> Tuple[int, float, List[int]]:
    """
    Check for exact-match contamination between train and val sets.

    Args:
        train_docs: list of train documents
        val_docs:   list of val documents

    Returns:
        (n_leaking, fraction_leaking, leaking_val_indices)
    """
    train_hashes: Set[str] = {document_hash(d) for d in train_docs}

    leaking_indices = []
    for i, doc in enumerate(val_docs):
        if document_hash(doc) in train_hashes:
            leaking_indices.append(i)

    n_leaking = len(leaking_indices)
    fraction = n_leaking / max(len(val_docs), 1)
    return n_leaking, fraction, leaking_indices


# ---------------------------------------------------------------------------
# N-gram overlap check
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    """Simple word tokenizer for n-gram analysis."""
    return re.findall(r"\b[a-z0-9']+\b", text.lower())


def build_ngram_set(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Build the set of all n-grams in a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_overlap(
    train_docs: List[str],
    val_docs: List[str],
    n: int = 8,
) -> Dict:
    """
    Compute n-gram overlap between train and val sets.

    For each val document, compute what fraction of its n-grams
    appear anywhere in the train set.

    This is O(n_val × n_train_ngrams) — may be slow for very large corpora.
    For production, use a more efficient approach (e.g., suffix arrays).

    Returns dict with:
      - per_doc_overlap_fractions: List[float]
      - n_docs_above_threshold: int (>10% overlap)
      - fraction_docs_above_threshold: float
      - mean_overlap_fraction: float
      - max_overlap_fraction: float
    """
    print(f"Building {n}-gram index over {len(train_docs)} train documents ...")

    # Build union of all train n-grams
    train_ngrams: Set[Tuple[str, ...]] = set()
    for doc in train_docs:
        tokens = tokenize_words(doc)
        train_ngrams.update(build_ngram_set(tokens, n))

    print(f"Train n-gram set size: {len(train_ngrams):,}")

    overlap_fracs: List[float] = []
    threshold = 0.10  # >10% overlap is flagged

    for doc in val_docs:
        tokens = tokenize_words(doc)
        val_ngrams = build_ngram_set(tokens, n)
        if not val_ngrams:
            overlap_fracs.append(0.0)
            continue
        matched = val_ngrams & train_ngrams
        frac = len(matched) / len(val_ngrams)
        overlap_fracs.append(frac)

    n_flagged = sum(1 for f in overlap_fracs if f > threshold)

    return {
        "n": n,
        "threshold": threshold,
        "n_val_docs": len(val_docs),
        "per_doc_overlap_fractions": overlap_fracs,
        "n_docs_above_threshold": n_flagged,
        "fraction_docs_above_threshold": n_flagged / max(len(val_docs), 1),
        "mean_overlap_fraction": sum(overlap_fracs) / max(len(overlap_fracs), 1),
        "max_overlap_fraction": max(overlap_fracs) if overlap_fracs else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    train_path: str,
    val_path: str,
    n_train: int,
    n_val: int,
    exact_n: int,
    exact_frac: float,
    leaking_indices: List[int],
    ngram_result: Dict | None,
    val_docs: List[str],
) -> None:
    """Print a formatted contamination report."""
    print("\n" + "=" * 70)
    print("MIKU CONTAMINATION CHECK REPORT")
    print("=" * 70)
    print(f"  Train file: {train_path}")
    print(f"  Val file:   {val_path}")
    print(f"  Train docs: {n_train:,}")
    print(f"  Val docs:   {n_val:,}")
    print()

    # ── Exact match ──────────────────────────────────────────────────
    print("EXACT-MATCH CHECK (SHA-256 of normalized document)")
    print("-" * 40)
    if exact_n == 0:
        print(f"  [OK] RESULT: 0 leaking documents (0.00%)")
        print(f"  CLAIM: Zero exact-match overlap between train and val.")
        print(f"  EVIDENCE: {n_train:,} train hashes checked against {n_val:,} val docs.")
    else:
        print(f"  [FAIL] RESULT: {exact_n} leaking documents ({exact_frac:.2%})")
        print(f"  WARNING: Val set is contaminated with train data.")
        print(f"  Leaking val doc indices: {leaking_indices[:20]}")
        if len(leaking_indices) > 20:
            print(f"    ... and {len(leaking_indices) - 20} more")
        print()
        print("  First 3 leaking val documents (truncated to 200 chars):")
        for idx in leaking_indices[:3]:
            print(f"    [{idx}] {val_docs[idx][:200]!r}")
    print()

    # ── N-gram overlap ────────────────────────────────────────────────
    if ngram_result:
        n = ngram_result["n"]
        threshold = ngram_result["threshold"]
        print(f"{n}-GRAM OVERLAP CHECK")
        print("-" * 40)
        print(f"  Threshold for flagging: >{threshold:.0%} of val doc n-grams in train set")
        print(f"  Mean overlap fraction:  {ngram_result['mean_overlap_fraction']:.3%}")
        print(f"  Max overlap fraction:   {ngram_result['max_overlap_fraction']:.3%}")
        print(
            f"  Docs above threshold:   "
            f"{ngram_result['n_docs_above_threshold']:,} "
            f"({ngram_result['fraction_docs_above_threshold']:.2%})"
        )
        if ngram_result["n_docs_above_threshold"] == 0:
            print(f"  [OK] RESULT: No val documents have >{threshold:.0%} n-gram overlap with train")
        else:
            print(
                f"  [WARN] RESULT: {ngram_result['n_docs_above_threshold']:,} val documents have "
                f">{threshold:.0%} n-gram overlap - potential near-duplicates"
            )
        print()

    # -- Final verdict -------------------------------------------------
    print("VERDICT")
    print("-" * 40)
    if exact_n == 0 and (ngram_result is None or ngram_result["n_docs_above_threshold"] == 0):
        print("  [OK] CLEAN: No contamination detected.")
        print("    Training may proceed.")
    elif exact_n > 0:
        print(f"  [FAIL] CONTAMINATED: {exact_n} exact-match leaks detected.")
        print("    DO NOT TRAIN until the split is fixed.")
        print("    Re-run: python data/prepare_corpus.py --stage dedup split verify")
    else:
        print(f"  [WARN] BORDERLINE: No exact matches, but some n-gram overlap detected.")
        print("    Review flagged documents before training.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MIKU train/val contamination checker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train", required=True,
        help="Path to corpus_train.txt"
    )
    p.add_argument(
        "--val", required=True,
        help="Path to corpus_val.txt"
    )
    p.add_argument(
        "--ngram", type=int, default=8,
        help="N-gram size for overlap check. Set to 0 to skip n-gram check."
    )
    p.add_argument(
        "--json-out", default=None,
        help="If set, write full results as JSON to this path"
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading train documents from {args.train} ...")
    train_docs = load_documents(args.train)
    print(f"  Loaded {len(train_docs):,} train documents")

    print(f"Loading val documents from {args.val} ...")
    val_docs = load_documents(args.val)
    print(f"  Loaded {len(val_docs):,} val documents")

    # Exact match
    exact_n, exact_frac, leaking_indices = exact_match_check(train_docs, val_docs)

    # N-gram overlap (optional)
    ngram_result = None
    if args.ngram > 0:
        ngram_result = compute_ngram_overlap(train_docs, val_docs, n=args.ngram)

    # Print report
    print_report(
        train_path=args.train,
        val_path=args.val,
        n_train=len(train_docs),
        n_val=len(val_docs),
        exact_n=exact_n,
        exact_frac=exact_frac,
        leaking_indices=leaking_indices,
        ngram_result=ngram_result,
        val_docs=val_docs,
    )

    # JSON output
    if args.json_out:
        report = {
            "train_file": args.train,
            "val_file": args.val,
            "n_train_docs": len(train_docs),
            "n_val_docs": len(val_docs),
            "exact_match": {
                "n_leaking": exact_n,
                "fraction_leaking": exact_frac,
                "leaking_val_indices": leaking_indices,
            },
            "ngram_overlap": ngram_result,
        }
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to {args.json_out}")

    # Exit with error code if contaminated (for CI integration)
    if exact_n > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
