"""
data/prepare_corpus.py — MIKU Corpus Preparation Pipeline
STATUS: IMPLEMENTED

Pipeline stages (run in order or all at once):
  1. clean    — strip Wikipedia markup + LaTeX, normalize whitespace
  2. dedup    — SHA-256 hash deduplication per document
  3. split    — train/val split with shuffle (no data leakage)
  4. verify   — exact-match contamination check after split
  5. stats    — category ratio report
  6. tokenize — BPE tokenize and write binary .bin files

File naming convention for data/raw/:
    reasoning_*.txt   → category: reasoning_logic_arithmetic
    wiki_*.txt        → category: encyclopedic
    general_*.txt     → category: general_text
    (other)           → category: general_text (with warning)

Each .txt file should contain one document per paragraph-separated block,
or continuous prose (the pipeline splits on double-newlines).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# ---------------------------------------------------------------------------
# WikiText-103 detokenization
# ---------------------------------------------------------------------------

# WikiText-103 uses space-separated punctuation escape tokens:
#   @-@  →  -   (hyphenated compounds: "four @-@ time" → "four-time")
#   @.@  →  .   (abbreviations/initials: "U.S.A @.@ born" → "U.S.A. born")
#   @,@  →  ,   (numeric grouping or quoted punctuation)
# These must be collapsed back to plain ASCII BEFORE any other cleaning.

_WIKITEXT_DETOK_PATTERNS = [
    # " @-@ " → "-" (hyphen, no surrounding spaces)
    (re.compile(r'\s@-@\s'), '-'),
    # " @.@ " → "." (period, no surrounding spaces)
    (re.compile(r'\s@\.@\s'), '.'),
    # " @,@ " → "," (comma, no surrounding spaces)
    (re.compile(r'\s@,@\s'), ','),
]


def detokenize_wikitext(text: str) -> str:
    """
    Convert WikiText-103 escaped punctuation tokens back to normal ASCII.

    WikiText-103's raw format stores hyphen/period/comma as space-padded
    escape tokens (e.g. "Jan @-@ Michael" instead of "Jan-Michael").
    This must be fixed BEFORE any downstream cleaning step runs, otherwise
    these tokens appear in the vocabulary and contaminate generated text.

    Only call this on text sourced from WikiText (wiki_*.txt files).
    """
    for pattern, replacement in _WIKITEXT_DETOK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def strip_wikipedia_markup(text: str) -> str:
    """
    Remove MediaWiki markup from text.

    Handles: templates, wikilinks, external links, bold/italic, headers,
    HTML tags, HTML entities, table markup, categories, redirects.
    """
    # Remove [[File:...]] and [[Image:...]] blocks
    text = re.sub(
        r'\[\[(?:File|Image|Fichier|Bild|Bestand|Plik|Arquivo):[^\]]*\]\]',
        '', text, flags=re.IGNORECASE
    )

    # Remove nested templates {{...}} iteratively (handles nesting up to ~5 deep)
    for _ in range(6):
        prev = text
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if text == prev:
            break  # No more templates found

    # Convert wikilinks [[target|display]] → display; [[target]] → target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)

    # Remove external links [http://... anchor text] → anchor text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    # Remove bare external links [http://...]
    text = re.sub(r'\[https?://\S+\]', '', text)

    # Remove bold/italic markers ('' and ''')
    text = re.sub(r"'{2,3}", '', text)

    # Convert headers (== Heading ==) to plain text with newline
    text = re.sub(r'={2,6}\s*(.*?)\s*={2,6}', r'\n\1\n', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Replace common HTML entities
    html_entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
        '&apos;': "'", '&nbsp;': ' ', '&ndash;': '–', '&mdash;': '—',
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)   # remaining named entities
    text = re.sub(r'&#\d+;', ' ', text)          # numeric entities

    # Remove table markup lines (lines starting with |, !, or {|)
    text = re.sub(r'^\s*[|!{][|!}\-][^\n]*', '', text, flags=re.MULTILINE)

    # Remove category/interwiki links
    text = re.sub(
        r'\[\[(?:Category|Kategorie|Catégorie|Categoría):[^\]]*\]\]',
        '', text, flags=re.IGNORECASE
    )

    # Remove redirect lines
    text = re.sub(r'^#REDIRECT\s.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove reference tags and citation templates
    text = re.sub(r'<ref[^/]*/>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)

    return text


def strip_latex(text: str) -> str:
    """
    Remove LaTeX mathematical notation from text, replacing with [MATH].

    Handles: display math, inline math, environments, commands.
    """
    # LaTeX environments: \begin{...}...\end{...}
    text = re.sub(
        r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',
        ' [MATH] ', text, flags=re.DOTALL
    )

    # Display math: $$...$$ (may span lines)
    text = re.sub(r'\$\$.*?\$\$', ' [MATH] ', text, flags=re.DOTALL)

    # LaTeX display math: \[...\]
    text = re.sub(r'\\\[.*?\\\]', ' [MATH] ', text, flags=re.DOTALL)

    # Inline math: $...$
    # Rules:
    # 1. Opening $ not followed by whitespace: (?!\s)
    # 2. Closing $ not preceded by whitespace: (?<!\s)
    # 3. Closing $ not immediately followed by digit: (?!\d) (avoids pairing $12 ... $3)
    def _replace_inline_math(match: re.Match) -> str:
        content = match.group(1)
        # If it contains LaTeX commands or math operators/symbols, it's math
        math_indicators = r'[\\[\]=+\-*/^_<>≤≥±≈≠~|{}]'
        if re.search(math_indicators, content):
            return ' [MATH] '
        # Single variable or alphanumeric expression without multiple words
        if re.fullmatch(r'[a-zA-Z0-9.,()]+', content.strip()):
            return ' [MATH] '
        # If it has spaces and no math operators, keep original
        return match.group(0)

    text = re.sub(r'\$(?!\s)([^\$\n]{1,200}?)(?<!\s)\$(?!\d)', _replace_inline_math, text)

    # LaTeX commands with arguments: \command{content} → content
    text = re.sub(r'\\[a-zA-Z]+\{([^}]{0,200})\}', r'\1', text)

    # Standalone LaTeX commands: \command
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)

    # Remove remaining curly braces and backslashes
    text = re.sub(r'[{}\\]', ' ', text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces/tabs, normalize
    line endings, remove trailing whitespace per line.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove trailing whitespace per line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    # Collapse runs of spaces/tabs (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading whitespace per line
    text = '\n'.join(line.lstrip() for line in text.split('\n'))
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_document(text: str) -> str:
    """Apply full cleaning pipeline to one document."""
    text = strip_wikipedia_markup(text)
    text = strip_latex(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Document segmentation
# ---------------------------------------------------------------------------

def segment_into_documents(text: str, min_chars: int = 80) -> List[str]:
    """
    Split a raw text file into individual documents.

    A document is a block of text separated by blank lines (double newline).
    Documents shorter than min_chars are discarded.
    """
    raw_docs = text.split('\n\n')
    docs = []
    for doc in raw_docs:
        doc = doc.strip()
        if len(doc) >= min_chars:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def document_hash(text: str) -> str:
    """SHA-256 hash of a document (case-normalized, whitespace-normalized)."""
    normalized = ' '.join(text.lower().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def deduplicate_documents(documents: List[str]) -> Tuple[List[str], int]:
    """
    Remove exact-duplicate documents using SHA-256 hashing.

    Returns:
        (unique_documents, n_removed)
    """
    seen: set = set()
    unique: List[str] = []
    for doc in documents:
        h = document_hash(doc)
        if h not in seen:
            seen.add(h)
            unique.append(doc)

    return unique, len(documents) - len(unique)


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

# Keyword sets for heuristic category detection
_REASONING_PATTERNS = re.compile(
    r'\b('
    r'therefore|because|since|consequently|hence|thus|implies|proves?|'
    r'let\s+x|solve|equals?|equation|sum|product|difference|quotient|'
    r'step\s*\d|step-by-step|first.*then|if.*then|'
    r'proof|theorem|lemma|corollary|hypothesis|conclude|'
    r'calculate|compute|determine|find\s+the\s+value|'
    r'logic|deductive|inductive|reasoning|argument|'
    r'[0-9]+\s*[\+\-\×\÷\*\/]\s*[0-9]|'
    r'percent|probability|fraction|ratio|'
    r'true\s+or\s+false|valid\s+argument'
    r')',
    re.IGNORECASE
)

_WIKI_PATTERNS = re.compile(
    r'\b('
    r'born\s+in|died\s+in|founded\s+in|established\s+in|'
    r'is\s+a\s+(?:city|country|town|river|mountain|island)|'
    r'wikipedia|encyclop|according\s+to|citation\s+needed|'
    r'population\s+of|located\s+in|capital\s+of|'
    r'during\s+the\s+\d{4}|in\s+the\s+\d{2}th\s+century'
    r')',
    re.IGNORECASE
)


def categorize_document(text: str) -> str:
    """
    Heuristically categorize a document into one of three categories.

    Returns one of: 'reasoning_logic_arithmetic', 'encyclopedic', 'general_text'
    """
    reasoning_hits = len(_REASONING_PATTERNS.findall(text))
    wiki_hits = len(_WIKI_PATTERNS.findall(text))

    words = len(text.split())
    reasoning_density = reasoning_hits / max(words, 1)
    wiki_density = wiki_hits / max(words, 1)

    if reasoning_density > 0.02:
        return "reasoning_logic_arithmetic"
    elif wiki_density > 0.01:
        return "encyclopedic"
    else:
        return "general_text"


def get_category_from_filename(filename: str) -> Optional[str]:
    """
    Infer category from filename prefix.

    reasoning_* → reasoning_logic_arithmetic
    wiki_*      → encyclopedic
    general_*   → general_text
    """
    name = Path(filename).name.lower()
    if name.startswith("reasoning_"):
        return "reasoning_logic_arithmetic"
    elif name.startswith("wiki_"):
        return "encyclopedic"
    elif name.startswith("general_"):
        return "general_text"
    return None


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------

def compute_corpus_stats(
    documents: List[str],
    categories: List[str],
) -> Dict:
    """
    Compute and return corpus statistics.

    Returns a dict with total counts, category breakdown, and char/word/token
    estimates.
    """
    total_chars = sum(len(d) for d in documents)
    total_words = sum(len(d.split()) for d in documents)

    cat_counts: Dict[str, int] = {}
    cat_chars: Dict[str, int] = {}
    for doc, cat in zip(documents, categories):
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_chars[cat] = cat_chars.get(cat, 0) + len(doc)

    n_docs = len(documents)
    stats = {
        "n_documents": n_docs,
        "total_chars": total_chars,
        "total_words": total_words,
        "estimated_tokens_approx": total_words // 1,   # ~1 token per word rough estimate
        "category_doc_counts": cat_counts,
        "category_char_counts": cat_chars,
        "category_doc_fractions": {
            cat: count / n_docs for cat, count in cat_counts.items()
        },
        "category_char_fractions": {
            cat: chars / max(total_chars, 1) for cat, chars in cat_chars.items()
        },
    }
    return stats


def print_stats_report(stats: Dict, target_mix: Optional[Dict] = None) -> None:
    """Print a formatted corpus stats report to stdout."""
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"  Documents:         {stats['n_documents']:>10,}")
    print(f"  Total chars:       {stats['total_chars']:>10,}")
    print(f"  Total words:       {stats['total_words']:>10,}")
    print(f"  Est. tokens:       {stats['estimated_tokens_approx']:>10,}")
    print()
    print("  CATEGORY BREAKDOWN (by document count):")
    for cat, frac in stats["category_doc_fractions"].items():
        count = stats["category_doc_counts"].get(cat, 0)
        bar = "#" * int(frac * 40)
        target_str = ""
        if target_mix and cat in target_mix:
            target_str = f" (target: {target_mix[cat]:.0%})"
        print(f"    {cat:<35s} {frac:6.1%} ({count:,}){target_str}")
        print(f"    {'':35s} {bar}")

    if target_mix:
        print()
        print("  MIX DEVIATION FROM TARGET:")
        for cat, target in target_mix.items():
            actual = stats["category_doc_fractions"].get(cat, 0.0)
            diff = actual - target
            flag = "[UNDER]" if diff < -0.05 else ("[OVER]" if diff > 0.05 else "[OK]")
            print(f"    {cat:<35s} actual={actual:.1%} target={target:.1%} {flag}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_train_val(
    documents: List[str],
    categories: List[str],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Shuffle documents and split into train/val sets.

    Shuffle happens BEFORE split so the val set is a random sample,
    not just the tail of the corpus.

    Returns:
        train_docs, val_docs, train_cats, val_cats
    """
    import random
    rng = random.Random(seed)

    paired = list(zip(documents, categories))
    rng.shuffle(paired)

    n_val = max(1, int(len(paired) * val_fraction))
    val_pairs = paired[:n_val]
    train_pairs = paired[n_val:]

    train_docs = [d for d, _ in train_pairs]
    train_cats = [c for _, c in train_pairs]
    val_docs = [d for d, _ in val_pairs]
    val_cats = [c for _, c in val_pairs]

    return train_docs, val_docs, train_cats, val_cats


# ---------------------------------------------------------------------------
# Contamination verification
# ---------------------------------------------------------------------------

def verify_zero_leakage(
    train_docs: List[str],
    val_docs: List[str],
) -> Tuple[int, float]:
    """
    Verify there is no exact-match overlap between train and val sets.

    Uses SHA-256 hashes of normalized documents.

    Returns:
        (n_leaking, fraction_leaking)
    """
    train_hashes = {document_hash(d) for d in train_docs}
    n_leaking = sum(1 for d in val_docs if document_hash(d) in train_hashes)
    fraction = n_leaking / max(len(val_docs), 1)
    return n_leaking, fraction


# ---------------------------------------------------------------------------
# Tokenization and binary serialization
# ---------------------------------------------------------------------------

def tokenize_and_save(
    documents: List[str],
    tokenizer_prefix: str,
    output_path: str,
    add_bos: bool = True,
    add_eos: bool = True,
) -> int:
    """
    Tokenize a list of documents using the trained BPE tokenizer and
    write all tokens sequentially to a binary file (uint16 numpy array).

    Format: flat uint16 array — each document is tokenized, BOS and EOS
    appended, then concatenated without padding.

    Returns:
        Total number of tokens written.
    """
    from model.tokenizer import MikuTokenizer
    from tqdm import tqdm

    tok = MikuTokenizer.load(tokenizer_prefix)

    all_tokens: List[int] = []
    for doc in tqdm(documents, desc=f"Tokenizing -> {output_path}", unit="doc"):
        ids = tok.encode(doc, add_bos=add_bos, add_eos=add_eos)
        all_tokens.extend(ids)

    # Validate vocab range (uint16 supports 0–65535, well above 32000)
    arr = np.array(all_tokens, dtype=np.uint16)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    arr.tofile(output_path)

    print(f"  Written {len(arr):,} tokens -> {output_path}")
    return len(arr)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw_files(raw_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load all .txt files from raw_dir.

    Returns:
        documents, categories, source_filenames
    """
    raw_path = Path(raw_dir)
    txt_files = sorted(raw_path.glob("*.txt"))

    if not txt_files:
        print(f"\n[WARNING] No .txt files found in {raw_dir}")
        print("  Place text files there named: reasoning_*.txt, wiki_*.txt, general_*.txt")
        print("  Each file should contain prose separated by blank lines (documents).\n")
        return [], [], []

    all_docs: List[str] = []
    all_cats: List[str] = []
    all_srcs: List[str] = []

    for fpath in txt_files:
        print(f"  Loading {fpath.name} ...", end=" ", flush=True)
        text = fpath.read_text(encoding="utf-8", errors="replace")

        # Determine category from filename first, fall back to content heuristic
        filename_cat = get_category_from_filename(fpath.name)
        is_wikitext = fpath.name.lower().startswith("wiki_")

        # Segment into documents BEFORE detokenization so we can print
        # per-document before/after samples.
        raw_docs = segment_into_documents(text, min_chars=80)

        n_samples_printed = 0
        for doc in raw_docs:
            # --- WikiText detokenization (wiki_*.txt only) ----------------
            # Must run BEFORE clean_document() so @-@/@.@/@,@ are removed
            # from the raw text before it reaches the vocabulary.
            if is_wikitext:
                doc_before = doc
                doc = detokenize_wikitext(doc)
                if (
                    getattr(load_raw_files, '_show_wiki_samples', False)
                    and n_samples_printed < 5
                    and doc_before != doc
                ):
                    print(f"\n  --- WikiText detokenization sample {n_samples_printed + 1} ---")
                    # Print just the first 300 chars of before/after
                    print(f"  BEFORE: {doc_before[:300]!r}")
                    print(f"  AFTER:  {doc[:300]!r}")
                    n_samples_printed += 1

            cleaned = clean_document(doc)
            if len(cleaned) < 80:
                continue  # too short after cleaning

            cat = filename_cat if filename_cat else categorize_document(cleaned)
            all_docs.append(cleaned)
            all_cats.append(cat)
            all_srcs.append(fpath.name)

        print(f"{len(raw_docs)} docs")

    return all_docs, all_cats, all_srcs


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full corpus preparation pipeline."""

    print("\n" + "=" * 60)
    print("MIKU CORPUS PREPARATION PIPELINE")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    stages = set(args.stage) if args.stage else {
        "clean", "dedup", "split", "verify", "stats", "tokenize"
    }

    # ── Stage: clean ──────────────────────────────────────────────────
    if "clean" in stages or "all" in stages:
        print("\n[STAGE 1] Load + clean raw corpus")
        documents, categories, sources = load_raw_files(args.raw_dir)
        if not documents:
            print("No documents loaded. Stopping.")
            return
        print(f"  Loaded {len(documents):,} documents from {args.raw_dir}")
    else:
        # Load previously saved clean corpus
        train_path = out_dir / "corpus_train.txt"
        val_path = out_dir / "corpus_val.txt"
        if not train_path.exists():
            print("No cleaned corpus found. Run with --stage clean first.")
            return
        print("[Skipped clean stage — loading from processed dir]")
        documents = (train_path.read_text(encoding="utf-8").split("\n\n") +
                     val_path.read_text(encoding="utf-8").split("\n\n"))
        categories = [categorize_document(d) for d in documents]
        sources = ["loaded"] * len(documents)

    # ── Stage: dedup ──────────────────────────────────────────────────
    if "dedup" in stages or "all" in stages:
        print(f"\n[STAGE 2] Deduplication")
        before = len(documents)
        paired = list(zip(documents, categories, sources))
        seen: set = set()
        deduped: List[Tuple] = []
        for doc, cat, src in paired:
            h = document_hash(doc)
            if h not in seen:
                seen.add(h)
                deduped.append((doc, cat, src))
        documents, categories, sources = zip(*deduped) if deduped else ([], [], [])
        documents, categories, sources = list(documents), list(categories), list(sources)
        removed = before - len(documents)
        print(f"  Removed {removed:,} duplicates ({removed/max(before,1):.1%})")
        print(f"  Unique documents: {len(documents):,}")

    # ── Stage: split ──────────────────────────────────────────────────
    if "split" in stages or "all" in stages:
        print(f"\n[STAGE 3] Train/val split (val={args.val_fraction:.0%}, seed={args.seed})")
        train_docs, val_docs, train_cats, val_cats = split_train_val(
            documents, categories,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(f"  Train: {len(train_docs):,} docs")
        print(f"  Val:   {len(val_docs):,} docs")

        # Save text files
        train_txt = out_dir / "corpus_train.txt"
        val_txt = out_dir / "corpus_val.txt"
        train_txt.write_text("\n\n".join(train_docs), encoding="utf-8")
        val_txt.write_text("\n\n".join(val_docs), encoding="utf-8")
        print(f"  Saved: {train_txt}")
        print(f"  Saved: {val_txt}")
    else:
        # Load from files for subsequent stages
        train_txt = out_dir / "corpus_train.txt"
        val_txt = out_dir / "corpus_val.txt"
        train_docs = train_txt.read_text(encoding="utf-8").split("\n\n") if train_txt.exists() else []
        val_docs = val_txt.read_text(encoding="utf-8").split("\n\n") if val_txt.exists() else []
        train_cats = [categorize_document(d) for d in train_docs]
        val_cats = [categorize_document(d) for d in val_docs]

    # ── Stage: verify ─────────────────────────────────────────────────
    if "verify" in stages or "all" in stages:
        print(f"\n[STAGE 4] Contamination verification (exact-match)")
        n_leaking, frac_leaking = verify_zero_leakage(train_docs, val_docs)
        print(f"  Val docs:         {len(val_docs):,}")
        print(f"  Leaking (exact):  {n_leaking:,} ({frac_leaking:.2%})")
        if n_leaking == 0:
            print("  [OK] ZERO EXACT-MATCH OVERLAP — split is clean")
        else:
            print(f"  [ERROR] WARNING: {n_leaking} val documents appear in train set!")
            print("  Do NOT train until this is resolved.")
            sys.exit(1)

    # ── Stage: stats ──────────────────────────────────────────────────
    if "stats" in stages or "all" in stages:
        print(f"\n[STAGE 5] Category statistics")
        train_stats = compute_corpus_stats(train_docs, train_cats)
        val_stats = compute_corpus_stats(val_docs, val_cats)

        target_mix = {
            "reasoning_logic_arithmetic": 0.25,
            "general_text": 0.40,
            "encyclopedic": 0.35,
        }
        print("TRAIN SET:")
        print_stats_report(train_stats, target_mix)
        print("VAL SET:")
        print_stats_report(val_stats)

        # Save stats JSON
        stats_out = out_dir / "stats.json"
        with open(stats_out, "w") as f:
            json.dump({"train": train_stats, "val": val_stats}, f, indent=2)
        print(f"  Stats saved to {stats_out}")

    # ── Stage: tokenize ───────────────────────────────────────────────
    if "tokenize" in stages or "all" in stages:
        print(f"\n[STAGE 6] Tokenization")
        tok_prefix = str(tok_dir / "miku_bpe")
        tok_model = tok_prefix + ".model"

        if not os.path.exists(tok_model):
            print(f"  Tokenizer model not found at {tok_model}")
            print(f"  Training BPE tokenizer from corpus_train.txt ...")
            from model.tokenizer import MikuTokenizer
            train_txt_path = out_dir / "corpus_train.txt"
            if not train_txt_path.exists():
                print("  ERROR: corpus_train.txt not found. Run --stage split first.")
                sys.exit(1)
            MikuTokenizer.train(
                input_file=str(train_txt_path),
                model_prefix=tok_prefix,
                vocab_size=args.vocab_size,
            )
        else:
            print(f"  Using existing tokenizer: {tok_model}")

        train_tokens = tokenize_and_save(
            train_docs, tok_prefix,
            str(out_dir / "train.bin"),
        )
        val_tokens = tokenize_and_save(
            val_docs, tok_prefix,
            str(out_dir / "val.bin"),
        )
        print(f"\n  Total tokens - Train: {train_tokens:,} | Val: {val_tokens:,}")

    print("\n[OK] Pipeline complete.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MIKU corpus preparation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Directory containing raw .txt files"
    )
    parser.add_argument(
        "--out-dir", default="data/processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05,
        help="Fraction of deduplicated corpus to use as validation set"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32_000,
        help="BPE vocabulary size for tokenizer training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffle/split"
    )
    parser.add_argument(
        "--stage", nargs="+",
        choices=["clean", "dedup", "split", "verify", "stats", "tokenize", "all"],
        default=None,
        help="Pipeline stages to run (default: all stages)"
    )
    parser.add_argument(
        "--show-wiki-samples", action="store_true",
        help="Print 5 before/after WikiText detokenization samples during clean stage"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    # Wire the --show-wiki-samples flag into load_raw_files via a function attribute
    # (avoids changing the function signature, which would break all callers).
    load_raw_files._show_wiki_samples = getattr(args, 'show_wiki_samples', False)
    run_pipeline(args)
