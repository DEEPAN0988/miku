"""
data/expand_corpus.py -- MIKU Corpus Expansion (Task 2 / Stage A Rebalance)
STATUS: EXPERIMENTAL

Expands training corpus from ~2.3M words to 6-9M by adding:
  1. More WikiText-103 articles (encyclopedic share)
  2. Project Gutenberg public-domain books (general prose share)

Does NOT touch reasoning_corpus.txt.

Sources:
  WikiText-103: HuggingFace datasets (CC BY-SA 3.0)
  Project Gutenberg: Public Domain (pre-1928 US works)
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
from typing import List, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent))

_WIKITEXT_DETOK = [
    (re.compile(r'\s@-@\s'), '-'),
    (re.compile(r'\s@\.@\s'), '.'),
    (re.compile(r'\s@,@\s'), ','),
]

def detokenize_wikitext(text):
    for pat, rep in _WIKITEXT_DETOK:
        text = pat.sub(rep, text)
    return text

def _is_section_header(line):
    s = line.strip()
    return s.startswith('=') and s.endswith('=')

def extract_wikitext_articles(raw_text, min_chars=300, max_articles=30000):
    paragraphs = []
    current_block = []

    def flush():
        if current_block:
            text = ' '.join(current_block).strip()
            text = detokenize_wikitext(text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) >= min_chars:
                paragraphs.append(f"Encyclopedic Record:\n{text}")
            current_block.clear()

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif _is_section_header(stripped):
            flush()
        else:
            current_block.append(stripped)
        if len(paragraphs) >= max_articles:
            break
    flush()
    return paragraphs[:max_articles]

def fetch_wikitext(target_paragraphs=25000):
    print("[WikiText-103] Loading from HuggingFace datasets...")
    try:
        from datasets import load_dataset
        # Try canonical repo name first (moved in datasets >= 2.x),
        # fall back to legacy short name.
        ds = None
        for repo_id in ["Salesforce/wikitext", "wikitext"]:
            try:
                ds = load_dataset(repo_id, "wikitext-103-raw-v1", trust_remote_code=False)
                print(f"  Loaded from repo: {repo_id}")
                break
            except Exception as inner_e:
                print(f"  [WARN] repo '{repo_id}' failed: {inner_e}")
        if ds is None:
            raise RuntimeError("All WikiText-103 repo IDs failed")
        print(f"  Splits: {list(ds.keys())}")
        all_text_parts = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                texts = [row["text"] for row in ds[split] if row["text"].strip()]
                all_text_parts.extend(texts)
                print(f"  {split}: {len(texts):,} non-empty rows")
        full_text = "\n".join(all_text_parts)
        paragraphs = extract_wikitext_articles(full_text, max_articles=target_paragraphs)
        desc = "WikiText-103 (wikitext-103-raw-v1 via HuggingFace, CC BY-SA 3.0)"
        print(f"  Extracted {len(paragraphs):,} paragraphs >= 300 chars")
        return paragraphs, desc
    except Exception as e:
        print(f"  [ERROR] {e}")
        return [], ""

GUTENBERG_BOOKS = [
    (84,    "Frankenstein - Mary Shelley"),
    (1342,  "Pride and Prejudice - Jane Austen"),
    (11,    "Alice in Wonderland - Lewis Carroll"),
    (46,    "A Christmas Carol - Charles Dickens"),
    (98,    "A Tale of Two Cities - Charles Dickens"),
    (1661,  "The Adventures of Sherlock Holmes - Arthur Conan Doyle"),
    (174,   "The Picture of Dorian Gray - Oscar Wilde"),
    (2701,  "Moby Dick - Herman Melville"),
    (844,   "The Importance of Being Earnest - Oscar Wilde"),
    (1400,  "Great Expectations - Charles Dickens"),
    (76,    "Adventures of Tom Sawyer - Mark Twain"),
    (345,   "Dracula - Bram Stoker"),
    (5200,  "Metamorphosis - Franz Kafka"),
    (219,   "Heart of Darkness - Joseph Conrad"),
    (2600,  "War and Peace - Leo Tolstoy"),
    (1232,  "The Prince - Niccolo Machiavelli"),
    (2554,  "Crime and Punishment - Fyodor Dostoevsky"),
    (25344, "The Scarlet Letter - Nathaniel Hawthorne"),
    (158,   "Emma - Jane Austen"),
    (768,   "Wuthering Heights - Emily Bronte"),
]

def fetch_gutenberg_book(book_id, title):
    import requests
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    text = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                text = resp.text
                break
        except Exception:
            continue
    if text is None:
        print(f"    [SKIP] Could not fetch {book_id}")
        return [], 0

    start = re.search(r'\*\*\* ?START OF .+?GUTENBERG.+?\*\*\*', text, re.IGNORECASE)
    end = re.search(r'\*\*\* ?END OF .+?GUTENBERG.+?\*\*\*', text, re.IGNORECASE)
    if start: text = text[start.end():]
    if end: text = text[:end.start()]

    # CRITICAL: normalize Windows \r\n line endings BEFORE splitting.
    # requests.get() returns \r\n for Windows-encoded Gutenberg files.
    # Without this, '\n\n'.split() finds nothing and the whole book
    # becomes one giant paragraph.
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    raw_paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraphs = []
    for para in raw_paras:
        para = re.sub(r'\n(?!\n)', ' ', para)
        para = re.sub(r'\s+', ' ', para).strip()
        if len(para) < 150: continue
        upper_ratio = sum(1 for c in para if c.isupper()) / max(len(para), 1)
        if upper_ratio > 0.5: continue
        paragraphs.append(para)
    n_words = sum(len(p.split()) for p in paragraphs)
    return paragraphs, n_words

def fetch_gutenberg(n_books=20):
    import time
    print(f"\n[Project Gutenberg] Fetching {n_books} public-domain books...")
    print("  License: Public Domain (pre-1928 US works)")
    all_paragraphs = []
    total_words = 0
    fetched = 0
    for book_id, title in GUTENBERG_BOOKS[:n_books]:
        print(f"  [{fetched+1}/{n_books}] {title} (ID {book_id})...", end=" ", flush=True)
        paras, words = fetch_gutenberg_book(book_id, title)
        if paras:
            all_paragraphs.extend(paras)
            total_words += words
            fetched += 1
            print(f"{len(paras)} paragraphs, ~{words:,} words")
        time.sleep(0.5)
    print(f"\n  Total Gutenberg: {len(all_paragraphs):,} paragraphs, ~{total_words:,} words")
    return all_paragraphs, f"Project Gutenberg ({fetched} books, Public Domain pre-1928)"

def run(args):
    raw_dir = Path("data/raw")
    wiki_path = raw_dir / "wiki_factual.txt"
    general_path = raw_dir / "general_prose.txt"

    print("\n" + "=" * 60)
    print("MIKU CORPUS EXPANSION")
    print("=" * 60)

    print("[CURRENT STATE]")
    for path in [raw_dir / "reasoning_corpus.txt", wiki_path, general_path]:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) >= 100]
            words = sum(len(p.split()) for p in paras)
            print(f"  {path.name}: {len(paras):,} docs, ~{words:,} words")

    print("\n[STEP 1] WikiText-103 expansion")
    wiki_paras, wiki_desc = fetch_wikitext(target_paragraphs=args.wiki_target)
    if wiki_paras:
        wiki_path.write_text("\n\n".join(wiki_paras), encoding="utf-8")
        wiki_words = sum(len(p.split()) for p in wiki_paras)
        print(f"  Wrote {len(wiki_paras):,} paragraphs (~{wiki_words:,} words) -> {wiki_path}")
        print(f"  Source: {wiki_desc}")
    else:
        print("  [WARN] WikiText expansion failed - keeping existing file")

    print("\n[STEP 2] Project Gutenberg general prose")
    gut_paras, gut_desc = fetch_gutenberg(n_books=args.gut_books)
    if gut_paras:
        existing = general_path.read_text(encoding="utf-8", errors="replace") if general_path.exists() else ""
        sep = "\n\n" if existing.strip() else ""
        general_path.write_text(existing + sep + "\n\n".join(gut_paras), encoding="utf-8")
        gut_words = sum(len(p.split()) for p in gut_paras)
        print(f"  Appended {len(gut_paras):,} paragraphs (~{gut_words:,} words) -> {general_path}")
        print(f"  Source: {gut_desc}")
    else:
        print("  [WARN] Gutenberg fetch returned no paragraphs")

    print("\n[FINAL INVENTORY]")
    total_words = 0
    for path in [raw_dir / "reasoning_corpus.txt", wiki_path, general_path]:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) >= 100]
            words = sum(len(p.split()) for p in paras)
            total_words += words
            print(f"  {path.name}: {len(paras):,} docs, ~{words:,} words")
    print(f"  TOTAL: ~{total_words:,} words")
    ok = "[OK]" if total_words >= 6_000_000 else "[WARN] Below 6M target"
    print(f"  {ok}")
    print("\n[DONE] Run: python data/prepare_corpus.py\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--wiki-target", type=int, default=25000)
    p.add_argument("--gut-books", type=int, default=20)
    run(p.parse_args())
