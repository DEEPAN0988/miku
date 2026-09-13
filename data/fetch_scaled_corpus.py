"""
data/fetch_scaled_corpus.py — Scaled Authentic Corpus Assembler for MIKU v0.1
STATUS: IMPLEMENTED

Scales the dataset to the maximum authentic scale strictly preserving the 40/35/25 ratio:
  1. Reasoning: ALL 7,473 chain-of-thought problems from GSM8K (40.0%)
  2. General Prose: 6,539 distinct paragraphs from WikiText-103 (35.0%)
  3. Wiki Factual: 4,670 distinct paragraphs from WikiText-103 (25.0%)
Total: 18,682 documents.
"""

import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
import pyarrow.parquet as pq


def fetch_all_gsm8k() -> list[str]:
    print("Fetching complete GSM8K train split (all ~7,473 problems)...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "MikuLM/0.1"})
    
    docs = []
    with urllib.request.urlopen(req, timeout=45) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            q = item["question"].strip()
            a = item["answer"].strip()
            
            # Clean calculator annotations <<...>> from GSM8k
            a_clean = re.sub(r"<<[^>]+>>", "", a)
            a_clean = a_clean.replace("####", "Therefore, the final answer is")
            
            # Use single newlines within document so double-newline separates docs
            doc = f"Problem: {q}\nStep-by-step Solution:\n{a_clean}"
            doc = "\n".join([l.strip() for l in doc.splitlines() if l.strip()])
            if len(doc) >= 80:
                docs.append(doc)
                
    print(f"  Loaded {len(docs):,} complete GSM8K reasoning documents.")
    return docs


def fetch_wikitext_103_scaled(n_general: int, n_wiki: int) -> tuple[list[str], list[str]]:
    total_needed = n_general + n_wiki
    print(f"Fetching {total_needed:,} paragraphs from WikiText-103 parquet on HuggingFace...")
    url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet"
    req = urllib.request.Request(url, headers={"User-Agent": "MikuLM/0.1"})
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = io.BytesIO(resp.read())
        table = pq.read_table(data)
        
    print(f"  WikiText-103 shard loaded: {table.num_rows:,} rows. Filtering paragraphs...")
    
    collected = []
    seen = set()
    for row in table["text"]:
        t = row.as_py().strip()
        if len(t) >= 180 and not t.startswith("="):
            # Key on first 60 chars to avoid duplicate boilerplate
            key = t[:60].lower()
            if key not in seen:
                seen.add(key)
                doc = "\n".join([l.strip() for l in t.splitlines() if l.strip()])
                collected.append(doc)
                if len(collected) >= total_needed:
                    break
                    
    print(f"  Extracted {len(collected):,} distinct paragraphs from WikiText-103.")
    if len(collected) < total_needed:
        raise ValueError(f"Need {total_needed} articles, but found {len(collected)}")
        
    general_docs = [f"Expository Essay:\n{d}" for d in collected[:n_general]]
    wiki_docs = [f"Encyclopedic Record:\n{d}" for d in collected[n_general:n_general + n_wiki]]
    
    return general_docs, wiki_docs


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Full GSM8K
    reasoning_docs = fetch_all_gsm8k()
    n_reasoning = len(reasoning_docs)
    
    # Calculate exact counts for 40/35/25 ratio
    # If reasoning is 40%, then general is 35%, wiki is 25%
    n_general = int(n_reasoning * 0.35 / 0.40)
    n_wiki = int(n_reasoning * 0.25 / 0.40)
    
    # 2. Scaled WikiText-103
    general_docs, wiki_docs = fetch_wikitext_103_scaled(n_general, n_wiki)
    
    total = len(reasoning_docs) + len(general_docs) + len(wiki_docs)
    print(f"\nScaled Corpus Assembly Summary:")
    print(f"  Reasoning docs : {len(reasoning_docs):>6,} ({len(reasoning_docs)/total:6.1%})")
    print(f"  General docs   : {len(general_docs):>6,} ({len(general_docs)/total:6.1%})")
    print(f"  Wiki docs      : {len(wiki_docs):>6,} ({len(wiki_docs)/total:6.1%})")
    print(f"  Total docs     : {total:>6,}")
    
    (raw_dir / "reasoning_corpus.txt").write_text("\n\n".join(reasoning_docs), encoding="utf-8")
    (raw_dir / "general_prose.txt").write_text("\n\n".join(general_docs), encoding="utf-8")
    (raw_dir / "wiki_factual.txt").write_text("\n\n".join(wiki_docs), encoding="utf-8")
    print(f"\nScaled seed corpus files successfully written to {raw_dir}/")


if __name__ == "__main__":
    main()
