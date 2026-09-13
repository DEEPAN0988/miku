"""
data/fetch_real_corpus.py — Production-Grade Authentic Corpus Assembler for MIKU v0.1
STATUS: IMPLEMENTED

Downloads and formats 100% authentic, real-world open data:
  1. Reasoning: 400 genuine chain-of-thought math problems from GSM8K (40.0%)
  2. Wiki Factual: 250 real encyclopedic articles from WikiText-2 (25.0%)
  3. General Prose: 350 real articles from WikiText-2 (35.0%)
Total: 1,000 unique, authentic documents.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path


def fetch_gsm8k_reasoning(target_count: int = 400) -> list[str]:
    print(f"Fetching {target_count} real reasoning problems from GSM8K...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "MikuLM/0.1"})
    
    docs = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            if len(docs) >= target_count:
                break
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
            if len(doc) >= 100:
                docs.append(doc)
                
    print(f"  Successfully loaded {len(docs)} GSM8K reasoning documents.")
    return docs


def fetch_wikitext_articles(target_wiki: int = 250, target_general: int = 350) -> tuple[list[str], list[str]]:
    total_needed = target_wiki + target_general
    print(f"Fetching {total_needed} real, non-overlapping articles from WikiText-2...")
    
    # Load from both train.txt and valid.txt to ensure abundant candidate pool
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    ]
    raw_articles = []
    for u in urls:
        req = urllib.request.Request(u, headers={"User-Agent": "MikuLM/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        raw_articles.extend(re.split(r'\n = [^=]+ = \n', text))
        
    def tokenize_words(t: str) -> list[str]:
        return re.findall(r"\b[a-z0-9']+\b", t.lower())
        
    def get_ngrams(tokens: list[str], n: int = 8) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    valid_articles = []
    all_ngrams: set = set()
    
    for a in raw_articles:
        if len(valid_articles) >= total_needed:
            break
        lines = [l.strip() for l in a.splitlines() if l.strip() and not l.startswith("=")]
        if not lines:
            continue
        article_text = " ".join(lines)
        if len(article_text) > 1500:
            article_text = article_text[:1500]
            last_period = article_text.rfind(".")
            if last_period > 200:
                article_text = article_text[:last_period + 1]
        if len(article_text) < 200:
            continue
            
        words = tokenize_words(article_text)
        ng = get_ngrams(words, 8)
        if not ng:
            continue
            
        # Reject near-duplicates (e.g. sister-ship articles with identical boiler specs)
        overlap = len(ng & all_ngrams) / len(ng)
        if overlap > 0.05:
            continue
            
        all_ngrams.update(ng)
        doc = f"Encyclopedic Article:\n{article_text}"
        doc = "\n".join([l.strip() for l in doc.splitlines() if l.strip()])
        valid_articles.append(doc)
            
    print(f"  Selected {len(valid_articles)} non-overlapping articles in WikiText-2.")
    if len(valid_articles) < total_needed:
        raise ValueError(f"Need {total_needed} articles, but found {len(valid_articles)}")
        
    wiki_docs = valid_articles[:target_wiki]
    general_docs = valid_articles[target_wiki:target_wiki + target_general]
    
    general_cleaned = []
    for d in general_docs:
        clean_d = d.replace("Encyclopedic Article:\n", "Expository Essay:\n")
        general_cleaned.append(clean_d)
        
    return wiki_docs, general_cleaned


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 400 real GSM8K math reasoning problems
    reasoning_docs = fetch_gsm8k_reasoning(target_count=400)
    
    # 2. 250 real encyclopedic articles & 350 real general articles from WikiText-2
    wiki_docs, general_docs = fetch_wikitext_articles(target_wiki=250, target_general=350)
    
    total = len(reasoning_docs) + len(general_docs) + len(wiki_docs)
    print(f"\nFinal Seed Corpus Assembly Summary:")
    print(f"  Reasoning docs : {len(reasoning_docs):>4} ({len(reasoning_docs)/total:6.1%})")
    print(f"  General docs   : {len(general_docs):>4} ({len(general_docs)/total:6.1%})")
    print(f"  Wiki docs      : {len(wiki_docs):>4} ({len(wiki_docs)/total:6.1%})")
    print(f"  Total docs     : {total:>4}")
    
    (raw_dir / "reasoning_corpus.txt").write_text("\n\n".join(reasoning_docs), encoding="utf-8")
    (raw_dir / "general_prose.txt").write_text("\n\n".join(general_docs), encoding="utf-8")
    (raw_dir / "wiki_factual.txt").write_text("\n\n".join(wiki_docs), encoding="utf-8")
    print(f"\nReal seed corpus files successfully written to {raw_dir}/")


if __name__ == "__main__":
    main()
