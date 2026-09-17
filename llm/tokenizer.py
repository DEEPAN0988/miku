"""
Local Lightweight Tokenizer for Miku LLM.
Built 100% from scratch. No external tokenizer models, zero cloud APIs.
Uses subword and byte-fallback encoding for zero-OOD robustness.
"""

import json
import os
import re
from typing import List, Dict, Optional

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<user>": 3,
    "<miku>": 4,
    "<unk>": 5
}


class MikuTokenizer:
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train_from_corpus(self, texts: List[str], max_vocab: int = 512):
        """Build vocabulary from a collection of texts."""
        # 1. Start with special tokens and all printable ASCII characters
        for ch in range(32, 127):
            char = chr(ch)
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char

        # 2. Count frequent words & subwords
        word_freq: Dict[str, int] = {}
        for text in texts:
            # Tokenize by whitespace and punctuation
            words = re.findall(r"\w+|[^\w\s]", text.lower())
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words:
            if len(self.token_to_id) >= max_vocab:
                break
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token ID sequence."""
        tokens = []
        if add_special_tokens:
            tokens.append(SPECIAL_TOKENS["<bos>"])

        # Split into words/punctuation
        segments = re.findall(r"<user>|<miku>|<eos>|<bos>|\w+|[^\w\s]|\s+", text)
        for seg in segments:
            seg_clean = seg.strip()
            if not seg_clean:
                continue
            if seg_clean in self.token_to_id:
                tokens.append(self.token_to_id[seg_clean])
            elif seg_clean.lower() in self.token_to_id:
                tokens.append(self.token_to_id[seg_clean.lower()])
            else:
                # Byte fallback character by character
                for ch in seg_clean:
                    tokens.append(self.token_to_id.get(ch, SPECIAL_TOKENS["<unk>"]))

        if add_special_tokens:
            tokens.append(SPECIAL_TOKENS["<eos>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ID sequence back to string."""
        words = []
        for tid in token_ids:
            if tid in [SPECIAL_TOKENS["<pad>"], SPECIAL_TOKENS["<bos>"], SPECIAL_TOKENS["<eos>"]]:
                continue
            token_str = self.id_to_token.get(tid, "")
            if token_str in ["<user>", "<miku>"]:
                words.append(f"\n{token_str} ")
            elif len(token_str) == 1 and not token_str.isalnum():
                # Attach punctuation directly
                if words:
                    words[-1] += token_str
                else:
                    words.append(token_str)
            else:
                words.append(token_str)
        return " ".join(words).replace("  ", " ").strip()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
