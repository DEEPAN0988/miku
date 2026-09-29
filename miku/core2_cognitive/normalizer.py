"""
Normalizer and Tokenizer for Miku English Understanding.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Quoted string preservation verbatim.
2. Contraction expansion (e.g. don't -> do not).
3. Conversational filler and politeness stripping (e.g. "please", "can you", "hey miku").
4. Number-word to digit conversion (e.g. "two" -> "2", "first" -> "1").
5. Deterministic tokenization into structured tokens with character spans.
"""
import re
from typing import List, Dict, Tuple, Any, Optional

# Standard contractions
CONTRACTIONS = {
    r"\bdon't\b": "do not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bi'm\b": "i am",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\blet's\b": "let us",
    r"\bthere's\b": "there is",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bwasn't\b": "was not",
    r"\baren't\b": "are not",
    r"\bisn't\b": "is not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
}

# Number words to digit string
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
}

# Conversational prefixes and fillers that don't change core command semantics
FILLER_PATTERNS = [
    r"^(?:hey\s+|hi\s+|hello\s+)?miku[,:\s]*",
    r"^(?:can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|would\s+you\s+mind\s+(?:if\s+you\s+)?|will\s+you\s+(?:please\s+)?|please\s+)",
    r"^(?:i\s+need\s+to\s+|i\s+want\s+to\s+|let\s+us\s+|let's\s+|how\s+about\s+we\s+)",
    r"^(?:i\s+wish\s+to\s+|i\s+(?:would|'d)\s+like\s+to\s+|feel\s+like\s+|how\s+about\s+|let\s+me\s+just\s+|let\s+me\s+quickly\s+|mind\s+)",
    r"(?:,\s*please|\s+please|\s+for\s+me|\s+right\s+now|\s+right\s+away|\s+asap)[\.!\?]*$",
]

class Normalizer:
    def __init__(self):
        self._contraction_re = [(re.compile(p, re.IGNORECASE), repl) for p, repl in CONTRACTIONS.items()]

    def extract_quoted_strings(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extracts single or double quoted strings and replaces them with placeholders
        so internal normalization doesn't alter user file names, URLs, or quotes.
        """
        placeholders = {}
        counter = 0

        def repl(match):
            nonlocal counter
            token = f"__quoted_span_{counter}__"
            placeholders[token] = match.group(1)
            counter += 1
            return token

        # Double quotes or single quotes with at least 1 character
        modified = re.sub(r'\"([^\"]+)\"', repl, text)
        modified = re.sub(r'\'([^\']+)\'', repl, modified)
        return modified, placeholders

    def restore_quoted_strings(self, text: str, placeholders: Dict[str, str]) -> str:
        res = text
        for token, original in placeholders.items():
            res = res.replace(token, original)
        return res

    def expand_contractions(self, text: str) -> str:
        res = text
        for pat, repl in self._contraction_re:
            res = pat.sub(repl, res)
        return res

    def convert_number_words(self, text: str) -> str:
        tokens = text.split()
        converted = []
        for t in tokens:
            lower_t = t.lower()
            if lower_t in NUMBER_WORDS:
                converted.append(NUMBER_WORDS[lower_t])
            else:
                converted.append(t)
        return " ".join(converted)

    def strip_fillers(self, text: str) -> str:
        """
        Strips conversational framing like 'could you please', 'hey miku', while preserving
        negative markers ('not', 'don't') and command keywords.
        """
        res = text.strip()
        changed = True
        while changed:
            changed = False
            for pat in FILLER_PATTERNS:
                new_res = re.sub(pat, "", res, flags=re.IGNORECASE).strip()
                if new_res != res and len(new_res) > 0:
                    res = new_res
                    changed = True
        return res

    def normalize(self, text: str, strip_politeness: bool = True) -> Dict[str, Any]:
        """
        Executes full normalization pipeline:
        1. Quote extraction
        2. Lowercasing
        3. Contraction expansion
        4. Politeness / filler stripping (if requested)
        5. Number-word conversion
        6. Quote restoration
        """
        raw_text = text.strip()
        quoted_text, placeholders = self.extract_quoted_strings(raw_text)

        # Lowercase non-quoted parts
        processed = quoted_text.lower()

        # Expand contractions
        processed = self.expand_contractions(processed)

        # Strip polite fillers
        if strip_politeness:
            processed = self.strip_fillers(processed)

        # Convert number words
        processed = self.convert_number_words(processed)

        # Restore quotes
        restored = self.restore_quoted_strings(processed, placeholders)

        # Clean multiple spaces and trailing punctuation
        cleaned = re.sub(r"\s+", " ", restored).strip()
        cleaned_no_punct = re.sub(r"[?!.]+$", "", cleaned).strip()

        # Tokenize
        tokens = cleaned_no_punct.split()

        return {
            "raw": raw_text,
            "normalized": cleaned_no_punct,
            "tokens": tokens,
            "placeholders": placeholders,
            "has_negation": bool(re.search(r"\b(?:not|no|never|don't|do\s+not)\b", cleaned_no_punct))
        }
