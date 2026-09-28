"""
Safe SymSpell-Style Typo Corrector with Destructive Action Safeguards.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Fast symmetric deletion candidate generation (edit distance <= 2).
2. Safety barrier: NEVER auto-corrects into a destructive command (delete, kill, format, remove, drop).
3. Ambiguity detector: If multiple valid candidates have equal or near-identical edit distance,
   refuses to guess and asks the user for clarification.
4. Confidence floor: Discards weak or distant candidates.
"""
from typing import Dict, List, Set, Tuple, Optional, Any

DESTRUCTIVE_WORDS = {"delete", "remove", "kill", "format", "wipe", "drop", "terminate", "destroy", "erase"}

# Command domain dictionary
DOMAIN_VOCABULARY = {
    # Action verbs
    "open": 100, "launch": 90, "start": 80, "run": 70, "close": 95, "shut": 75, "quit": 65, "exit": 60,
    "volume": 90, "sound": 85, "audio": 80, "mute": 85, "unmute": 80, "increase": 70, "decrease": 70,
    "turn": 85, "crank": 60, "lower": 70, "raise": 70, "boost": 60, "silence": 75,
    "screenshot": 90, "screen": 80, "capture": 75, "snap": 70,
    "camera": 75, "webcam": 70, "inspect": 60, "check": 80,
    "file": 90, "folder": 80, "document": 70, "note": 75, "create": 80, "make": 75, "search": 85, "find": 85,
    "browse": 75, "navigate": 75, "extract": 70, "scrape": 70,
    "status": 80, "report": 75, "health": 70, "plan": 80, "schedule": 75, "task": 80,
    "automation": 70, "stop": 85, "abort": 80, "cancel": 85,
    "maximize": 70, "minimize": 70, "restore": 65, "window": 80,
    "button": 75, "icon": 75, "click": 80, "press": 75,
    "red": 70, "blue": 70, "green": 70, "yellow": 70,

    # App entities
    "notepad": 95, "calculator": 95, "edge": 95, "chrome": 90,
    "wuthering": 85, "waves": 85, "spotify": 80, "vlc": 80, "terminal": 80, "powershell": 80,
}

class SafeSpeller:
    def __init__(self, vocab: Dict[str, int] = DOMAIN_VOCABULARY, known_words: Optional[Set[str]] = None):
        self.vocab = vocab
        if known_words is None:
            try:
                from miku.core2_cognitive.english_lexicon import EnglishLexicon
                self.known_words = EnglishLexicon().all_words
            except Exception:
                self.known_words = set(self.vocab.keys())
        else:
            self.known_words = known_words
        self.deletes_dict: Dict[str, Set[str]] = {}
        self._build_index()

    def _get_deletes(self, word: str, max_distance: int = 2) -> Set[str]:
        deletes = set()
        queue = {word}
        for _ in range(max_distance):
            temp = set()
            for w in queue:
                if len(w) > 1:
                    for i in range(len(w)):
                        del_w = w[:i] + w[i+1:]
                        deletes.add(del_w)
                        temp.add(del_w)
            queue = temp
        return deletes

    def _build_index(self):
        for word in self.vocab:
            del_set = self._get_deletes(word, max_distance=2)
            for d in del_set:
                if d not in self.deletes_dict:
                    self.deletes_dict[d] = set()
                self.deletes_dict[d].add(word)

    def _levenshtein(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insert = prev[j + 1] + 1
                delete = curr[j] + 1
                subst = prev[j] + (0 if c1 == c2 else 1)
                curr.append(min(insert, delete, subst))
            prev = curr
        return prev[-1]

    def correct_token(self, token: str) -> Tuple[str, str, List[str]]:
        """
        Corrects a single token.
        Returns (corrected_token, status, candidate_list)
        Status: 'exact', 'corrected', 'ambiguous', 'unknown', 'blocked_destructive'
        """
        t = token.lower().strip()
        if not t or len(t) < 3 or t in self.vocab or t in self.known_words:
            return token, "exact", [token]

        # Check symmetric deletes
        candidates: Dict[str, int] = {} # word -> edit_distance

        # Direct deletes match
        if t in self.deletes_dict:
            for cand in self.deletes_dict[t]:
                dist = self._levenshtein(t, cand)
                if dist <= 2:
                    candidates[cand] = dist

        # Deletes of token matching deletes of dictionary words
        token_deletes = self._get_deletes(t, max_distance=2)
        for td in token_deletes:
            if td in self.deletes_dict:
                for cand in self.deletes_dict[td]:
                    dist = self._levenshtein(t, cand)
                    if dist <= 2:
                        candidates[cand] = dist

        if not candidates:
            return token, "unknown", []

        # Sort candidates by: 1) smallest edit distance, 2) highest frequency
        sorted_cands = sorted(candidates.keys(), key=lambda c: (candidates[c], -self.vocab.get(c, 0)))

        best_cand = sorted_cands[0]
        best_dist = candidates[best_cand]

        # Safety check: Is best candidate a destructive word?
        if best_cand in DESTRUCTIVE_WORDS:
            return token, "blocked_destructive", sorted_cands[:3]

        # Ambiguity check: If top-2 candidates share the same edit distance and close frequency
        if len(sorted_cands) > 1:
            second_cand = sorted_cands[1]
            second_dist = candidates[second_cand]
            if best_dist == second_dist and abs(self.vocab.get(best_cand, 0) - self.vocab.get(second_cand, 0)) < 15:
                # Ambiguous
                return token, "ambiguous", sorted_cands[:3]

        return best_cand, "corrected", sorted_cands[:3]

    def correct_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Corrects full sentence safely.
        Returns:
        {
            "original": sentence,
            "corrected": corrected_sentence,
            "has_changes": bool,
            "ambiguities": list,
            "blocked_destructive": list
        }
        """
        words = sentence.split()
        corrected_words = []
        ambiguities = []
        blocked = []
        changed = False

        for w in words:
            # Strip punctuation attached to word
            clean_w = w.strip(".,!?\"'")
            if not clean_w:
                corrected_words.append(w)
                continue

            corr, status, cands = self.correct_token(clean_w)
            if status == "corrected":
                # Preserve surrounding punctuation
                prefix = w[:w.find(clean_w)]
                suffix = w[w.find(clean_w) + len(clean_w):]
                corrected_words.append(f"{prefix}{corr}{suffix}")
                changed = True
            elif status == "ambiguous":
                corrected_words.append(w)
                ambiguities.append({"word": clean_w, "candidates": cands})
            elif status == "blocked_destructive":
                corrected_words.append(w)
                blocked.append({"word": clean_w, "candidate": corr})
            else:
                corrected_words.append(w)

        return {
            "original": sentence,
            "corrected": " ".join(corrected_words),
            "has_changes": changed,
            "ambiguities": ambiguities,
            "blocked_destructive": blocked
        }
