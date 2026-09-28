"""
Safe Speller with Damerau-Levenshtein edit distance.
Core 2: Cognitive Router (Brain and Memory)

Provides:
1. Fast symmetric deletion candidate generation (edit distance <= 2).
2. Transposition pre-pass (Damerau extension): swapped-adjacent-char variants
   are generated alongside deletion variants, so 'floder'->'folder' resolves
   by rule, not by frequency tuning.
3. Safety barrier: NEVER auto-corrects into a destructive command word.
4. Ambiguity: when top-2 candidates are equally distant AND within
   AMBIGUITY_FREQ_THRESHOLD frequency score, return 'ambiguous'.
5. General English word list (EnglishLexicon.all_words) prevents correcting
   valid common words ('sing', 'lower', 'dance', etc.) to domain words.
"""
from typing import Dict, List, Set, Tuple, Optional, Any

DESTRUCTIVE_WORDS = {
    "delete", "remove", "kill", "format", "wipe", "drop",
    "terminate", "destroy", "erase",
}

# Ambiguity threshold: if top-2 candidates share same Damerau distance and
# their frequency scores differ by less than this, ask instead of guessing.
AMBIGUITY_FREQ_THRESHOLD = 15

# Command domain vocabulary: word -> priority score (higher = more likely correction target).
# IMPORTANT: Do NOT tune these scores to fix specific test strings.
# The score reflects genuine relative frequency in the command domain.
# To add a word, add it here with a principled score; to prevent false correction
# of a common English word, add it to known_words (via EnglishLexicon).
DOMAIN_VOCABULARY: Dict[str, int] = {
    # Primary action verbs
    "open":       100, "launch":    90, "start":  80, "run":      70,
    "close":       95, "shut":      75, "quit":   65, "exit":     60,
    "terminate":   90, "kill":      80,   # close synonyms — keep here so they are not corrected away
    # Volume / audio
    "volume":      90, "sound":     85, "audio":  80,
    "mute":        85, "unmute":    80,
    "increase":    70, "decrease":  70,
    "turn":        85, "boost":     60, "silence": 75,
    "lower":       70, "raise":     70, "crank":   60,
    # Screen / camera
    "screenshot":  90, "screen":    80, "capture": 75, "snap":    70,
    "camera":      75, "webcam":    70,
    # File / folder
    "file":        90, "folder":    80, "document": 70, "note":   75,
    # Actions
    "create":      80, "make":      75, "search":   85, "find":   85,
    "browse":      75, "navigate":  75, "extract":  70, "scrape": 70,
    "inspect":     60, "check":     80, "show":     75,
    "delete":      80, "remove":    75, "erase":    70,
    # Planning / tasks
    "status":      80, "report":    75, "health":   70,
    "plan":        80, "schedule":  75, "task":     80,
    "automation":  70, "stop":      85, "abort":    80, "cancel": 85,
    # Window management
    "maximize":    70, "minimize":  70, "restore":  65, "window": 80,
    # UI interaction
    "button":      75, "icon":      75, "click":    80, "press":  75,
    # Colors (for UI interactions)
    "red":         70, "blue":      70, "green":    70, "yellow": 70,
    # App entities
    "notepad":     95, "calculator": 95, "edge":    95, "chrome": 90,
    "wuthering":   85, "waves":     85, "spotify":  80, "vlc":    80,
    "terminal":    80, "powershell": 80,
    # Downloads / common folder names
    "downloads":   75, "documents": 70, "desktop":  70, "pictures": 65,
}


def _damerau_levenshtein(s1: str, s2: str) -> int:
    """
    Optimal string alignment (restricted Damerau-Levenshtein) distance.
    Handles substitutions, insertions, deletions, AND adjacent transpositions.
    e.g. damerau('floder', 'folder') == 1  (transpose e<->r)
         levenshtein('floder', 'folder') == 2  (delete e, insert r elsewhere)
    """
    m, n = len(s1), len(s2)
    # dp[i][j] = OSA distance between s1[:i] and s2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
            # Transposition
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
    return dp[m][n]


COMMON_ENGLISH_WORDS = {
    "sing", "song", "dance", "cook", "hug", "tune", "melody", "flower",
    "joke", "story", "weather", "book", "coffee", "tea", "food",
    "help", "show", "tell", "give", "play", "read", "write",
    "try", "need", "want", "like", "use", "see", "hear",
    "word", "name", "time", "list", "view", "set", "get", "loader",
}

class SafeSpeller:
    def __init__(
        self,
        vocab: Dict[str, int] = DOMAIN_VOCABULARY,
        known_words: Optional[Set[str]] = None,
    ):
        self.vocab = vocab
        # known_words: tokens that are correct as-is and should not be corrected.
        # Populated from EnglishLexicon (common English words like 'sing', 'lower', etc.)
        if known_words is None:
            try:
                from miku.core2_cognitive.english_lexicon import EnglishLexicon
                self.known_words: Set[str] = EnglishLexicon().all_words
            except Exception:
                self.known_words = set(self.vocab.keys())
        else:
            self.known_words = known_words
            
        self.known_words.update(COMMON_ENGLISH_WORDS)

        # Index: deletion_variant -> set of vocab words that produce it
        self.deletes_dict: Dict[str, Set[str]] = {}
        self._build_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _get_deletes(self, word: str, max_distance: int = 2) -> Set[str]:
        """Generate all strings reachable by up to max_distance deletions."""
        deletes: Set[str] = set()
        queue = {word}
        for _ in range(max_distance):
            temp: Set[str] = set()
            for w in queue:
                if len(w) > 1:
                    for i in range(len(w)):
                        del_w = w[:i] + w[i + 1:]
                        deletes.add(del_w)
                        temp.add(del_w)
            queue = temp
        return deletes

    def _get_transpositions(self, word: str) -> Set[str]:
        """Generate all strings reachable by exactly one adjacent transposition."""
        result: Set[str] = set()
        for i in range(len(word) - 1):
            t = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            result.add(t)
        return result

    def _build_index(self):
        for word in self.vocab:
            del_set = self._get_deletes(word, max_distance=2)
            for d in del_set:
                self.deletes_dict.setdefault(d, set()).add(word)

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct_token(self, token: str) -> Tuple[str, str, List[str]]:
        """
        Correct a single token.

        Returns: (corrected_token, status, candidates[:3])
          status in {'exact', 'corrected', 'ambiguous', 'unknown', 'blocked_destructive'}
        """
        t = token.lower().strip()
        if not t or len(t) < 3:
            return token, "exact", [token]
        if t in self.vocab or t in self.known_words:
            return token, "exact", [token]

        candidates: Dict[str, int] = {}  # word -> damerau distance

        def _add_candidate(word: str):
            if word in candidates:
                return
            dist = _damerau_levenshtein(t, word)
            if dist <= 2:
                candidates[word] = dist

        # Path 1: direct deletion match (t is a deletion of a vocab word)
        if t in self.deletes_dict:
            for cand in self.deletes_dict[t]:
                _add_candidate(cand)

        # Path 2: deletion of t matches deletions of vocab words
        for td in self._get_deletes(t, max_distance=2):
            if td in self.deletes_dict:
                for cand in self.deletes_dict[td]:
                    _add_candidate(cand)

        # Path 3 (Damerau extension): transpositions of t may be exact vocab words
        # or may hit the deletion index. This catches 'floder' -> transpose -> 'folder'.
        for tp in self._get_transpositions(t):
            if tp in self.vocab:
                _add_candidate(tp)
            if tp in self.deletes_dict:
                for cand in self.deletes_dict[tp]:
                    _add_candidate(cand)
            # Also try deletions of transpositions (dist-2 transposition+deletion)
            for td in self._get_deletes(tp, max_distance=1):
                if td in self.deletes_dict:
                    for cand in self.deletes_dict[td]:
                        _add_candidate(cand)

        if not candidates:
            return token, "unknown", []

        # Sort by (distance ASC, frequency DESC)
        sorted_cands = sorted(
            candidates.keys(),
            key=lambda c: (candidates[c], -self.vocab.get(c, 0))
        )

        best_cand = sorted_cands[0]
        best_dist = candidates[best_cand]

        # Safety: never correct into a destructive keyword
        if best_cand in DESTRUCTIVE_WORDS:
            return token, "blocked_destructive", sorted_cands[:3]

        # Ambiguity: top-2 same distance AND close frequency
        if len(sorted_cands) > 1:
            second_cand = sorted_cands[1]
            second_dist = candidates[second_cand]
            freq_gap = abs(self.vocab.get(best_cand, 0) - self.vocab.get(second_cand, 0))
            if best_dist == second_dist and freq_gap < AMBIGUITY_FREQ_THRESHOLD:
                return token, "ambiguous", sorted_cands[:3]

        return best_cand, "corrected", sorted_cands[:3]

    def correct_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Correct a full sentence, word by word.
        Returns:
          corrected    : corrected sentence string
          has_changes  : bool
          ambiguities  : list of {word, candidates}
          blocked_destructive : list of {word, candidate}
        """
        words = sentence.split()
        corrected_words = []
        ambiguities = []
        blocked = []
        changed = False

        for w in words:
            clean_w = w.strip(".,!?\"'")
            if not clean_w:
                corrected_words.append(w)
                continue

            corr, status, cands = self.correct_token(clean_w)
            if status == "corrected":
                prefix = w[: w.find(clean_w)]
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
            "blocked_destructive": blocked,
        }
