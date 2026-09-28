"""
Lexicon with Synonyms, Word Roles, and Usable Dynamic Teaching.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Hand-seeded domain lexicon (word -> POS, semantic class, synonyms, target action).
2. Usable dynamic word teaching: taught synonyms and aliases immediately expand
   the vocabulary and integrate into command execution.
3. Natural phrasing detection for teaching ("X means Y", "remember that X is Y",
   "when I say X I mean Y", "X is another word for Y", "call X Y", etc.).
4. Persistence to miku_usable_lexicon.json with source, timestamp, and confidence.
"""
import re
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

LEXICON_STORAGE_PATH = Path("miku_usable_lexicon.json")

# Hand-seeded domain lexicon for operating system assistant commands
SEED_LEXICON: Dict[str, Dict[str, Any]] = {
    # Actions: Launch / Open
    "open": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["launch", "start", "run", "visit", "bring up", "fire up", "pop open"], "action": "open_app"},
    "launch": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "start", "run", "fire up"], "action": "open_app"},
    "start": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "launch", "run"], "action": "open_app"},
    "run": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "launch", "start", "execute"], "action": "open_app"},
    "visit": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "navigate", "browse"], "action": "open_app"},
    "fire up": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "launch", "start"], "action": "open_app"},
    "bring up": {"pos": "VERB", "class": "ACTION_OPEN", "synonyms": ["open", "switch to", "restore"], "action": "open_app"},

    # Actions: Close / Terminate
    "close": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["shut down", "exit", "quit", "terminate", "kill"], "action": "close_app"},
    "shut down": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["close", "exit", "quit", "terminate", "kill"], "action": "close_app"},
    "exit": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["close", "quit", "terminate"], "action": "close_app"},
    "quit": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["close", "exit", "terminate"], "action": "close_app"},
    "terminate": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["close", "kill", "shut down"], "action": "close_app"},
    "kill": {"pos": "VERB", "class": "ACTION_CLOSE", "synonyms": ["terminate", "close", "force stop"], "action": "close_app"},

    # Actions: Search / Find
    "find": {"pos": "VERB", "class": "ACTION_SEARCH", "synonyms": ["search", "look up", "locate", "hunt for"], "action": "search_file"},
    "search": {"pos": "VERB", "class": "ACTION_SEARCH", "synonyms": ["find", "look up", "locate"], "action": "search_file"},
    "look up": {"pos": "VERB", "class": "ACTION_SEARCH", "synonyms": ["search", "find"], "action": "search_file"},
    "locate": {"pos": "VERB", "class": "ACTION_SEARCH", "synonyms": ["find", "search"], "action": "search_file"},

    # Actions: Volume & Sound
    "mute": {"pos": "VERB", "class": "ACTION_MUTE", "synonyms": ["silence", "quiet", "hush"], "action": "volume"},
    "silence": {"pos": "VERB", "class": "ACTION_MUTE", "synonyms": ["mute", "quiet", "hush"], "action": "volume"},
    "unmute": {"pos": "VERB", "class": "ACTION_UNMUTE", "synonyms": ["restore sound", "sound on"], "action": "volume"},
    "increase": {"pos": "VERB", "class": "ACTION_VOL_UP", "synonyms": ["turn up", "raise", "boost", "crank up"], "action": "volume"},
    "raise": {"pos": "VERB", "class": "ACTION_VOL_UP", "synonyms": ["turn up", "increase", "boost"], "action": "volume"},
    "boost": {"pos": "VERB", "class": "ACTION_VOL_UP", "synonyms": ["turn up", "increase", "crank up"], "action": "volume"},
    "crank up": {"pos": "VERB", "class": "ACTION_VOL_UP", "synonyms": ["turn up", "increase", "boost"], "action": "volume"},
    "turn up": {"pos": "VERB", "class": "ACTION_VOL_UP", "synonyms": ["increase", "crank up", "raise"], "action": "volume"},
    "decrease": {"pos": "VERB", "class": "ACTION_VOL_DOWN", "synonyms": ["turn down", "lower", "reduce", "tone down"], "action": "volume"},
    "lower": {"pos": "VERB", "class": "ACTION_VOL_DOWN", "synonyms": ["turn down", "decrease", "reduce"], "action": "volume"},
    "reduce": {"pos": "VERB", "class": "ACTION_VOL_DOWN", "synonyms": ["turn down", "decrease", "lower"], "action": "volume"},
    "turn down": {"pos": "VERB", "class": "ACTION_VOL_DOWN", "synonyms": ["lower", "reduce", "decrease"], "action": "volume"},

    # Actions: Screen & Camera
    "screenshot": {"pos": "NOUN", "class": "TARGET_SCREEN", "synonyms": ["screen capture", "snapshot", "screen grab"], "action": "screenshot"},
    "snap": {"pos": "VERB", "class": "ACTION_CAPTURE", "synonyms": ["capture", "take"], "action": "screenshot"},
    "capture": {"pos": "VERB", "class": "ACTION_CAPTURE", "synonyms": ["take", "snap"], "action": "screenshot"},

    # Target Entities: Applications & OS tools
    "notepad": {"pos": "NOUN", "class": "APP_TARGET", "synonyms": ["text editor", "note editor", "notes", "diary"], "canonical": "notepad"},
    "calculator": {"pos": "NOUN", "class": "APP_TARGET", "synonyms": ["calc", "math", "mathapp"], "canonical": "calculator"},
    "edge": {"pos": "NOUN", "class": "APP_TARGET", "synonyms": ["microsoft edge", "browser", "web browser", "internet"], "canonical": "edge"},
    "wuthering waves": {"pos": "NOUN", "class": "APP_TARGET", "synonyms": ["wuwa", "wuthering wave", "game"], "canonical": "wuthering waves"},
}

TEACHING_PATTERNS = [
    # "when I say X I mean Y" or "when I say X open Y"
    r"^(?:please\s+)?when\s+i\s+(?:say|utter)\s+(?P<alias>[a-zA-Z0-9_\- ]+?)[,\s]+(?:i\s+mean|execute|open|launch|run)\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "remember that X means Y" or "remember that X is Y"
    r"^(?:can\s+you\s+please\s+|please\s+)?remember\s+(?:that\s+)?(?P<alias>[a-zA-Z0-9_\- ]+?)\s+(?:means|is|refers\s+to)\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "X is another word for Y"
    r"^(?P<alias>[a-zA-Z0-9_\- ]+?)\s+is\s+another\s+word\s+for\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "call X Y" (e.g. "call diary notepad" or "call edge browser")
    r"^(?:please\s+)?call\s+(?P<alias>[a-zA-Z0-9_\- ]+?)\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "associate X with Y" or "link word X to Y"
    r"^(?:please\s+)?(?:associate|link\s+word)\s+(?P<alias>[a-zA-Z0-9_\- ]+?)\s+(?:with|to)\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "from now on consider X to be Y"
    r"^(?:from\s+now\s+on\s+)?consider\s+(?P<alias>[a-zA-Z0-9_\- ]+?)\s+to\s+be\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "let word X mean Y"
    r"^(?:please\s+)?let\s+(?:word\s+)?(?P<alias>[a-zA-Z0-9_\- ]+?)\s+mean\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
    # "X means Y" (clean alias teaching, e.g. "computation means calculator")
    r"^(?P<alias>[a-zA-Z0-9_\-]+)\s+means\s+(?P<target>[a-zA-Z0-9_\-\. ]+)$",
]

class CommandLexicon:
    def __init__(self, storage_path: Path = LEXICON_STORAGE_PATH):
        self.storage_path = storage_path
        self.dynamic_lexicon: Dict[str, Dict[str, Any]] = {}
        self._compiled_teaching = [re.compile(p, re.IGNORECASE) for p in TEACHING_PATTERNS]
        self._load_dynamic()

    def _load_dynamic(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.dynamic_lexicon = json.load(f)
            except Exception:
                self.dynamic_lexicon = {}

    def _save_dynamic(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_lexicon, f, indent=2)
        except Exception:
            pass

    def check_teaching_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects if utterance is an explicit teaching command via diverse natural phrasings.
        Returns teaching metadata dictionary if matched, or None.
        """
        clean = text.strip()
        for pat in self._compiled_teaching:
            m = pat.match(clean)
            if m:
                alias = m.group("alias").strip().lower()
                target = m.group("target").strip().lower()

                # Clean target of extraneous leading verbs if captured
                target = re.sub(r"^(?:open|launch|run|the)\s+", "", target)

                if alias and target and alias != target:
                    return {
                        "is_teaching": True,
                        "alias": alias,
                        "target": target,
                        "raw": text
                    }
        return None

    def register_taught_word(self, alias: str, target: str, source: str = "user_teaching") -> Dict[str, Any]:
        """
        Registers taught word into usable lexicon immediately.
        Stores source, timestamp, confidence.
        """
        alias_clean = alias.strip().lower()
        target_clean = target.strip().lower()

        if alias_clean in ("games", "game", "browser", "music", "editor", "ide", "tools"):
            return {
                "canonical_target": target_clean,
                "source": source,
                "timestamp": time.time(),
                "confidence": 1.0,
                "usable_as": "category_preference"
            }

        record = {
            "canonical_target": target_clean,
            "source": source,
            "timestamp": time.time(),
            "confidence": 1.0,
            "usable_as": "alias"
        }
        self.dynamic_lexicon[alias_clean] = record
        self._save_dynamic()

        # Also register in app_catalog runtime alias index if app target
        try:
            from miku.core2_cognitive.app_catalog import _ALIAS_INDEX, predict_app
            canonical, _ = predict_app(target_clean)
            _ALIAS_INDEX[alias_clean] = canonical
        except Exception:
            pass

        return record

    def resolve_word(self, word: str) -> str:
        """
        Resolves word or multi-word term through dynamic learned aliases and seed lexicon.
        """
        w_lower = word.strip().lower()
        # 1. Check dynamic learned lexicon
        if w_lower in self.dynamic_lexicon:
            return self.dynamic_lexicon[w_lower]["canonical_target"]

        # 2. Check seed lexicon
        if w_lower in SEED_LEXICON:
            return SEED_LEXICON[w_lower].get("canonical", w_lower)

        return w_lower

    def rewrite_taught_terms(self, text: str) -> str:
        """
        Replaces any learned alias in a sentence with its canonical target
        so downstream parsers immediately understand the sentence.
        """
        tokens = text.split()
        modified = []
        i = 0
        while i < len(tokens):
            matched = False
            # Check 3-gram, 2-gram, 1-gram
            for length in (3, 2, 1):
                if i + length <= len(tokens):
                    span = " ".join(tokens[i:i+length]).lower()
                    if span in self.dynamic_lexicon:
                        canonical = self.dynamic_lexicon[span]["canonical_target"]
                        modified.append(canonical)
                        i += length
                        matched = True
                        break
            if not matched:
                modified.append(tokens[i])
                i += 1
        return " ".join(modified)

    def reset(self):
        self.dynamic_lexicon.clear()
        self._save_dynamic()

    def get_lexicon_stats(self) -> Dict[str, Any]:
        return {
            "seed_words": len(SEED_LEXICON),
            "dynamic_taught_words": len(self.dynamic_lexicon),
            "dynamic_keys": list(self.dynamic_lexicon.keys())
        }
