"""
Lexicon with Synonyms, Word Roles, and Usable Dynamic Teaching.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Hand-seeded domain lexicon (word -> POS, semantic class, synonyms, target action).
2. Usable dynamic word teaching: taught synonyms and aliases expand the vocabulary
   and integrate into command execution ONLY after explicit confirmation.
3. Natural phrasing detection for teaching ("X means Y", "remember that X is Y",
   "when I say X I mean Y", "X is another word for Y", "call X Y", etc.).
4. Atomic and schema-validated persistence to miku_usable_lexicon.json.
5. Forget word and show learned commands.
6. Validation against empty keys, >4 word keys, core verb shadowing, and self-references.
"""
import os
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

CORE_VERBS_AND_RESERVED = {
    "open", "launch", "start", "run", "visit", "fire up", "bring up",
    "close", "shut down", "exit", "quit", "terminate", "kill",
    "find", "search", "look up", "locate",
    "mute", "silence", "unmute", "increase", "raise", "boost", "crank up", "turn up", "decrease", "lower", "reduce", "turn down", "volume", "sound",
    "screenshot", "snap", "capture",
    "delete", "remove", "erase", "format", "wipe", "destroy",
    "create", "make", "write", "craft",
    "click", "press", "tap",
    "plan", "schedule", "task", "add",
    "browse", "navigate", "extract", "scrape",
    "status", "health", "abort", "stop", "cancel", "halt",
    "yes", "no", "confirm", "deny",
    "forget", "learn", "teach", "remember",
    "games", "game", "browser", "music", "editor", "ide", "tools", "tool", "social"
}

TEACHING_PATTERNS = [
    # "when I say X I mean Y" or "when I say X open Y" or "whenever I say X then open Y"
    r"^(?:please\s+)?(?:when|whenever)\s+i\s+(?:say|utter)\s+[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?[,\s]+(?:i\s+mean\s+|then\s+)?(?:open|launch|run|execute|mean)?\s*[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "remember that X means Y" or "remember that X is Y" or "remember X refers to Y"
    r"^(?:can\s+you\s+please\s+|please\s+)?remember\s+(?:that\s+)?(?:my\s+)?[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+(?:means|is|refers\s+to)\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "X is another word for Y"
    r"^[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+is\s+another\s+word\s+for\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "call X Y" (e.g. "call diary notepad" or "call edge browser")
    r"^(?:please\s+)?call\s+[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "associate X with Y" or "link word X to Y"
    r"^(?:please\s+)?(?:associate|link\s+(?:word\s+)?)\s*[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+(?:with|to)\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "from now on consider X to be Y"
    r"^(?:from\s+now\s+on\s+)?consider\s+[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+to\s+be\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "let word X mean Y"
    r"^(?:please\s+)?let\s+(?:word\s+)?[\"']?(?P<alias>[a-zA-Z0-9_\- ]+?)[\"']?\s+mean\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "teach word X means Y" / "learn definition X means Y" / "memorize that X means Y"
    r"^(?:please\s+)?(?:teach\s+(?:word\s+)?|learn\s+(?:definition\s+)?|memorize\s+(?:that\s+)?)(?:word\s+)?[\"']?(?P<alias>[a-zA-Z0-9_\-]+)[\"']?\s+(?:means|as|definition)\s+[\"']?(?P<target>.+?)[\"']?$",
    # "please learn that X refers to Y" / "please learn that X is Y"
    r"^(?:please\s+)?learn\s+that\s+[\"']?(?P<alias>[a-zA-Z0-9_\-]+)[\"']?\s+(?:refers?\s+to|is|means)\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
    # "define the word X as Y" / "can you store word X definition Y"
    r"^(?:can\s+you\s+)?(?:store\s+word|define\s+(?:the\s+)?word)\s+[\"']?(?P<alias>[a-zA-Z0-9_\-]+)[\"']?\s+(?:definition|as)\s+[\"']?(?P<target>.+?)[\"']?$",
    # "X means Y" (clean alias teaching, e.g. "computation means calculator")
    r"^[\"']?(?P<alias>[a-zA-Z0-9_\-]+)[\"']?\s+means\s+[\"']?(?P<target>[a-zA-Z0-9_\-\. ]+?)[\"']?$",
]

def validate_teaching(alias: str, target: str) -> Tuple[bool, str]:
    """
    Validates teaching parameters strictly:
    1. Reject empty keys or targets.
    2. Reject keys longer than 4 words.
    3. Reject keys that shadow existing commands, core verbs, or reserved words.
    4. Reject self-references (alias == target).
    5. Reject garbage phrases containing 'i mean', 'whenever', etc.
    """
    alias_clean = alias.strip().lower()
    target_clean = target.strip().lower()

    if not alias_clean or not target_clean:
        return False, "Alias and target cannot be empty."

    # Guard against parsing bug where 'i mean' was accidentally captured in alias
    if "i mean" in alias_clean or alias_clean.endswith(" mean"):
        return False, "Malformed alias key containing 'i mean'."

    words = alias_clean.split()
    if len(words) > 4:
        return False, f"Teaching key is too long ({len(words)} words, maximum allowed is 4 words)."

    if alias_clean in CORE_VERBS_AND_RESERVED or any(w in CORE_VERBS_AND_RESERVED for w in words if len(words) == 1):
        return False, f"Teaching key '{alias_clean}' shadows an existing core command or verb."

    if alias_clean == target_clean:
        return False, f"Self-referential teaching is not permitted ('{alias_clean}' cannot point to itself)."

    # Reject malformed connectors
    if re.search(r"\b(?:but|and|or|then|mean|says?|utter)\b", alias_clean):
        return False, f"Teaching key '{alias_clean}' contains invalid connector or operator word."

    return True, "Valid"

def validate_schema(data: Dict[str, Any]) -> bool:
    """
    Validates that a dynamic lexicon dictionary adheres strictly to the required schema.
    """
    if not isinstance(data, dict):
        return False
    for k, v in data.items():
        if not isinstance(k, str) or not k.strip():
            return False
        if len(k.strip().split()) > 4:
            return False
        if not isinstance(v, dict):
            return False
        if "canonical_target" not in v or not isinstance(v["canonical_target"], str):
            return False
        if "timestamp" not in v or not isinstance(v["timestamp"], (int, float)):
            return False
        if "confidence" not in v or not isinstance(v["confidence"], (int, float)):
            return False
    return True

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
                    data = json.load(f)
                    if validate_schema(data):
                        self.dynamic_lexicon = data
                    else:
                        self.dynamic_lexicon = {}
            except Exception:
                self.dynamic_lexicon = {}

    def _save_dynamic_atomic(self):
        """
        Saves dynamic lexicon atomically and schema-validated.
        Writes to a temporary file, verifies schema, then atomically renames (os.replace).
        """
        if not validate_schema(self.dynamic_lexicon):
            raise ValueError("Schema validation failed for dynamic lexicon prior to write.")
        tmp_path = self.storage_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_lexicon, f, indent=2)
            os.replace(tmp_path, self.storage_path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

    def check_teaching_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detects if utterance is a teaching candidate via diverse natural phrasings.
        Returns teaching metadata dictionary with validation status.
        """
        clean = text.strip()
        for pat in self._compiled_teaching:
            m = pat.match(clean)
            if m:
                raw_alias = m.group("alias").strip()
                raw_target = m.group("target").strip()

                # Clean any lingering filler
                alias = re.sub(r"^(?:that|my|word)\s+", "", raw_alias, flags=re.I).strip()
                alias = re.sub(r"\s+i\s+mean$", "", alias, flags=re.I).strip()

                target = re.sub(r"^(?:open|launch|run|the)\s+", "", raw_target, flags=re.I).strip()
                target = re.sub(r"^(?:definition|meaning|means)\s+", "", target, flags=re.I).strip()

                is_valid, reason = validate_teaching(alias, target)
                return {
                    "is_teaching": True,
                    "is_valid": is_valid,
                    "validation_reason": reason,
                    "alias": alias.lower(),
                    "target": target.lower(),
                    "raw": text
                }
        return None

    def register_taught_word(self, alias: str, target: str, source: str = "user_teaching") -> Dict[str, Any]:
        """
        Registers taught word into usable lexicon atomically after confirmation.
        """
        alias_clean = alias.strip().lower()
        target_clean = target.strip().lower()

        is_valid, reason = validate_teaching(alias_clean, target_clean)
        if not is_valid:
            raise ValueError(f"Cannot register invalid teaching '{alias_clean}' -> '{target_clean}': {reason}")

        record = {
            "canonical_target": target_clean,
            "source": source,
            "timestamp": time.time(),
            "confidence": 1.0,
            "usable_as": "alias"
        }
        self.dynamic_lexicon[alias_clean] = record
        self._save_dynamic_atomic()

        # Update app catalog index if app target
        try:
            from miku.core2_cognitive.app_catalog import _ALIAS_INDEX, predict_app
            canonical, _ = predict_app(target_clean)
            _ALIAS_INDEX[alias_clean] = canonical
        except Exception:
            pass

        return record

    def forget_word(self, word: str) -> bool:
        """
        Deletes a word or alias from dynamic memory atomically.
        """
        w = word.strip().lower()
        if w in self.dynamic_lexicon:
            del self.dynamic_lexicon[w]
            self._save_dynamic_atomic()
            try:
                from miku.core2_cognitive.app_catalog import _ALIAS_INDEX
                if w in _ALIAS_INDEX:
                    del _ALIAS_INDEX[w]
            except Exception:
                pass
            return True
        return False

    def show_learned(self) -> str:
        """
        Returns a formatted listing of all learned words and aliases.
        """
        if not self.dynamic_lexicon:
            return "I haven't learned any custom aliases or words yet."
        lines = ["Learned Knowledge & Dynamic Lexicon:"]
        for k, v in sorted(self.dynamic_lexicon.items()):
            target = v.get("canonical_target", "")
            usable = v.get("usable_as", "alias")
            lines.append(f"  • '{k}' -> '{target}' ({usable})")
        return "\n".join(lines)

    def resolve_word(self, word: str) -> str:
        w_lower = word.strip().lower()
        if w_lower in self.dynamic_lexicon:
            return self.dynamic_lexicon[w_lower]["canonical_target"]
        if w_lower in SEED_LEXICON:
            return SEED_LEXICON[w_lower].get("canonical", w_lower)
        return w_lower

    def rewrite_taught_terms(self, text: str) -> str:
        tokens = text.split()
        modified = []
        i = 0
        while i < len(tokens):
            matched = False
            for length in (4, 3, 2, 1):
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
        self._save_dynamic_atomic()

    def get_lexicon_stats(self) -> Dict[str, Any]:
        return {
            "seed_words": len(SEED_LEXICON),
            "dynamic_taught_words": len(self.dynamic_lexicon),
            "dynamic_keys": list(self.dynamic_lexicon.keys())
        }
