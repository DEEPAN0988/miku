"""
MIKU: Real-Time Dynamic Learning & Continuous Knowledge Store.
Core 2: Cognitive Router (Brain & Memory)
Persists learned user preferences, custom aliases, and category mappings in real time.
"""
import os
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

LEARNED_MEMORY_FILE = Path("c:/miku/miku_learned_memory.json")

TEACH_PATTERNS = [
    # "when I say games open wuthering wave" / "whenever I say play game then open wuwa"
    r"^(?:when\s+i\s+say|whenever\s+i\s+say)\s+[\"']?(?P<trigger>.+?)[\"']?\s+(?:then\s+)?(?:open|launch|start|run)\s+[\"']?(?P<target>.+?)[\"']?$",
    # "remember that my game is wuthering waves" / "remember my favorite browser is brave"
    r"^(?:remember\s+(?:that\s+)?(?:my\s+)?)(?P<trigger>favorite\s+game|game|browser|music|editor|ide|player)\s+is\s+[\"']?(?P<target>.+?)[\"']?$",
    # "remember notes means notepad" / "remember wuwa means wuthering waves"
    r"^(?:remember\s+(?:that\s+)?)(?P<trigger>.+?)\s+means\s+[\"']?(?P<target>.+?)[\"']?$",
    # "call diary notepad" / "alias wuwa to wuthering waves"
    r"^(?:call|alias)\s+[\"']?(?P<trigger>.+?)[\"']?\s+(?:as|to|is|\:)\s+[\"']?(?P<target>.+?)[\"']?$",
    r"^(?:call)\s+(?P<trigger>\S+)\s+(?P<target>\S+)$",
    # "set default browser to brave" / "set default game to wuthering waves"
    r"^(?:set\s+default\s+)(?P<trigger>game|browser|music|editor)\s+to\s+[\"']?(?P<target>.+?)[\"']?$"
]

class RealtimeLearner:
    def __init__(self, filepath: Path = LEARNED_MEMORY_FILE):
        self.filepath = filepath
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "category_preferences": {},  # e.g. {"games": "wuthering waves", "browser": "edge"}
            "custom_aliases": {},        # e.g. {"diary": "notepad", "my game": "wuthering waves"}
            "history": []
        }

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def check_teaching_intent(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Detects if user is actively teaching Miku in real time.
        Returns (trigger, target, confirmation_message) or None.
        """
        clean = text.strip()
        for pat in TEACH_PATTERNS:
            m = re.match(pat, clean, re.I)
            if m:
                raw_trigger = m.group("trigger").strip().lower()
                raw_target = m.group("target").strip().lower()
                
                # Normalize triggers like "favorite game" -> "games"
                if "game" in raw_trigger:
                    trigger = "games"
                elif "browser" in raw_trigger:
                    trigger = "browser"
                elif "music" in raw_trigger:
                    trigger = "music"
                elif "editor" in raw_trigger or "ide" in raw_trigger:
                    trigger = "editor"
                else:
                    trigger = raw_trigger

                # Save association
                if trigger in ("games", "browser", "music", "editor", "social"):
                    self.set_category_preference(trigger, raw_target)
                    self.set_custom_alias(trigger, raw_target)
                    msg = f"Got it! Whenever you say '{trigger}', I will open '{raw_target}'."
                else:
                    self.set_custom_alias(trigger, raw_target)
                    msg = f"Learned! I've linked '{trigger}' to '{raw_target}'."

                return trigger, raw_target, msg
        return None

    def set_category_preference(self, category: str, app_name: str):
        cat = category.strip().lower()
        if cat in ("game", "games"):
            cat = "games"
        self.data["category_preferences"][cat] = app_name.strip()
        self.data["history"].append({
            "type": "category_preference",
            "category": cat,
            "target": app_name.strip(),
            "time": time.time()
        })
        self._save()

    def get_category_preference(self, category: str) -> Optional[str]:
        cat = category.strip().lower()
        if cat in ("game", "games"):
            cat = "games"
        return self.data["category_preferences"].get(cat)

    def set_custom_alias(self, alias: str, target: str):
        al = alias.strip().lower()
        self.data["custom_aliases"][al] = target.strip()
        self.data["history"].append({
            "type": "custom_alias",
            "alias": al,
            "target": target.strip(),
            "time": time.time()
        })
        self._save()

    def resolve_alias(self, query: str) -> Optional[str]:
        clean = query.strip().lower()
        return self.data["custom_aliases"].get(clean)

    def get_learned_summary(self) -> str:
        prefs = self.data.get("category_preferences", {})
        aliases = self.data.get("custom_aliases", {})
        lines = []
        if prefs:
            lines.append("Category Preferences:")
            for k, v in prefs.items():
                lines.append(f"  • {k.capitalize()} -> {v}")
        if aliases:
            lines.append("Custom Aliases:")
            for k, v in aliases.items():
                lines.append(f"  • '{k}' -> '{v}'")
        if not lines:
            return "I haven't learned any custom preferences yet. You can teach me anytime (e.g. 'when I say games open wuthering wave')!"
        return "\n".join(lines)

    def reset(self):
        self.data = {
            "category_preferences": {},
            "custom_aliases": {},
            "history": []
        }
        self._save()
