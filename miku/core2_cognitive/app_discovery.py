"""
MIKU: Dynamic System Application & Game Discovery.
Core 2: Cognitive Router (Brain & Memory)
Scans Start Menu, App Paths, and common locations to identify installed software and games.
"""
import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

CATEGORY_KEYWORDS = {
    "games": [
        "wuthering", "game", "steam", "genshin", "epic games", "riot",
        "valorant", "minecraft", "roblox", "battle.net", "ea app", "ubisoft",
        "honkai", "gta", "cyberpunk", "witcher", "league of legends"
    ],
    "browser": [
        "chrome", "edge", "brave", "firefox", "opera", "vivaldi", "browser"
    ],
    "developer": [
        "code", "cursor", "visual studio", "python", "git", "terminal", "pycharm",
        "intellij", "sublime", "android studio"
    ],
    "social": [
        "discord", "whatsapp", "telegram", "slack", "teams", "skype", "zoom", "instagram"
    ],
    "music": [
        "spotify", "itunes", "music", "vlc", "foobar", "winamp"
    ],
    "productivity": [
        "excel", "word", "powerpoint", "onenote", "notepad", "calculator", "paint"
    ]
}

class AppDiscovery:
    def __init__(self):
        self.cached_apps: Dict[str, Dict[str, Any]] = {}
        self.category_index: Dict[str, List[str]] = {
            "games": [],
            "browser": [],
            "developer": [],
            "social": [],
            "music": [],
            "productivity": []
        }
        self.scan()

    def scan(self):
        """
        Scans Windows Start Menu shortcuts and program directories.
        """
        dirs = [
            os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
            os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs")
        ]

        found_shortcuts = []
        for d in dirs:
            if os.path.exists(d):
                found_shortcuts.extend(glob.glob(os.path.join(d, "**", "*.lnk"), recursive=True))

        for lnk in found_shortcuts:
            base_name = os.path.splitext(os.path.basename(lnk))[0].strip()
            lower_name = base_name.lower()
            if "uninstall" in lower_name or "remove" in lower_name or "setup" in lower_name:
                continue

            entry = {
                "name": base_name,
                "path": lnk,
                "type": "shortcut"
            }
            self.cached_apps[lower_name] = entry

            # Categorize
            for cat, kws in CATEGORY_KEYWORDS.items():
                if any(kw in lower_name for kw in kws):
                    if base_name not in self.category_index[cat]:
                        self.category_index[cat].append(base_name)

        # Ensure known installed games are present
        if os.path.exists(r"C:\Program Files\Wuthering Waves") or os.path.exists(r"C:\Program Files\Wuthering Waves\launcher.exe"):
            if "Wuthering Waves" not in self.category_index["games"]:
                self.category_index["games"].insert(0, "Wuthering Waves")

    def get_apps_in_category(self, category: str) -> List[str]:
        cat = category.strip().lower()
        if cat in ("game", "games"):
            cat = "games"
        return self.category_index.get(cat, [])

    def find_app(self, query: str) -> Optional[Dict[str, Any]]:
        q = query.strip().lower()
        if q in self.cached_apps:
            return self.cached_apps[q]
        for name, entry in self.cached_apps.items():
            if q in name or name in q:
                return entry
        import shutil
        which_path = shutil.which(query)
        if which_path:
            return {"name": query, "path": which_path, "type": "executable"}
        return None
