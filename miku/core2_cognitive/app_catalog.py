"""
MIKU: Predictive Application & Service Catalog.
Core 2: Cognitive Router (Brain & Memory)
Provides predictive alias resolution, fuzzy typo-matching, and multi-modal application dispatch.
"""
import re
import difflib
from typing import Dict, Any, Tuple, Optional, List

# Conversational prefix and suffix cleaners
CONVERSATIONAL_PREFIX_RE = re.compile(
    r"^(?:(?:hey\s+)?miku[,:\s]+|please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|would\s+you\s+(?:please\s+)?|will\s+you\s+|i\s+want\s+(?:you\s+)?to\s+|kindly\s+|go\s+ahead\s+and\s+|help\s+me\s+)+",
    re.I
)
CONVERSATIONAL_SUFFIX_RE = re.compile(
    r"(?:\s+please|\s+now|\s+for\s+me|\s+thanks|\s+thank\s+you|\s+right\s+now)+$",
    re.I
)

APP_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Social & Web Platforms ---
    "instagram": {
        "display_name": "Instagram",
        "aliases": ["insta", "ig", "instagram", "instgram", "instagramm", "insta gram"],
        "protocol": "instagram:",
        "url": "https://www.instagram.com",
        "category": "social"
    },
    "youtube": {
        "display_name": "YouTube",
        "aliases": ["yt", "youtube", "youtub", "yotube", "ytube", "you tube"],
        "url": "https://www.youtube.com",
        "category": "video"
    },
    "whatsapp": {
        "display_name": "WhatsApp",
        "aliases": ["wp", "wa", "whatsapp", "whats app", "whatsap", "watsapp"],
        "protocol": "whatsapp:",
        "executable": "WhatsApp.exe",
        "url": "https://web.whatsapp.com",
        "category": "messaging"
    },
    "twitter": {
        "display_name": "Twitter / X",
        "aliases": ["tw", "twt", "twitter", "x", "twtr", "x.com"],
        "url": "https://x.com",
        "category": "social"
    },
    "facebook": {
        "display_name": "Facebook",
        "aliases": ["fb", "facebook", "face book"],
        "url": "https://www.facebook.com",
        "category": "social"
    },
    "reddit": {
        "display_name": "Reddit",
        "aliases": ["reddit", "rdt", "reddt"],
        "url": "https://www.reddit.com",
        "category": "social"
    },
    "linkedin": {
        "display_name": "LinkedIn",
        "aliases": ["linkedin", "linked in"],
        "url": "https://www.linkedin.com",
        "category": "professional"
    },
    "telegram": {
        "display_name": "Telegram",
        "aliases": ["tg", "telegram", "tgram"],
        "protocol": "tg:",
        "url": "https://web.telegram.org",
        "category": "messaging"
    },
    "discord": {
        "display_name": "Discord",
        "aliases": ["discord", "dc", "disc"],
        "protocol": "discord:",
        "executable": "Discord.exe",
        "url": "https://discord.com/app",
        "category": "messaging"
    },
    "spotify": {
        "display_name": "Spotify",
        "aliases": ["spotify", "spotfy", "music"],
        "protocol": "spotify:",
        "executable": "Spotify.exe",
        "url": "https://open.spotify.com",
        "category": "music"
    },
    "netflix": {
        "display_name": "Netflix",
        "aliases": ["netflix", "nflx", "net flix"],
        "url": "https://www.netflix.com",
        "category": "streaming"
    },
    "prime": {
        "display_name": "Prime Video",
        "aliases": ["prime", "prime video", "amazon prime"],
        "url": "https://www.primevideo.com",
        "category": "streaming"
    },
    "chatgpt": {
        "display_name": "ChatGPT",
        "aliases": ["gpt", "chatgpt", "chat gpt", "openai", "ai chat"],
        "url": "https://chatgpt.com",
        "category": "ai"
    },
    "github": {
        "display_name": "GitHub",
        "aliases": ["gh", "git", "github", "git hub"],
        "url": "https://github.com",
        "category": "developer"
    },
    "google": {
        "display_name": "Google",
        "aliases": ["google", "googl", "search engine"],
        "url": "https://www.google.com",
        "category": "search"
    },
    "gmail": {
        "display_name": "Gmail",
        "aliases": ["gmail", "mail", "email", "google mail"],
        "url": "https://mail.google.com",
        "category": "productivity"
    },
    "drive": {
        "display_name": "Google Drive",
        "aliases": ["drive", "google drive", "gdrive"],
        "url": "https://drive.google.com",
        "category": "storage"
    },
    "docs": {
        "display_name": "Google Docs",
        "aliases": ["docs", "google docs", "gdocs"],
        "url": "https://docs.google.com",
        "category": "productivity"
    },
    "sheets": {
        "display_name": "Google Sheets",
        "aliases": ["sheets", "google sheets", "excel online"],
        "url": "https://sheets.google.com",
        "category": "productivity"
    },
    "amazon": {
        "display_name": "Amazon",
        "aliases": ["amazon", "amzn", "shopping"],
        "url": "https://www.amazon.com",
        "category": "shopping"
    },

    # --- Windows Desktop OS Applications ---
    "notepad": {
        "display_name": "Notepad",
        "aliases": ["notepad", "notes", "note", "notepd", "text editor"],
        "executable": "notepad.exe",
        "close_process": ["notepad.exe"],
        "category": "desktop"
    },
    "calculator": {
        "display_name": "Calculator",
        "aliases": ["calculator", "calc", "calcultor", "calcy", "math"],
        "executable": "calc.exe",
        "protocol": "calc:",
        "close_process": ["CalculatorApp.exe", "calc.exe"],
        "category": "desktop"
    },
    "edge": {
        "display_name": "Microsoft Edge",
        "aliases": ["edge", "browser", "web browser", "internet", "microsoft edge", "msedge"],
        "protocol": "microsoft-edge:",
        "executable": "msedge.exe",
        "close_process": ["msedge.exe"],
        "category": "browser"
    },
    "chrome": {
        "display_name": "Google Chrome",
        "aliases": ["chrome", "google chrome", "crhome"],
        "executable": "chrome.exe",
        "close_process": ["chrome.exe"],
        "category": "browser"
    },
    "cmd": {
        "display_name": "Command Prompt",
        "aliases": ["cmd", "command prompt", "terminal", "prompt"],
        "executable": "cmd.exe",
        "close_process": ["cmd.exe"],
        "category": "desktop"
    },
    "powershell": {
        "display_name": "PowerShell",
        "aliases": ["powershell", "ps", "pwsh"],
        "executable": "powershell.exe",
        "close_process": ["powershell.exe", "pwsh.exe"],
        "category": "desktop"
    },
    "terminal": {
        "display_name": "Windows Terminal",
        "aliases": ["wt", "windows terminal"],
        "executable": "wt.exe",
        "close_process": ["WindowsTerminal.exe", "wt.exe"],
        "category": "desktop"
    },
    "vscode": {
        "display_name": "Visual Studio Code",
        "aliases": ["vscode", "code", "vs code", "visual studio code", "vscod"],
        "executable": "code",
        "close_process": ["Code.exe"],
        "category": "developer"
    },
    "cursor": {
        "display_name": "Cursor",
        "aliases": ["cursor", "cursor editor", "cursor ide"],
        "executable": "cursor",
        "close_process": ["cursor.exe"],
        "category": "developer"
    },
    "paint": {
        "display_name": "Paint",
        "aliases": ["paint", "mspaint", "draw", "drawing"],
        "executable": "mspaint.exe",
        "close_process": ["mspaint.exe"],
        "category": "desktop"
    },
    "explorer": {
        "display_name": "File Explorer",
        "aliases": ["explorer", "file explorer", "files", "my computer", "this pc"],
        "executable": "explorer.exe",
        "close_process": ["explorer.exe"],
        "category": "desktop"
    },
    "task manager": {
        "display_name": "Task Manager",
        "aliases": ["task manager", "taskmgr", "tasks"],
        "executable": "taskmgr.exe",
        "close_process": ["taskmgr.exe"],
        "category": "desktop"
    },
    "settings": {
        "display_name": "Windows Settings",
        "aliases": ["settings", "windows settings", "preferences", "config", "control panel"],
        "protocol": "ms-settings:",
        "close_process": ["SystemSettings.exe"],
        "category": "desktop"
    },
    # --- Games & Gaming Launchers ---
    "wuthering waves": {
        "display_name": "Wuthering Waves",
        "aliases": [
            "wuthering waves",
            "wuthering wave",
            "wuwa",
            "wuthering",
            "kuro games",
            "wuthering waves game",
            "wuthering waves launcher"
        ],
        "paths": [
            r"C:\Program Files\Wuthering Waves\launcher.exe",
            r"C:\Program Files\Wuthering Waves\Wuthering Waves Game\Wuthering Waves.exe",
            r"C:\Program Files\Wuthering Waves\Wuthering Waves Game\Client\Binaries\Win64\Client-Win64-Shipping.exe",
            r"C:\Wuthering Waves\launcher.exe",
            r"D:\Wuthering Waves\launcher.exe",
            r"E:\Wuthering Waves\launcher.exe",
        ],
        "executable": "Wuthering Waves.exe",
        "close_process": [
            "Client-Win64-Shipping.exe",
            "Wuthering Waves.exe",
            "launcher.exe",
            "launcher_main.exe"
        ],
        "url": "https://wutheringwaves.kurogames.com",
        "category": "game"
    },
    "steam": {
        "display_name": "Steam",
        "aliases": ["steam", "valve"],
        "protocol": "steam:",
        "executable": "steam.exe",
        "close_process": ["steam.exe"],
        "url": "https://store.steampowered.com",
        "category": "gaming"
    },
    "genshin impact": {
        "display_name": "Genshin Impact",
        "aliases": ["genshin", "genshin impact", "gi", "hoyoverse"],
        "executable": "GenshinImpact.exe",
        "close_process": ["GenshinImpact.exe"],
        "url": "https://genshin.hoyoverse.com",
        "category": "game"
    }
}

# Invert alias index for O(1) exact matching
_ALIAS_INDEX: Dict[str, str] = {}
for _canonical, _data in APP_CATALOG.items():
    _ALIAS_INDEX[_canonical] = _canonical
    for _al in _data.get("aliases", []):
        _ALIAS_INDEX[_al.lower()] = _canonical

def normalize_command_text(text: str) -> str:
    """
    Cleans leading/trailing conversational fillers from raw user input.
    E.g. 'hey miku please open insta now' -> 'open insta'
    """
    t = text.strip()
    t = CONVERSATIONAL_PREFIX_RE.sub("", t).strip()
    t = CONVERSATIONAL_SUFFIX_RE.sub("", t).strip()
    return t

def predict_app(raw_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Predicts and resolves user app name to canonical app name and metadata.
    Handles exact aliases, word overlap, prefix completion, and fuzzy typo matching.
    Guarantees no false positive cross-substring collisions (e.g. 'wuthering wave' never matches 'wa').
    """
    clean = raw_name.strip().lower()

    # Strip prefixes like 'the '
    if clean.startswith("the "):
        clean = clean[4:].strip()

    # Strip suffixes like ' application', ' app', ' game', ' launcher', ' website', ' site', ' page'
    for suffix in (" application", " app", " game", " launcher", " website", " site", " page"):
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)].strip()
            break

    # 1. Exact match in alias index
    if clean in _ALIAS_INDEX:
        canonical = _ALIAS_INDEX[clean]
        return canonical, APP_CATALOG[canonical]

    # Normalize spaces (e.g. 'whats app' -> 'whatsapp')
    clean_nospace = clean.replace(" ", "")
    if clean_nospace in _ALIAS_INDEX:
        canonical = _ALIAS_INDEX[clean_nospace]
        return canonical, APP_CATALOG[canonical]

    # 2. Multi-word phrase overlap (e.g. 'wuthering wave' -> 'wuthering waves')
    clean_words = set(clean.split())
    if len(clean_words) > 1:
        best_canonical = None
        best_score = 0
        for canonical, data in APP_CATALOG.items():
            for al in data.get("aliases", []):
                al_words = set(al.split())
                if len(al_words) > 1:
                    overlap = len(clean_words.intersection(al_words))
                    if overlap >= len(clean_words) - 1 and overlap > best_score:
                        best_score = overlap
                        best_canonical = canonical
        if best_canonical:
            return best_canonical, APP_CATALOG[best_canonical]

    # 3. Clean is a prefix of an alias (e.g. 'insta' -> 'instagram', 'calcu' -> 'calculator')
    # Guard: clean must be >= 4 chars, and alias must START with clean
    if len(clean) >= 4:
        for al, canonical in _ALIAS_INDEX.items():
            if al.startswith(clean):
                return canonical, APP_CATALOG[canonical]

    # 4. Fuzzy typo matching (only for words of comparable length, cutoff=0.72)
    candidates = [k for k in _ALIAS_INDEX.keys() if abs(len(clean) - len(k)) <= 3]
    if candidates:
        close_matches = difflib.get_close_matches(clean, candidates, n=1, cutoff=0.72)
        if close_matches:
            canonical = _ALIAS_INDEX[close_matches[0]]
            return canonical, APP_CATALOG[canonical]

    # 5. Unknown target: return as generic app entry
    return clean, {
        "display_name": clean.capitalize(),
        "aliases": [clean],
        "executable": clean,
        "category": "unknown"
    }
