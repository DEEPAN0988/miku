"""
Sequence Slot Tagger and Entity Validator.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Slot extraction for app names, file paths, URLs, times, volume directions, colors, and queries.
2. Ground-truth validation of extracted slots against actual OS state (app catalog & installed discovery).
3. Safeguards against executing actions on phantom/hallucinated applications or invalid paths.
"""
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from miku.core2_cognitive.app_catalog import APP_CATALOG, predict_app, _ALIAS_INDEX

# Common volume keywords
VOL_UP_TERMS = {"up", "turn up", "crank up", "increase", "raise", "boost", "more", "max"}
VOL_DOWN_TERMS = {"down", "turn down", "decrease", "lower", "reduce", "less", "tone down"}
VOL_MUTE_TERMS = {"mute", "silence", "quiet", "hush"}
VOL_UNMUTE_TERMS = {"unmute", "restore"}

class SlotTagger:
    def __init__(self):
        pass

    def extract_slots(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Extracts structured slot values based on intent and sentence structure.
        """
        slots: Dict[str, Any] = {}
        clean = text.strip()
        lower = clean.lower()

        # 1. URL extraction
        url_match = re.search(r"\b(https?://\S+|www\.\S+|[a-zA-Z0-9_\-]+\.(?:com|org|net|io|edu|gov|co|app|ai|dev)(?:/\S*)?)\b", clean)
        if url_match:
            slots["url"] = url_match.group(1)

        # 2. Time extraction
        time_match = re.search(r"\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}\s*o'?clock)\b", lower)
        if time_match:
            slots["time"] = time_match.group(1).replace("at ", "").strip()

        # 3. Volume direction
        if intent == "SYSTEM_VOLUME":
            for term in VOL_MUTE_TERMS:
                if re.search(r"\b" + re.escape(term) + r"\b", lower):
                    slots["direction"] = "mute"
                    break
            if "direction" not in slots:
                for term in VOL_UNMUTE_TERMS:
                    if re.search(r"\b" + re.escape(term) + r"\b", lower):
                        slots["direction"] = "unmute"
                        break
            if "direction" not in slots:
                for term in VOL_UP_TERMS:
                    if re.search(r"\b" + re.escape(term) + r"\b", lower):
                        slots["direction"] = "up"
                        break
            if "direction" not in slots:
                for term in VOL_DOWN_TERMS:
                    if re.search(r"\b" + re.escape(term) + r"\b", lower):
                        slots["direction"] = "down"
                        break

        # 4. Color & visual target
        if intent == "CLICK_TARGET":
            color_m = re.search(r"\b(red|green|blue|yellow|orange|purple)\b", lower)
            if color_m:
                slots["color"] = color_m.group(1)
            target_m = re.search(r"\b(icon|button|window|box|link|menu)\b", lower)
            if target_m:
                slots["target"] = target_m.group(1)

        # 5. File / Folder operations
        if intent in ("OPEN_FILE", "DELETE_FILE", "SEARCH_FILE", "CREATE_FILE"):
            file_m = re.search(r"(?:file|folder|document|note)\s+([a-zA-Z0-9_\-\.\/\\:]+)", clean)
            if file_m:
                val = file_m.group(1)
                if intent == "CREATE_FILE":
                    slots["filename"] = val
                elif intent == "SEARCH_FILE":
                    slots["query"] = val
                else:
                    slots["path"] = val
            content_m = re.search(r"with\s+content\s+(.+)$", clean)
            if content_m:
                slots["content"] = content_m.group(1).strip()

        # 6. Task creation
        if intent == "ADD_TASK":
            task_m = re.search(r"(?:add\s+task|schedule|enqueue\s+a\s+task)\s+(.+?)(?:\s+(?:at|by)\s+\d{1,2}(?:am|pm)?|$)", clean, re.IGNORECASE)
            if task_m:
                slots["task"] = task_m.group(1).strip()

        # 7. App extraction (for OPEN_APP and CLOSE_APP)
        if intent in ("OPEN_APP", "CLOSE_APP"):
            # Strip command verb prefix
            app_cand = re.sub(r"^(?:open(?:\s+up)?|launch|start|fire\s+up|visit|switch\s+to|bring\s+up|close(?:\s+down)?|shut(?:\s+down)?|exit(?:\s+from)?|quit|terminate|kill)\s+(?:the\s+)?", "", lower).strip()
            # Strip trailing fillers like "app", "application", "browser", "window", "right now", "please"
            app_cand = re.sub(r"\s+(?:application|app|window|right\s+now|please|immediately|asap)$", "", app_cand).strip()

            if app_cand:
                slots["raw_app"] = app_cand
                if intent == "OPEN_APP":
                    slots["app"] = app_cand
                else:
                    slots["target"] = app_cand

        return slots

    def validate_app_slot(self, raw_app: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates if raw_app corresponds to a known installed application or catalog entry.
        Returns:
        (is_valid, canonical_name, app_info)
        """
        clean_app = raw_app.strip().lower()
        if not clean_app or clean_app in ("something", "anything", "whatever", "app", "program"):
            return False, clean_app, None

        # Predict against APP_CATALOG and runtime aliases
        canonical, info = predict_app(clean_app)

        # Check if canonical is in APP_CATALOG or known discovery
        if canonical in APP_CATALOG:
            return True, canonical, info

        # Check discovered apps
        try:
            from miku.core2_cognitive.app_discovery import AppDiscoveryService
            discovery = AppDiscoveryService()
            all_apps = discovery.get_all_apps()
            for name, meta in all_apps.items():
                if clean_app == name.lower() or canonical == name.lower():
                    return True, name, meta
        except Exception:
            pass

        # If it was matched with high certainty via predict_app
        if info and info.get("match_type") in ("exact_canonical", "alias_match", "executable_stem"):
            return True, canonical, info

        # Unknown / non-existent application
        return False, clean_app, None
