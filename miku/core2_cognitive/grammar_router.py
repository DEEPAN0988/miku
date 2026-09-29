"""
Deterministic Grammar and Pattern Matching Intent Parser.
Core 2: Cognitive Router (Brain & Memory)
Provides bounded, predictable intent routing with exact entity extraction.
"""
import re
from typing import Dict, Any, Optional, Tuple, List

from miku.core2_cognitive.app_catalog import predict_app, normalize_command_text, _ALIAS_INDEX

class DeterministicGrammarRouter:
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self._init_grammars()
        self.total_queries = 0
        self.matched_queries = 0

    def _init_grammars(self):
        # Format: (pattern, intent_name, action_type, default_params)
        grammars = [
            # Browser Session Automation (CDP) & URLs (Matched before general open app)
            (r"^(?:open\s+url|navigate\s+to|browse\s+to|browse|navigate)\s+(?P<url>https?://\S+|\S+\.\S+)$", "BROWSER_NAVIGATE", "browser_navigate", {}),
            (r"^(?:open)\s+(?P<url>(?:https?://\S+|www\.\S+|[a-zA-Z0-9_\-]+\.(?:com|org|net|io|edu|gov|co|app|ai|me|dev)(?:/\S*)?))$", "BROWSER_NAVIGATE", "browser_navigate", {}),
            (r"^(?:extract\s+page\s+text|read\s+page|scrape\s+page)$", "BROWSER_EXTRACT", "browser_extract", {}),
            # CAPTCHA / anti-bot evasion is out of scope — route to refusal, never to an action.
            (r"^(?:solve|bypass|crack|beat|defeat|get\s+past)\s+(?:the\s+)?captcha$", "REFUSE_CAPTCHA_REQUEST", "refuse_oos", {}),
            (r"^(?:enable\s+)?(?:stealth|anti[- ]bot)(?:\s+mode)?$", "REFUSE_CAPTCHA_REQUEST", "refuse_oos", {}),
            (r".*(?:captcha|anti[- ]?bot|bot check).*", "REFUSE_CAPTCHA_REQUEST", "refuse_oos", {}),

            # File & Folder Operations
            (r"^(?:open|show)\s+(?:file|folder)\s+(?P<path>.+)$", "OPEN_FILE", "open_file", {}),
            (r"^(?:delete|remove)\s+(?:file|folder)\s+(?P<path>.+)$", "DELETE_FILE", "delete_file", {"needs_confirmation": True}),
            (r"^(?:find|search)\s+(?:file|folder)\s+(?P<query>.+)$", "SEARCH_FILE", "search_file", {}),
            (r"^(?:create|make)\s+(?:file|note)\s+(?P<filename>[a-zA-Z0-9_\-\.]+)(?:\s+with\s+(?P<content>.+))?$", "CREATE_FILE", "create_file", {}),

            # Window State
            (r"^(?:maximize|minimize|restore)\s+(?:the\s+)?(?:window|app)$", "WINDOW_STATE", "window_state", {}),

            (r"^(?:open(?:\s+up)?|launch|start|fire\s+up|go\s+to|visit|take\s+me\s+to|bring\s+up|switch\s+to|restore)\s+(?:the\s+)?(?P<app>[a-zA-Z0-9_\-\. ]+?)(?:\s+application|\s+app|\s+website|\s+site|\s+page)?$", "OPEN_APP", "open_app", {}),
            # Close / terminate
            (r"^(?:close(?:\s+down)?|shut(?:\s+down)?|exit(?:\s+from)?|terminate|kill|quit)\s+(?:the\s+)?(?P<target>[a-zA-Z0-9_\-\. ]+?)(?:\s+application|\s+app)?(?:\s+(?:right\s+away|now|please|immediately))?$", "CLOSE_APP", "close_app", {}),
            
            # System Volume & Media (Natural phrasing: volume up, turn up volume, increase sound, mute, unmute)
            (r"^(?:volume|sound)\s+(?P<direction>up|down|mute|unmute)$", "SYSTEM_VOLUME", "volume", {}),
            (r"^(?:turn\s+up|crank\s+up|increase|raise|boost)\s+(?:the\s+)?(?:volume|sound|audio)$", "SYSTEM_VOLUME", "volume", {"direction": "up"}),
            (r"^(?:turn\s+down|decrease|lower|reduce)\s+(?:the\s+)?(?:volume|sound|audio)$", "SYSTEM_VOLUME", "volume", {"direction": "down"}),
            (r"^(?:mute|silence)\s+(?:the\s+)?(?:volume|sound|audio)?$", "SYSTEM_VOLUME", "volume", {"direction": "mute"}),
            (r"^(?:unmute)\s+(?:the\s+)?(?:volume|sound|audio)?$", "SYSTEM_VOLUME", "volume", {"direction": "unmute"}),
            
            # Task & Day Planning (Logic / rule engine based)
            (r"^(?:plan\s+my\s+day|show\s+my\s+schedule|what\s+do\s+i\s+have\s+today)$", "PLAN_DAY", "plan_day", {}),
            (r"^(?:add\s+task|schedule)\s+(?P<task>.+?)(?:\s+at\s+(?P<time>.+))?$", "ADD_TASK", "add_task", {}),

            # Screen & Visual Perceptions
            (r"^(?:click|press)\s+(?:the\s+)?(?P<color>red|green|blue|yellow)?\s*(?P<target>icon|button|window|box)$", "CLICK_TARGET", "click_target", {}),
            (r"^(?:take|capture|snap)\s+(?:a\s+)?(?:screenshot|screen|snapshot)$", "SCREENSHOT", "screenshot", {}),
            (r"^screenshot$", "SCREENSHOT", "screenshot", {}),
            (r"^(?:look\s+at|check)\s+(?:the\s+)?(?:camera|webcam|room)$", "INSPECT_CAMERA", "inspect_camera", {}),

            # Status & Automation Control
            (r"^(?:status\s+report|system\s+status|health\s+check)$", "SYSTEM_STATUS", "status_report", {}),
            (r"^(?:stop|abort|cancel|halt)(?:\s+automation)?$", "ABORT_AUTOMATION", "abort_automation", {})
        ]

        for pat, intent, action, extra in grammars:
            self.rules.append({
                "regex": re.compile(pat, re.IGNORECASE),
                "intent": intent,
                "action": action,
                "extra": extra
            })

    def parse(self, text: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any], float]:
        """
        Parses text deterministically with conversational normalization and predictive entity resolution.
        Returns (intent, action_type, params, linguistic_score)
        If unknown, returns (None, None, {}, 0.0) -> fails predictably.
        """
        clean_text = normalize_command_text(text.strip().lower())
        self.total_queries += 1

        for rule in self.rules:
            match = rule["regex"].match(clean_text)
            if match:
                self.matched_queries += 1
                params = rule["extra"].copy()
                params.update({k: v for k, v in match.groupdict().items() if v is not None})

                # Predictive entity resolution for applications
                if "app" in params:
                    raw_app = params["app"]
                    canonical, info = predict_app(raw_app)
                    params["original_app"] = raw_app
                    params["app"] = canonical
                    params["app_info"] = info

                if "target" in params and rule["action"] == "close_app":
                    raw_target = params["target"]
                    if raw_target not in ("window", "active window", "current window", "app"):
                        canonical, info = predict_app(raw_target)
                        params["original_target"] = raw_target
                        params["target"] = canonical
                        params["app_info"] = info

                return rule["intent"], rule["action"], params, 1.0

        # Standalone app name prediction (e.g. user just typed 'insta' or 'instagram' or 'calculator')
        if clean_text in _ALIAS_INDEX:
            canonical = _ALIAS_INDEX[clean_text]
            info = predict_app(clean_text)[1]
            self.matched_queries += 1
            return "OPEN_APP", "open_app", {"app": canonical, "original_app": clean_text, "app_info": info}, 1.0

        return None, None, {}, 0.0

    @property
    def coverage_ratio(self) -> float:
        if self.total_queries == 0:
            return 1.0
        return self.matched_queries / self.total_queries
