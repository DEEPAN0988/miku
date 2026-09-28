"""
Deterministic Grammar and Pattern Matching Intent Parser.
Core 2: Cognitive Router (Brain & Memory)
Provides bounded, predictable intent routing with exact entity extraction.
"""
import re
from typing import Dict, Any, Optional, Tuple, List

class DeterministicGrammarRouter:
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self._init_grammars()
        self.total_queries = 0
        self.matched_queries = 0

    def _init_grammars(self):
        # Format: (pattern, intent_name, action_type, default_params)
        grammars = [
            # OS & Application Control
            (r"^(?:open|launch|start)\s+(?P<app>notepad|calculator|cmd|chrome|powershell|paint|explorer)$", "OPEN_APP", "open_app", {}),
            (r"^(?:close|exit|terminate)\s+(?P<target>window|notepad|calculator|cmd|chrome|app)$", "CLOSE_APP", "close_app", {}),
            (r"^(?:maximize|minimize|restore)\s+(?:window|app)$", "WINDOW_STATE", "window_state", {}),
            
            # System Volume & Media
            (r"^(?:volume|sound)\s+(?P<direction>up|down|mute|unmute)$", "SYSTEM_VOLUME", "volume", {}),
            
            # File Operations (Explicitly confirmation-gated in PRD)
            (r"^(?:delete|remove)\s+(?:file|folder)\s+(?P<path>.+)$", "DELETE_FILE", "delete_file", {"needs_confirmation": True}),
            (r"^(?:find|search)\s+(?:file|folder)\s+(?P<query>.+)$", "SEARCH_FILE", "search_file", {}),
            (r"^(?:create|make)\s+(?:file|note)\s+(?P<filename>[a-zA-Z0-9_\-\.]+)(?:\s+with\s+(?P<content>.+))?$", "CREATE_FILE", "create_file", {}),

            # Task & Day Planning (Logic / rule engine based)
            (r"^(?:plan\s+my\s+day|show\s+my\s+schedule|what\s+do\s+i\s+have\s+today)$", "PLAN_DAY", "plan_day", {}),
            (r"^(?:add\s+task|schedule)\s+(?P<task>.+?)(?:\s+at\s+(?P<time>.+))?$", "ADD_TASK", "add_task", {}),

            # Screen & Visual Perceptions
            (r"^(?:click|press)\s+(?:the\s+)?(?P<color>red|green|blue|yellow)?\s*(?P<target>icon|button|window|box)$", "CLICK_TARGET", "click_target", {}),
            (r"^(?:take|capture)\s+(?:a\s+)?(?:screenshot|screen)$", "SCREENSHOT", "screenshot", {}),
            (r"^(?:look\s+at|check)\s+(?:the\s+)?(?:camera|webcam|room)$", "INSPECT_CAMERA", "inspect_camera", {}),

            # Status & Automation Control
            (r"^(?:status\s+report|system\s+status|health\s+check)$", "SYSTEM_STATUS", "status_report", {}),
            (r"^(?:stop|abort|cancel|halt)(?:\s+automation)?$", "ABORT_AUTOMATION", "abort_automation", {}),

            # Browser Session Automation (CDP)
            (r"^(?:navigate\s+to|browse\s+to|open\s+url)\s+(?P<url>https?://\S+|\S+\.\S+)$", "BROWSER_NAVIGATE", "browser_navigate", {}),
            (r"^(?:extract\s+page\s+text|read\s+page|scrape\s+page)$", "BROWSER_EXTRACT", "browser_extract", {})
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
        Parses text deterministically.
        Returns (intent, action_type, params, linguistic_score)
        If unknown, returns (None, None, {}, 0.0) -> fails predictably.
        """
        clean_text = text.strip().lower()
        self.total_queries += 1

        for rule in self.rules:
            match = rule["regex"].match(clean_text)
            if match:
                self.matched_queries += 1
                params = rule["extra"].copy()
                params.update({k: v for k, v in match.groupdict().items() if v is not None})
                return rule["intent"], rule["action"], params, 1.0

        return None, None, {}, 0.0

    @property
    def coverage_ratio(self) -> float:
        if self.total_queries == 0:
            return 1.0
        return self.matched_queries / self.total_queries
