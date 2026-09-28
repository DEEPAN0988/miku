"""
Anti-Bot Evasion & Stealth Browser Profile Subsystem.
Core 3: Execution Engine (Hands)
Provides CDP stealth script injection and humanized interaction synthesis
to prevent automated sessions from being flagged as headless bots.
"""
import time
import random
from typing import Dict, Any, List, Tuple
from miku.core3_execution.motion import minimum_jerk_trajectory

STEALTH_JS_INJECTION = r"""
// Overwrite the 'webdriver' property to prevent bot detection
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
});

// Mock plugins array
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5]
});

// Mock languages
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en']
});

// Mock window.chrome object
if (!window.chrome) {
    window.chrome = {
        runtime: {},
        loadTimes: function() {},
        csi: function() {},
        app: {}
    };
}
"""

class AntiBotProfile:
    def __init__(self):
        self.stealth_script = STEALTH_JS_INJECTION

    def get_cdp_stealth_payload(self) -> Dict[str, Any]:
        """
        Returns CDP Page.addScriptToEvaluateOnNewDocument payload.
        """
        return {
            "method": "Page.addScriptToEvaluateOnNewDocument",
            "params": {
                "source": self.stealth_script
            }
        }

    def generate_humanized_keystrokes(self, text: str) -> List[Tuple[str, float]]:
        """
        Generates sequence of (char, delay_seconds) with natural human typing cadence.
        Mimics 60-90 WPM with natural inter-key jitter.
        """
        events = []
        for ch in text:
            # Baseline delay between 50ms and 120ms
            delay = random.uniform(0.05, 0.12)
            # Occasional pause for punctuation or space
            if ch in " .,!?:;":
                delay += random.uniform(0.08, 0.18)
            events.append((ch, round(delay, 3)))
        return events

    def generate_humanized_mouse_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        steps: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Computes minimum-jerk trajectory with subtle micro-jitter to prevent linear bot triggers.
        """
        base_path = minimum_jerk_trajectory(start, end, steps=steps)
        human_path = []
        for i, (x, y) in enumerate(base_path):
            if 0 < i < len(base_path) - 1:
                # Add micro-variation (-1 to +1 pixel)
                jx = x + random.choice([-1, 0, 1])
                jy = y + random.choice([-1, 0, 1])
                human_path.append((jx, jy))
            else:
                human_path.append((x, y))
        return human_path
