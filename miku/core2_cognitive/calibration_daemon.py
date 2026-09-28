"""
Day-Zero Calibration Daemon and Incremental Quality Tracker.
Core 2: Cognitive Router (Brain & Memory)
Addresses Bare-Metal Bottleneck #2 (The Cold Start Calibration Void).
"""
from ast import Tuple
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from miku.config import CALIBRATION_DIR

PHONETIC_BALANCED_SCRIPT = [
    "open notepad and write daily thoughts",
    "close the calculator application now",
    "adjust system volume up by ten percent",
    "mute audio output immediately",
    "find the quarterly report document in downloads",
    "click the blue confirmation button on screen",
    "take a screenshot of the main workspace",
    "check the room with the camera",
    "stop all current automation tasks",
    "plan my meetings and focus time today"
]

class CalibrationDaemon:
    def __init__(self, calib_dir: Path = CALIBRATION_DIR, target_phrases: int = 10):
        self.calib_dir = calib_dir
        self.state_file = self.calib_dir / "calibration_state.json"
        self.corrections_log = self.calib_dir / "corrections.jsonl"
        self.target_phrases = target_phrases
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "voice_enrolled_phrases": [],
            "vision_enrolled_classes": {},
            "min_viable_floor": 0.50,
            "created_at": time.time(),
            "updated_at": time.time()
        }

    def _save_state(self):
        self.state["updated_at"] = time.time()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def enroll_voice_phrase(self, phrase: str):
        p = phrase.strip().lower()
        if p not in self.state["voice_enrolled_phrases"]:
            self.state["voice_enrolled_phrases"].append(p)
            self._save_state()

    def enroll_vision_class(self, class_name: str, num_samples: int):
        cur = self.state["vision_enrolled_classes"].get(class_name, 0)
        self.state["vision_enrolled_classes"][class_name] = cur + num_samples
        self._save_state()

    def log_correction(self, user_input: str, wrong_action: str, correct_action: str):
        """
        Logs every post-launch correction as additional labeled calibration data.
        """
        entry = {
            "timestamp": time.time(),
            "input": user_input,
            "wrong_action": wrong_action,
            "correct_action": correct_action
        }
        with open(self.corrections_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_calibration_metrics(self) -> Dict[str, Any]:
        enrolled_count = len(self.state["voice_enrolled_phrases"])
        voice_coverage = min(1.0, enrolled_count / max(self.target_phrases, 1))
        
        vision_samples = sum(self.state["vision_enrolled_classes"].values())
        vision_coverage = min(1.0, vision_samples / 20.0)

        is_usable = voice_coverage >= self.state["min_viable_floor"]

        return {
            "voice_enrolled_count": enrolled_count,
            "voice_target": self.target_phrases,
            "voice_coverage_pct": round(voice_coverage * 100, 1),
            "vision_enrolled_classes": self.state["vision_enrolled_classes"],
            "vision_samples_total": vision_samples,
            "vision_coverage_pct": round(vision_coverage * 100, 1),
            "cleared_min_viable_floor": is_usable,
            "display_string": f"voice model: {enrolled_count}/{self.target_phrases} phrases enrolled, coverage estimate {round(voice_coverage * 100)}%"
        }

    def check_action_permitted(self, action_type: str) -> Tuple[bool, str]:
        """
        Enforces minimum-viable calibration floor before enabling action.
        """
        metrics = self.get_calibration_metrics()
        if not metrics["cleared_min_viable_floor"]:
            return False, f"Calibration floor not met ({metrics['display_string']}). Run onboarding first."
        return True, "OK"
