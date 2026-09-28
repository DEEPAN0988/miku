"""
Active Learning Manager: Transparent Learning from Failures and User Corrections.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Audit logging of every failed, ambiguous, or refused utterance with reason and timestamp.
2. Review command ('show what you didn't understand', 'review failures') ranked by frequency.
3. Explicit user promotion: Nothing is learned silently; user confirms promotion of an utterance
   to a designated intent, which incrementally retrains the classical classifier.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

FAILED_LOG_PATH = Path("miku_failed_utterances.json")

class ActiveLearningManager:
    def __init__(self, log_path: Path = FAILED_LOG_PATH):
        self.log_path = log_path
        self.failures: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.failures = json.load(f)
            except Exception:
                self.failures = {}

    def _save(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.failures, f, indent=2)
        except Exception:
            pass

    def log_failure(self, utterance: str, reason: str, confidence: float = 0.0, top_intent: Optional[str] = None):
        """
        Logs an unhandled, rejected, or low-confidence utterance.
        """
        clean = utterance.strip()
        if not clean:
            return

        if clean in self.failures:
            self.failures[clean]["count"] += 1
            self.failures[clean]["last_seen"] = time.time()
            self.failures[clean]["last_reason"] = reason
            self.failures[clean]["last_confidence"] = confidence
        else:
            self.failures[clean] = {
                "count": 1,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "last_reason": reason,
                "last_confidence": confidence,
                "candidate_intent": top_intent
            }
        self._save()

    def get_ranked_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        items = []
        for text, data in self.failures.items():
            item = dict(data)
            item["utterance"] = text
            items.append(item)
        # Sort by count descending, then by last_seen descending
        items.sort(key=lambda x: (-x["count"], -x["last_seen"]))
        return items[:limit]

    def format_review_summary(self, limit: int = 5) -> str:
        ranked = self.get_ranked_failures(limit)
        if not ranked:
            return "No logged comprehension failures. All commands have been understood or handled!"

        lines = [f"I have logged {len(self.failures)} utterances I struggled with (ranked by frequency):"]
        for idx, r in enumerate(ranked, 1):
            cand = f" [Guess: {r['candidate_intent']}]" if r.get("candidate_intent") else ""
            lines.append(f"  {idx}. \"{r['utterance']}\" (Seen {r['count']}x) - Reason: {r['last_reason']}{cand}")
        lines.append("\nTo teach me one, say: 'learn [phrase] as [intent]' (e.g. 'learn surf net as open_app').")
        return "\n".join(lines)

    def promote_to_training(self, utterance: str, target_intent: str, classifier: Any) -> bool:
        """
        Promotes a previously failed utterance into the classical classifier training set
        and retrains the model on disk.
        """
        clean = utterance.strip()
        if not clean:
            return False

        # Add to classifier and retrain
        classifier.add_training_example_and_retrain(clean, target_intent)

        # Remove from failures list now that it is resolved
        if clean in self.failures:
            del self.failures[clean]
            self._save()

        return True
