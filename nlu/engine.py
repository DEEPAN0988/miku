"""
Unified NLU Engine for Miku.
Combines deterministic rule matching with a trainable statistical classifier.
"""

from typing import Dict, Any, Optional
from .rules import RuleMatcher
from .classifier import IntentClassifier


class NLUEngine:
    def __init__(self, classifier_path: Optional[str] = None):
        self.classifier = IntentClassifier(classifier_path) if classifier_path else IntentClassifier()

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse raw user utterance into structured intent and entities.
        Rules take precedence for deterministic structures;
        Classifier handles fuzzier/general utterances.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "raw_text": text,
                "intent": "unknown",
                "entities": {},
                "confidence": 0.0,
                "source": "none"
            }

        # 1. Try rule matcher
        rule_res = RuleMatcher.match(clean_text)
        if rule_res:
            return {
                "raw_text": clean_text,
                "intent": rule_res["intent"],
                "entities": rule_res["entities"],
                "confidence": rule_res["confidence"],
                "source": "rule"
            }

        # 2. Fall back to trained local statistical classifier
        intent, conf = self.classifier.predict(clean_text)
        entities = {}
        
        # Simple heuristic entity extraction for fallback
        lower = clean_text.lower()
        if intent == "launch_app":
            for token in ["open", "launch", "start"]:
                if token in lower:
                    app = lower.split(token, 1)[1].strip()
                    entities["app"] = app
                    break

        return {
            "raw_text": clean_text,
            "intent": intent,
            "entities": entities,
            "confidence": conf,
            "source": "classifier"
        }
