"""
Classical Intent Classifier (Trained Locally, Zero-Cloud, Zero-Pretrained).
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Feature extraction using combined word (1-3) and character (3-5) n-gram TF-IDF.
2. Calibrated regularized Logistic Regression classifier with class balancing.
3. Top-3 candidates with calibrated probability scores.
4. Confidence floor check: if confidence < floor, flags for clarification offering top candidates.
5. Loads both hand-written and templated datasets while maintaining them separately.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

MODEL_FILE = Path("miku_intent_model.pkl")
# Calibrated confidence floor
CONFIDENCE_FLOOR = 0.45

class ClassicalIntentClassifier:
    def __init__(self, model_path: Path = MODEL_FILE, confidence_floor: float = CONFIDENCE_FLOOR):
        self.model_path = model_path
        self.confidence_floor = confidence_floor
        self.vectorizer: Optional[FeatureUnion] = None
        self.clf: Optional[LogisticRegression] = None
        self.classes_: List[str] = []
        self.training_examples: List[Dict[str, str]] = []

        if self.model_path.exists():
            self._load()
        else:
            self._train_default()

    def _train_default(self):
        data_dir = Path(__file__).parents[2] / "eval" / "data"
        train_path = data_dir / "train.jsonl"
        templated_path = data_dir / "train_templated.jsonl"
        
        paths = []
        if train_path.exists():
            paths.append(train_path)
        if templated_path.exists():
            paths.append(templated_path)
        if paths:
            self.train_from_jsonls(paths)

    def train_from_jsonls(self, jsonl_paths: List[Path]):
        texts = []
        labels = []
        self.training_examples = []
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    intent = record.get("intent", record.get("expected_intent", "")).strip()
                    if text and intent:
                        texts.append(text)
                        labels.append(intent)
                        self.training_examples.append({"text": text, "intent": intent})

        self.fit(texts, labels)
        self._save()

    def train_from_jsonl(self, jsonl_path: Path):
        self.train_from_jsonls([jsonl_path])

    def fit(self, texts: List[str], labels: List[str]):
        """
        Fits TF-IDF (word n-grams 1-3 and char_wb n-grams 3-5) with regularized Logistic Regression.
        """
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer="word"
        )
        char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
            analyzer="char_wb"
        )
        self.vectorizer = FeatureUnion([
            ("word", word_vec),
            ("char", char_vec)
        ])

        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(
            C=3.0,
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text and returns:
        - top_intent
        - confidence
        - top3_candidates
        - is_confident (confidence >= confidence_floor)
        - clarification_prompt: suggested prompt offering top candidates if ambiguous
        """
        if not self.vectorizer or not self.clf:
            return {
                "top_intent": "UNKNOWN",
                "confidence": 0.0,
                "top3": [],
                "is_confident": False,
                "clarification_prompt": "Could you clarify what you would like to do?"
            }

        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        top3 = []
        for idx in top_indices:
            cls_name = str(self.classes_[idx])
            prob = float(probs[idx])
            top3.append((cls_name, prob))

        best_intent, best_prob = top3[0]
        is_confident = (best_prob >= self.confidence_floor)

        # Build candidate clarification prompt
        candidate_readable = []
        readable_map = {
            "OPEN_APP": "open an app",
            "CLOSE_APP": "close an app",
            "SYSTEM_VOLUME": "adjust volume",
            "BROWSER_NAVIGATE": "navigate to a website",
            "BROWSER_EXTRACT": "extract webpage text",
            "OPEN_FILE": "open a file",
            "SEARCH_FILE": "search for a file",
            "CREATE_FILE": "create a file",
            "PLAN_DAY": "check your schedule",
            "ADD_TASK": "add a calendar task",
            "SCREENSHOT": "take a screenshot",
            "CLICK_TARGET": "click a screen target",
            "INSPECT_CAMERA": "check the camera",
            "SYSTEM_STATUS": "view system status",
            "ABORT_AUTOMATION": "halt automation",
            "WINDOW_STATE": "change window state",
            "OUT_OF_SCOPE": "chat conversation"
        }
        for cls_name, _ in top3[:3]:
            candidate_readable.append(readable_map.get(cls_name, cls_name.lower().replace("_", " ")))

        clarification_prompt = f"Did you mean to {candidate_readable[0]} or {candidate_readable[1]}?"

        return {
            "top_intent": best_intent,
            "confidence": best_prob,
            "top3": top3,
            "is_confident": is_confident,
            "clarification_prompt": clarification_prompt
        }

    def add_training_example_and_retrain(self, text: str, intent: str):
        self.training_examples.append({"text": text.strip(), "intent": intent.strip()})
        texts = [ex["text"] for ex in self.training_examples]
        labels = [ex["intent"] for ex in self.training_examples]
        self.fit(texts, labels)
        self._save()

    def _save(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "clf": self.clf,
                    "classes": self.classes_,
                    "training_examples": self.training_examples
                }, f)
        except Exception:
            pass

    def _load(self):
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.clf = data["clf"]
                self.classes_ = data["classes"]
                self.training_examples = data.get("training_examples", [])
        except Exception:
            self._train_default()
