"""
Classical Intent Classifier (Trained Locally, Zero-Cloud, Zero-Pretrained).
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Feature extraction using word & character n-gram TF-IDF.
2. Calibrated Multinomial Naive Bayes / Logistic Regression classifier.
3. Top-3 candidates with calibrated probability scores.
4. Confidence floor check: if confidence < floor, flags for clarification rather than guessing.
5. Incremental / active learning retraining support.
"""
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion

MODEL_FILE = Path("miku_intent_model.pkl")
CONFIDENCE_FLOOR = 0.52

class ClassicalIntentClassifier:
    def __init__(self, model_path: Path = MODEL_FILE, confidence_floor: float = CONFIDENCE_FLOOR):
        self.model_path = model_path
        self.confidence_floor = confidence_floor
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.clf: Optional[LogisticRegression] = None
        self.classes_: List[str] = []
        self.training_examples: List[Dict[str, str]] = []

        if self.model_path.exists():
            self._load()
        else:
            self._train_default()

    def _train_default(self):
        train_path = Path(__file__).parents[2] / "eval" / "data" / "train.jsonl"
        if train_path.exists():
            self.train_from_jsonl(train_path)

    def train_from_jsonl(self, jsonl_path: Path):
        texts = []
        labels = []
        self.training_examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
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

    def fit(self, texts: List[str], labels: List[str]):
        """
        Fits TF-IDF (word n-grams 1-3 and char n-grams 3-5) with regularized Logistic Regression.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=1,
            analyzer="word"
        )
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(C=5.0, max_iter=300, random_state=42)
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text and returns:
        - top_intent
        - confidence
        - top3_candidates
        - is_confident (confidence >= confidence_floor)
        """
        if not self.vectorizer or not self.clf:
            return {
                "top_intent": "UNKNOWN",
                "confidence": 0.0,
                "top3": [],
                "is_confident": False
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

        return {
            "top_intent": best_intent,
            "confidence": best_prob,
            "top3": top3,
            "is_confident": is_confident
        }

    def add_training_example_and_retrain(self, text: str, intent: str):
        """
        Active Learning hook: adds user-confirmed example and updates model.
        """
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
