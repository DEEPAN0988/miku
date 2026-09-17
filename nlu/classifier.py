"""
Trainable Local Intent Classifier for Miku NLU.
Trained 100% from scratch using scikit-learn (TF-IDF + LogisticRegression / SGD).
Zero pretrained weights, zero API keys. Fully retrainable on device.
"""

import os
import pickle
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nlu_model.pkl")

# Built-in seed training dataset for bootstrapping the local model from scratch
SEED_DATASET = [
    # launch_app
    ("open notepad", "launch_app"),
    ("launch calculator", "launch_app"),
    ("start paint", "launch_app"),
    ("open browser", "launch_app"),
    ("open file manager", "launch_app"),
    ("launch terminal", "launch_app"),
    ("start notepad application", "launch_app"),
    ("open calc", "launch_app"),
    
    # get_time / get_date
    ("what is the time right now", "get_time"),
    ("tell me the current time", "get_time"),
    ("what time is it", "get_time"),
    ("time please", "get_time"),
    ("what is today's date", "get_date"),
    ("tell me the date", "get_date"),
    ("what day is it today", "get_date"),
    
    # plan_day / plan_goal
    ("plan my day today", "plan_day"),
    ("show my schedule for the day", "plan_day"),
    ("what is my agenda today", "plan_day"),
    ("what do i have to do today", "plan_day"),
    ("how should i train for a marathon", "plan_goal"),
    ("plan my training for fitness", "plan_goal"),
    ("how should i prepare for python study", "plan_goal"),
    
    # task management
    ("create a new task buy groceries", "create_task"),
    ("add task finish report with priority high", "create_task"),
    ("add a task call mom", "create_task"),
    ("list all my pending tasks", "list_tasks"),
    ("show my tasks", "list_tasks"),
    ("what tasks do i have pending", "list_tasks"),
    
    # vision
    ("take a look with the camera", "vision_camera"),
    ("what do you see in the room", "vision_camera"),
    ("check camera", "vision_camera"),
    ("capture my current screen", "vision_screen"),
    ("read what is on the screen", "vision_screen"),
    ("inspect the active window on screen", "vision_screen"),
    
    # connectivity
    ("scan for bluetooth devices", "connect_scan"),
    ("find nearby wifi networks", "connect_scan"),
    ("list available devices", "connect_scan"),
    
    # system control
    ("what is the cpu and ram usage", "system_status"),
    ("check my pc status and battery", "system_status"),
    ("show system performance", "system_status"),
    ("delete file test.txt", "delete_file"),
    ("erase file old_data.log", "delete_file"),
    ("kill process notepad", "kill_process"),
    ("terminate process task", "kill_process"),
    
    # confirmation
    ("yes please proceed", "confirm_yes"),
    ("i confirm", "confirm_yes"),
    ("do it", "confirm_yes"),
    ("no do not do that", "confirm_no"),
    ("cancel operation", "confirm_no"),
    ("abort action", "confirm_no")
]


class IntentClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                return
            except Exception:
                pass
        self.train(SEED_DATASET)

    def train(self, dataset: list = None) -> Dict[str, Any]:
        """Train the classifier from scratch using TF-IDF + LogisticRegression."""
        data = dataset or SEED_DATASET
        texts = [x[0] for x in data]
        labels = [x[1] for x in data]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=1000)),
            ("clf", MultinomialNB(alpha=0.1))
        ])

        self.pipeline.fit(texts, labels)
        
        # Save model locally
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

        return {
            "samples": len(texts),
            "classes": len(set(labels)),
            "model_path": self.model_path
        }

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent and estimated probability score."""
        if not self.pipeline:
            self.train()

        probs = self.pipeline.predict_proba([text])[0]
        max_idx = probs.argmax()
        label = self.pipeline.classes_[max_idx]
        confidence = float(probs[max_idx])
        return label, confidence
