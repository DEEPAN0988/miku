import json
import time
import uuid
import collections
from pathlib import Path

MISSES_LOG_FILE = Path("c:/miku/miku_misses.jsonl")
LEARNED_EXAMPLES_FILE = Path("c:/miku/eval/data/learned_examples.jsonl")
REGRESSION_SET = Path("c:/miku/eval/data/dev.jsonl")

def log_miss(text, outcome, confidence, top3_candidates):
    record = {
        "id": str(uuid.uuid4()),
        "text": text,
        "outcome": outcome,
        "confidence": confidence,
        "top3": top3_candidates,
        "timestamp": time.time()
    }
    with open(MISSES_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

class MissReviewer:
    def __init__(self):
        self.misses = []
        if MISSES_LOG_FILE.exists():
            with open(MISSES_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.misses.append(json.loads(line))

    def get_grouped_misses(self):
        grouped = collections.defaultdict(list)
        for m in self.misses:
            grouped[m["text"].lower().strip()].append(m)
        
        ranked = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)
        return ranked

    def run_review_loop(self):
        ranked = self.get_grouped_misses()
        if not ranked:
            print("No misses to review.")
            return

        from miku.core2_cognitive.intent_classifier import ClassicalIntentClassifier
        classifier = ClassicalIntentClassifier()

        print(f"Found {len(ranked)} unique phrases to review.")
        for text, items in ranked:
            print(f"\n[{len(items)}x] '{text}'")
            print(f"Recent outcome: {items[-1]['outcome']}")
            
            top3 = items[-1].get("top3", [])
            if not top3:
                res = classifier.predict(text)
                top3 = res["top3"]
            
            print("Top candidates:")
            for i, (intent, prob) in enumerate(top3, 1):
                print(f"  {i}. {intent} ({prob:.2f})")
            print("  4. Custom intent / Edit")
            print("  5. Skip/Reject")
            
            ans = input("Select an option (1-5) or type intent: ").strip()
            if ans in ("5", "", "skip", "reject"):
                continue
            
            chosen_intent = None
            if ans == "1": chosen_intent = top3[0][0]
            elif ans == "2": chosen_intent = top3[1][0]
            elif ans == "3": chosen_intent = top3[2][0]
            elif ans == "4":
                chosen_intent = input("Enter intent name: ").strip().upper()
            else:
                chosen_intent = ans.upper()
                
            if not chosen_intent:
                continue
                
            is_destructive = any(d in chosen_intent for d in ["DELETE", "CLOSE", "KILL", "FORMAT", "REMOVE"])
            if is_destructive:
                conf2 = input(f"WARNING: {chosen_intent} is a destructive intent. Promoting this example requires explicit second confirmation. Proceed? (yes/no): ").strip().lower()
                if conf2 != "yes":
                    print("Skipped.")
                    continue
            
            # Auto-revert logic: measure accuracy before and after
            acc_before = self.measure_regression(classifier)
            
            # Save learned example
            ex_id = str(uuid.uuid4())
            record = {
                "id": ex_id,
                "text": text,
                "intent": chosen_intent,
                "source": "reviewed",
                "timestamp": time.time()
            }
            
            with open(LEARNED_EXAMPLES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                
            print(f"Approved! Added example ID: {ex_id}")
            
            # Retrain
            classifier.train_from_jsonls([
                Path("c:/miku/eval/data/train.jsonl"), 
                Path("c:/miku/eval/data/train_templated.jsonl"),
                LEARNED_EXAMPLES_FILE
            ])
            
            acc_after = self.measure_regression(classifier)
            print(f"Regression Accuracy: {acc_before:.2f}% -> {acc_after:.2f}%")
            if acc_after < acc_before:
                print("WARNING: Accuracy dropped! Auto-reverting...")
                self.forget_example(ex_id)
                classifier.train_from_jsonls([
                    Path("c:/miku/eval/data/train.jsonl"), 
                    Path("c:/miku/eval/data/train_templated.jsonl"),
                    LEARNED_EXAMPLES_FILE
                ])
                print("Reverted successfully.")

    def measure_regression(self, classifier):
        if not REGRESSION_SET.exists():
            return 100.0
        correct = 0
        total = 0
        with open(REGRESSION_SET, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                res = classifier.predict(d["text"])
                if res["top_intent"] == d.get("intent", d.get("expected_intent")):
                    correct += 1
                total += 1
        return (correct / total * 100) if total > 0 else 100.0

    def forget_example(self, ex_id):
        if not LEARNED_EXAMPLES_FILE.exists():
            return False
            
        lines = []
        found = False
        with open(LEARNED_EXAMPLES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                if d.get("id") == ex_id:
                    found = True
                    continue
                # schema check
                if "text" in d and "intent" in d:
                    lines.append(line)
        
        if found:
            with open(LEARNED_EXAMPLES_FILE, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line)
            print(f"Forgot example {ex_id}.")
            return True
        else:
            print(f"Example {ex_id} not found.")
            return False

    def weekly_summary(self):
        learned = []
        if LEARNED_EXAMPLES_FILE.exists():
            with open(LEARNED_EXAMPLES_FILE, "r") as f:
                learned = [json.loads(l) for l in f if l.strip()]
        
        from miku.core2_cognitive.intent_classifier import ClassicalIntentClassifier
        c = ClassicalIntentClassifier()
        acc = self.measure_regression(c)
        
        print("=== WEEKLY SUMMARY ===")
        print(f"Total misses logged: {len(self.misses)}")
        print(f"Total examples promoted: {len(learned)}")
        print(f"Current Regression Set Accuracy: {acc:.2f}%")
        print("======================")
