import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
from pathlib import Path

from miku.core2_cognitive.grammar_router import DeterministicGrammarRouter

def get_destructive_words():
    # from cognitive_daemon.py
    return ["delete", "remove", "format", "wipe", "destroy", "erase", "overwrite", "kill", "terminate", "close", "quit", "exit", "shut", "rm", "drop", "transfer", "send"]

def load_training_phrasings():
    phrases = {}
    data_dir = Path(r"c:\miku\eval\data")
    for f in ["train.jsonl", "train_templated.jsonl"]:
        p = data_dir / f
        if p.exists():
            with open(p, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip(): continue
                    d = json.loads(line)
                    i = d.get("intent", d.get("expected_intent"))
                    if i:
                        phrases.setdefault(i, set()).add(d.get("text", ""))
    return phrases

def main():
    router = DeterministicGrammarRouter()
    destructive_words = get_destructive_words()
    
    phrasings = load_training_phrasings()
    
    capabilities = {}
    
    for rule in router.rules:
        intent = rule["intent"]
        if intent == "REFUSE_CAPTCHA_REQUEST":
            continue
        if intent not in capabilities:
            capabilities[intent] = {
                "intent": intent,
                "action": rule["action"],
                "slots": set(),
                "is_destructive": False,
                "phrasings_covered": [],
                "flag_needs_more_phrasings": False
            }
        
        cap = capabilities[intent]
        
        # slots
        cap["slots"].update(rule["regex"].groupindex.keys())
        cap["slots"].update(rule["extra"].keys())
        
        # destructive
        pat = rule["regex"].pattern.lower()
        if any(re.search(r"\b" + re.escape(w) + r"\b", pat) for w in destructive_words) or rule["extra"].get("needs_confirmation"):
            cap["is_destructive"] = True
            
    # Add non-grammar intents
    for intent in ["TEACH_WORD_ALIAS", "DEFINE_WORD_QUERY", "FORGET_WORD_ALIAS", "COMPOUND_COMMAND", "PRONOUN_COMMAND"]:
        if intent not in capabilities:
            capabilities[intent] = {
                "intent": intent,
                "action": None,
                "slots": set(),
                "is_destructive": False,
                "phrasings_covered": [],
                "flag_needs_more_phrasings": False
            }
            
    for intent, cap in capabilities.items():
        cap["slots"] = list(cap["slots"])
        covered = list(phrasings.get(intent, []))
        cap["phrasings_covered"] = covered
        cap["flag_needs_more_phrasings"] = len(covered) < 10

    with open(r"c:\miku\capabilities.json", "w", encoding="utf-8") as f:
        json.dump(capabilities, f, indent=2)

if __name__ == "__main__":
    main()
