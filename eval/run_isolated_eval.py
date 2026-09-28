import sys
import json
from pathlib import Path
from isolated_harness import evaluate_item_isolated
import traceback

def load_dataset(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def evaluate_dataset(dataset_path: str):
    records = load_dataset(dataset_path)
    total = len(records)
    
    correct_intents = 0
    slot_true_positives = 0
    slot_false_positives = 0
    slot_false_negatives = 0
    false_actions = 0
    
    for item in records:
        text = item["text"]
        expected_intent = item.get("expected_intent") or item.get("intent", "")
        expected_action = item.get("expected_action") or item.get("action", "")
        should_refuse_or_ask = item.get("should_refuse_or_ask", False)
        is_destructive = item.get("is_destructive", False)
        category = item.get("category", "general")
        expected_slots = item.get("expected_slots") or item.get("slots") or {}
        
        # Check for multi-turn script in item (this simulates the script conversation)
        previous_messages = item.get("previous_messages", [])
        
        try:
            res = evaluate_item_isolated(item, previous_messages=previous_messages)
        except Exception as e:
            traceback.print_exc()
            continue
            
        status = res["status"]
        actual_action = res["actual_action"]
        
        is_false_action = False
        if should_refuse_or_ask:
            if status == "dispatched" and actual_action not in (None, "noop"):
                if not (category == "negation" and expected_action == "open_app"):
                    is_false_action = True
                    false_actions += 1
                    
        intent_match = False
        if expected_intent in ("OUT_OF_SCOPE", "UNKNOWN"):
            intent_match = (status in ("conversational_response", "clarification_needed") and not is_false_action)
        elif expected_intent == "NEGATION_REFUSAL":
            intent_match = (status == "negation_refusal" or (status in ("conversational_response", "clarification_needed") and not is_false_action))
        elif expected_intent == "CONFIRMATION_REQUIRED":
            intent_match = (status == "confirmation_required")
        elif expected_intent == "CLARIFICATION_NEEDED":
            intent_match = (status == "clarification_needed")
        elif expected_intent == "COMPOUND_COMMAND":
            intent_match = (status == "dispatched" and (actual_action == "compound" or "compound" in str(res.get("message", "")).lower()))
        elif expected_intent == "PRONOUN_COMMAND":
            # dispatched = resolved entity found and action taken
            # confirmation_required = resolved destructive entity found, awaiting yes/no
            # clarification_needed = no entity in context; asking is the correct safe behavior
            intent_match = (
                status == "dispatched"
                or (is_destructive and status == "confirmation_required")
                or status == "clarification_needed"
            )
        elif expected_intent == "TEACH_WORD_ALIAS":
            # Accept Step-1 (confirmation_required) OR Step-2 (learned confirmed).
            # Both are correct system behaviors; items in the dataset capture one or the other.
            msg_lower = str(res.get("message", "")).lower()
            intent_match = (
                status == "confirmation_required"  # Step 1: awaiting yes/no
                or (status == "conversational_response" and ("learned" in msg_lower or "got it" in msg_lower or "linked" in msg_lower))
            )
        elif expected_intent == "DEFINE_WORD_QUERY":
            intent_match = (status == "conversational_response" and ("mean" in str(res.get("message", "")).lower() or ":" in str(res.get("message", "")).lower()))
        else:
            intent_match = (status == "dispatched" and actual_action == expected_action)
            
        if intent_match:
            correct_intents += 1

    intent_acc = (correct_intents / total) * 100.0
    false_action_rate = (false_actions / total) * 100.0
    
    print(f"Dataset: {dataset_path}")
    print(f"Total Items: {total}")
    print(f"Intent Accuracy: {intent_acc:.2f}%")
    print(f"False-Action Rate: {false_action_rate:.2f}% ({false_actions} total)")
    print("-" * 50)

if __name__ == "__main__":
    evaluate_dataset("c:/miku/eval/data/dev_blind_v1.jsonl")
    evaluate_dataset("c:/miku/eval/data/blind_v2.jsonl")
