import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import uuid
from miku.core2_cognitive.app_catalog import APP_CATALOG
from collections import defaultdict

random.seed(42)

APPS = list(APP_CATALOG.keys())
if not APPS: APPS = ["notepad", "calculator", "edge", "chrome"]

POLITE_PREFIXES = ["", "please ", "can you ", "could you ", "could you please ", "can you please ", "hey miku, ", "i want to ", "i need you to ", "would you mind "]
SUFFIXES = ["", " please", " for me", " now", " real quick", " immediately", " right away"]

VERBS = {
    "OPEN_APP": ["open", "launch", "start", "run", "fire up", "bring up", "pull up", "load", "boot up"],
    "CLOSE_APP": ["close", "quit", "exit", "shut", "shut down", "kill", "terminate"],
    "SEARCH_FILE": ["search", "find", "look for", "look up"],
    "SYSTEM_VOLUME_UP": ["raise", "turn up", "crank up", "increase", "boost"],
    "SYSTEM_VOLUME_DOWN": ["lower", "turn down", "reduce", "decrease"],
    "SYSTEM_VOLUME_MUTE": ["mute", "silence"],
    "DELETE_FILE": ["delete", "remove", "destroy", "wipe", "format", "erase"],
    "BROWSER_NAVIGATE": ["navigate to", "browse to", "open url", "go to"],
}

HELD_OUT_VERBS = ["pull up", "crank up", "fire up", "shut down", "look up", "boot up"]
HELD_OUT_SUFFIXES = ["real quick", "right away"]

def introduce_typo(text):
    if len(text) < 4: return text
    idx = random.randint(1, len(text)-2)
    chars = list(text)
    typo_type = random.choice(["transpose", "drop", "double"])
    if typo_type == "transpose":
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif typo_type == "drop":
        chars.pop(idx)
    elif typo_type == "double":
        chars.insert(idx, chars[idx])
    return "".join(chars)

def generate():
    out_all = []
    out_held = []
    
    # helper
    def add_ex(text, intent, source_tid, is_heldout):
        clean = text.lower().strip()
        ex = {"text": clean, "intent": intent, "source": "template", "template_id": source_tid}
        
        target_list = out_held if is_heldout else out_all
        target_list.append(ex)
        
        # Typos at low rate (10%)
        if random.random() < 0.1 and not is_heldout:
            ex_typo = ex.copy()
            ex_typo["text"] = introduce_typo(clean)
            target_list.append(ex_typo)

    # 1. OPEN APP
    for app in APPS:
        for verb in VERBS["OPEN_APP"]:
            for pfx in POLITE_PREFIXES:
                for sfx in SUFFIXES:
                    held = (verb in HELD_OUT_VERBS) or (sfx.strip() in HELD_OUT_SUFFIXES)
                    # "open notepad"
                    add_ex(f"{pfx}{verb} {app}{sfx}", "OPEN_APP", "open_app_1", held)
                    # "notepad, open it"
                    if not pfx and not held:
                        add_ex(f"{app}, {verb} it{sfx}", "OPEN_APP", "open_app_2", False)
                        add_ex(f"get {app} {verb}{sfx}", "OPEN_APP", "open_app_3", False)

    # 2. CLOSE APP
    for app in APPS:
        for verb in VERBS["CLOSE_APP"]:
            for pfx in POLITE_PREFIXES:
                for sfx in SUFFIXES:
                    held = (verb in HELD_OUT_VERBS) or (sfx.strip() in HELD_OUT_SUFFIXES)
                    add_ex(f"{pfx}{verb} {app}{sfx}", "CLOSE_APP", "close_app_1", held)

    # 3. DELETE_FILE
    for verb in VERBS["DELETE_FILE"]:
        for fn in ["report.pdf", "notes.txt", "project_v2", "old_folder", "data.csv"]:
            for pfx in POLITE_PREFIXES:
                for sfx in SUFFIXES:
                    held = (verb in HELD_OUT_VERBS) or (sfx.strip() in HELD_OUT_SUFFIXES)
                    add_ex(f"{pfx}{verb} {fn}{sfx}", "DELETE_FILE", "delete_file_1", held)
                    add_ex(f"{pfx}{verb} file {fn}{sfx}", "DELETE_FILE", "delete_file_2", held)

    # 4. SYSTEM_VOLUME
    for v_type, v_intent in [("SYSTEM_VOLUME_UP", "SYSTEM_VOLUME"), ("SYSTEM_VOLUME_DOWN", "SYSTEM_VOLUME"), ("SYSTEM_VOLUME_MUTE", "SYSTEM_VOLUME")]:
        for verb in VERBS[v_type]:
            for tgt in ["volume", "sound", "audio", "it"]:
                for pfx in POLITE_PREFIXES:
                    for sfx in SUFFIXES:
                        held = (verb in HELD_OUT_VERBS) or (sfx.strip() in HELD_OUT_SUFFIXES)
                        add_ex(f"{pfx}{verb} {tgt}{sfx}", v_intent, f"vol_{v_type}", held)
                        if "mute" in verb:
                            add_ex(f"{pfx}{verb}{sfx}", v_intent, f"vol_{v_type}_no_tgt", held)

    # 5. NEGATIVES (Refusals, Incomplete, OOS)
    for app in APPS:
        for pfx in POLITE_PREFIXES:
            add_ex(f"{pfx}don't open {app}", "NEGATION_REFUSAL", "neg_1", False)
            add_ex(f"{pfx}never mind about {app}", "NEGATION_REFUSAL", "neg_2", False)
            add_ex(f"{pfx}stop", "ABORT_AUTOMATION", "neg_3", False)
    
    for v in ["open", "close", "delete", "search for"]:
        add_ex(f"please {v}", "CLARIFICATION_NEEDED", "inc_1", False)
        add_ex(f"{v}", "CLARIFICATION_NEEDED", "inc_2", False)

    for oos in ["write me an essay", "tell me a joke", "sing a song", "write a poem", "how do i cook pasta", "give me financial advice", "what is the meaning of life"]:
        for pfx in POLITE_PREFIXES:
            add_ex(f"{pfx}{oos}", "OUT_OF_SCOPE", "oos_1", False)

    # REFUSE_CAPTCHA_REQUEST — classifier examples (captcha/anti-bot evasion is OOS; these teach refusal)
    for i in range(1):
        app = f"app"
        al = f"alias"
        word = f"word"
        add_ex(f"solve this captcha", "REFUSE_CAPTCHA_REQUEST", "t1", False)
        add_ex(f"click the captcha", "REFUSE_CAPTCHA_REQUEST", "t1", False)
        add_ex(f"do the bot check", "REFUSE_CAPTCHA_REQUEST", "t2", False)
        add_ex(f"pass the anti bot", "REFUSE_CAPTCHA_REQUEST", "t2", False)
        add_ex(f"remember that {app} is {al}", "TEACH_WORD_ALIAS", "t3", False)
        add_ex(f"associate {app} with {al}", "TEACH_WORD_ALIAS", "t3", False)
        add_ex(f"when i say {al} i mean {app}", "TEACH_WORD_ALIAS", "t3", False)
        add_ex(f"what does {word} mean", "DEFINE_WORD_QUERY", "t4", False)
        add_ex(f"define {word}", "DEFINE_WORD_QUERY", "t4", False)
        add_ex(f"forget word {word}", "FORGET_WORD_ALIAS", "t5", False)
        add_ex(f"forget alias {al}", "FORGET_WORD_ALIAS", "t5", False)
        add_ex(f"open {app} and close notepad", "COMPOUND_COMMAND", "t6", False)
        add_ex(f"open {app} and search for dogs", "COMPOUND_COMMAND", "t6", False)
        add_ex(f"open {app} and then search", "COMPOUND_COMMAND", "t6", False)
        add_ex(f"open notepad and close {app}", "COMPOUND_COMMAND", "t6", False)
        add_ex(f"close it", "PRONOUN_COMMAND", "t7", False)
        add_ex(f"open it", "PRONOUN_COMMAND", "t7", False)
        add_ex(f"delete it", "PRONOUN_COMMAND", "t7", False)

    for i in range(1):
        add_ex(f"navigate to example.com", "BROWSER_NAVIGATE", "t8", False)
        add_ex(f"go to example.com", "BROWSER_NAVIGATE", "t8", False)
        add_ex(f"extract text from page", "BROWSER_EXTRACT", "t9", False)
        add_ex(f"scrape page content", "BROWSER_EXTRACT", "t9", False)
        add_ex(f"open document.txt", "OPEN_FILE", "t10", False)
        add_ex(f"view file", "OPEN_FILE", "t10", False)
        add_ex(f"find file", "SEARCH_FILE", "t11", False)
        add_ex(f"search for document", "SEARCH_FILE", "t11", False)
        add_ex(f"create file", "CREATE_FILE", "t12", False)
        add_ex(f"make new document", "CREATE_FILE", "t12", False)
        add_ex(f"maximize window", "WINDOW_STATE", "t13", False)
        add_ex(f"minimize window", "WINDOW_STATE", "t13", False)
        add_ex(f"what is my schedule", "PLAN_DAY", "t14", False)
        add_ex(f"check my plan for today", "PLAN_DAY", "t14", False)
        add_ex(f"add task to list", "ADD_TASK", "t15", False)
        add_ex(f"remind me to", "ADD_TASK", "t15", False)
        add_ex(f"click on", "CLICK_TARGET", "t16", False)
        add_ex(f"press", "CLICK_TARGET", "t16", False)
        add_ex(f"take screenshot", "SCREENSHOT", "t17", False)
        add_ex(f"capture screen", "SCREENSHOT", "t17", False)
        add_ex(f"check camera", "INSPECT_CAMERA", "t18", False)
        add_ex(f"view webcam", "INSPECT_CAMERA", "t18", False)
        add_ex(f"check system status", "SYSTEM_STATUS", "t19", False)
        add_ex(f"how is memory", "SYSTEM_STATUS", "t19", False)
    counts = defaultdict(list)
    for ex in out_all:
        counts[ex["template_id"]].append(ex)
    
    final_all = []
    for tid, items in counts.items():
        if len(items) > 500:
            final_all.extend(random.sample(items, 500))
        else:
            final_all.extend(items)
            
    random.shuffle(final_all)
    random.shuffle(out_held)
    
    print(f"Generated {len(final_all)} train examples, {len(out_held)} held-out examples.")
    
    with open(r"c:\miku\eval\data\train_templated.jsonl", "w", encoding="utf-8") as f:
        for ex in final_all:
            f.write(json.dumps(ex) + "\n")
            
    with open(r"c:\miku\eval\data\train_templated_heldout.jsonl", "w", encoding="utf-8") as f:
        for ex in out_held:
            f.write(json.dumps(ex) + "\n")

    # Class balance report
    dist = defaultdict(int)
    for ex in final_all:
        dist[ex["intent"]] += 1
    
    with open(r"c:\miku\class_balance_report.txt", "w") as f:
        f.write("Class Balance:\n")
        for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    generate()
