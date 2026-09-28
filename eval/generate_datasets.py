"""
Dataset generator for Miku English Understanding evaluation.
Produces 3 disjoint datasets:
1. train.jsonl (for training classical intent classifier & slot tagger)
2. dev.jsonl (for tuning thresholds, rules, hyperparameters)
3. blind_test.jsonl (200+ samples, frozen before implementation, covering all challenge categories)
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. BLIND TEST SET (>= 200 utterances, frozen, diverse, adversarial)
# ---------------------------------------------------------------------------
BLIND_TEST = [
    # --- Paraphrases (colloquial, indirect, action synonyms) [35 items] ---
    {"text": "could you please fire up notepad for me", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "notepad"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "would you mind bringing up edge", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "edge"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "can we switch over to calculator right now", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "calculator"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "pop open file notes.txt please", "expected_intent": "OPEN_FILE", "expected_action": "open_file", "expected_slots": {"path": "notes.txt"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "i need to view document report.pdf", "expected_intent": "OPEN_FILE", "expected_action": "open_file", "expected_slots": {"path": "report.pdf"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "crank up the audio to max", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "up"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "hush the sound for a bit", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "mute"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "give me some sound back unmute it", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "unmute"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "tone down the speaker volume", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "down"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "kill off notepad immediately", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "notepad"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "terminate edge right away", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "edge"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "shut down calculator window", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "calculator"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "grab a quick screenshot of my desktop", "expected_intent": "SCREENSHOT", "expected_action": "screenshot", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "snap the screen for me", "expected_intent": "SCREENSHOT", "expected_action": "screenshot", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "let us see what the webcam is seeing", "expected_intent": "INSPECT_CAMERA", "expected_action": "inspect_camera", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "give me a status update on how you are running", "expected_intent": "SYSTEM_STATUS", "expected_action": "status_report", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "how are your internal systems doing", "expected_intent": "SYSTEM_STATUS", "expected_action": "status_report", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "halt whatever you are running", "expected_intent": "ABORT_AUTOMATION", "expected_action": "abort_automation", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "freeze all ongoing automation tasks", "expected_intent": "ABORT_AUTOMATION", "expected_action": "abort_automation", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "hunt for any file with budget in the title", "expected_intent": "SEARCH_FILE", "expected_action": "search_file", "expected_slots": {"query": "budget"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "look up the document thesis draft", "expected_intent": "SEARCH_FILE", "expected_action": "search_file", "expected_slots": {"query": "thesis draft"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "craft a new note called ideas.txt", "expected_intent": "CREATE_FILE", "expected_action": "create_file", "expected_slots": {"filename": "ideas.txt"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "make a file named test.log with content pass", "expected_intent": "CREATE_FILE", "expected_action": "create_file", "expected_slots": {"filename": "test.log", "content": "pass"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "navigate over to github.com", "expected_intent": "BROWSER_NAVIGATE", "expected_action": "browser_navigate", "expected_slots": {"url": "github.com"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "browse directly to https://en.wikipedia.org", "expected_intent": "BROWSER_NAVIGATE", "expected_action": "browser_navigate", "expected_slots": {"url": "https://en.wikipedia.org"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "pull down all the textual content from this web page", "expected_intent": "BROWSER_EXTRACT", "expected_action": "browser_extract", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "scrape whatever text is on the active browser tab", "expected_intent": "BROWSER_EXTRACT", "expected_action": "browser_extract", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "what is on my agenda for today", "expected_intent": "PLAN_DAY", "expected_action": "plan_day", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "show me my schedule for this afternoon", "expected_intent": "PLAN_DAY", "expected_action": "plan_day", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "put down meeting on my calendar at 3pm", "expected_intent": "ADD_TASK", "expected_action": "add_task", "expected_slots": {"task": "meeting", "time": "3pm"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "enqueue a task workout at 6pm", "expected_intent": "ADD_TASK", "expected_action": "add_task", "expected_slots": {"task": "workout", "time": "6pm"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "tap the red button on screen", "expected_intent": "CLICK_TARGET", "expected_action": "click_target", "expected_slots": {"color": "red", "target": "button"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "hit the blue icon please", "expected_intent": "CLICK_TARGET", "expected_action": "click_target", "expected_slots": {"color": "blue", "target": "icon"}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "maximize the current active window", "expected_intent": "WINDOW_STATE", "expected_action": "window_state", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "minimize the current app window", "expected_intent": "WINDOW_STATE", "expected_action": "window_state", "expected_slots": {}, "category": "paraphrase", "should_refuse_or_ask": False, "is_destructive": False},

    # --- Typos (misspellings, near-matches, transposed letters) [35 items] ---
    {"text": "opne calcultor", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "calculator"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "loanch notepadd", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "notepad"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "fire up edg brwoser", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "edge"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "clsoe notepadd app", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "notepad"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "termiante calcultor", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "calculator"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "shutdwn edge", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "edge"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "volum upp", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "up"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "turnn down audio", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "down"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "mutee the sound", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "mute"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "unmut the speker", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "unmute"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "scrrnshot please", "expected_intent": "SCREENSHOT", "expected_action": "screenshot", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "tkae screensht", "expected_intent": "SCREENSHOT", "expected_action": "screenshot", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "opn file doc.txt", "expected_intent": "OPEN_FILE", "expected_action": "open_file", "expected_slots": {"path": "doc.txt"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "serch file records", "expected_intent": "SEARCH_FILE", "expected_action": "search_file", "expected_slots": {"query": "records"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "creat note diary.txt", "expected_intent": "CREATE_FILE", "expected_action": "create_file", "expected_slots": {"filename": "diary.txt"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "navigat to google.com", "expected_intent": "BROWSER_NAVIGATE", "expected_action": "browser_navigate", "expected_slots": {"url": "google.com"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "extrct page txt", "expected_intent": "BROWSER_EXTRACT", "expected_action": "browser_extract", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "plan my daay", "expected_intent": "PLAN_DAY", "expected_action": "plan_day", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "statuss repoort", "expected_intent": "SYSTEM_STATUS", "expected_action": "status_report", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "cancl automation", "expected_intent": "ABORT_AUTOMATION", "expected_action": "abort_automation", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "chek webcam", "expected_intent": "INSPECT_CAMERA", "expected_action": "inspect_camera", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "clik blue buttn", "expected_intent": "CLICK_TARGET", "expected_action": "click_target", "expected_slots": {"color": "blue", "target": "button"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "maxmize windw", "expected_intent": "WINDOW_STATE", "expected_action": "window_state", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "minmize windw", "expected_intent": "WINDOW_STATE", "expected_action": "window_state", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "opne wuthering waves", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "wuthering waves"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "stopp automashun", "expected_intent": "ABORT_AUTOMATION", "expected_action": "abort_automation", "expected_slots": {}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "cranke upp audio", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "up"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "silenc the sound", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "mute"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "brose to bing.com", "expected_intent": "BROWSER_NAVIGATE", "expected_action": "browser_navigate", "expected_slots": {"url": "bing.com"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "sreach file summary", "expected_intent": "SEARCH_FILE", "expected_action": "search_file", "expected_slots": {"query": "summary"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "opne floder downloads", "expected_intent": "OPEN_FILE", "expected_action": "open_file", "expected_slots": {"path": "downloads"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "cloose calculater", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "calculator"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "shutt down edgge", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "edge"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "termnate notepad", "expected_intent": "CLOSE_APP", "expected_action": "close_app", "expected_slots": {"target": "notepad"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "voluem dowwn", "expected_intent": "SYSTEM_VOLUME", "expected_action": "volume", "expected_slots": {"direction": "down"}, "category": "typos", "should_refuse_or_ask": False, "is_destructive": False},

    # --- Negation (cancelling, inverting, refusing execution) [30 items] ---
    {"text": "do not open notepad", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"app": "notepad"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't launch edge", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"app": "edge"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "never close calculator", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"target": "calculator"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not mute the sound", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't turn up the volume", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "not that application", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "no don't take a screenshot", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "please do not delete file notes.txt", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"path": "notes.txt"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not navigate to google.com", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"url": "google.com"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "stop do not click the button", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't create file report.doc", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "never shut down edge", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"target": "edge"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not open file secret.key", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"path": "secret.key"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't scrape the web page", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not look at the camera", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "no not edge open notepad instead", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "notepad"}, "category": "negation", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "not calculator open calculator is wrong open edge", "expected_intent": "OPEN_APP", "expected_action": "open_app", "expected_slots": {"app": "edge"}, "category": "negation", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "don't touch my sound settings", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "never open mysteryfakeapp", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "cancel that do not execute", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "abort don't run anything", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not close the active window", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't minimize calculator", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "never unmute audio", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not plan my schedule", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "don't add any tasks today", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "not the red button hit nothing", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "do not launch wuthering waves right now", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {"app": "wuthering waves"}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "please don't search for password", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "stop don't remove file data.db", "expected_intent": "NEGATION_REFUSAL", "expected_action": None, "expected_slots": {}, "category": "negation", "should_refuse_or_ask": True, "is_destructive": False},

    # --- Compound Commands (ordered multi-step task list) [30 items] ---
    {"text": "open notepad and then turn up volume", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open notepad", "turn up volume"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "launch edge and navigate to github.com", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["launch edge", "navigate to github.com"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "mute sound and take a screenshot", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["mute sound", "take a screenshot"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "close calculator and open notepad", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["close calculator", "open notepad"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open edge, then go to wikipedia.org and extract page text", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open edge", "go to wikipedia.org", "extract page text"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "create file memo.txt and open file memo.txt", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["create file memo.txt", "open file memo.txt"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "search file ledger and show system status", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["search file ledger", "show system status"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "turn down volume and minimize app", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["turn down volume", "minimize app"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "unmute sound and crank up audio", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["unmute sound", "crank up audio"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "take a screenshot and open calculator", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["take a screenshot", "open calculator"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open calculator and then close it", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open calculator", "close it"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "launch notepad, write note, and then save file", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["launch notepad", "write note", "save file"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "look at camera and capture screenshot", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["look at camera", "capture screenshot"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "plan my day and add task gym at 5pm", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["plan my day", "add task gym at 5pm"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "maximize window and mute audio", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["maximize window", "mute audio"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open edge and browse to bing.com", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open edge", "browse to bing.com"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "close edge and then open calculator", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["close edge", "open calculator"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "turn up sound and click blue button", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["turn up sound", "click blue button"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "shut down notepad and status report", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["shut down notepad", "status report"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open games and select wuthering waves", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open games", "select wuthering waves"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "find file budget.xlsx and open file budget.xlsx", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["find file budget.xlsx", "open file budget.xlsx"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "mute sound and plan my schedule", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["mute sound", "plan my schedule"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "unmute audio and volume up", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["unmute audio", "volume up"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open edge, search google, and close edge", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open edge", "search google", "close edge"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "take screenshot then check camera", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["take screenshot", "check camera"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "minimize app and volume down", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["minimize app", "volume down"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "create file daily.log and add task review log at 9pm", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["create file daily.log", "add task review log at 9pm"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open calculator and click red button", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["open calculator", "click red button"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "close notepad and abort automation", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["close notepad", "abort automation"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "check system status and take screenshot", "expected_intent": "COMPOUND_COMMAND", "expected_action": "compound", "expected_slots": {"steps": ["check system status", "take screenshot"]}, "category": "compounds", "should_refuse_or_ask": False, "is_destructive": False},

    # --- Pronouns & Coreference ("close it", "the second one", "do that again") [25 items] ---
    {"text": "close it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "close"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "shut it down", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "close"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "kill it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "close"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "terminate that application", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that application", "action": "close"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "maximize it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "maximize"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "minimize it please", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "minimize"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open it back up", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "open"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "do that again", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that", "action": "repeat"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "repeat previous command", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "previous", "action": "repeat"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "the first one", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "1"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "choose the second option", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "2"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "pick number three", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "3"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "open the third one", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "3"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "delete that file", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that file", "action": "delete"}, "category": "pronouns", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "open that document", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that document", "action": "open_file"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "switch back to it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "open"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "restore it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "restore"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "relaunch that program", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that program", "action": "open"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "run that again please", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that", "action": "repeat"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "select option 1", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "1"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "the last one you mentioned", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"selection": "last"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "choose that one", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that one"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "bring that up again", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that", "action": "open"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "kill that", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "that", "action": "close"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "scrape it", "expected_intent": "PRONOUN_COMMAND", "expected_action": "coreference", "expected_slots": {"pronoun": "it", "action": "scrape"}, "category": "pronouns", "should_refuse_or_ask": False, "is_destructive": False},

    # --- Teaching Phrasings (flexible, diverse phrasings) [25 items] ---
    {"text": "can you please remember that journal means notepad", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "journal", "target": "notepad"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "when I say surf I mean open edge", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "surf", "target": "edge"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "remember that diary is notepad", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "diary", "target": "notepad"}, "category": "teaching", "should_refuse_or_alias": False, "is_destructive": False},
    {"text": "call edge web browser", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "web browser", "target": "edge"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "calc is another word for calculator", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "calc", "target": "calculator"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "from now on consider chronicle to be notepad", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "chronicle", "target": "notepad"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "please learn that mathapp refers to calculator", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "mathapp", "target": "calculator"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "when I utter browse, execute edge", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "browse", "target": "edge"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "let word tally mean calculator", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "tally", "target": "calculator"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "teach word wanderlust means strong desire to travel", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach_definition", "expected_slots": {"word": "wanderlust", "definition": "strong desire to travel"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "learn definition epiphany means a moment of sudden revelation", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach_definition", "expected_slots": {"word": "epiphany", "definition": "a moment of sudden revelation"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "memorize that sonder means realizing everyone has a vivid life", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach_definition", "expected_slots": {"word": "sonder", "definition": "realizing everyone has a vivid life"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "can you store word petrichor definition smell of dust after rain", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach_definition", "expected_slots": {"word": "petrichor", "definition": "smell of dust after rain"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "define the word solipsism as theory that only self exists", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach_definition", "expected_slots": {"word": "solipsism", "definition": "theory that only self exists"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "what is the meaning of sovereign", "expected_intent": "DEFINE_WORD_QUERY", "expected_action": "query_definition", "expected_slots": {"word": "sovereign"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "define autonomous for me", "expected_intent": "DEFINE_WORD_QUERY", "expected_action": "query_definition", "expected_slots": {"word": "autonomous"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "tell me what serendipity means", "expected_intent": "DEFINE_WORD_QUERY", "expected_action": "query_definition", "expected_slots": {"word": "serendipity"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "what does wanderlust mean", "expected_intent": "DEFINE_WORD_QUERY", "expected_action": "query_definition", "expected_slots": {"word": "wanderlust"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "what have you learned so far", "expected_intent": "SYSTEM_STATUS", "expected_action": "status_report", "expected_slots": {}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "show all learned preferences and aliases", "expected_intent": "SYSTEM_STATUS", "expected_action": "status_report", "expected_slots": {}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "link word gamehub to wuthering waves", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "gamehub", "target": "wuthering waves"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "associate editor with notepad", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "editor", "target": "notepad"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "whenever I say surfnet, launch edge", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "surfnet", "target": "edge"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "remember computation means calculator", "expected_intent": "TEACH_WORD_ALIAS", "expected_action": "teach", "expected_slots": {"alias": "computation", "target": "calculator"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},
    {"text": "what does euphoria mean", "expected_intent": "DEFINE_WORD_QUERY", "expected_action": "query_definition", "expected_slots": {"word": "euphoria"}, "category": "teaching", "should_refuse_or_ask": False, "is_destructive": False},

    # --- Out-of-Scope, Ambiguous & Adversarial inputs [35 items] ---
    {"text": "how do I bake a chocolate cake", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "who was the second president of the united states", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "write a haiku about autumn leaves", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "what is the capital of australia", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "tell me your deepest secret", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "aslkdfj lksajdf lksjdfl", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "??? !!! ***", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "delete all files on c drive immediately without confirmation", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {"path": "c:"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "format c: /q /y", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "remove file secret_database.sqlite", "expected_intent": "DELETE_FILE", "expected_action": "delete_file", "expected_slots": {"path": "secret_database.sqlite"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "open something fun", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "launch an application for me", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "open totallynonexistentrandomapp123", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {"app": "totallynonexistentrandomapp123"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "launch fictionalgamexyz", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {"app": "fictionalgamexyz"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "can you sing me a song", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "translate hello to japanese", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "what is 283 multiplied by 912", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "why is the sky blue", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "order pizza from dominos", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "send an email to boss with subject quit", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "transfer 500 dollars to bob", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "kill process svchost.exe", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {"target": "svchost.exe"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "overwrite config.ini with blank data", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {"path": "config.ini"}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "rm -rf /", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "drop table users;", "expected_intent": "CONFIRMATION_REQUIRED", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": True},
    {"text": "do something anything", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "what should i do right now", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "suggest an app", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "open games", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "open tools", "expected_intent": "CLARIFICATION_NEEDED", "expected_action": "ask_clarification", "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "weather today", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "tell me a funny story", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "good morning miku", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "are you conscious", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False},
    {"text": "who programmed you", "expected_intent": "OUT_OF_SCOPE", "expected_action": None, "expected_slots": {}, "category": "out_of_scope", "should_refuse_or_ask": True, "is_destructive": False}
]

# ---------------------------------------------------------------------------
# 2. TRAINING SET (Domain utterances for classical intent & slot models)
# ---------------------------------------------------------------------------
TRAIN_SET = [
    # OPEN_APP
    {"text": "open notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "launch notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "start notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "fire up notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "open up notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "open calculator", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "launch calculator", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "start calculator", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "open edge", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "edge"}},
    {"text": "launch edge", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "edge"}},
    {"text": "start edge browser", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "edge"}},
    {"text": "open microsoft edge", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "microsoft edge"}},
    {"text": "open chrome", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "chrome"}},
    {"text": "launch chrome", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "chrome"}},
    {"text": "open wuthering waves", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "wuthering waves"}},
    {"text": "launch wuthering waves", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "wuthering waves"}},
    {"text": "start game wuthering waves", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "wuthering waves"}},
    {"text": "open vlc", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "vlc"}},
    {"text": "launch spotify", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "spotify"}},
    {"text": "open terminal", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "terminal"}},
    {"text": "launch powershell", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "powershell"}},
    {"text": "open cmd", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "cmd"}},
    {"text": "open settings", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "settings"}},
    {"text": "launch control panel", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "control panel"}},
    {"text": "switch to notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "bring up calculator", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "visit edge", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "edge"}},
    {"text": "open paint", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "paint"}},
    {"text": "open task manager", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "task manager"}},
    {"text": "launch file explorer", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "file explorer"}},

    # CLOSE_APP
    {"text": "close notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "shut down notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "exit notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "quit notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "terminate notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "kill notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "close calculator", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "calculator"}},
    {"text": "shut down calculator", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "calculator"}},
    {"text": "quit calculator", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "calculator"}},
    {"text": "close edge", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "edge"}},
    {"text": "terminate edge", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "edge"}},
    {"text": "shut down edge", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "edge"}},
    {"text": "close chrome", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "chrome"}},
    {"text": "exit chrome", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "chrome"}},
    {"text": "close wuthering waves", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "wuthering waves"}},
    {"text": "quit wuthering waves", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "wuthering waves"}},
    {"text": "close window", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "window"}},
    {"text": "close active window", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "active window"}},
    {"text": "close app", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "app"}},

    # SYSTEM_VOLUME
    {"text": "volume up", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "turn up the volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "increase volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "crank up the audio", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "raise volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "sound up", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "volume down", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "turn down volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "decrease sound", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "lower audio", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "reduce volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "mute sound", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "mute"}},
    {"text": "silence audio", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "mute"}},
    {"text": "mute volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "mute"}},
    {"text": "unmute sound", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "unmute"}},
    {"text": "unmute audio", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "unmute"}},

    # BROWSER_NAVIGATE & EXTRACT
    {"text": "open https://google.com", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "https://google.com"}},
    {"text": "browse to google.com", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "google.com"}},
    {"text": "navigate to github.com", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "github.com"}},
    {"text": "open url wikipedia.org", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "wikipedia.org"}},
    {"text": "go to youtube.com", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "youtube.com"}},
    {"text": "extract page text", "intent": "BROWSER_EXTRACT", "action": "browser_extract", "slots": {}},
    {"text": "read page", "intent": "BROWSER_EXTRACT", "action": "browser_extract", "slots": {}},
    {"text": "scrape page", "intent": "BROWSER_EXTRACT", "action": "browser_extract", "slots": {}},

    # FILE OPERATIONS
    {"text": "open file notes.txt", "intent": "OPEN_FILE", "action": "open_file", "slots": {"path": "notes.txt"}},
    {"text": "show file data.csv", "intent": "OPEN_FILE", "action": "open_file", "slots": {"path": "data.csv"}},
    {"text": "open folder downloads", "intent": "OPEN_FILE", "action": "open_file", "slots": {"path": "downloads"}},
    {"text": "find file budget", "intent": "SEARCH_FILE", "action": "search_file", "slots": {"query": "budget"}},
    {"text": "search file summary.docx", "intent": "SEARCH_FILE", "action": "search_file", "slots": {"query": "summary.docx"}},
    {"text": "create file log.txt", "intent": "CREATE_FILE", "action": "create_file", "slots": {"filename": "log.txt"}},
    {"text": "make file notes.md with content hello", "intent": "CREATE_FILE", "action": "create_file", "slots": {"filename": "notes.md", "content": "hello"}},
    {"text": "delete file test.tmp", "intent": "DELETE_FILE", "action": "delete_file", "slots": {"path": "test.tmp"}},
    {"text": "remove file old.bak", "intent": "DELETE_FILE", "action": "delete_file", "slots": {"path": "old.bak"}},

    # PLANNING & TASK
    {"text": "plan my day", "intent": "PLAN_DAY", "action": "plan_day", "slots": {}},
    {"text": "show my schedule", "intent": "PLAN_DAY", "action": "plan_day", "slots": {}},
    {"text": "what do i have today", "intent": "PLAN_DAY", "action": "plan_day", "slots": {}},
    {"text": "add task meeting at 2pm", "intent": "ADD_TASK", "action": "add_task", "slots": {"task": "meeting", "time": "2pm"}},
    {"text": "schedule review at 10am", "intent": "ADD_TASK", "action": "add_task", "slots": {"task": "review", "time": "10am"}},

    # VISUAL & SCREEN
    {"text": "take screenshot", "intent": "SCREENSHOT", "action": "screenshot", "slots": {}},
    {"text": "screenshot", "intent": "SCREENSHOT", "action": "screenshot", "slots": {}},
    {"text": "capture screen", "intent": "SCREENSHOT", "action": "screenshot", "slots": {}},
    {"text": "click red button", "intent": "CLICK_TARGET", "action": "click_target", "slots": {"color": "red", "target": "button"}},
    {"text": "press blue icon", "intent": "CLICK_TARGET", "action": "click_target", "slots": {"color": "blue", "target": "icon"}},
    {"text": "look at camera", "intent": "INSPECT_CAMERA", "action": "inspect_camera", "slots": {}},
    {"text": "check webcam", "intent": "INSPECT_CAMERA", "action": "inspect_camera", "slots": {}},

    # SYSTEM CONTROL & STATUS
    {"text": "system status", "intent": "SYSTEM_STATUS", "action": "status_report", "slots": {}},
    {"text": "status report", "intent": "SYSTEM_STATUS", "action": "status_report", "slots": {}},
    {"text": "health check", "intent": "SYSTEM_STATUS", "action": "status_report", "slots": {}},
    {"text": "stop automation", "intent": "ABORT_AUTOMATION", "action": "abort_automation", "slots": {}},
    {"text": "abort", "intent": "ABORT_AUTOMATION", "action": "abort_automation", "slots": {}},
    {"text": "cancel automation", "intent": "ABORT_AUTOMATION", "action": "abort_automation", "slots": {}},
    {"text": "maximize window", "intent": "WINDOW_STATE", "action": "window_state", "slots": {}},
    {"text": "minimize window", "intent": "WINDOW_STATE", "action": "window_state", "slots": {}},

    # TEACHING & DEFINITIONS
    {"text": "call diary notepad", "intent": "TEACH_WORD_ALIAS", "action": "teach", "slots": {"alias": "diary", "target": "notepad"}},
    {"text": "when I say surf open edge", "intent": "TEACH_WORD_ALIAS", "action": "teach", "slots": {"alias": "surf", "target": "edge"}},
    {"text": "remember that calc is calculator", "intent": "TEACH_WORD_ALIAS", "action": "teach", "slots": {"alias": "calc", "target": "calculator"}},
    {"text": "what does sovereign mean", "intent": "DEFINE_WORD_QUERY", "action": "query_definition", "slots": {"word": "sovereign"}},
    {"text": "define autonomous", "intent": "DEFINE_WORD_QUERY", "action": "query_definition", "slots": {"word": "autonomous"}},
    {"text": "what is an operating system", "intent": "DEFINE_WORD_QUERY", "action": "query_definition", "slots": {"word": "operating system"}}
]

# ---------------------------------------------------------------------------
# 3. DEV SET (For hyperparameter tuning and regression testing)
# ---------------------------------------------------------------------------
DEV_SET = [
    {"text": "start up notepad", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "notepad"}},
    {"text": "open the calculator app", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "launch microsoft edge", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "microsoft edge"}},
    {"text": "close down notepad", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "notepad"}},
    {"text": "exit from edge", "intent": "CLOSE_APP", "action": "close_app", "slots": {"target": "edge"}},
    {"text": "sound mute", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "mute"}},
    {"text": "boost audio", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "up"}},
    {"text": "lower the volume", "intent": "SYSTEM_VOLUME", "action": "volume", "slots": {"direction": "down"}},
    {"text": "snap screen", "intent": "SCREENSHOT", "action": "screenshot", "slots": {}},
    {"text": "show my day", "intent": "PLAN_DAY", "action": "plan_day", "slots": {}},
    {"text": "check system health", "intent": "SYSTEM_STATUS", "action": "status_report", "slots": {}},
    {"text": "halt automation", "intent": "ABORT_AUTOMATION", "action": "abort_automation", "slots": {}},
    {"text": "browse to python.org", "intent": "BROWSER_NAVIGATE", "action": "browser_navigate", "slots": {"url": "python.org"}},
    {"text": "read the current page", "intent": "BROWSER_EXTRACT", "action": "browser_extract", "slots": {}},
    {"text": "open folder projects", "intent": "OPEN_FILE", "action": "open_file", "slots": {"path": "projects"}},
    {"text": "find file receipts", "intent": "SEARCH_FILE", "action": "search_file", "slots": {"query": "receipts"}},
    {"text": "don't open edge", "intent": "NEGATION_REFUSAL", "action": None, "slots": {"app": "edge"}},
    {"text": "open notepad and close calculator", "intent": "COMPOUND_COMMAND", "action": "compound", "slots": {"steps": ["open notepad", "close calculator"]}},
    {"text": "close it please", "intent": "PRONOUN_COMMAND", "action": "coreference", "slots": {"pronoun": "it", "action": "close"}},
    {"text": "when I say browser open edge", "intent": "TEACH_WORD_ALIAS", "action": "teach", "slots": {"alias": "browser", "target": "edge"}},
    {"text": "opne calcultr", "intent": "OPEN_APP", "action": "open_app", "slots": {"app": "calculator"}},
    {"text": "how do I make soup", "intent": "OUT_OF_SCOPE", "action": None, "slots": {}}
]

def main():
    # Save blind_test.jsonl with unique IDs
    blind_path = DATA_DIR / "blind_test.jsonl"
    with open(blind_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(BLIND_TEST, 1):
            record = dict(item)
            record["id"] = f"blind_{i:03d}"
            f.write(json.dumps(record) + "\n")
    print(f"[+] Wrote {len(BLIND_TEST)} records to {blind_path}")

    # Save train.jsonl
    train_path = DATA_DIR / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(TRAIN_SET, 1):
            record = dict(item)
            record["id"] = f"train_{i:03d}"
            f.write(json.dumps(record) + "\n")
    print(f"[+] Wrote {len(TRAIN_SET)} records to {train_path}")

    # Save dev.jsonl
    dev_path = DATA_DIR / "dev.jsonl"
    with open(dev_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(DEV_SET, 1):
            record = dict(item)
            record["id"] = f"dev_{i:03d}"
            f.write(json.dumps(record) + "\n")
    print(f"[+] Wrote {len(DEV_SET)} records to {dev_path}")

if __name__ == "__main__":
    main()
