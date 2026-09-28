"""
Generates templated training data for Miku's Classical Intent Classifier.
Templates: (verb synonyms) x (targets) x (politeness wrappers) x (common typos).
Stores in eval/data/train_templated.jsonl (kept separate from train.jsonl).
"""
import json
from pathlib import Path
from typing import List, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT_DIR / "eval" / "data" / "train_templated.jsonl"

POLITENESS_PREFIXES = [
    "", "please ", "can you ", "could you please ", "kindly ",
    "hey miku ", "would you mind ", "go ahead and ", "i want you to "
]

POLITENESS_SUFFIXES = [
    "", " please", " now", " for me", " right away", " right now"
]

TEMPLATES = [
    # OPEN_APP
    ("OPEN_APP", [
        ("open", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "spotify", "discord", "terminal"]),
        ("launch", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "spotify", "discord", "browser"]),
        ("start", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "spotify", "discord"]),
        ("fire up", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "spotify"]),
        ("bring up", ["notepad", "calculator", "edge", "chrome", "wuthering waves"]),
        ("pop open", ["notepad", "calculator", "edge", "chrome"]),
        ("switch to", ["notepad", "calculator", "edge", "chrome", "wuthering waves"]),
        ("opne", ["calculator", "notepad", "edge"]),
        ("lauch", ["notepad", "calculator", "edge"]),
    ]),

    # CLOSE_APP
    ("CLOSE_APP", [
        ("close", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "the window", "active app"]),
        ("shut down", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "the program"]),
        ("terminate", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "the app", "calculator right now"]),
        ("kill", ["notepad", "calculator", "edge", "chrome", "wuthering waves", "the process"]),
        ("exit", ["notepad", "calculator", "edge", "chrome", "wuthering waves"]),
        ("quit", ["notepad", "calculator", "edge", "chrome", "wuthering waves"]),
        ("termiante", ["calculator", "notepad", "edge"]),
        ("shutdwn", ["edge", "chrome", "notepad"]),
        ("cloose", ["calculator", "notepad"]),
        ("termnate", ["notepad", "calculator"]),
    ]),

    # SYSTEM_VOLUME
    ("SYSTEM_VOLUME", [
        ("turn up", ["volume", "sound", "audio", "speaker"]),
        ("turn down", ["volume", "sound", "audio", "speaker"]),
        ("crank up", ["volume", "sound", "audio"]),
        ("lower", ["volume", "sound", "audio"]),
        ("raise", ["volume", "sound", "audio"]),
        ("increase", ["volume", "sound", "audio"]),
        ("decrease", ["volume", "sound", "audio"]),
        ("mute", ["sound", "audio", "speaker", "volume", "all audio"]),
        ("unmute", ["sound", "audio", "speaker", "volume", "the speaker"]),
        ("silence", ["volume", "sound", "audio"]),
        ("unmut", ["the speker", "sound", "volume"]),
    ]),

    # BROWSER_NAVIGATE
    ("BROWSER_NAVIGATE", [
        ("browse to", ["https://en.wikipedia.org", "google.com", "github.com", "reddit.com", "youtube.com"]),
        ("navigate to", ["https://en.wikipedia.org", "wikipedia.org", "google.com", "python.org"]),
        ("go to", ["https://en.wikipedia.org", "youtube.com", "github.com", "cnn.com"]),
        ("visit", ["https://en.wikipedia.org", "google.com", "nytimes.com", "wikipedia.org"]),
        ("open website", ["https://en.wikipedia.org", "github.com", "google.com"]),
    ]),

    # BROWSER_EXTRACT
    ("BROWSER_EXTRACT", [
        ("extract", ["page text", "text from this page", "all text on current tab", "article content"]),
        ("scrape", ["page text", "whatever text is on the active browser tab", "current website", "active tab"]),
        ("pull down", ["all the textual content from this web page", "text from page", "article text"]),
        ("read", ["page text", "text on this site", "content of web page"]),
        ("extrct", ["page txt", "text from site"]),
    ]),

    # OPEN_FILE
    ("OPEN_FILE", [
        ("open file", ["notes.txt", "report.pdf", "data.csv", "document.docx", "ideas.txt"]),
        ("pop open file", ["notes.txt", "report.pdf", "todo.txt"]),
        ("view document", ["report.pdf", "notes.txt", "budget.xlsx", "thesis.pdf"]),
        ("display file", ["report.pdf", "notes.txt", "summary.md"]),
        ("open folder", ["downloads", "documents", "desktop", "projects"]),
        ("opne floder", ["downloads", "documents"]),
    ]),

    # SEARCH_FILE
    ("SEARCH_FILE", [
        ("search for file", ["budget.xlsx", "thesis draft", "report.pdf", "invoice.pdf"]),
        ("find file", ["budget", "thesis", "photos", "receipts", "passwords"]),
        ("hunt for any file with", ["budget in the title", "draft in the name", "summary in title"]),
        ("look up the document", ["thesis draft", "quarterly report", "tax return"]),
        ("locate document", ["thesis draft", "project proposal"]),
    ]),

    # CREATE_FILE
    ("CREATE_FILE", [
        ("create file", ["notes.txt", "todo.md", "draft.txt", "ideas.txt", "log.txt"]),
        ("make a file named", ["test.log with content pass", "notes.txt", "todo.txt", "output.txt"]),
        ("craft a new note called", ["ideas.txt", "meeting.txt", "shopping.txt"]),
        ("write note", ["ideas.txt", "daily_plan.txt", "journal.txt"]),
    ]),

    # DELETE_FILE
    ("DELETE_FILE", [
        ("delete file", ["temp.txt", "old_cache.log", "secret_database.sqlite", "unused.zip"]),
        ("remove file", ["secret_database.sqlite", "junk.tmp", "scratch.py"]),
        ("erase document", ["draft.txt", "old_notes.txt"]),
    ]),

    # PLAN_DAY
    ("PLAN_DAY", [
        ("plan my day", ["", "today", "for tomorrow", "this morning"]),
        ("what is on my agenda", ["for today", "this afternoon", "today", "for tomorrow"]),
        ("show me my schedule", ["for today", "for this afternoon", "for tomorrow morning"]),
        ("what do i have scheduled", ["today", "this afternoon", "on my calendar"]),
        ("plan my daay", ["", "today"]),
    ]),

    # ADD_TASK
    ("ADD_TASK", [
        ("add task", ["meeting at 3pm", "buy groceries at 5pm", "call doctor tomorrow"]),
        ("put down", ["meeting on my calendar at 3pm", "workout at 6pm", "lunch at noon"]),
        ("enqueue a task", ["workout at 6pm", "finish slides at 4pm", "send report at 2pm"]),
        ("schedule", ["meeting at 3pm", "dinner at 7pm", "doctor appointment at 10am"]),
    ]),

    # SCREENSHOT
    ("SCREENSHOT", [
        ("take a screenshot", ["", "of the desktop", "of current window", "right now"]),
        ("capture the screen", ["", "for me", "now", "please"]),
        ("grab a quick screenshot", ["of my desktop", "of this window", "now"]),
        ("snap the screen", ["for me", "now", "please"]),
        ("capture screenshot", ["", "of display"]),
    ]),

    # CLICK_TARGET
    ("CLICK_TARGET", [
        ("click the", ["red button", "blue icon", "green box", "submit button", "confirm button"]),
        ("tap the", ["red button on screen", "blue icon", "close icon", "save button"]),
        ("hit the", ["blue icon please", "red button", "next button", "login icon"]),
        ("press", ["the red button", "the blue button", "submit", "the ok icon"]),
    ]),

    # INSPECT_CAMERA
    ("INSPECT_CAMERA", [
        ("look at the camera", ["", "please", "feed", "now"]),
        ("check the webcam", ["", "right now", "feed", "view"]),
        ("let us see what the webcam is seeing", ["", "now", "please"]),
        ("inspect camera", ["", "view", "feed"]),
        ("view camera stream", ["", "now"]),
    ]),

    # SYSTEM_STATUS
    ("SYSTEM_STATUS", [
        ("system status", ["", "report", "check"]),
        ("give me a status update on how you are running", ["", "now"]),
        ("how are your internal systems doing", ["", "today"]),
        ("health check", ["", "on all cores", "of miku"]),
        ("status report", ["", "now", "please"]),
    ]),

    # ABORT_AUTOMATION
    ("ABORT_AUTOMATION", [
        ("halt whatever you are running", ["", "right now", "immediately"]),
        ("freeze all ongoing automation tasks", ["", "now"]),
        ("stop automation", ["", "immediately", "right now"]),
        ("abort automation", ["", "now", "please"]),
        ("cancel all running tasks", ["", "immediately"]),
        ("stopp automashun", ["", "now"]),
    ]),

    # WINDOW_STATE
    ("WINDOW_STATE", [
        ("maximize the current active window", ["", "please"]),
        ("minimize the current app window", ["", "now"]),
        ("restore the window", ["", "to normal size"]),
        ("maximize window", ["", "now"]),
        ("minimize window", ["", "now"]),
    ]),

    # OUT_OF_SCOPE
    ("OUT_OF_SCOPE", [
        ("who was", ["the second president of the united states", "albert einstein", "the first emperor of rome"]),
        ("how do i", ["bake a chocolate cake", "solve a quadratic equation", "learn guitar"]),
        ("tell me a", ["joke", "story", "poem", "riddle"]),
        ("what is", ["the speed of light", "the weather outside", "the capital of france"]),
        ("why is", ["the sky blue", "water wet", "grass green"]),
    ])
]

def generate_templated_dataset() -> int:
    records: List[Dict[str, str]] = []
    seen = set()

    for intent, verb_groups in TEMPLATES:
        for verb, targets in verb_groups:
            for target in targets:
                for pre in ["", "please ", "can you ", "kindly "]:
                    for suf in ["", " please", " now"]:
                        core = f"{verb} {target}".strip()
                        phrase = f"{pre}{core}{suf}".strip().lower()
                        if phrase and phrase not in seen:
                            seen.add(phrase)
                            records.append({
                                "text": phrase,
                                "intent": intent
                            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[+] Wrote {len(records)} templated training samples to {OUT_PATH}")
    return len(records)

if __name__ == "__main__":
    generate_templated_dataset()
