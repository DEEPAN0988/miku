"""
train/prepare_tool_data.py — SFT Dataset Generator for Media & Expanded Tool-Calling (Phase 8).

Expands tool coverage to 14 tools:
  - Get Time (Argument: None) — LOW risk
  - Get Date (Argument: None) — LOW risk
  - Get Battery (Argument: None) — LOW risk
  - Get Volume (Argument: None) — LOW risk
  - List Running Apps (Argument: None) — LOW risk
  - Play Media (Argument: None) — LOW risk
  - Pause Media (Argument: None) — LOW risk
  - Next Track (Argument: None) — LOW risk
  - Previous Track (Argument: None) — LOW risk
  - Search Web (Argument: <query>) — MEDIUM risk
  - Set Volume (Argument: <level>) — MEDIUM risk
  - Play Query (Argument: <query>) — MEDIUM risk
  - Open App (Argument: <app>) — HIGH risk
  - Close App (Argument: <app>) — HIGH risk

STRICTLY MAINTAINS 2:1 CONVERSATIONAL-TO-TOOL RATIO (66.7% Replay / 33.3% Tools)
to prevent tool-bias regression on general dialogue.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import MikuTokenizer


# ==============================================================================
# 1. HELD-OUT TEST POOLS (STRICTLY ISOLATED FROM TRAINING)
# ==============================================================================

HELD_OUT_APPS: Set[str] = {"slack", "spotify", "photoshop", "blender", "steam"}
HELD_OUT_QUERIES: Set[str] = {
    "quantum computing papers",
    "mars rover latest photos",
    "history of ancient rome",
    "best coffee brew methods"
}
HELD_OUT_VOLUME_LEVELS: Set[str] = {"45%", "90%", "15%", "silent"}
HELD_OUT_PLAY_QUERIES: Set[str] = {
    "bohemian rhapsody",
    "cyberpunk ambient soundtrack",
    "chopin nocturne op 9 no 2",
    "hans zimmer interstellar theme"
}

HELD_OUT_PHRASINGS = {
    "Get Time": [
        "Do you have the time on you?",
        "Could you check the current time?",
        "What's the clock saying right now?",
    ],
    "Get Date": [
        "What's on the calendar for today's date?",
        "Mind checking what date we're on?",
        "Can you look up what today's date is?",
    ],
    "Get Battery": [
        "Can you see how much power is remaining in the battery?",
        "What's the current charge level?",
        "Is the device running on battery power right now?",
    ],
    "Get Volume": [
        "Could you inspect the audio level on my machine?",
        "What is the master speaker volume?",
        "Check if my audio is muted.",
    ],
    "List Running Apps": [
        "Give me an inventory of open software.",
        "What's currently active on my desktop?",
        "Inspect the system to see what applications are active.",
    ],
    "Play Media": [
        "Get the audio moving again.",
        "Hit the play toggle on the media player.",
        "Could you restart playing the audio?",
    ],
    "Pause Media": [
        "Cease audio output for a second.",
        "Put the tunes on hold.",
        "Temporarily freeze playback right now.",
    ],
    "Next Track": [
        "Forward to the subsequent tune.",
        "Jump ahead one title.",
        "Can we advance past this audio track?",
    ],
    "Previous Track": [
        "Revisit the preceding title.",
        "Roll back to the prior tune.",
        "Can we rewind to the last piece?",
    ],
    "Search Web": [
        "Do a web search regarding {query}.",
        "Search online to find out about {query}.",
        "Query the internet for {query}.",
    ],
    "Set Volume": [
        "Tune the audio output to {level}.",
        "Switch the master volume to {level}.",
        "Dial the sound to {level}.",
    ],
    "Play Query": [
        "Cue up {query}.",
        "Fire up the recording {query}.",
        "Spin the track {query} on the speakers.",
    ],
    "Open App": [
        "Fire up {app}.",
        "Can you boot up {app}?",
        "Please execute {app} for me.",
    ],
    "Close App": [
        "End the {app} process.",
        "Force close {app} right now.",
        "Shut off {app} for me.",
    ]
}


# ==============================================================================
# 2. TRAINING POOLS
# ==============================================================================

TRAIN_APPS: List[str] = [
    "calculator", "terminal", "browser", "notepad", "settings",
    "calendar", "clock", "camera", "files", "editor",
    "weather", "music", "maps", "mail", "gallery",
    "notes", "contacts", "tasks", "calculator app", "terminal window"
]

TRAIN_QUERIES: List[str] = [
    "weather forecast", "python tutorial", "world news today", "dinner recipes",
    "machine learning basics", "stock market today", "flight status", "nearby restaurants",
    "traffic conditions", "bread baking tips", "top movies 2026", "how to tie a tie",
    "solar eclipse dates", "best hiking trails", "workout routine", "guitar chords",
    "gardening tips for spring", "local gas prices", "translate english to spanish", "fastest land animal"
]

TRAIN_VOLUME_LEVELS: List[str] = [
    "50%", "75%", "20%", "30%", "80%", "100%", "mute", "unmute", "maximum", "zero", "40%", "60%"
]

TRAIN_PLAY_QUERIES: List[str] = [
    "beethoven symphony 5", "lofi hip hop radio", "jazz relaxing piano",
    "taylor swift shake it off", "daft punk get lucky", "classical focus playlist",
    "rock classics", "synthwave chill", "acoustic guitar", "ambient sleep sounds",
    "queen don't stop me now", "miles davis blue in green", "nature rain sounds",
    "edm festival mix", "chillhop beats", "the beatles yesterday", "relaxing cello",
    "piano sonata", "coffee shop ambiance", "instrumental study beats"
]

TEMPLATES = {
    "Get Time": [
        "What time is it?", "What time is it right now?", "Can you tell me the current time?",
        "Tell me the time.", "What is the current time?", "Do you know what time it is?",
        "What's the time?", "Current time please.", "Give me the time right now.",
        "Check the time for me.", "Could you tell me what time it is?", "Time please.",
        "What is the time on the clock?", "Can you read the clock?", "Tell me the local time.",
        "What time do you have?", "Time check right now.", "Let me know the time.",
        "Check system clock time.", "What hour and minute is it?"
    ],
    "Get Date": [
        "What is today's date?", "What's the date today?", "Could you tell me what date it is?",
        "What day is it today?", "Current date please.", "Do you know what date it is?",
        "What's today's date?", "Tell me the date.", "What date is it?", "Date please.",
        "What is the date on the calendar?", "Check today's date.", "What day of the week is it?",
        "Tell me the day and date.", "Check the calendar date.", "What month and day is today?",
        "Look at the date today.", "Today's calendar date please.", "Give me the current date."
    ],
    "Get Battery": [
        "What is my battery level?", "How much battery is left?", "Check battery status.",
        "What's my battery percentage?", "Is the laptop plugged in?", "How much charge do I have left?",
        "Battery status please.", "Tell me how much battery I have.", "What is the battery charge?",
        "Check laptop battery.", "Is the battery charging?", "Battery level check.",
        "Check battery power.", "How much juice is left in the battery?", "Is the power cord connected?",
        "Show me battery life.", "What's the remaining battery charge?", "Is the battery full?",
        "Report battery status.", "Check the power level of the laptop.", "How full is the battery?",
        "How many minutes of battery remain?", "Is my computer on AC power or battery?", "Battery percentage check.",
        "Check if the laptop is plugged into the wall.", "How is the laptop power holding up?", "Check remaining power."
    ],
    "Get Volume": [
        "What is the volume level?", "Check the volume.", "What's the audio volume right now?",
        "Is the sound muted?", "What is the current volume?", "How loud is the sound set to?",
        "Volume check.", "What's the system volume level?", "Tell me the current sound volume.",
        "Is my audio muted?", "Check system volume.", "Current volume level please.",
        "Inspect the audio volume.", "What is the speaker volume?", "Check master volume.",
        "Is sound turned on or muted?", "What decibel or percentage is sound set to?", "How loud are the speakers?",
        "Check the audio output level.", "What is the master speaker volume?", "Tell me if the sound is muted.",
        "Inspect system audio level.", "Is speaker sound active?", "What's the sound level on my computer?",
        "Audio volume check please.", "How high is the volume set?", "Check sound loudness."
    ],
    "List Running Apps": [
        "What apps are currently running?", "List running applications.", "Show me what programs are open.",
        "What processes are active?", "Which apps are open right now?", "List active apps.",
        "What programs are running?", "Show open apps.", "List the running programs.",
        "Can you see what applications are active?", "What software is currently running?", "List open apps.",
        "Show active processes.", "What software do I have active?", "List all running tasks.",
        "Show running processes.", "What applications do I have open?", "View open software.",
        "Check running software.", "List all active applications.", "Show me all open windows.",
        "What is running on my computer?", "Give me a list of open apps.", "List active computer software."
    ],
    "Play Media": [
        "Play media.", "Play the music.", "Resume playback.", "Resume playing the track.",
        "Unpause the music.", "Unpause playback.", "Start the music playing.", "Continue playing the music.",
        "Can you play the current song?", "Resume audio playback.", "Start playing the audio.", "Unpause.",
        "Resume the song.", "Hit play on the media.", "Continue playback.", "Start playing track.",
        "Turn playback back on.", "Play the audio.", "Resume track.", "Resume my playlist.",
        "Start the player.", "Play current track.", "Get the music going again.", "Can you unpause playback?",
        "Resume the sound.", "Keep playing the song.", "Continue the audio track.", "Start media playback."
    ],
    "Pause Media": [
        "Pause the music.", "Pause playback.", "Pause media.", "Pause the song.",
        "Can you pause playback?", "Freeze playback.", "Halt the music.", "Stop playing the audio.",
        "Pause current track.", "Pause the sound.", "Put the music on hold.", "Pause the audio right now.",
        "Hold playback.", "Can you pause the music for me?", "Pause the track.", "Stop media playback.",
        "Halt audio playback.", "Pause the player.", "Please pause the current song.", "Take a break from music, pause it.",
        "Pause audio.", "Hit pause on the player.", "Hold the music.", "Pause song.",
        "Temporarily stop the music.", "Pause my playlist.", "Freeze the audio.", "Pause playback please."
    ],
    "Next Track": [
        "Skip this song.", "Next track please.", "Play the next song.", "Skip to next track.",
        "Can you go to the next song?", "Next track.", "Skip this track.", "Jump to the next song.",
        "Advance to the next track.", "Go forward one song.", "Next song.", "Skip ahead to the next track.",
        "Change to the next song.", "Please skip this song.", "Move to the next track.", "Play the next track on the playlist.",
        "Fast forward to next track.", "Skip forward.", "Go to the next track.", "Forward to the next song.",
        "Switch to the next song.", "Play next.", "Next song on the list.", "Skip this track for me.",
        "Next audio track.", "Hop to the next song.", "Skip to the next.", "Next song please."
    ],
    "Previous Track": [
        "Go back to the previous song.", "Previous track please.", "Play the previous track.", "Rewind to the last song.",
        "Can you play the previous track?", "Previous track.", "Go back one song.", "Play the last track again.",
        "Restart the previous song.", "Backtrack to previous track.", "Return to the previous song.", "Play previous song.",
        "Previous song.", "Back up one track.", "Go to the previous song on playlist.", "Jump back to the previous track.",
        "Skip back one track.", "Play the prior song.", "Go back to the song before this.", "Step back to the previous track.",
        "Reverse to the previous song.", "Play the preceding track.", "Back to previous song.", "Can you go back to the previous song?",
        "Previous audio track.", "Switch back to the previous track.", "Rewind track.", "Go back a song."
    ],
    "Play Query": [
        "Play {query}.", "Can you play {query}?", "Put on {query}.", "Stream {query}.",
        "I want to listen to {query}.", "Play some {query}.", "Queue up {query}.", "Play the song {query}.",
        "Find and play {query}.", "Start playing {query}.", "Listen to {query}.", "Put {query} on the music player.",
        "Can you stream {query}?", "Play the album {query}.", "Please play {query}.", "Search and play {query}.",
        "Spin the track {query}.", "Play music by {query}.", "Queue up the song {query}.", "Let's listen to {query}.",
        "Play {query} on speakers.", "Can you put on {query}?", "Play the artist {query}.", "Stream the track {query}.",
        "Play {query} right now.", "Put on some {query} music.", "Start {query} playback.", "Play {query} for me."
    ],
    "Open App": [
        "Open the {app}.", "Open {app}.", "Launch {app}.", "Can you open {app}?",
        "Please open {app}.", "Start {app}.", "Launch the {app} program.", "Bring up {app}.",
        "Run {app}.", "Can you start {app} for me?", "Please launch {app}.", "Open up {app} please.",
        "Start running {app}.", "Execute the {app} program.", "Start up {app}.",
        "Please open the {app} application.", "Can you launch {app} right now?", "Bring up the {app} window.",
        "I want to use {app}, open it.", "Open up {app} software."
    ],
    "Close App": [
        "Close the {app}.", "Close {app}.", "Quit {app}.", "Shut down {app}.",
        "Exit {app}.", "Terminate {app}.", "Kill {app}.", "Please close {app}.",
        "Stop the {app} application.", "Close down {app}.", "Can you close {app}?", "Exit {app} program.",
        "Shut off {app}.", "Kill the {app} task.", "Stop running {app}.", "Terminate the {app} software.",
        "Close out {app}.", "End {app}.", "Please shut down {app} for me.", "Close the {app} window."
    ],
    "Search Web": [
        "Search for {query}.", "Search the web for {query}.", "Look up {query} on the web.",
        "Can you search for {query}?", "Please search the web for {query}.", "Find information about {query}.",
        "Search online for {query}.", "Look up {query}.", "Google {query}.", "Search the internet for {query}.",
        "Can you look up {query}?", "Please find info on {query}.", "Search the web to find {query}.",
        "Find details about {query} online.", "Do a search for {query}.", "Search about {query}.",
        "Can you find {query} on the web?", "Look online for {query}.", "Perform a web search for {query}."
    ],
    "Set Volume": [
        "Set volume to {level}.", "Change volume to {level}.", "Adjust sound to {level}.",
        "Turn volume to {level}.", "Put the volume at {level}.", "Please set the volume to {level}.",
        "Make the volume {level}.", "Set audio to {level}.", "Change sound to {level}.", "Set speaker volume to {level}.",
        "Turn sound to {level}.", "Switch volume to {level}.", "Set the system volume to {level}.",
        "Adjust audio loudness to {level}.", "Set the speaker sound to {level}.", "Put sound volume to {level}."
    ]
}


def generate_tool_samples(count_per_tool: int = 180) -> List[Dict[str, str]]:
    samples = []

    for tool_name, tmpls in TEMPLATES.items():
        for _ in range(count_per_tool):
            tmpl = random.choice(tmpls)
            if tool_name in (
                "Get Time", "Get Date", "Get Battery", "Get Volume", "List Running Apps",
                "Play Media", "Pause Media", "Next Track", "Previous Track"
            ):
                user_text = tmpl
                response = f"Action: {tool_name}\nArgument: None"
            elif tool_name == "Play Query":
                q = random.choice(TRAIN_PLAY_QUERIES)
                user_text = tmpl.format(query=q)
                response = f"Action: Play Query\nArgument: {q}"
            elif tool_name == "Open App":
                app = random.choice(TRAIN_APPS)
                user_text = tmpl.format(app=app)
                clean_arg = app.replace(" app", "").replace(" the ", "").replace(" window", "").strip()
                response = f"Action: Open App\nArgument: {clean_arg}"
            elif tool_name == "Close App":
                app = random.choice(TRAIN_APPS)
                user_text = tmpl.format(app=app)
                clean_arg = app.replace(" app", "").replace(" the ", "").replace(" window", "").strip()
                response = f"Action: Close App\nArgument: {clean_arg}"
            elif tool_name == "Search Web":
                query = random.choice(TRAIN_QUERIES)
                user_text = tmpl.format(query=query)
                response = f"Action: Search Web\nArgument: {query}"
            elif tool_name == "Set Volume":
                lvl = random.choice(TRAIN_VOLUME_LEVELS)
                user_text = tmpl.format(level=lvl)
                response = f"Action: Set Volume\nArgument: {lvl}"
            else:
                continue

            prompt = f"Instruction:\n{user_text}\n\nResponse:\n"
            cat = f"tool_{tool_name.lower().replace(' ', '_')}"
            samples.append({"prompt": prompt, "response": response, "category": cat})

    return samples


def load_conversational_replay(dolly_count: int = 3300, multiturn_count: int = 1800) -> List[Dict[str, str]]:
    """Loads conversational QA and multi-turn dialogue to strictly enforce 2:1 ratio."""
    replay = []

    # 1. From Dolly-15k (single-turn QA, general knowledge, creative)
    dolly_path = Path("data/processed/sft/train_sft.jsonl")
    if dolly_path.exists():
        with open(dolly_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if len(d["instruction"]) < 140 and len(d["response"]) < 140 and not d.get("context"):
                    replay.append({
                        "prompt": f"Instruction:\n{d['instruction']}\n\nResponse:\n",
                        "response": d["response"],
                        "category": f"replay_dolly_{d.get('category', 'qa')}"
                    })

    random.seed(42)
    random.shuffle(replay)
    selected_replay = replay[:dolly_count]

    # 2. From Multi-turn Diversified dataset (to preserve multi-turn context reference)
    multiturn_replay = []
    multiturn_path = Path("data/processed/sft_diversified/train_diversified.jsonl")
    if multiturn_path.exists():
        with open(multiturn_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if "Conversation:" in d["prompt"]:
                    multiturn_replay.append({
                        "prompt": d["prompt"],
                        "response": d["response"],
                        "category": "replay_multiturn"
                    })

    random.seed(42)
    random.shuffle(multiturn_replay)
    selected_replay.extend(multiturn_replay[:multiturn_count])

    random.shuffle(selected_replay)
    return selected_replay


def main():
    random.seed(42)
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")

    print("=" * 60)
    print("GENERATING SFT DATASET FOR MEDIA & EXPANDED TOOL CALLING (PHASE 8)")
    print("=" * 60)

    # Audits
    assert not (set(TRAIN_APPS) & HELD_OUT_APPS), "Leakage in app names!"
    assert not (set(TRAIN_QUERIES) & HELD_OUT_QUERIES), "Leakage in queries!"
    assert not (set(TRAIN_VOLUME_LEVELS) & HELD_OUT_VOLUME_LEVELS), "Leakage in volume levels!"
    assert not (set(TRAIN_PLAY_QUERIES) & HELD_OUT_PLAY_QUERIES), "Leakage in play queries!"

    # 180 samples per tool * 14 tools = 2,520 tool samples (33.1%)
    # 5,100 replay samples = 3,400 Dolly QA + 1,700 Multi-turn (66.9%)
    tool_samples = generate_tool_samples(count_per_tool=180)
    replay_samples = load_conversational_replay(dolly_count=3400, multiturn_count=1700)

    all_samples = tool_samples + replay_samples
    random.shuffle(all_samples)

    # Validate token lengths (must fit seq_len 220)
    valid_samples = []
    category_counts = {}
    for s in all_samples:
        full_text = s["prompt"] + s["response"]
        tok_ids = tok.encode(full_text, add_bos=True, add_eos=True)
        if len(tok_ids) <= 220:
            valid_samples.append(s)
            cat = s["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    total_valid = len(valid_samples)
    tool_count = sum(c for k, c in category_counts.items() if k.startswith("tool_"))
    replay_count = sum(c for k, c in category_counts.items() if k.startswith("replay_"))

    print(f"\nTotal Valid Samples: {total_valid:,}")
    print(f"  Tool Samples : {tool_count:,} ({tool_count / total_valid * 100:.1f}%)")
    print(f"  Replay Samples: {replay_count:,} ({replay_count / total_valid * 100:.1f}%)")
    print(f"  Ratio (Replay:Tool): {replay_count / max(1, tool_count):.2f}:1")

    print("\nCategory Breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:32}: {count:5d} ({count / total_valid * 100:.1f}%)")

    # 5% validation split
    n_val = int(total_valid * 0.05)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]

    out_dir = Path("data/processed/sft_tool")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_tool.jsonl", "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s) + "\n")

    with open(out_dir / "val_tool.jsonl", "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved datasets to {out_dir}:")
    print(f"  Train: {len(train_set):,} samples")
    print(f"  Val  : {len(val_set):,} samples")

    # Generate held-out test evaluation set
    eval_dir = Path("eval/data")
    eval_dir.mkdir(parents=True, exist_ok=True)
    held_out_cases = []

    for tool_name, phrasings in HELD_OUT_PHRASINGS.items():
        for p_tmpl in phrasings:
            if tool_name in (
                "Get Time", "Get Date", "Get Battery", "Get Volume", "List Running Apps",
                "Play Media", "Pause Media", "Next Track", "Previous Track"
            ):
                held_out_cases.append({
                    "instruction": p_tmpl,
                    "expected_action": tool_name,
                    "expected_argument": "None",
                    "category": f"held_out_phrasing_{tool_name.lower().replace(' ', '_')}"
                })
            elif tool_name == "Play Query":
                for q in HELD_OUT_PLAY_QUERIES:
                    p = p_tmpl.format(query=q)
                    held_out_cases.append({
                        "instruction": p,
                        "expected_action": "Play Query",
                        "expected_argument": q,
                        "category": "held_out_novel_play_query"
                    })
            elif tool_name == "Open App":
                for app in HELD_OUT_APPS:
                    p = p_tmpl.format(app=app)
                    held_out_cases.append({
                        "instruction": p,
                        "expected_action": "Open App",
                        "expected_argument": app,
                        "category": "held_out_novel_open_app"
                    })
            elif tool_name == "Close App":
                for app in HELD_OUT_APPS:
                    p = p_tmpl.format(app=app)
                    held_out_cases.append({
                        "instruction": p,
                        "expected_action": "Close App",
                        "expected_argument": app,
                        "category": "held_out_novel_close_app"
                    })
            elif tool_name == "Search Web":
                for q in HELD_OUT_QUERIES:
                    p = p_tmpl.format(query=q)
                    held_out_cases.append({
                        "instruction": p,
                        "expected_action": "Search Web",
                        "expected_argument": q,
                        "category": "held_out_novel_search"
                    })
            elif tool_name == "Set Volume":
                for lvl in HELD_OUT_VOLUME_LEVELS:
                    p = p_tmpl.format(level=lvl)
                    held_out_cases.append({
                        "instruction": p,
                        "expected_action": "Set Volume",
                        "expected_argument": lvl,
                        "category": "held_out_novel_set_volume"
                    })

    with open(eval_dir / "held_out_tool_cases.json", "w", encoding="utf-8") as f:
        json.dump(held_out_cases, f, indent=2)

    print(f"  Held-out test cases: {len(held_out_cases)} cases saved to {eval_dir / 'held_out_tool_cases.json'}")


if __name__ == "__main__":
    main()
