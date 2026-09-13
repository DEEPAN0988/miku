import math
import re
from collections import Counter
from typing import Dict, List, Tuple

TOOL_DESCRIPTIONS = {
    "Get Time": "time clock hour minute local time current now",
    "Get Date": "date calendar day today month year",
    "Get Battery": "battery charge plugged power laptop level remaining juice",
    "Get Volume": "volume audio sound level speaker loudness mute muted inspect machine master",
    "List Running Apps": "running apps applications software processes programs inventory desktop tasks open active",
    "Play Media": "play media resume playback unpause audio music start playing track sound moving again toggle",
    "Pause Media": "pause playback music audio media freeze halt stop playing track hold tunes",
    "Next Track": "next track song skip advance forward subsequent ahead title",
    "Previous Track": "previous track song go back rewind prior backtrack preceding title last revisit",
    "Play Query": "play song artist album stream listen to queue up specific track music recording bohemian rhapsody cue",
    "Open App": "open launch start application software program execute boot up fire up calculator terminal",
    "Close App": "close exit quit terminate kill process shut down shut off end application software",
    "Search Web": "search web google online internet look up query find information details about weather forecast",
    "Set Volume": "set volume adjust change sound audio level speaker loudness dial tune to percent output"
}

ALL_TEST_SUITES = {
    "in_distribution": [
        ("What time is it right now?", "Get Time"),
        ("What is today's date?", "Get Date"),
        ("What is my battery level?", "Get Battery"),
        ("What is the volume level?", "Get Volume"),
        ("What apps are currently running?", "List Running Apps"),
        ("Play media.", "Play Media"),
        ("Pause the music.", "Pause Media"),
        ("Skip this song.", "Next Track"),
        ("Go back to the previous song.", "Previous Track"),
        ("Play Beethoven Symphony 5.", "Play Query"),
        ("Open the calculator.", "Open App"),
        ("Close terminal.", "Close App"),
        ("Set volume to 50%.", "Set Volume"),
        ("Search the web for weather forecast.", "Search Web"),
    ],
    "ood_phrasings": [
        ("Do you have the time on you?", "Get Time"),
        ("What's on the calendar for today's date?", "Get Date"),
        ("Can you see how much power is remaining in the battery?", "Get Battery"),
        ("Could you inspect the audio level on my machine?", "Get Volume"),
        ("Give me an inventory of open software.", "List Running Apps"),
        ("Get the audio moving again.", "Play Media"),
        ("Put the tunes on hold.", "Pause Media"),
        ("Forward to the subsequent tune.", "Next Track"),
        ("Revisit the preceding title.", "Previous Track"),
        ("Cue up bohemian rhapsody.", "Play Query"),
        ("Fire up the calculator.", "Open App"),
        ("End the terminal process.", "Close App"),
        ("Tune the audio output to 50%.", "Set Volume"),
        ("Search online to find out about weather forecast.", "Search Web"),
    ],
    "novel_arguments": [
        ("Open slack.", "Open App"),
        ("Launch blender.", "Open App"),
        ("Close spotify.", "Close App"),
        ("Quit photoshop.", "Close App"),
        ("Set volume to 45%.", "Set Volume"),
        ("Set volume to silent.", "Set Volume"),
        ("Search the web for quantum computing papers.", "Search Web"),
        ("Look up mars rover latest photos.", "Search Web"),
        ("Play bohemian rhapsody.", "Play Query"),
        ("Listen to cyberpunk ambient soundtrack.", "Play Query"),
        ("Stream chopin nocturne op 9 no 2.", "Play Query"),
    ]
}

def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-zA-Z0-9%]+\b", text.lower()) if len(w) > 1]

corpus = {tool: tokenize(desc) for tool, desc in TOOL_DESCRIPTIONS.items()}
vocab = set(w for doc in corpus.values() for w in doc)
N = len(corpus)

idf = {}
for w in vocab:
    df = sum(1 for doc in corpus.values() if w in doc)
    idf[w] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

k1 = 1.5
b = 0.75
avgdl = sum(len(doc) for doc in corpus.values()) / N

def bm25_rank(query: str) -> List[Tuple[str, float]]:
    q_tokens = tokenize(query)
    scores = {}
    for tool, doc in corpus.items():
        score = 0.0
        doc_len = len(doc)
        counts = Counter(doc)
        for t in q_tokens:
            if t in idf and t in counts:
                tf = counts[t]
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * (doc_len / avgdl))
                score += idf[t] * (num / den)
        scores[tool] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

for suite_name, cases in ALL_TEST_SUITES.items():
    top1_correct = 0
    top3_correct = 0
    print(f"\n=== BM25 EVALUATION: {suite_name.upper()} (N={len(cases)}) ===")
    for q, exp in cases:
        ranked = bm25_rank(q)
        top1 = ranked[0][0] if ranked and ranked[0][1] > 0 else "NONE"
        top3 = [t for t, s in ranked[:3] if s > 0]
        m1 = (top1 == exp)
        m3 = (exp in top3)
        if m1:
            top1_correct += 1
        if m3:
            top3_correct += 1
        print(f"\"{q}\" -> Expected: {exp:18} | Top-1: {top1:18} (Match: {m1}) | Top-3: {m3}")
    print(f"Summary for {suite_name}: Top-1={top1_correct}/{len(cases)} ({top1_correct/len(cases)*100:.1f}%), Top-3 Recall={top3_correct}/{len(cases)} ({top3_correct/len(cases)*100:.1f}%)")
