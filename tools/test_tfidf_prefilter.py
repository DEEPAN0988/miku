import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

# Descriptions for all 14 tools
TOOL_DESCRIPTIONS = {
    "Get Time": "get the current system time clock hour minute local time right now",
    "Get Date": "get today's calendar date day of week month year today",
    "Get Battery": "check laptop battery level percentage remaining power charge plugged in ac power juice",
    "Get Volume": "inspect audio sound volume master speaker level loudness muted sound level",
    "List Running Apps": "list open software running applications active processes desktop programs inventory tasks",
    "Play Media": "play media resume playback unpause audio music start playing track sound",
    "Pause Media": "pause playback music audio media freeze halt stop playing track hold",
    "Next Track": "skip to next track song advance forward skip ahead next audio track title",
    "Previous Track": "previous track song go back rewind prior song backtrack preceding title last",
    "Play Query": "play song artist album stream listen to queue up specific track music recording bohemian rhapsody",
    "Open App": "open launch start application software program execute boot up fire up calculator terminal",
    "Close App": "close exit quit terminate kill process shut down shut off end application software",
    "Search Web": "search web google online internet look up query find information details about topic",
    "Set Volume": "set volume adjust change sound audio level speaker loudness dial tune to percent mute"
}

OOD_QUERIES = [
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
]

def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-zA-Z0-9%]+\b", text.lower()) if len(w) > 1]

def run_tfidf_test():
    # Build corpus
    corpus = {tool: tokenize(desc) for tool, desc in TOOL_DESCRIPTIONS.items()}
    vocab = set(w for doc in corpus.values() for w in doc)
    N = len(corpus)

    # Compute IDF
    idf = {}
    for w in vocab:
        df = sum(1 for doc in corpus.values() if w in doc)
        idf[w] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avgdl = sum(len(doc) for doc in corpus.values()) / N

    def bm25_score(query: str) -> List[Tuple[str, float]]:
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

    print("=== BM25 / TF-IDF SIMILARITY TEST ON 14 OOD QUERIES ===")
    top1_correct = 0
    top3_correct = 0

    for q, expected in OOD_QUERIES:
        ranked = bm25_score(q)
        top1 = ranked[0][0] if ranked and ranked[0][1] > 0 else "NONE"
        top3 = [t for t, s in ranked[:3] if s > 0]
        in_top1 = (top1 == expected)
        in_top3 = (expected in top3)
        if in_top1:
            top1_correct += 1
        if in_top3:
            top3_correct += 1
        print(f"Query: \"{q}\"")
        print(f"  Expected: {expected}")
        print(f"  Top 1: {top1} (Score: {ranked[0][1]:.2f}) -> Match: {in_top1}")
        print(f"  Top 3: {[(t, round(s, 2)) for t, s in ranked[:3]]} -> In Top 3: {in_top3}")
        print("-" * 60)

    print(f"Top-1 Accuracy: {top1_correct}/14 ({top1_correct/14*100:.1f}%)")
    print(f"Top-3 Recall:   {top3_correct}/14 ({top3_correct/14*100:.1f}%)")


if __name__ == "__main__":
    run_tfidf_test()

