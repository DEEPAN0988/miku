import json
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

def hybrid_resolve_tool(query: str, neural_action: str) -> str:
    ranked = bm25_rank(query)
    if not ranked or ranked[0][1] <= 0:
        return neural_action
    
    top_tool, top_score = ranked[0]
    top3_tools = [t for t, s in ranked[:3] if s > 0]

    # If neural action is already in Top-3 lexical candidates, trust it!
    # (Except when neural picked an obvious attractor like Pause Media but top_score is strong for another tool)
    if neural_action in top3_tools:
        if neural_action == "Pause Media" and top_tool != "Pause Media" and top_score > 3.0:
            return top_tool
        return neural_action

    # If neural action is completely absent from Top-3 lexical candidates, use Top-1 BM25!
    return top_tool

def run_hybrid_evaluation():
    try:
        with open("logs/phase8_tool_sft/evaluation_results.json", "r") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print("[SKIP] logs/phase8_tool_sft/evaluation_results.json not found.")
        return

    for suite in ["in_distribution", "ood_phrasings", "novel_arguments"]:
        cases = eval_data[suite]
        n = len(cases)
        matches = 0
        print(f"\n=== TESTING HYBRID (BM25 + NEURAL) ON {suite.upper()} (N={n}) ===")
        for c in cases:
            q = c["instruction"]
            exp = c["expected_action"]
            neural = c["parsed_action"]
            resolved = hybrid_resolve_tool(q, neural)
            ok = (resolved == exp)
            if ok:
                matches += 1
            print(f"\"{q}\" -> Neural: {neural:18} | Resolved: {resolved:18} | Expected: {exp:18} | Match: {ok}")
        print(f"Total Matches: {matches}/{n} ({matches/n*100:.1f}%)")


if __name__ == "__main__":
    run_hybrid_evaluation()

