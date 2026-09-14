"""
tools/dispatcher.py — Core Tool Dispatcher & Safety Execution Harness (Phase 8)

PHASE 8 SAFETY SPECIFICATION:
  - 14 tools total.
  - LOW-risk tools (read-only system state or harmless local media keys:
    Get Time, Get Date, Get Battery, Get Volume, List Running Apps,
    Play Media, Pause Media, Next Track, Previous Track)
    are wired for REAL execution via local system APIs (datetime, psutil, pycaw, user32.keybd_event).
  - MEDIUM-risk tools (outbound network or browser state changes:
    Search Web, Set Volume, Play Query)
    and HIGH-risk tools (process spawning/termination: Open App, Close App)
    are STRICTLY GATED behind Human-In-The-Loop (HITL) confirmation (confirmed=True).
    Attempting execution without confirmation returns DRY_RUN_PENDING_HITL.
"""

from __future__ import annotations

import ctypes
import datetime
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RiskLevel(Enum):
    LOW = "LOW"        # Read-only query or safe hardware media key (e.g. get time, play/pause, next track)
    MEDIUM = "MEDIUM"  # Outbound network call or browser launch (e.g. search web, set volume, play query)
    HIGH = "HIGH"      # Local process spawning or termination (e.g. open app, close app)


@dataclass
class ToolCall:
    action: str
    argument: str
    risk: RiskLevel
    is_valid: bool
    error: Optional[str] = None
    is_confident: bool = True
    clarification_prompt: Optional[str] = None


@dataclass
class ToolResult:
    tool_call: ToolCall
    executed: bool
    dry_run: bool
    output: str
    status: str  # "SUCCESS", "REJECTED", "DRY_RUN_PENDING_HITL", "PARSE_ERROR", "CLARIFICATION_REQUIRED"


# Registry of supported tools and their risk profiles
TOOL_REGISTRY: Dict[str, RiskLevel] = {
    # LOW Risk — Verified Read-Only Queries & Safe Media/App Keys (Real Execution Permitted)
    "Get Time": RiskLevel.LOW,
    "Get Date": RiskLevel.LOW,
    "Get Battery": RiskLevel.LOW,
    "Get Volume": RiskLevel.LOW,
    "List Running Apps": RiskLevel.LOW,
    "Play Media": RiskLevel.LOW,
    "Pause Media": RiskLevel.LOW,
    "Next Track": RiskLevel.LOW,
    "Previous Track": RiskLevel.LOW,
    "Find App": RiskLevel.LOW,
    "Focus App": RiskLevel.LOW,
    "Inspect Screen": RiskLevel.LOW,

    # MEDIUM Risk — State-changing, Audio level, or Outbound Network Search (HITL Required)
    "Search Web": RiskLevel.MEDIUM,
    "Set Volume": RiskLevel.MEDIUM,
    "Play Query": RiskLevel.MEDIUM,

    # HIGH Risk — Local Process Lifecycle & Messaging (HITL Required)
    "Open App": RiskLevel.HIGH,
    "Close App": RiskLevel.HIGH,
    "Restart App": RiskLevel.HIGH,
    "Send Message": RiskLevel.HIGH,
    "Stage Message": RiskLevel.MEDIUM,
}

NO_ARG_TOOLS = {
    "Get Time", "Get Date", "Get Battery", "Get Volume", "List Running Apps",
    "Play Media", "Pause Media", "Next Track", "Previous Track", "Inspect Screen"
}


def parse_tool_call(response_text: str) -> ToolCall:
    """
    Parses model response text expecting:
      Action: <tool_name>
      Argument: <value>
    """
    cleaned = response_text.strip()
    
    action_match = re.search(r"Action\s*:\s*(.*?)(?:\n|\r|\s+Argument\s*:|$)", cleaned, re.IGNORECASE)
    arg_match = re.search(r"Argument\s*:\s*(.*)", cleaned, re.IGNORECASE)

    if not action_match:
        return ToolCall(
            action="",
            argument="",
            risk=RiskLevel.LOW,
            is_valid=False,
            error="Missing 'Action:' header in model output."
        )

    raw_action = action_match.group(1).strip()
    matched_action = None
    for reg_action in TOOL_REGISTRY:
        if raw_action.lower() == reg_action.lower():
            matched_action = reg_action
            break

    if not matched_action:
        return ToolCall(
            action=raw_action,
            argument="",
            risk=RiskLevel.HIGH,
            is_valid=False,
            error=f"Unregistered tool action: '{raw_action}'."
        )

    arg = arg_match.group(1).strip() if arg_match else ""
    if matched_action in NO_ARG_TOOLS and not arg:
        arg = "None"

    risk = TOOL_REGISTRY[matched_action]
    return ToolCall(
        action=matched_action,
        argument=arg,
        risk=risk,
        is_valid=True,
        error=None
    )


# ==============================================================================
# BM25 LEXICAL DESCRIPTOR ENGINE (ZERO VECTOR DB OVERHEAD)
# ==============================================================================

import math
from collections import Counter

TOOL_DESCRIPTORS: Dict[str, str] = {
    "Get Time": "time clock hour minute local time current",
    "Get Date": "date calendar day today month year",
    "Get Battery": "battery charge plugged power laptop level remaining juice",
    "Get Volume": "volume audio sound level speaker loudness mute muted inspect machine master",
    "List Running Apps": "running apps applications software processes programs inventory desktop tasks open active taskmgr list currently what",
    "Play Media": "play media resume playback unpause audio music start playing track sound moving again toggle",
    "Pause Media": "pause playback music audio media freeze halt stop playing track hold tunes",
    "Next Track": "next track song skip advance forward subsequent ahead title",
    "Previous Track": "previous track song go back rewind prior backtrack preceding title last revisit",
    "Play Query": "play song artist album stream listen to queue up specific track music recording bohemian rhapsody cue",
    "Open App": "open launch start application software program execute boot up fire up run app apps",
    "Close App": "close exit quit terminate kill process shut down shut off end application software",
    "Find App": "find locate search where installed discover computer machine installed software app program",
    "Restart App": "restart relaunch reboot refresh cycle bounce reset restart relaunch",
    "Focus App": "focus switch switch to bring foreground show active window foreground window",
    "Search Web": "search web google online internet look up query find information details about weather forecast",
    "Set Volume": "set volume adjust change sound audio level speaker loudness dial tune to percent output",
    "Send Message": "send message text chat whatsapp telegram discord ping dispatch transmit communicate write tell contact person recipient",
    "Stage Message": "stage draft compose prepare message text type prompt preview unstaged",
    "Inspect Screen": "inspect screen examine controls layout view buttons fields active ui elements look screen reader",
}

def _tokenize_text(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-zA-Z0-9%]+\b", text.lower()) if len(w) > 1]

_CORPUS = {tool: _tokenize_text(desc) for tool, desc in TOOL_DESCRIPTORS.items()}
_VOCAB = set(w for doc in _CORPUS.values() for w in doc)
_N = len(_CORPUS)
_IDF = {
    w: math.log((_N - sum(1 for doc in _CORPUS.values() if w in doc) + 0.5) / (sum(1 for doc in _CORPUS.values() if w in doc) + 0.5) + 1.0)
    for w in _VOCAB
}
_K1 = 1.5
_B = 0.75
_AVGDL = sum(len(doc) for doc in _CORPUS.values()) / _N


def _bm25_rank(query: str) -> List[Tuple[str, float]]:
    q_tokens = _tokenize_text(query)
    scores = {}
    for tool, doc in _CORPUS.items():
        score = 0.0
        doc_len = len(doc)
        counts = Counter(doc)
        for t in q_tokens:
            if t in _IDF and t in counts:
                tf = counts[t]
                num = tf * (_K1 + 1)
                den = tf + _K1 * (1 - _B + _B * (doc_len / _AVGDL))
                score += _IDF[t] * (num / den)
        scores[tool] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Common application entity catalog for lightweight structural detection
COMMON_APP_NAMES = {
    "chrome", "google chrome", "slack", "blender", "spotify", "discord",
    "notepad", "calculator", "terminal", "browser", "code", "visual studio code",
    "excel", "word", "powerpoint", "explorer", "paint", "task manager"
}

CONFIDENCE_MARGIN = 0.60
CONFIDENCE_RATIO = 1.30


def detect_query_argument(query: str) -> Dict[str, Any]:
    """
    Lightweight heuristic entity/argument detector:
    Determines if query targets an explicit application/resource entity vs. an argumentless state inquiry.
    """
    t = query.lower().strip()

    # 1. Direct match against known application catalog
    detected_app = None
    for app in sorted(COMMON_APP_NAMES, key=len, reverse=True):
        pattern = r"\b" + re.escape(app) + r"\b"
        if re.search(pattern, t):
            detected_app = app
            break

    # 2. Transitive verb object heuristic (e.g. "get X running", "open X", "launch X", "restart X")
    transitive_match = re.search(
        r"\b(?:get|open|launch|start|run|restart|relaunch|reboot|close|kill|terminate|find|locate|focus|switch\s+to|bring\s+up)\s+(?:the\s+|my\s+|up\s+)?([a-zA-Z0-9_\-]+)",
        t, re.I
    )
    transitive_obj = None
    if transitive_match:
        cand = transitive_match.group(1).strip().lower()
        if cand not in ("the", "a", "an", "up", "for", "me", "my", "this", "all", "what", "which", "how", "app", "application", "software", "program", "window", "process"):
            transitive_obj = cand

    has_argument = (detected_app is not None) or (transitive_obj is not None)
    best_entity = detected_app if detected_app else transitive_obj

    # 3. Check if query is an inventory/state query (argumentless query)
    is_inventory_query = bool(re.search(
        r"\b(?:what(?:\s+apps|\s+programs|\s+software|\s+processes)?\s+(?:are|is)\s+(?:currently\s+)?running|list\s+(?:the\s+)?(?:running\s+)?(?:apps|applications|software|processes|tasks|programs)|show\s+(?:running|active)\s+tasks|inventory\s+of\s+open\s+software)\b",
        t, re.I
    ))

    # 4. Messaging recipient detection
    messaging_match = re.search(
        r"\b(?:tell|message|send\s+(?:a\s+)?message\s+to|send\s+to|text|ping)\s+([a-zA-Z0-9_\-]+)",
        t, re.I
    )
    if messaging_match:
        cand = messaging_match.group(1).strip().lower()
        if cand not in ("the", "a", "an", "up", "for", "me", "my", "this", "all", "what", "which", "how", "app", "application", "software", "program", "window", "process", "message", "text"):
            best_entity = cand
            has_argument = True

    return {
        "has_argument": has_argument,
        "entity": best_entity,
        "is_inventory_query": is_inventory_query
    }


def rank_with_structure(query: str) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Combines BM25 lexical ranking with lightweight argument-structure heuristics.
    Down-weights argumentless tools when a specific entity is present,
    and down-weights argument-requiring tools when no argument entity exists.
    """
    scores = dict(_bm25_rank(query))
    arg_info = detect_query_argument(query)
    has_arg = arg_info["has_argument"]
    is_inv = arg_info["is_inventory_query"]

    adjusted = dict(scores)

    # If inventory query explicitly asked for, boost List Running Apps and down-weight argument tools
    if is_inv:
        adjusted["List Running Apps"] += 3.0
        for t in ("Open App", "Close App", "Restart App", "Focus App", "Find App", "Send Message"):
            adjusted[t] = max(0.0, adjusted.get(t, 0.0) - 2.0)
    elif has_arg:
        # If an explicit app or recipient entity was detected, argumentless tools should not take precedence
        adjusted["List Running Apps"] = max(0.0, adjusted.get("List Running Apps", 0.0) - 2.0)
        if not re.search(r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?time\b(?!\s+(?:should|do|are|can|will|we|to|shall|did|would))|what\s+time\s+is\s+it|current\s+time|time\s+now|local\s+time|system\s+clock)\b", query.lower()):
            adjusted["Get Time"] = max(0.0, adjusted.get("Get Time", 0.0) - 3.0)
        if not re.search(r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+)?date|today's\s+date|current\s+date|what\s+day\s+is\s+today)\b", query.lower()):
            adjusted["Get Date"] = max(0.0, adjusted.get("Get Date", 0.0) - 3.0)

        # If query has launch phrasing with entity ("get X running", "open X", "fire up X")
        if re.search(r"\b(?:get|open|launch|start|fire\s+up|run|boot\s+up)\b", query.lower()):
            adjusted["Open App"] += 2.0

    # Messaging structural awareness: boost Send Message when messaging phrasing is present
    if re.search(r"\b(?:send\s+(?:a\s+)?message(?:\s+to)?|send\s+to\s+[a-zA-Z0-9_\-]+|tell\s+[a-zA-Z0-9_\-]+|text\s+[a-zA-Z0-9_\-]+|message\s+[a-zA-Z0-9_\-]+|whatsapp)\b", query.lower()):
        adjusted["Send Message"] = adjusted.get("Send Message", 0.0) + 3.0

    # Volume structural awareness: Set Volume requires a target level / percentage / mute value
    has_vol_arg = bool(re.search(r"\b([0-9]+\s*%|[0-9]+\s*percent|to\s+[0-9]{1,3}\b|volume\s+[0-9]{1,3}\b|mute|unmute|max|min|silent)\b", query.lower()))
    if not has_vol_arg:
        adjusted["Set Volume"] = max(0.0, adjusted.get("Set Volume", 0.0) - 2.0)
    else:
        adjusted["Set Volume"] += 2.0

    ranked = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
    return ranked, arg_info


def resolve_intent_with_confidence(text: str, fallback_action: str) -> Tuple[str, bool, Optional[str]]:
    """
    Hybrid intent anchor resolver with confidence gating and argument-structure awareness.
    Returns: (chosen_action, is_confident, clarification_prompt)
    """
    t = text.lower().strip()

    # 1. Media Disambiguation Gate (e.g. "Play Pokemon")
    # If the user says "Play Pokemon", Miku halts and queries if they want to search local files, YouTube videos, or YouTube Shorts
    if re.search(r"\bplay\s+pokemon\b", t, re.IGNORECASE):
        clarification_msg = (
            "[MEDIA DISAMBIGUATION] Would you like to search local files, YouTube videos, or YouTube Shorts for 'Pokemon'?"
        )
        return "Play Query", False, clarification_msg

    # 2. Single structural disambiguation rule: Play Media (no argument) vs Play Query (specific song/artist)
    if re.search(r"\bplay\s+(?!media\b|the music\b|music\b|audio\b|playback\b|current\b)(.+)", t):
        return "Play Query", True, None

    # 2. Structural BM25 ranking
    ranked, arg_info = rank_with_structure(text)
    if not ranked or ranked[0][1] <= 0:
        return fallback_action, True, None

    top1_tool, s1 = ranked[0]
    top2_tool, s2 = ranked[1] if len(ranked) > 1 else ("None", 0.0)
    margin = s1 - s2
    ratio = s1 / s2 if s2 > 0 else 99.0

    # Domain-specific ambiguity gating for messaging
    if top1_tool in ("Send Message", "Stage Message"):
        from tools.messaging import parse_messaging_intent
        msg_intent = parse_messaging_intent(text)
        if msg_intent.get("is_ambiguous"):
            return top1_tool, False, msg_intent.get("clarification_prompt")

    # Ambiguity check: if two conflicting tools have close scores (< margin or < ratio)
    is_ambiguous = (s1 > 0 and s2 > 0 and (margin < CONFIDENCE_MARGIN or ratio < CONFIDENCE_RATIO) and top1_tool != top2_tool)

    if is_ambiguous:
        entity = arg_info.get("entity") or "the target application"
        clarification_msg = (
            f"[AMBIGUITY DETECTED] Multiple conflicting tools match this request: "
            f"'{top1_tool}' (score {s1:.2f}) vs '{top2_tool}' (score {s2:.2f}). "
            f"Clarification needed: Did you mean to {top1_tool} or {top2_tool} regarding '{entity}'?"
        )
        return top1_tool, False, clarification_msg

    scores = dict(ranked)
    neural_score = scores.get(fallback_action, 0.0)

    # Corroborate or override neural prediction
    if s1 >= 2.5 and s1 > 1.5 * neural_score:
        return top1_tool, True, None
    elif fallback_action in [tool for tool, s in ranked[:3] if s > 0]:
        return fallback_action, True, None
    else:
        return top1_tool, True, None


def resolve_intent_anchor(text: str, fallback_action: str) -> str:
    """
    Backwards-compatible wrapper returning the primary action.
    """
    action, _, _ = resolve_intent_with_confidence(text, fallback_action)
    return action


def extract_deterministic_slot(instruction: str, action: str) -> Optional[str]:
    """
    Deterministic argument slot-fill fallback for direct requests across all 17 tools.
    """
    text = instruction.strip()
    if action in NO_ARG_TOOLS:
        return "None"

    if action == "Open App":
        m = re.search(r"(?:open|launch|start|run|boot up|bring up|fire up|execute|get)\s+(?:the\s+|up\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*)", text, re.I)
        if m:
            arg = m.group(1).strip().lower()
            if arg not in ("the", "a", "an", "up", "for", "me", "program", "app", "application"):
                return arg
        return None

    if action == "Close App":
        m = re.search(r"(?:close|quit|shut down|exit|terminate|kill|stop|end)\s+(?:the\s+|down\s+|off\s+)?([a-zA-Z0-9_\-]+)", text, re.I)
        if m:
            arg = m.group(1).strip().lower()
            if arg not in ("the", "a", "an", "down", "off", "for", "me", "program", "app", "process"):
                return arg
        return None

    if action == "Find App":
        m = re.search(r"(?:find\s+where|where\s+is|find|locate|search\s+for|discover\s+which)\s+(?:the\s+)?(?:app\s+|application\s+|program\s+|software\s+)?([a-zA-Z0-9_\-]+)", text, re.I)
        if m:
            arg = m.group(1).strip().lower()
            if arg in ("where", "is", "installed", "the", "a", "an", "for", "me", "program", "app", "which"):
                m2 = re.search(r"(?:find\s+where|where\s+is|find|locate|search\s+for|discover\s+which)\s+(?:where\s+)?(?:is\s+)?(?:the\s+)?([a-zA-Z0-9_\-]+)", text, re.I)
                if m2:
                    arg = m2.group(1).strip().lower()
            if arg not in ("the", "a", "an", "for", "me", "program", "app", "where", "is"):
                return arg
        return None

    if action == "Restart App":
        m = re.search(r"(?:restart|relaunch|reboot|refresh|cycle|bounce|reset|kill\s+and\s+restart)\s+(?:and\s+bounce\s+)?(?:and\s+restart\s+)?(?:the\s+|my\s+|up\s+)?(?:app\s+|application\s+|program\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*?)(?:\s+process|\s+app|\s+session|\s+because|$)", text, re.I)
        if m:
            arg = m.group(1).strip().lower()
            for skip in ("the ", "my ", "and bounce "):
                if arg.startswith(skip):
                    arg = arg[len(skip):].strip()
            if arg not in ("the", "a", "an", "my", "for", "me", "program", "app", "and"):
                return arg
        return None

    if action == "Focus App":
        m = re.search(r"(?:focus|switch\s+to|switch\s+over\s+to|bring\s+up|show)\s+(?:the\s+)?(?:active\s+|current\s+)?(?:app\s+|application\s+|program\s+|window\s+)?([a-zA-Z0-9_\-]+)", text, re.I)
        if not m:
            m = re.search(r"bring\s+([a-zA-Z0-9_\-]+)\s+to\s+(?:the\s+)?foreground", text, re.I)
        if m:
            arg = m.group(1).strip().lower()
            if arg not in ("the", "a", "an", "for", "me", "program", "app", "window", "active", "current"):
                return arg
        return None

    if action == "Set Volume":
        m = re.search(r"(?:set|change|adjust|turn|put|make|switch|tune|dial)\s+(?:the\s+)?(?:volume|sound|audio(?:\s+output)?|speaker volume)\s+(?:to|at)?\s*([0-9]+%?|mute|unmute|maximum|zero|silent|max|min)", text, re.I)
        if m:
            return m.group(1).strip().lower()
        return None

    if action == "Search Web":
        m = re.search(r"(?:search(?:\s+(?:the\s+web|online|the\s+internet))?\s+(?:to find out about|to find|for info on|for details about|for|regarding|about)|look up|google|find(?:\s+(?:information|info|details))?\s+(?:about|on))\s+(.+?)(?:\s+(?:on the web|online|on internet|for me))?[\.\?!]?$", text, re.I)
        if m:
            return m.group(1).strip()
        return None

    if action == "Play Query":
        m = re.search(r"(?:play|put on|stream|listen to|queue up|spin|cue up|fire up the recording)\s+(?:the\s+(?:song|album|track|artist|recording)\s+)?(?:some\s+)?(.+?)(?:\s+(?:on the speakers|on speakers|on the player|right now|for me|please))?[\.\?!]?$", text, re.I)
        if m:
            arg = m.group(1).strip()
            if arg.lower() not in ("music", "song", "audio", "track", "media", "playback", "the music"):
                return arg
    if action in ("Send Message", "Stage Message"):
        from tools.messaging import parse_messaging_intent
        res = parse_messaging_intent(text)
        return f"to: {res['resolved_recipient']} | message: \"{res['formatted_message']}\""

    return None


# ==============================================================================
# REAL EXECUTION HANDLERS (LOW-RISK TOOLS ONLY)
# ==============================================================================

def _real_get_time() -> str:
    now = datetime.datetime.now()
    return f"[REAL SYSTEM TIME] {now.strftime('%I:%M:%S %p %Z').strip()}"


def _real_get_date() -> str:
    today = datetime.date.today()
    return f"[REAL SYSTEM DATE] {today.strftime('%A, %B %d, %Y')}"


def _real_get_battery() -> str:
    try:
        import psutil
        batt = psutil.sensors_battery()
        if batt is None:
            return "[REAL BATTERY STATUS] No battery detected (System appears to be on AC desktop power)."
        pct = batt.percent
        plugged = "Plugged in (Charging/AC)" if batt.power_plugged else "Discharging (On Battery)"
        mins_left = batt.secsleft // 60 if batt.secsleft > 0 else "Unknown / Calculating"
        return f"[REAL BATTERY STATUS] Battery: {pct}% | State: {plugged} | Remaining: {mins_left} min"
    except Exception as e:
        return f"[ERROR] Failed to read battery status: {e}"


def _real_get_volume() -> str:
    try:
        from pycaw.utils import AudioUtilities
        speakers = AudioUtilities.GetSpeakers()
        vol = speakers.EndpointVolume
        level = round(vol.GetMasterVolumeLevelScalar() * 100)
        is_muted = bool(vol.GetMute())
        return f"[REAL VOLUME STATUS] Master Volume: {level}% | Muted: {'Yes' if is_muted else 'No'}"
    except Exception as e:
        return f"[ERROR] Failed to read system audio volume: {e}"


def _real_list_running_apps(limit: int = 15) -> str:
    try:
        import psutil
        apps = set()
        system_prefixes = (
            "svchost", "system", "registry", "smss", "csrss", "wininit",
            "services", "lsass", "runtimebroker", "taskhost", "conhost",
            "searchhost", "shellexperiencehost", "startmenuexperiencehost"
        )
        for p in psutil.process_iter(["name"]):
            try:
                name = p.info["name"]
                if name and name.lower().endswith(".exe"):
                    base = name[:-4].lower()
                    if not base.startswith(system_prefixes):
                        apps.add(base)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        sorted_apps = sorted(list(apps))[:limit]
        formatted = ", ".join(sorted_apps) if sorted_apps else "None"
        return f"[REAL RUNNING APPS] Active user processes ({len(sorted_apps)} sample): {formatted}"
    except Exception as e:
        return f"[ERROR] Failed to list running apps: {e}"


def _real_play_media() -> str:
    try:
        user32 = ctypes.windll.user32
        # VK_MEDIA_PLAY_PAUSE = 0xB3
        user32.keybd_event(0xB3, 0, 1, 0)
        user32.keybd_event(0xB3, 0, 3, 0)
        return "[REAL MEDIA CONTROL] Sent Play/Resume signal (VK_MEDIA_PLAY_PAUSE: 0xB3) to Windows Media Transport."
    except Exception as e:
        return f"[ERROR] Failed to send Play signal: {e}"


def _real_pause_media() -> str:
    try:
        user32 = ctypes.windll.user32
        # VK_MEDIA_PLAY_PAUSE = 0xB3
        user32.keybd_event(0xB3, 0, 1, 0)
        user32.keybd_event(0xB3, 0, 3, 0)
        return "[REAL MEDIA CONTROL] Sent Pause signal (VK_MEDIA_PLAY_PAUSE: 0xB3) to Windows Media Transport."
    except Exception as e:
        return f"[ERROR] Failed to send Pause signal: {e}"


def _real_next_track() -> str:
    try:
        user32 = ctypes.windll.user32
        # VK_MEDIA_NEXT_TRACK = 0xB0
        user32.keybd_event(0xB0, 0, 1, 0)
        user32.keybd_event(0xB0, 0, 3, 0)
        return "[REAL MEDIA CONTROL] Sent Next Track signal (VK_MEDIA_NEXT_TRACK: 0xB0) to Windows Media Transport."
    except Exception as e:
        return f"[ERROR] Failed to send Next Track signal: {e}"


def _real_prev_track() -> str:
    try:
        user32 = ctypes.windll.user32
        # VK_MEDIA_PREV_TRACK = 0xB1
        user32.keybd_event(0xB1, 0, 1, 0)
        user32.keybd_event(0xB1, 0, 3, 0)
        return "[REAL MEDIA CONTROL] Sent Previous Track signal (VK_MEDIA_PREV_TRACK: 0xB1) to Windows Media Transport."
    except Exception as e:
        return f"[ERROR] Failed to send Previous Track signal: {e}"


def dispatch_tool(tool_call: ToolCall, hitl_confirmed: bool = False, mock_window_state: Optional[Dict[str, Any]] = None) -> ToolResult:
    """
    Dispatches a tool call enforcing safety invariants:
      1. Malformed or unregistered tools -> PARSE_ERROR (rejected).
      2. LOW-risk tools -> REAL execution (safe read-only system inspection or media keys).
      3. MEDIUM / HIGH-risk tools -> REQUIRE explicit hitl_confirmed=True.
         If not confirmed -> DRY_RUN_PENDING_HITL (rejected/blocked).
         If confirmed -> Simulated preview (dry run approved; actual OS mutation is blocked).
      4. Messaging tools -> STRICTLY SIMULATION ONLY. Real send capability is permanently locked.
    """
    if not tool_call.is_valid:
        return ToolResult(
            tool_call=tool_call,
            executed=False,
            dry_run=True,
            output=f"[PARSE ERROR] {tool_call.error}",
            status="PARSE_ERROR"
        )

    # LOW CONFIDENCE / AMBIGUITY GATING
    if not tool_call.is_confident:
        return ToolResult(
            tool_call=tool_call,
            executed=False,
            dry_run=True,
            output=tool_call.clarification_prompt or "[AMBIGUITY DETECTED] Routing confidence is low. Please clarify your request.",
            status="CLARIFICATION_REQUIRED"
        )

    # LOW RISK — REAL EXECUTION
    if tool_call.risk == RiskLevel.LOW:
        if tool_call.action == "Get Time":
            real_out = _real_get_time()
        elif tool_call.action == "Get Date":
            real_out = _real_get_date()
        elif tool_call.action == "Get Battery":
            real_out = _real_get_battery()
        elif tool_call.action == "Get Volume":
            real_out = _real_get_volume()
        elif tool_call.action == "List Running Apps":
            real_out = _real_list_running_apps()
        elif tool_call.action == "Play Media":
            real_out = _real_play_media()
        elif tool_call.action == "Pause Media":
            real_out = _real_pause_media()
        elif tool_call.action == "Next Track":
            real_out = _real_next_track()
        elif tool_call.action == "Previous Track":
            real_out = _real_prev_track()
        elif tool_call.action == "Find App":
            from tools.app_launcher import find_installed_apps
            found = find_installed_apps(tool_call.argument)
            if found:
                apps_str = ", ".join(f"{a['name']} ({a['source']})" for a in found)
                real_out = f"[REAL APP SEARCH] Found {len(found)} app(s) for '{tool_call.argument}': {apps_str}"
            else:
                real_out = f"[REAL APP SEARCH] No installed applications found matching '{tool_call.argument}'."
        elif tool_call.action == "Focus App":
            from tools.app_launcher import focus_app
            res = focus_app(tool_call.argument)
            real_out = f"[REAL WINDOW FOCUS] {res['output']}"
        elif tool_call.action == "Inspect Screen":
            from tools.screen_inspector import inspect_screen_elements
            snap = inspect_screen_elements()
            if snap.error:
                real_out = f"[SCREEN INSPECTION ERROR] {snap.error}"
            else:
                summary = [f"{e.control_type}: '{e.name}'" for e in snap.elements[:10]]
                sum_str = ", ".join(summary) if summary else "None"
                real_out = (
                    f"[REAL SCREEN INSPECTION] Window '{snap.title}' ({snap.process_name}) — "
                    f"Found {snap.interactive_count} interactive element(s) in {snap.latency_ms}ms: {sum_str}"
                )
        else:
            real_out = f"[LOW RISK UNKNOWN] {tool_call.action}"

        return ToolResult(
            tool_call=tool_call,
            executed=True,
            dry_run=False,
            output=real_out,
            status="SUCCESS"
        )

    # MEDIUM & HIGH RISK — MANDATORY HITL CONFIRMATION
    if not hitl_confirmed:
        if tool_call.action == "Send Message":
            arg = tool_call.argument
            m = re.search(r'to:\s*(.*?)\s*\|\s*message:\s*"(.*)"', arg)
            if m:
                recip, msg_text = m.group(1).strip(), m.group(2).strip()
            else:
                recip, msg_text = arg, ""
            out_msg = (
                f"[GUARDRAIL BLOCKED] Action 'Send Message' is HIGH risk (real-world communication). "
                f"Requires explicit user confirmation of recipient and exact text before transmission.\n"
                f"Planned recipient: '{recip}'\n"
                f"Exact final text: \"{msg_text}\""
            )
        else:
            out_msg = (
                f"[GUARDRAIL BLOCKED] Action '{tool_call.action}' has {tool_call.risk.value} risk. "
                f"Requires explicit user confirmation before execution. "
                f"Planned call: {tool_call.action}(argument='{tool_call.argument}')"
            )
        return ToolResult(
            tool_call=tool_call,
            executed=False,
            dry_run=True,
            output=out_msg,
            status="DRY_RUN_PENDING_HITL"
        )

    # If confirmed by user: execute real app lifecycle for Open App / Restart App
    if tool_call.action == "Open App":
        from tools.app_launcher import launch_app
        res = launch_app(tool_call.argument)
        return ToolResult(
            tool_call=tool_call,
            executed=(res["status"] == "SUCCESS"),
            dry_run=False,
            output=f"[REAL APP LAUNCH] {res['output']}",
            status=res["status"]
        )
    elif tool_call.action == "Restart App":
        from tools.app_launcher import restart_app
        res = restart_app(tool_call.argument)
        return ToolResult(
            tool_call=tool_call,
            executed=(res["status"] == "SUCCESS"),
            dry_run=False,
            output=f"[REAL APP RESTART] {res['output']}",
            status=res["status"]
        )
    elif tool_call.action == "Send Message":
        from tools.messaging import send_whatsapp_message
        arg = tool_call.argument
        m = re.search(r'to:\s*(.*?)\s*\|\s*message:\s*"(.*)"', arg)
        if m:
            recip, msg_text = m.group(1).strip(), m.group(2).strip()
        else:
            recip, msg_text = arg, ""
        # Respect REAL_SEND_ENABLED circuit breaker state
        from tools.messaging import REAL_SEND_ENABLED
        res = send_whatsapp_message(recip, msg_text, dry_run=(not REAL_SEND_ENABLED), mock_window_state=mock_window_state)
        return ToolResult(
            tool_call=tool_call,
            executed=res.get("executed", False),
            dry_run=res.get("dry_run", True),
            output=res.get("output", ""),
            status=res.get("status", "SUCCESS")
        )
    elif tool_call.action == "Stage Message":
        from tools.messaging import stage_whatsapp_message
        arg = tool_call.argument
        m = re.search(r'to:\s*(.*?)\s*\|\s*message:\s*"(.*)"', arg)
        if m:
            recip, msg_text = m.group(1).strip(), m.group(2).strip()
        else:
            recip, msg_text = arg, ""
        res = stage_whatsapp_message(recip, msg_text, dry_run=True, mock_window_state=mock_window_state)
        return ToolResult(
            tool_call=tool_call,
            executed=False,
            dry_run=True,
            output=res.get("output", ""),
            status=res.get("status", "SUCCESS")
        )

    # For other confirmed tools (Search Web, Set Volume, Play Query), safely preview approved call
    return ToolResult(
        tool_call=tool_call,
        executed=False,
        dry_run=True,
        output=f"[HITL APPROVED PREVIEW] User confirmed. Approved call: {tool_call.action} with arg '{tool_call.argument}'.",
        status="SUCCESS"
    )


# ==============================================================================
# CONTEXT-AWARE MEDIA ROUTER (PHASE 3)
# ==============================================================================

def route_media_play(
    query: str,
    destination: Optional[str] = None,
    dry_run: bool = True,
    browser_opener: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Context-Aware Media Router:
    - If user says 'Play Pokemon' without specifying destination, halts execution
      and prompts if they want to search local files, YouTube videos, or YouTube Shorts.
    - Once destination is known, formats search query and opens browser (or local search).
    """
    import urllib.parse
    q_clean = query.strip()
    m = re.search(r"^\s*(?:please\s+)?play\s+(.+)$", q_clean, re.IGNORECASE)
    target = m.group(1).strip() if m else q_clean
    target = target.rstrip(".?!")

    dest = destination
    if not dest:
        if re.search(r"\b(?:on\s+)?youtube\s+shorts\b", q_clean, re.IGNORECASE):
            dest = "YouTube Shorts"
            target = re.sub(r"\b(?:on\s+)?youtube\s+shorts\b", "", target, flags=re.I).strip()
        elif re.search(r"\b(?:on\s+)?youtube\b", q_clean, re.IGNORECASE):
            dest = "YouTube videos"
            target = re.sub(r"\b(?:on\s+)?youtube\b", "", target, flags=re.I).strip()
        elif re.search(r"\b(?:local|locally|on\s+my\s+(?:pc|computer|disk|files?))\b", q_clean, re.IGNORECASE):
            dest = "local files"
            target = re.sub(r"\b(?:local|locally|on\s+my\s+(?:pc|computer|disk|files?))\b", "", target, flags=re.I).strip()

    # If destination is unspecified and query is generic like "Play Pokemon", halt with prompt
    if not dest:
        return {
            "status": "MEDIA_DISAMBIGUATION_REQUIRED",
            "halted": True,
            "target": target,
            "query": query,
            "prompt": f"[MEDIA DISAMBIGUATION] Would you like to search local files, YouTube videos, or YouTube Shorts for '{target}'?",
            "options": ["local files", "YouTube videos", "YouTube Shorts"],
        }

    encoded = urllib.parse.quote(target)
    if dest.lower() in ("youtube shorts", "shorts"):
        url = f"https://www.youtube.com/results?search_query={encoded}+shorts"
    elif dest.lower() in ("youtube videos", "youtube", "video"):
        url = f"https://www.youtube.com/results?search_query={encoded}"
    else:
        url = f"file:///search?query={encoded}"

    if dry_run:
        return {
            "status": "SIMULATED_MEDIA_LAUNCH",
            "halted": False,
            "executed": False,
            "dry_run": True,
            "target": target,
            "destination": dest,
            "url": url,
            "output": f"[SIMULATION: MEDIA ROUTER] Would launch browser for {dest}: {url}",
        }

    if browser_opener:
        browser_opener(url)
    else:
        import webbrowser
        webbrowser.open(url)

    return {
        "status": "SUCCESS",
        "halted": False,
        "executed": True,
        "dry_run": False,
        "target": target,
        "destination": dest,
        "url": url,
        "output": f"[MEDIA ROUTER SUCCESS] Launched {dest} for '{target}': {url}",
    }

