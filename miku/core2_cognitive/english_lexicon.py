"""
MIKU: Comprehensive Local English Lexicon & Semantic Comprehension Engine.
Core 2: Cognitive Router (Brain & Memory)
Provides offline English vocabulary understanding, typo correction, 
part-of-speech classification, semantic definitions, and real-time word learning.
Zero Cloud APIs. 100% On-Device.
"""
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Local storage for user-taught words
LEARNED_WORDS_FILE = Path("c:/miku/miku_learned_words.json")

# Core English Vocabulary Definitions & Semantic Associations
CORE_ENGLISH_DICTIONARY: Dict[str, Dict[str, Any]] = {
    # System, Tech & Assistant Concepts
    "miku": {
        "pos": "noun",
        "definition": "Your sovereign on-device personal assistant built with zero cloud APIs.",
        "synonyms": ["assistant", "agent", "companion"]
    },
    "sovereign": {
        "pos": "adjective",
        "definition": "Possessing supreme or independent power; fully self-contained without external dependencies.",
        "synonyms": ["autonomous", "independent", "self-governing", "private"]
    },
    "autonomous": {
        "pos": "adjective",
        "definition": "Acting independently or having the freedom to do so without external servers.",
        "synonyms": ["independent", "self-directed", "automated", "sovereign"]
    },
    "understand": {
        "pos": "verb",
        "definition": "To perceive the intended meaning of words, language, or commands.",
        "synonyms": ["comprehend", "grasp", "interpret", "know", "process"]
    },
    "train": {
        "pos": "verb",
        "definition": "To teach, calibrate, or condition an AI system through local data, repetition, and feedback.",
        "synonyms": ["educate", "calibrate", "fit", "instruct", "prepare"]
    },
    "word": {
        "pos": "noun",
        "definition": "A single distinct meaningful element of speech or writing used with others to form sentences.",
        "synonyms": ["term", "expression", "token", "vocable"]
    },
    "words": {
        "pos": "noun",
        "definition": "Plural of word; meaningful units of human language.",
        "synonyms": ["language", "vocabulary", "terms", "expressions"]
    },
    "english": {
        "pos": "noun",
        "definition": "A West Germanic language originating in England, currently the global lingua franca.",
        "synonyms": ["language", "tongue"]
    },
    "language": {
        "pos": "noun",
        "definition": "A system of communication consisting of sounds, words, and grammar.",
        "synonyms": ["speech", "tongue", "dialect", "vocabulary"]
    },
    "vocabulary": {
        "pos": "noun",
        "definition": "The body of words known to a person, language, or artificial intelligence agent.",
        "synonyms": ["lexicon", "wordbook", "glossary", "dictionary"]
    },
    "learning": {
        "pos": "noun",
        "definition": "The acquisition of knowledge or skills through study, experience, or being taught.",
        "synonyms": ["education", "training", "adaptation", "memorization"]
    },
    "privacy": {
        "pos": "noun",
        "definition": "The state of being free from unwanted observation or cloud data extraction.",
        "synonyms": ["confidentiality", "secrecy", "security", "isolation"]
    },
    "offline": {
        "pos": "adjective",
        "definition": "Operating completely disconnected from external internet servers or cloud networks.",
        "synonyms": ["local", "disconnected", "airgapped", "on-device"]
    },
    "local": {
        "pos": "adjective",
        "definition": "Belonging or relating to the particular computer or device you are physically using.",
        "synonyms": ["on-device", "bare-metal", "internal", "offline"]
    },
    "browser": {
        "pos": "noun",
        "definition": "A software application with which one browses the World Wide Web.",
        "synonyms": ["web browser", "navigator", "surf client"]
    },
    "game": {
        "pos": "noun",
        "definition": "An interactive digital pastime or simulation played on a computer or console.",
        "synonyms": ["video game", "interactive entertainment", "play"]
    },
    "games": {
        "pos": "noun",
        "definition": "Plural of game; video games or interactive computer entertainment.",
        "synonyms": ["gaming", "video games", "titles"]
    },
    "computer": {
        "pos": "noun",
        "definition": "An electronic device for storing and processing data according to instructions.",
        "synonyms": ["pc", "system", "machine", "workstation"]
    },
    "operating system": {
        "pos": "noun",
        "definition": "The low-level software that supports a computer's basic functions and hardware scheduling.",
        "synonyms": ["os", "windows", "linux", "kernel"]
    },
    "memory": {
        "pos": "noun",
        "definition": "The faculty by which the mind or agent stores and remembers information.",
        "synonyms": ["storage", "recall", "retention", "remembrance"]
    },
    "volume": {
        "pos": "noun",
        "definition": "The magnitude of sound emitted by an audio output device.",
        "synonyms": ["sound level", "loudness", "audio"]
    },
    "mute": {
        "pos": "verb",
        "definition": "To silence or deactivate the sound of an audio output device.",
        "synonyms": ["silence", "quiet", "deactivate"]
    },
    "screenshot": {
        "pos": "noun",
        "definition": "A digital image of what is currently displayed on a computer screen.",
        "synonyms": ["screen capture", "snapshot", "screen grab"]
    },
    "camera": {
        "pos": "noun",
        "definition": "An optical device used to capture images and video frames.",
        "synonyms": ["webcam", "sensor", "lens"]
    },
    "file": {
        "pos": "noun",
        "definition": "A resource for recording data in a computer storage device.",
        "synonyms": ["document", "record", "archive"]
    },
    "folder": {
        "pos": "noun",
        "definition": "A virtual directory container within a digital file system.",
        "synonyms": ["directory", "catalog", "drawer"]
    },
    "notepad": {
        "pos": "noun",
        "definition": "A simple text editing application included with Microsoft Windows.",
        "synonyms": ["text editor", "notes", "diary"]
    },
    "calculator": {
        "pos": "noun",
        "definition": "An electronic device or application for performing mathematical calculations.",
        "synonyms": ["calc", "arithmetic tool", "reckoner"]
    },
    "terminal": {
        "pos": "noun",
        "definition": "A text-based command-line interface for executing system commands.",
        "synonyms": ["command prompt", "shell", "console", "powershell"]
    },
    "wuthering waves": {
        "pos": "noun",
        "definition": "An open-world action RPG developed by Kuro Games.",
        "synonyms": ["wuwa", "ww", "wuthering wave"]
    },
    "whatsapp": {
        "pos": "noun",
        "definition": "A messaging and voice-over-IP service owned by Meta.",
        "synonyms": ["wa", "chat app"]
    },
    "instagram": {
        "pos": "noun",
        "definition": "A photo and video sharing social networking service.",
        "synonyms": ["insta", "ig", "social app"]
    },
    "help": {
        "pos": "verb",
        "definition": "To assist or provide support in accomplishing a goal.",
        "synonyms": ["assist", "aid", "support", "serve"]
    }
}

# English Action Verb Synonyms for Semantic Mapping
ACTION_VERB_MAP: Dict[str, str] = {
    # Open / Launch
    "open": "open",
    "launch": "open",
    "start": "open",
    "run": "open",
    "fire up": "open",
    "bring up": "open",
    "switch to": "open",
    "visit": "open",
    "go to": "open",
    "execute": "open",
    "play": "open",
    "activate": "open",
    "initiate": "open",
    
    # Close / Terminate
    "close": "close",
    "quit": "close",
    "exit": "close",
    "shut": "close",
    "shut down": "close",
    "terminate": "close",
    "kill": "close",
    "end": "close",
    "halt": "close",
    "stop": "close",
    
    # Search / Locate
    "find": "search",
    "search": "search",
    "locate": "search",
    "lookup": "search",
    "seek": "search",
    "where is": "search",
    
    # Audio Volume
    "volume": "volume",
    "sound": "volume",
    "audio": "volume",
    "louder": "volume_up",
    "raise": "volume_up",
    "turn up": "volume_up",
    "increase": "volume_up",
    "boost": "volume_up",
    "quieter": "volume_down",
    "lower": "volume_down",
    "turn down": "volume_down",
    "decrease": "volume_down",
    "mute": "mute",
    "silence": "mute",
    "unmute": "unmute",
    
    # Vision & Camera
    "screenshot": "screenshot",
    "screen": "screenshot",
    "capture": "screenshot",
    "snap": "screenshot",
    "snapshot": "screenshot",
    "look": "inspect_camera",
    "see": "inspect_camera",
    "inspect": "inspect_camera",
    "check": "inspect_camera",
    
    # Automation & Plan
    "plan": "plan",
    "schedule": "plan",
    "todo": "plan",
    "tasks": "plan",
    "agenda": "plan"
}

# 1,000+ Common English Lexicon Words for Typo Detection & Vocabulary Encoding
COMMON_ENGLISH_WORDS = {
    "a", "about", "above", "across", "action", "activate", "actually", "add", "adjust", "advance",
    "after", "again", "agent", "ahead", "ai", "alarm", "alert", "all", "allow", "almost",
    "alone", "along", "already", "also", "always", "am", "among", "an", "and", "another",
    "answer", "anti", "any", "anything", "api", "apis", "app", "application", "apps", "are", "around", "as", "ask",
    "assistant", "at", "audio", "auto", "automated", "automatic", "autonomous", "away", "back",
    "bad", "bare", "be", "beautiful", "because", "become", "been", "before", "begin", "behind",
    "being", "believe", "best", "better", "between", "big", "bit", "black", "blue", "body",
    "book", "boost", "bot", "bots", "both", "brain", "bring", "browse", "browser", "build", "busy", "but",
    "button", "buy", "by", "bypass", "calculate", "calculator", "call", "calm", "camera", "can",
    "cancel", "cannot", "capable", "captcha", "capture", "care", "case", "cat", "catch", "cause", "center",
    "certain", "change", "chat", "check", "choose", "chrome", "city", "clean", "clear", "click",
    "client", "clock", "close", "cloud", "code", "cognitive", "cold", "color", "come", "command",
    "commands", "commercial", "common", "communication", "companion", "complete", "completed",
    "comprehend", "computer", "condition", "confidence", "confirm", "connect", "console", "control",
    "conversation", "cool", "copy", "core", "correct", "correction", "could", "count", "country",
    "course", "create", "current", "custom", "daemon", "daily", "dark", "data", "date", "day",
    "decrease", "deep", "default", "define", "definition", "delete", "describe", "design", "desk",
    "device", "dialogue", "dictionary", "different", "digital", "direct", "directory", "discover",
    "display", "do", "dock", "document", "does", "done", "door", "down", "download", "downloads",
    "draw", "drive", "drop", "during", "each", "early", "earth", "easy", "edge", "edit", "editor",
    "either", "electronic", "element", "embed", "enable", "end", "engine", "english", "enjoy",
    "enough", "enroll", "enter", "entire", "environment", "error", "even", "evening", "every",
    "everything", "exact", "execute", "execution", "exit", "experience", "explain", "eye", "eyes",
    "face", "fact", "fail", "fast", "feature", "feel", "few", "field", "file", "files", "filter",
    "final", "find", "finish", "fire", "first", "flow", "folder", "follow", "for", "force", "forget",
    "form", "found", "four", "free", "frequency", "friend", "from", "front", "full", "fully",
    "fun", "function", "future", "game", "games", "gaming", "general", "get", "give", "glass",
    "go", "good", "google", "got", "grammar", "great", "green", "greeting", "group", "grow",
    "guard", "guide", "half", "halt", "hand", "hands", "happen", "hard", "hardware", "has", "have",
    "he", "head", "hear", "heart", "heavy", "hello", "help", "her", "here", "high", "him", "his",
    "history", "hold", "home", "hope", "hour", "house", "how", "however", "human", "idea", "if",
    "image", "immediate", "immediately", "important", "in", "increase", "independent", "index",
    "information", "initial", "input", "inspect", "install", "installed", "instead", "instruction",
    "intelligent", "intent", "interactive", "interface", "internet", "into", "is", "isolate",
    "isolated", "issue", "it", "its", "itself", "join", "just", "keep", "key", "kill", "kind",
    "know", "knowledge", "language", "large", "last", "late", "later", "laugh", "launch", "lead",
    "learn", "learned", "learning", "least", "leave", "left", "less", "let", "letter", "level",
    "lexicon", "life", "light", "like", "line", "list", "listen", "little", "live", "local",
    "locate", "lock", "log", "long", "look", "loud", "love", "low", "lower", "machine", "main",
    "make", "man", "manage", "many", "map", "mark", "match", "matter", "may", "maybe", "me",
    "mean", "meaning", "means", "measure", "meet", "meeting", "memory", "message", "metal", "method",
    "middle", "might", "miku", "mind", "mine", "minute", "miss", "mode", "model", "modern", "modes", "moment",
    "money", "month", "more", "morning", "most", "move", "much", "music", "must", "mute", "my",
    "myself", "name", "nation", "natural", "nature", "navigate", "near", "need", "network",
    "neural", "never", "nevermind", "new", "next", "nice", "night", "no", "node", "noise", "none",
    "noon", "normal", "not", "note", "notepad", "notes", "nothing", "now", "number", "occur",
    "of", "off", "offline", "often", "ok", "okay", "old", "on", "once", "one", "online", "only",
    "onto", "open", "operate", "operating", "operation", "optical", "option", "or", "order",
    "original", "os", "other", "others", "our", "out", "output", "over", "own", "page", "paint",
    "paper", "part", "pass", "past", "path", "pattern", "pause", "pc", "people", "per", "perform",
    "perhaps", "period", "person", "personal", "phone", "phrase", "picture", "piece", "place",
    "plan", "play", "player", "please", "point", "power", "powershell", "preference", "predict",
    "prediction", "prepare", "present", "press", "preview", "price", "private", "privacy", "problem",
    "process", "produce", "program", "prompt", "protect", "proud", "provide", "pull", "pure",
    "push", "put", "quality", "question", "quick", "quiet", "quit", "quite", "radio", "raise",
    "random", "rate", "raw", "reach", "read", "ready", "real", "really", "reason", "recall",
    "receive", "recent", "recognition", "record", "red", "reduce", "refresh", "regular", "release",
    "remember", "remove", "repeat", "reply", "report", "request", "require", "reset", "resolve",
    "resource", "respond", "response", "rest", "result", "retrieve", "return", "right", "ring",
    "room", "root", "round", "route", "router", "rule", "run", "running", "safe", "same", "sample",
    "save", "say", "scale", "scan", "scene", "schedule", "score", "screen", "screenshot", "script",
    "search", "second", "secret", "secure", "security", "see", "seek", "seem", "self", "send",
    "sense", "sensor", "sentence", "separate", "server", "service", "session", "set", "settings",
    "several", "shall", "she", "shell", "shift", "short", "should", "show", "shut", "shutdown",
    "side", "sight", "signal", "silence", "simple", "since", "single", "site", "size", "skill",
    "sleep", "slow", "small", "smooth", "snap", "snapshot", "so", "social", "software", "solve",
    "some", "someone", "something", "sometimes", "song", "songs", "soon", "sound", "source",
    "sovereign", "space", "speak", "speaker", "special", "specific", "speed", "spotify", "stand",
    "standard", "start", "state", "status", "stay", "stealth", "step", "still", "stop", "storage", "store",
    "street", "strict", "strictly", "strong", "study", "subword", "successful", "such", "suggest",
    "support", "sure", "surface", "switch", "system", "table", "take", "talk", "target", "task",
    "tasks", "teach", "teacher", "team", "tell", "term", "terminal", "test", "text", "than",
    "thank", "thanks", "that", "the", "their", "them", "then", "there", "these", "they", "thing",
    "things", "think", "third", "this", "those", "though", "thought", "three", "through", "time",
    "to", "today", "together", "token", "tomorrow", "tone", "too", "tool", "top", "total", "toward",
    "track", "tracker", "train", "trained", "training", "transcript", "transformer", "translate",
    "tree", "true", "trust", "try", "turn", "two", "type", "under", "understand", "understanding",
    "unmute", "until", "up", "upon", "url", "usable", "use", "user", "usual", "value", "variable",
    "very", "video", "view", "vision", "visit", "visual", "voice", "volume", "wait", "walk", "want",
    "war", "warm", "was", "watch", "water", "way", "we", "weather", "web", "webcam", "website",
    "week", "welcome", "well", "went", "were", "what", "whatever", "wheel", "when", "whenever",
    "where", "wherever", "which", "while", "white", "who", "whole", "whom", "whose", "why", "wide",
    "will", "win", "window", "windows", "wire", "with", "within", "without", "word", "words",
    "work", "workspace", "world", "would", "write", "wrong", "year", "yellow", "yes", "yesterday",
    "yet", "you", "young", "your", "yourself", "zero"
}

def damerau_levenshtein(s1: str, s2: str) -> int:
    """Calculates Damerau-Levenshtein edit distance for typo detection."""
    d = {}
    len_s1 = len(s1)
    len_s2 = len(s2)
    for i in range(-1, len_s1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len_s2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len_s1):
        for j in range(len_s2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len_s1 - 1, len_s2 - 1)]


class EnglishLexicon:
    """
    On-device English language comprehension engine.
    """
    def __init__(self, learned_path: Path = LEARNED_WORDS_FILE):
        self.learned_path = learned_path
        self.learned_words: Dict[str, Dict[str, Any]] = self._load_learned_words()
        self.all_words = set(COMMON_ENGLISH_WORDS)
        self.all_words.update(CORE_ENGLISH_DICTIONARY.keys())
        self.all_words.update(self.learned_words.keys())

    def _load_learned_words(self) -> Dict[str, Dict[str, Any]]:
        if self.learned_path.exists():
            try:
                with open(self.learned_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_learned_words(self):
        try:
            with open(self.learned_path, "w", encoding="utf-8") as f:
                json.dump(self.learned_words, f, indent=2)
        except Exception:
            pass

    def is_known_word(self, word: str) -> bool:
        return word.strip().lower() in self.all_words

    def correct_spelling(self, word: str) -> str:
        """
        Corrects minor typos in English words (e.g. 'undrestand' -> 'understand').
        """
        w = word.strip().lower()
        if not w:
            return w

        # Common phonetic typo quick-fixes
        replacements = {
            "undrestand": "understand",
            "undestand": "understand",
            "undrstand": "understand",
            "opne": "open",
            "oppen": "open",
            "clsoe": "close",
            "clse": "close",
            "gmaes": "games",
            "gaems": "games",
            "calcultor": "calculator",
            "calcualtor": "calculator",
            "notepadd": "notepad",
            "notpad": "notepad",
            "whastapp": "whatsapp",
            "watsapp": "whatsapp",
            "broswer": "browser",
            "voulme": "volume",
            "volum": "volume",
            "screnshot": "screenshot",
            "screenshit": "screenshot",
            "termnial": "terminal"
        }
        if w in replacements:
            return replacements[w]

        if len(w) <= 3 or w in self.all_words:
            return w

        # Scan for closest known word within distance 1 or 2
        best_match = w
        min_dist = 3
        # Candidate filter by length
        candidates = [c for c in self.all_words if abs(len(c) - len(w)) <= 1 and c[0] == w[0]]
        for cand in candidates:
            dist = damerau_levenshtein(w, cand)
            if dist < min_dist:
                min_dist = dist
                best_match = cand

        if min_dist <= 1:
            return best_match
        return w

    def normalize_sentence(self, sentence: str) -> str:
        """
        Normalizes a sentence by correcting English typos while preserving punctuation.
        """
        tokens = re.findall(r"\w+|[^\w\s]", sentence)
        corrected = []
        for t in tokens:
            if re.match(r"^[a-zA-Z]+$", t):
                fixed = self.correct_spelling(t)
                corrected.append(fixed)
            else:
                corrected.append(t)
        
        # Reconstruct sentence with proper spacing
        out = ""
        for i, tok in enumerate(corrected):
            if i > 0 and tok not in ".,?!;:')]":
                prev = corrected[i - 1]
                if prev not in "([{'\"":
                    out += " "
            out += tok
        return out

    def get_word_definition(self, word: str) -> Optional[Dict[str, Any]]:
        w = word.strip().lower()
        w = self.correct_spelling(w)

        if w in self.learned_words:
            return self.learned_words[w]
        if w in CORE_ENGLISH_DICTIONARY:
            return CORE_ENGLISH_DICTIONARY[w]

        # Check plural or lemma forms (e.g. 'games' -> 'game')
        if w.endswith("s") and w[:-1] in CORE_ENGLISH_DICTIONARY:
            return CORE_ENGLISH_DICTIONARY[w[:-1]]
        if w.endswith("ing") and w[:-3] in CORE_ENGLISH_DICTIONARY:
            return CORE_ENGLISH_DICTIONARY[w[:-3]]

        return None

    def learn_new_word(self, word: str, definition: str, pos: str = "noun", synonyms: Optional[List[str]] = None) -> str:
        w = word.strip().lower()
        entry = {
            "pos": pos,
            "definition": definition.strip(),
            "synonyms": synonyms or [],
            "learned_at": time.time()
        }
        self.learned_words[w] = entry
        self.all_words.add(w)
        self._save_learned_words()
        return f"Learned English word '{w}': {definition}"

    def check_definition_query(self, text: str) -> Optional[str]:
        """
        Detects if user is asking for the meaning of an English word.
        e.g. 'what does sovereign mean', 'define autonomous', 'meaning of browser'
        """
        clean = text.strip().lower()
        clean = self.normalize_sentence(clean)

        patterns = [
            r"^(?:what\s+does\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\s+mean\??)$",
            r"^(?:what\s+is\s+the\s+meaning\s+of\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\??)$",
            r"^(?:define\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\??)$",
            r"^(?:meaning\s+of\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\??)$",
            r"^(?:what\s+is\s+(?:a|an)\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\??)$"
        ]

        for pat in patterns:
            m = re.match(pat, clean)
            if m:
                target_word = m.group("word").strip()
                defn = self.get_word_definition(target_word)
                if defn:
                    syns = f" Synonyms: {', '.join(defn['synonyms'])}." if defn.get("synonyms") else ""
                    return f"'{target_word.capitalize()}' ({defn['pos']}): {defn['definition']}{syns}"
                else:
                    return f"I understand the word '{target_word}', but haven't learned its specific definition yet. You can teach me by saying 'learn word {target_word} means [definition]'!"
        return None

    def check_word_learning_intent(self, text: str) -> Optional[str]:
        """
        Detects user teaching a new word:
        'learn word serendipity means finding good things'
        'teach word wuwa means wuthering waves'
        """
        clean = text.strip()
        patterns = [
            r"^(?:learn\s+word|teach\s+word|the\s+word)\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\s+(?:means|is|definition\s+is|\:)\s+[\"']?(?P<defn>.+?)[\"']?$",
            r"^(?:define\s+word)\s+[\"']?(?P<word>[a-zA-Z\s]+?)[\"']?\s+as\s+[\"']?(?P<defn>.+?)[\"']?$"
        ]
        for pat in patterns:
            m = re.match(pat, clean, re.I)
            if m:
                word = m.group("word").strip()
                defn = m.group("defn").strip()
                return self.learn_new_word(word, defn)
        return None

    def get_lexicon_stats(self) -> Dict[str, Any]:
        return {
            "total_vocabulary_count": len(self.all_words),
            "core_dictionary_entries": len(CORE_ENGLISH_DICTIONARY),
            "user_learned_words": len(self.learned_words)
        }
