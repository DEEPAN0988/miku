"""
Paraphrase Rewriter for Miku's Cognitive Pipeline.
Core 2: Cognitive Router (Brain & Memory)

Maps indirect, idiomatic, and rambling English utterances to canonical
Miku command forms before grammar routing so that phrasings outside the
grammar's fixed verb set are still understood.

Design principles:
  - Ordered rules: most-specific patterns first, catch-all last.
  - Non-destructive: only rewrite when confident; never guess app names
    from ambiguous fragments — let the grammar/classifier handle those.
  - Each rule is a (compiled_pattern, replacement_or_callable) pair.
    If replacement is a string it is used directly as re.sub repl arg.
    If it is callable it receives the match and returns the rewritten string.
"""

import re
from typing import List, Tuple, Union, Callable


_Repl = Union[str, Callable[[re.Match], str]]
_Rule = Tuple[re.Pattern, _Repl]


def _r(pattern: str, repl: _Repl, flags: int = re.IGNORECASE) -> _Rule:
    return re.compile(pattern, flags), repl


# ---------------------------------------------------------------------------
# OPEN_APP paraphrase rules  (each rewrites to "open <app>")
# ---------------------------------------------------------------------------
_OPEN_RULES: List[_Rule] = [
    # "give <X> a spin / a go / a try / a whirl"
    _r(r"^give\s+(?:the\s+)?(.+?)\s+a\s+(?:spin|go|try|whirl|shot)(?:\s+on\s+my\s+screen)?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "get <X> going / running / started / up / up and running"
    _r(r"^get\s+(?:the\s+)?(.+?)\s+(?:going|running|started|up\s+and\s+running|up)(?:\s+for\s+me)?(?:\s+right\s+now)?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "summon <X> [window]"
    _r(r"^summon\s+(?:the\s+)?(.+?)(?:\s+window)?(?:\s+for\s+me)?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "display / show / show me <X>"
    _r(r"^(?:display|show\s+me|show)\s+(?:the\s+)?(.+?)(?:\s+(?:window|program|application|app))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "spin up <X>"
    _r(r"^spin\s+up\s+(?:the\s+)?(.+?)(?:\s+(?:app|application|window|program))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "i wish to / i want to [use|open|run|access|...] <X>"
    _r(r"^i\s+(?:wish|want)\s+to\s+(?:use|work\s+in|take\s+notes\s+in|open|run|access|launch|view|see|look\s+at)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program|window|tool))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "i'd like to / i would like to open/use/run <X>"
    _r(r"^i(?:'d|\s+would)\s+like\s+to\s+(?:open|use|run|launch|start|access)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "feel like / fancy using / opening <X>"
    _r(r"^(?:i\s+)?(?:feel|fancy)\s+(?:like\s+)?(?:using|opening|running|launching)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "kindly launch / open <X>"
    _r(r"^kindly\s+(?:launch|open|start|run|fire\s+up|bring\s+up)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program|window))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "pull up / load up / boot up <X>"
    _r(r"^(?:pull|load|boot)\s+up\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program|window))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # Rambling prefix: "ok so ... could you maybe open the music thing"
    _r(r"^(?:ok(?:ay)?[\s,]+)?(?:so[\s,]+)?(?:i\s+was\s+thinking[\s,]+)?(?:(?:could|can|would)\s+you\s+maybe\s+)?(?:open|launch|start|fire\s+up|pull\s+up)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program|window|thing|thingy))?(?:\s+please)?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "let me use / access / check out <X>"
    _r(r"^let\s+me\s+(?:use|access|check\s+out|have\s+a\s+look\s+at)\s+(?:the\s+)?(.+?)(?:\s+(?:application|app|program))?$",
       lambda m: f"open {m.group(1).strip()}"),

    # "hop into / jump into <X>"
    _r(r"^(?:hop|jump)\s+into\s+(?:the\s+)?(.+?)(?:\s+(?:app|window))?$",
       lambda m: f"open {m.group(1).strip()}"),
]

# ---------------------------------------------------------------------------
# CLOSE_APP paraphrase rules  (each rewrites to "close <app>")
# ---------------------------------------------------------------------------
_CLOSE_RULES: List[_Rule] = [
    # "get rid of / eliminate / ditch / scrap <X>"
    _r(r"^(?:get\s+rid\s+of|eliminate|ditch|scrap)\s+(?:the\s+)?(.+?)(?:\s+(?:right\s+away|now|immediately|please|app|application|window))*$",
       lambda m: f"close {m.group(1).strip()}"),

    # "dismiss <X> [from (the) screen]"
    _r(r"^dismiss\s+(?:the\s+)?(.+?)(?:\s+from\s+(?:the\s+)?(?:screen|display))?(?:\s+please)?$",
       lambda m: f"close {m.group(1).strip()}"),

    # "take <X> off [the screen]"
    _r(r"^take\s+(?:the\s+)?(.+?)\s+off(?:\s+(?:the\s+|my\s+)?(?:screen|display))?$",
       lambda m: f"close {m.group(1).strip()}"),

    # "put <X> away"
    _r(r"^put\s+(?:the\s+)?(.+?)\s+away$",
       lambda m: f"close {m.group(1).strip()}"),

    # "remove <X> from (the) screen"
    _r(r"^remove\s+(?:the\s+)?(.+?)\s+from\s+(?:the\s+)?(?:screen|display)$",
       lambda m: f"close {m.group(1).strip()}"),

    # "wipe <X> off / from screen"
    _r(r"^wipe\s+(?:the\s+)?(.+?)\s+(?:from\s+(?:the\s+)?screen|off\s+(?:the\s+)?screen)$",
       lambda m: f"close {m.group(1).strip()}"),

    # "make <X> go away"
    _r(r"^make\s+(?:the\s+)?(.+?)\s+go\s+away$",
       lambda m: f"close {m.group(1).strip()}"),
]

# ---------------------------------------------------------------------------
# VOLUME paraphrase rules
# ---------------------------------------------------------------------------
_VOLUME_RULES: List[_Rule] = [
    _r(r"^make\s+(?:it|the\s+(?:sound|volume|audio))\s+(?:louder|higher|bigger)$", "volume up"),
    _r(r"^make\s+(?:it|the\s+(?:sound|volume|audio))\s+(?:quieter|softer|lower|smaller)$", "volume down"),
    _r(r"^quiet\s+(?:it|the\s+(?:sound|audio|volume))?(?:\s+down)?$", "volume down"),
    _r(r"^(?:cut|kill)\s+(?:the\s+)?(?:sound|audio|volume|music)$", "mute"),
    _r(r"^pump\s+(?:it\s+up|up\s+(?:the\s+)?(?:volume|sound|audio))$", "volume up"),
    _r(r"^blast\s+(?:the\s+)?(?:music|audio|sound|volume)$", "volume up"),
    _r(r"^(?:shush|shh|sh|shhh)$", "mute"),
]

# ---------------------------------------------------------------------------
# WINDOW_STATE paraphrase rules
# ---------------------------------------------------------------------------
_WINDOW_RULES: List[_Rule] = [
    _r(r"^make\s+(?:the\s+)?(?:window|app)\s+(?:full\s+screen|fullscreen|big|bigger|large)$",
       "maximize the window"),
    _r(r"^make\s+(?:the\s+)?(?:window|app)\s+(?:small(?:er)?|tiny|little)$",
       "minimize the window"),
]

# Rule groups — window/volume/close checked before open (more specific)
_ALL_RULE_GROUPS: List[List[_Rule]] = [
    _WINDOW_RULES,
    _VOLUME_RULES,
    _CLOSE_RULES,
    _OPEN_RULES,
]


class ParaphraseRewriter:
    """
    Deterministic paraphrase rewriter.

    Usage::

        rewriter = ParaphraseRewriter()
        rewritten, did_rewrite = rewriter.rewrite("get rid of notepad")
        # rewritten == "close notepad", did_rewrite == True
    """

    def rewrite(self, text: str) -> Tuple[str, bool]:
        """
        Attempt to rewrite *text* to a canonical Miku command form.

        Returns:
            (rewritten_text, did_rewrite)
            If no rule matched, returns ``(text, False)`` unchanged.
        """
        t = text.strip()
        for rule_group in _ALL_RULE_GROUPS:
            for pattern, replacement in rule_group:
                if callable(replacement):
                    m = pattern.match(t)
                    if m:
                        result = replacement(m).strip()
                        # Guard: must produce a non-empty, different, multi-token string
                        if result and result != t and len(result.split()) >= 2:
                            return result, True
                else:
                    new_t = pattern.sub(replacement, t)
                    if new_t != t:
                        return new_t.strip(), True
        return t, False
