"""
Discourse Engine: Negation, Compound Command Splitting, and Coreference Resolution.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Strict negation handling ("don't open chrome", "never touch that", "not that one") -> refuses execution.
2. Correction negation ("not edge open notepad instead") -> extracts true desired command.
3. Compound command decomposition ("open notepad and then turn up volume") into ordered tasks.
4. Pronoun and entity coreference ("close it", "do that again", "the second one") via recent-entity stack.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

class DiscourseManager:
    def __init__(self):
        pass

    def analyze_negation(self, text: str) -> Dict[str, Any]:
        """
        Analyzes sentence for negation markers.
        Distinguishes between:
        1. Pure refusal / prohibition ("don't open chrome", "do not mute") -> MUST NOT EXECUTE.
        2. Inverted correction ("not edge, open notepad instead") -> EXECUTE notepad, REFUSE edge.
        """
        clean = text.strip()
        lower = clean.lower()

        # Check for correction pattern: "not X, open Y instead" or "no not X open Y"
        corr_m = re.search(r"\b(?:no\s+not|not)\s+([a-zA-Z0-9_\-]+)[,\s]+(?:open|launch|run)\s+([a-zA-Z0-9_\-]+)(?:\s+instead)?", lower)
        if corr_m:
            rejected = corr_m.group(1).strip()
            accepted = corr_m.group(2).strip()
            return {
                "has_negation": True,
                "is_correction": True,
                "rejected_target": rejected,
                "target_command": f"open {accepted}",
                "refusal_reason": f"Rejected '{rejected}' and switched to '{accepted}'"
            }

        # Pure negation patterns
        neg_patterns = [
            r"\b(?:do\s+not|don't|never)\s+(?:open|launch|start|run|close|shut\s+down|mute|unmute|turn\s+up|turn\s+down|delete|remove|click|take|create|navigate|scrape)\b",
            r"\b(?:not\s+that\s+(?:one|app|application|file)|no\s+do\s+not|stop\s+do\s+not)\b",
            r"\b(?:cancel\s+that\s+do\s+not\s+execute|abort\s+don't\s+run\s+anything)\b",
            r"\b(?:do\s+not\s+touch|never\s+open|never\s+close)\b",
            r"\b(?:don't|do\s+not)\s+[a-zA-Z0-9_\-\. ]+$"
        ]

        for pat in neg_patterns:
            if re.search(pat, lower):
                return {
                    "has_negation": True,
                    "is_correction": False,
                    "rejected_target": None,
                    "target_command": None,
                    "refusal_reason": f"Negation detected: user explicitly commanded NOT to execute ('{text}')"
                }

        # Leading 'no' or 'not'
        if re.match(r"^(?:no|not|never)\b", lower):
            return {
                "has_negation": True,
                "is_correction": False,
                "rejected_target": None,
                "target_command": None,
                "refusal_reason": f"Negative refusal detected ('{text}')"
            }

        return {
            "has_negation": False,
            "is_correction": False,
            "rejected_target": None,
            "target_command": None,
            "refusal_reason": None
        }

    def split_compound(self, text: str) -> List[str]:
        """
        Splits compound sentences into ordered atomic sub-commands.
        Handles connectors:
        - "and then"
        - "then"
        - "and" (when followed by command verbs)
        - commas separating verb phrases
        """
        clean = text.strip()

        # Preserve quotes during split
        # Check explicit sequential connectors
        connectors = [
            r"\s+and\s+then\s+",
            r"\s+then\s+",
            r"\s*,\s*then\s+",
            r"\s+after\s+that\s+",
            r"\s*,\s*and\s+",
        ]

        # First try strong connectors
        for conn in connectors:
            if re.search(conn, clean, re.IGNORECASE):
                parts = re.split(conn, clean, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    return parts

        # Check 'and' or comma followed by command verb
        cmd_verbs = r"(?:open|launch|start|run|close|shut|quit|exit|kill|terminate|mute|unmute|turn|crank|sound|volume|take|capture|snap|look|check|create|make|delete|remove|search|find|plan|add|navigate|browse|extract|scrape|click|select|maximize|minimize|restore|status)"
        
        # 'and <verb>'
        and_verb_pat = rf"\s+and\s+(?={cmd_verbs}\b)"
        if re.search(and_verb_pat, clean, re.IGNORECASE):
            parts = re.split(and_verb_pat, clean, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                return parts

        # ', <verb>'
        comma_verb_pat = rf"\s*,\s*(?={cmd_verbs}\b)"
        if re.search(comma_verb_pat, clean, re.IGNORECASE):
            parts = re.split(comma_verb_pat, clean, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                return parts

        return [clean]

    def resolve_coreference(self, text: str, entity_stack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves pronouns ('it', 'that', 'the first one') using recent entity stack.
        Returns resolution dictionary with reconstructed explicit command.
        """
        clean = text.strip()
        lower = clean.lower()

        # Pronoun targets: "it", "that", "that application", "that document"
        pronoun_match = re.search(r"\b(close|shut\s+down|terminate|kill|maximize|minimize|restore|open|scrape)\s+(it|that|that\s+application|that\s+app|that\s+program|that\s+document|that\s+file)\b", lower)
        if pronoun_match:
            verb = pronoun_match.group(1).strip()
            pronoun = pronoun_match.group(2).strip()

            target_entity = None
            if "file" in pronoun or "document" in pronoun:
                target_entity = entity_stack.get("last_file")
            else:
                target_entity = entity_stack.get("last_app") or entity_stack.get("last_target")

            if target_entity:
                reconstructed = f"{verb} {target_entity}"
                return {
                    "is_resolved": True,
                    "reconstructed": reconstructed,
                    "original_pronoun": pronoun,
                    "resolved_entity": target_entity,
                    "action": verb
                }
            else:
                return {
                    "is_resolved": False,
                    "reconstructed": None,
                    "error": f"No recent application or entity found in context to resolve '{pronoun}'."
                }

        # Repeat previous command: "do that again", "repeat previous command", "run that again"
        if re.search(r"\b(?:do\s+that\s+again|run\s+that\s+again|repeat\s+previous|repeat\s+last)\b", lower):
            last_cmd = entity_stack.get("last_command")
            if last_cmd:
                return {
                    "is_resolved": True,
                    "reconstructed": last_cmd,
                    "original_pronoun": "that",
                    "resolved_entity": last_cmd,
                    "action": "repeat"
                }
            return {
                "is_resolved": False,
                "reconstructed": None,
                "error": "No previous command to repeat."
            }

        # Selection choices: "the first one", "choose the second option", "pick number three", "option 1"
        sel_match = re.search(r"\b(?:the\s+)?(1|2|3|first|second|third|last)\s*(?:one|option|choice)?\b", lower)
        if sel_match and ("options" in entity_stack or "last_options" in entity_stack):
            options = entity_stack.get("options") or entity_stack.get("last_options", [])
            sel_str = sel_match.group(1)
            idx = 0
            if sel_str in ("1", "first"):
                idx = 0
            elif sel_str in ("2", "second"):
                idx = 1
            elif sel_str in ("3", "third"):
                idx = 2
            elif sel_str == "last":
                idx = len(options) - 1

            if 0 <= idx < len(options):
                chosen = options[idx]
                return {
                    "is_resolved": True,
                    "reconstructed": f"open {chosen}",
                    "original_pronoun": sel_str,
                    "resolved_entity": chosen,
                    "action": "open"
                }

        return {
            "is_resolved": False,
            "reconstructed": clean,
            "error": None
        }
