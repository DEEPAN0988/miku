"""
Discourse Engine: Negation Scope, Compound Command Splitting, and Coreference Resolution.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Strict negation handling ("don't open chrome", "never touch that", "never mind") -> refuses execution.
2. Inverted correction and negation scope ("not edge open notepad", "don't open edge but open chrome", "open chrome, not edge") -> executes accepted target, refuses rejected target.
3. Quote-aware compound command decomposition without splitting inside quoted strings.
4. Pronoun and entity coreference with mandatory explicit confirmation naming the resolved target for destructive or close actions.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

class DiscourseManager:
    def __init__(self):
        pass

    def analyze_negation(self, text: str) -> Dict[str, Any]:
        """
        Analyzes sentence for negation markers and negation scope.
        Distinguishes between:
        1. Pure refusal / prohibition ("don't open chrome", "never touch that", "never mind") -> MUST NOT EXECUTE.
        2. Inverted correction ("not edge, open notepad instead", "don't open edge but open chrome", "open chrome, not edge") -> EXECUTE accepted, REFUSE rejected.
        """
        clean = text.strip()
        lower = clean.lower()

        # Cancellation / nevermind
        if lower in ("never mind", "nevermind", "cancel", "abort", "forget it", "stop that", "don't do that"):
            return {
                "has_negation": True,
                "is_correction": False,
                "rejected_target": None,
                "target_command": None,
                "refusal_reason": "User commanded cancellation / never mind."
            }

        # Pattern A: "don't open X but open Y" or "do not launch X, but open Y"
        scope_a = re.search(r"\b(?:don't|do\s+not)\s+(?:open|launch|run)\s+([a-zA-Z0-9_\-\. ]+?)[,\s]+(?:but|instead\s+(?:of\s+that\s+)?(?:please\s+)?)[,\s]*(?:open|launch|run)\s+([a-zA-Z0-9_\-\. ]+)$", lower)

        if scope_a:
            rejected = scope_a.group(1).strip()
            accepted = scope_a.group(2).strip()
            return {
                "has_negation": True,
                "is_correction": True,
                "rejected_target": rejected,
                "target_command": f"open {accepted}",
                "refusal_reason": f"Negated '{rejected}' in favor of '{accepted}'."
            }

        # Pattern B: "open X, not Y" or "launch X not Y" or "open X and not Y"
        scope_b = re.search(r"\b(?:open|launch|run)\s+([a-zA-Z0-9_\-\. ]+?)[,\s]+(?:and\s+)?not\s+([a-zA-Z0-9_\-\. ]+)$", lower)
        if scope_b:
            accepted = scope_b.group(1).strip()
            rejected = scope_b.group(2).strip()
            return {
                "has_negation": True,
                "is_correction": True,
                "rejected_target": rejected,
                "target_command": f"open {accepted}",
                "refusal_reason": f"Accepted '{accepted}', negated '{rejected}'."
            }

        # Pattern C: "not X, open Y instead" or "no not X open Y"
        # NOTE: Requires single-word rejected target to avoid false fires on complex sentences.
        scope_c = re.search(r"\b(?:no\s+not|not)\s+([a-zA-Z0-9_\-]+)[,\s]+(?:open|launch|run)\s+([a-zA-Z0-9_\-]+)(?:\s+instead)?$", lower)
        if scope_c:
            rejected = scope_c.group(1).strip()
            accepted = scope_c.group(2).strip()
            # Only treat as correction when the rejected token is a single word (not a multi-word sentence)
            if " " not in rejected:
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
        CRITICAL: Never splits inside quotes (single or double quotes)!
        E.g. "create file 'then and now.txt'" remains ONE command.
        """
        clean = text.strip()

        # Step 1: Mask quoted segments
        quotes = []
        def mask_quote(match):
            placeholder = f"__QUOTED_SEGMENT_{len(quotes)}__"
            quotes.append(match.group(0))
            return placeholder

        # Match single or double quoted strings
        masked_text = re.sub(r"\"[^\"]*\"|'[^']*'", mask_quote, clean)

        # Connectors for splitting
        connectors = [
            r"\s+and\s+then\s+",
            r"\s+then\s+",
            r"\s*,\s*then\s+",
            r"\s+after\s+that\s+",
            r"\s*,\s*and\s+",
        ]

        parts = None
        for conn in connectors:
            if re.search(conn, masked_text, re.IGNORECASE):
                sub = re.split(conn, masked_text, flags=re.IGNORECASE)
                sub = [p.strip() for p in sub if p.strip()]
                if len(sub) > 1:
                    parts = sub
                    break

        if not parts:
            cmd_verbs = r"(?:open|launch|start|run|close|shut|quit|exit|kill|terminate|mute|unmute|turn|crank|sound|volume|take|capture|snap|look|check|create|make|craft|delete|remove|search|find|plan|add|navigate|browse|extract|scrape|click|select|maximize|minimize|restore|show|display|status|abort|stop|report)"
            and_verb_pat = rf"\s+and\s+(?={cmd_verbs}\b)"
            if re.search(and_verb_pat, masked_text, re.IGNORECASE):
                sub = re.split(and_verb_pat, masked_text, flags=re.IGNORECASE)
                sub = [p.strip() for p in sub if p.strip()]
                if len(sub) > 1:
                    parts = sub

        if not parts:
            comma_verb_pat = rf"\s*,\s*(?={cmd_verbs}\b)"
            if re.search(comma_verb_pat, masked_text, re.IGNORECASE):
                sub = re.split(comma_verb_pat, masked_text, flags=re.IGNORECASE)
                sub = [p.strip() for p in sub if p.strip()]
                if len(sub) > 1:
                    parts = sub

        if not parts:
            parts = [masked_text]

        # Step 2: Unmask quoted segments in every part
        unmasked_parts = []
        for part in parts:
            p_unmasked = part
            for idx, orig_q in enumerate(quotes):
                p_unmasked = p_unmasked.replace(f"__QUOTED_SEGMENT_{idx}__", orig_q)
            unmasked_parts.append(p_unmasked.strip())

        return unmasked_parts

    def resolve_coreference(self, text: str, entity_stack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves pronouns ('it', 'that', 'the second one') using recent entity stack.
        CRITICAL SAFETY: If resolved action is destructive or closing an application,
        sets 'confirmation_required' with prompt naming the resolved target ("Close Notepad?").
        """
        clean = text.strip()
        lower = clean.lower()

        # Pronoun targets: "close it", "terminate that application", "delete it", "maximize it", "scrape it"
        # Also catches: "open it back up", "bring that up again", "restore it", "run that again"
        # Match pattern: <action/verb> ... <pronoun> ...
        p_pattern = (
            r"^(?:please\s+)?(?P<verb>close|shut\s+down|terminate|kill|delete|remove|destroy|format|wipe|"
            r"maximize|minimize|restore|open|scrape|extract|switch\s+(?:back\s+)?to|relaunch|bring|run)\s+"
            r"(?:the\s+)?(?P<pronoun>it|that|that\s+application|that\s+app|that\s+program|that\s+document|that\s+file|that\s+one)"
            r"(?:\s+again|\s+back|\s+back\s+up|\s+up|\s+up\s+again|\s+please)*$"
        )
        m = re.match(p_pattern, lower)
        if m:
            verb = m.group("verb").strip()
            pronoun = m.group("pronoun").strip()

            target_entity = None
            if "file" in pronoun or "document" in pronoun or verb in ("delete", "remove", "destroy", "format", "wipe", "erase"):
                target_entity = entity_stack.get("last_file") or entity_stack.get("last_app") or entity_stack.get("last_target")
            else:
                target_entity = entity_stack.get("last_app") or entity_stack.get("last_target") or entity_stack.get("last_file")


            if target_entity:
                target_display = target_entity.capitalize() if target_entity else "target"

                # Check if action could lose data or terminates an app
                is_destructive = verb in ("close", "shut down", "terminate", "kill", "delete", "remove", "destroy", "format", "wipe")
                
                # Normalize verbs like "switch back to" -> "open", "relaunch" -> "open", "bring" -> "open"
                exec_verb = "open" if verb in ("switch to", "switch back to", "relaunch", "bring") else verb

                reconstructed = f"{exec_verb} {target_entity}"
                confirmation_prompt = f"{verb.capitalize()} {target_display}? (yes/no)" if is_destructive else None

                return {
                    "is_resolved": True,
                    "reconstructed": reconstructed,
                    "original_pronoun": pronoun,
                    "resolved_entity": target_entity,
                    "action": exec_verb,
                    "requires_confirmation": is_destructive,
                    "confirmation_prompt": confirmation_prompt
                }
            else:
                return {
                    "is_resolved": False,
                    "reconstructed": None,
                    "error": f"No recent application or entity found in context to resolve '{pronoun}'."
                }

        # Standalone selection: "pick number three", "choose the second option", "the first one", "option 2"
        # STRICT MATCH: Only when the utterance is explicitly selecting an option!
        sel_match = re.match(
            r"^(?:please\s+)?(?:pick|choose|select|open|take|give\s+me)?\s*(?:the\s+)?(?:option\s+|number\s+)?(?P<choice>1|2|3|first|second|third|last)(?:\s+(?:one|option|choice))?$",
            lower
        )
        if sel_match and ("options" in entity_stack or "last_options" in entity_stack):
            options = entity_stack.get("options") or entity_stack.get("last_options", [])
            sel_str = sel_match.group("choice")
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
                    "action": "open",
                    "requires_confirmation": False,
                    "confirmation_prompt": None
                }

        # Repeat last command (e.g. "do it again", "run that again")
        if re.match(r"^(?:do\s+(?:it|that)\s+again|run\s+that\s+again(?:\s+please)?|repeat\s+(?:previous|last)(?:\s+command)?|that\s+again(?:\s+please)?)$", lower):
            last_cmd = entity_stack.get("last_command")
            if last_cmd:
                # Check if the last command was destructive
                is_destr = any(last_cmd.lower().startswith(d) for d in ("delete ", "remove ", "kill ", "terminate ", "close ", "shut down ", "format ", "wipe ", "destroy ", "erase "))
                return {
                    "is_resolved": True,
                    "reconstructed": last_cmd,
                    "original_pronoun": "that",
                    "resolved_entity": last_cmd,
                    "action": "repeat",
                    "requires_confirmation": is_destr,
                    "confirmation_prompt": f"Repeat destructive action '{last_cmd}'? (yes/no)" if is_destr else None
                }

        # Contextual selection: "the last one you mentioned", "choose that one", "bring that up again"
        if re.match(r"^(?:the\s+last\s+one(?:\s+(?:you\s+)?(?:mentioned|said|told\s+me))?|pick\s+the\s+last\s+one)$", lower):
            last_cmd = entity_stack.get("last_command") or entity_stack.get("last_app")
            if last_cmd:
                return {
                    "is_resolved": True,
                    "reconstructed": f"open {last_cmd}",
                    "original_pronoun": "the last one",
                    "resolved_entity": last_cmd,
                    "action": "open",
                    "requires_confirmation": False,
                    "confirmation_prompt": None
                }

        # Contextual selection: "choose that one", "select that one"
        if re.match(r"^(?:choose|select|take|use|pick)\s+that\s+(?:one|option|choice)?$", lower):
            options = entity_stack.get("options") or entity_stack.get("last_options", [])
            if options:
                chosen = options[-1]  # default to last option in context
                return {
                    "is_resolved": True,
                    "reconstructed": f"open {chosen}",
                    "original_pronoun": "that one",
                    "resolved_entity": chosen,
                    "action": "open",
                    "requires_confirmation": False,
                    "confirmation_prompt": None
                }

        return {
            "is_resolved": False,
            "reconstructed": clean,
            "error": None
        }
