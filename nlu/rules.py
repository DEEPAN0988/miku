"""
Rule-Based Slot Filler & Pattern Matcher for Miku NLU.
Extracts intent and named slots from deterministic utterance structures.
Zero external models, zero API keys.
"""

import re
from typing import Optional, Dict, Any


class RuleMatcher:
    @staticmethod
    def match(text: str) -> Optional[Dict[str, Any]]:
        raw = text.strip()
        lower = raw.lower()

        # 0. Greetings & Identity & Help (Natural conversation)
        if re.search(r"^(?:hi|hello|hey|hey\s+miku|yo|good\s+(?:morning|evening|afternoon|day)|greetings)(?:[!\.\?,\s]|$)", lower):
            return {"intent": "greeting", "entities": {}, "confidence": 0.99}

        if re.search(r"\b(who are you|what('s| is) your name|what are you|introduce yourself|tell me about yourself)\b", lower):
            return {"intent": "identity", "entities": {}, "confidence": 0.99}

        if re.search(r"^(?:help|what can you do|show commands|list commands|help me|how to use)\b", lower):
            return {"intent": "help", "entities": {}, "confidence": 0.99}

        if re.search(r"\b(how are you|are you (?:there|listening|online)|status check)\b", lower):
            return {"intent": "status_query", "entities": {}, "confidence": 0.98}

        # 1. Compositional command: "open <app> and write/type <content> [about <topic>]"
        comp_match = re.search(r"open\s+([a-zA-Z0-9\s]+?)\s+and\s+(?:write|type)\s+(?:an?\s+)?([a-zA-Z0-9\s]+?)(?:\s+about\s+(.+))?$", lower)
        if comp_match:
            app = comp_match.group(1).strip()
            action_content = comp_match.group(2).strip()
            topic = comp_match.group(3).strip() if comp_match.group(3) else ""
            return {
                "intent": "compositional_app_write",
                "entities": {
                    "app": app,
                    "document_type": action_content,
                    "topic": topic
                },
                "confidence": 0.98
            }

        # 2. Simple App Launch: "open <app>", "launch <app>", "start <app>"
        open_match = re.match(r"^(?:please\s+)?(?:open|launch|start)\s+([a-zA-Z0-9\s\.\-_]+)$", lower)
        if open_match:
            app = open_match.group(1).strip()
            if not any(k in app for k in ["camera", "notepad and", "task"]):
                return {
                    "intent": "launch_app",
                    "entities": {"app": app},
                    "confidence": 0.95
                }

        # App Termination: "close <app>", "exit <app>"
        close_match = re.match(r"^(?:please\s+)?(?:close|exit|quit|shut)\s+([a-zA-Z0-9\s\.\-_]+)$", lower)
        if close_match:
            app = close_match.group(1).strip()
            return {
                "intent": "kill_process",
                "entities": {"process": app},
                "confidence": 0.95
            }

        # 3. Time & Date queries: "what time is it", "current time", "what day is it"
        if re.search(r"\b(what('s| is) the time|current time|tell me the time|what time is it|time please)\b", lower):
            return {"intent": "get_time", "entities": {}, "confidence": 0.99}

        if re.search(r"\b(what('s| is) (the |today's )?date|what day is (it|today)|date please)\b", lower):
            return {"intent": "get_date", "entities": {}, "confidence": 0.99}

        # 4. Day & Task Planning: "plan my day", "show schedule", "what do i have today"
        if re.search(r"\b(plan my day|show my (day|schedule|tasks)|what('s| is) my schedule|today's plan|what do i have today)\b", lower):
            return {"intent": "plan_day", "entities": {}, "confidence": 0.96}

        # 5. Goal decomposition: "how should i train for <goal>", "plan goal <goal>", "plan my training for <goal>"
        goal_match = re.search(r"\b(?:plan (?:my )?training for|how should i (?:train|prepare|study) for|plan goal)\s+(.+)$", lower)
        if goal_match:
            return {
                "intent": "plan_goal",
                "entities": {"goal": goal_match.group(1).strip()},
                "confidence": 0.95
            }

        # 6. Task Management:
        # "create task <title>", "add task <title> with priority <priority>"
        task_match = re.search(r"\b(?:create|add)\s+task\s+(.+?)(?:\s+with priority\s+(low|medium|high))?$", lower)
        if task_match:
            title = task_match.group(1).strip()
            priority = task_match.group(2) if task_match.group(2) else "medium"
            return {
                "intent": "create_task",
                "entities": {"title": title, "priority": priority},
                "confidence": 0.95
            }

        if re.search(r"\b(list tasks|show tasks|what are my tasks|pending tasks|my tasks)\b", lower):
            return {"intent": "list_tasks", "entities": {}, "confidence": 0.98}

        # 7. Sensitive / Confirmation-Gated Actions
        # File deletion: "delete file <path>", "remove file <path>"
        del_match = re.search(r"\b(?:delete|remove)\s+(?:file|folder)\s+(.+)$", lower)
        if del_match:
            return {
                "intent": "delete_file",
                "entities": {"target": del_match.group(1).strip()},
                "confidence": 0.95
            }

        # Process termination: "kill process <name/pid>", "stop process <name>"
        kill_match = re.search(r"\b(?:kill|terminate|stop)\s+process\s+(.+)$", lower)
        if kill_match:
            return {
                "intent": "kill_process",
                "entities": {"process": kill_match.group(1).strip()},
                "confidence": 0.95
            }

        # 8. Vision / Camera / Screen
        if re.search(r"\b(take a photo|look through camera|what do you see|camera status|check camera)\b", lower):
            return {"intent": "vision_camera", "entities": {}, "confidence": 0.95}

        if re.search(r"\b(read screen|what's on (my |the )?screen|capture screen|inspect screen)\b", lower):
            return {"intent": "vision_screen", "entities": {}, "confidence": 0.95}

        # 9. Device Connectivity:
        if re.search(r"\b(scan bluetooth|scan wifi|find devices|list devices)\b", lower):
            dev_type = "bluetooth" if "bluetooth" in lower else ("wifi" if "wifi" in lower else "all")
            return {
                "intent": "connect_scan",
                "entities": {"device_type": dev_type},
                "confidence": 0.95
            }

        # 10. System Status
        if re.search(r"\b(system status|cpu usage|battery level|memory usage|pc status|how is my pc)\b", lower):
            return {"intent": "system_status", "entities": {}, "confidence": 0.95}

        # 11. Confirmation responses: "yes", "confirm", "proceed", "no", "cancel"
        if lower in ["yes", "confirm", "proceed", "sure", "do it", "i confirm", "affirmative", "ok", "okay"]:
            return {"intent": "confirm_yes", "entities": {}, "confidence": 1.0}

        if lower in ["no", "cancel", "stop", "abort", "don't do it", "never mind", "negative"]:
            return {"intent": "confirm_no", "entities": {}, "confidence": 1.0}

        return None
