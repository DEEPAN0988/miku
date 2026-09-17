"""
Central Orchestrator & Router for Miku Assistant.
Dispatches intents to subsystems, coordinates multi-step actions,
and enforces safety confirmation gates.
"""

import time
import os
import psutil
from datetime import datetime
from typing import Dict, Any, Optional

from control.gate import RiskGate, RiskLevel
from control.powershell import PowerShellEngine
from control.system import SystemControl
from control.input_sim import InputSimulator
from planner.engine import PlannerEngine
from connect.manager import ConnectManager
from vision.engine import VisionEngine
from tts.engine import TTSEngine
from llm.generator import MikuLLM
from .dialogue import DialogueManager


class Orchestrator:
    def __init__(self, tts_enabled: bool = True):
        self.dialogue = DialogueManager()
        self.powershell = PowerShellEngine()
        self.system = SystemControl()
        self.input_sim = InputSimulator()
        self.planner = PlannerEngine()
        self.connect = ConnectManager()
        self.vision = VisionEngine()
        self.tts = TTSEngine() if tts_enabled else None
        self.llm = MikuLLM()

    def respond(self, message: str, speak: bool = True) -> str:
        """Log, speak (if enabled), and return response string."""
        if speak and self.tts:
            self.tts.speak(message, block=False)
        return message

    def handle_intent(self, nlu_result: Dict[str, Any], confirmed: bool = False) -> Dict[str, Any]:
        """
        Process parsed NLU result and dispatch to the correct subsystem.
        """
        intent = nlu_result.get("intent", "unknown")
        entities = nlu_result.get("entities", {})
        raw_text = nlu_result.get("raw_text", "")

        # 1. Handle confirmation responses when an action is pending
        if self.dialogue.has_pending_confirmation():
            pending = self.dialogue.pending_confirmation
            if intent == "confirm_yes":
                self.dialogue.clear_pending_confirmation()
                return self._execute_confirmed_action(pending)
            elif intent == "confirm_no":
                self.dialogue.clear_pending_confirmation()
                msg = self.respond("Action cancelled. No changes were made.")
                return {"success": True, "action": "cancelled", "response": msg}

        # 1.5 Conversational & Helper Intents
        if intent == "greeting":
            msg = self.respond("Hello! I am Miku, your offline personal assistant. What would you like to do?")
            return {"success": True, "response": msg}

        if intent == "identity":
            msg = self.respond("I am Miku, an offline-first AI voice assistant running locally on your PC without external API keys or cloud models.")
            return {"success": True, "response": msg}

        if intent == "status_query":
            msg = self.respond("I am online, listening, and ready for your commands.")
            return {"success": True, "response": msg}

        if intent == "help":
            help_text = (
                "You can ask me to:\n"
                " - Check time/date: 'what is the time'\n"
                " - Plan your day/goals: 'plan my day' or 'plan my training for marathon'\n"
                " - Manage tasks: 'create task finish code with priority high'\n"
                " - Launch & write: 'open notepad and write an essay about robotics'\n"
                " - System performance: 'system status'\n"
                " - Vision: 'take a photo' or 'read screen'\n"
                " - Devices: 'scan bluetooth' or 'scan wifi'\n"
                " - Manage files & apps: 'delete file <path>' or 'close <app>'"
            )
            msg = self.respond(help_text)
            return {"success": True, "response": msg}

        if intent == "thanks":
            msg = self.respond("You're very welcome! Let me know if you need anything else.")
            return {"success": True, "response": msg}

        if intent == "appreciation":
            msg = self.respond("Thank you! I'm here to help.")
            return {"success": True, "response": msg}

        if intent == "goodbye":
            msg = self.respond("Goodbye! Have a great day.")
            return {"success": True, "response": msg}

        # 2. Compositional Command: "open <app> and write an essay about <topic>"
        if intent == "compositional_app_write":
            app = entities.get("app", "notepad")
            doc_type = entities.get("document_type", "essay")
            topic = entities.get("topic", "the future of AI")
            return self._handle_compositional_write(app, doc_type, topic)

        # 3. App Launching
        if intent == "launch_app":
            app = entities.get("app", "").strip()
            if not app or len(app) < 2:
                msg = self.respond("Which application would you like me to open?")
                return {"success": False, "response": msg}
            res = self.system.launch_app(app)
            if res.get("success"):
                msg = self.respond(f"Opened {app}.")
                return {"success": True, "response": msg}
            else:
                msg = self.respond(f"Could not open {app}: {res.get('error')}")
                return {"success": False, "response": msg}

        # 4. Time & Date
        if intent == "get_time":
            now_str = datetime.now().strftime("%I:%M %p")
            msg = self.respond(f"The current time is {now_str}.")
            return {"success": True, "response": msg}

        if intent == "get_date":
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            msg = self.respond(f"Today is {date_str}.")
            return {"success": True, "response": msg}

        # 5. Day Planning & Scheduling
        if intent == "plan_day":
            plan = self.planner.get_day_plan()
            summary = plan["summary"]
            msg = self.respond(summary)
            return {"success": True, "plan": plan, "response": msg}

        if intent == "plan_goal":
            goal = entities.get("goal", "your goal")
            subtasks = self.planner.plan_goal(goal)
            msg = self.respond(f"I have created a {len(subtasks)}-step plan for {goal}.")
            return {"success": True, "tasks": subtasks, "response": msg}

        # 6. Task Management
        if intent == "create_task":
            title = entities.get("title", "New Task")
            priority = entities.get("priority", "medium")
            res = self.planner.create_task(title=title, priority=priority)
            msg = self.respond(f"Added task: '{title}' with {priority} priority.")
            return {"success": True, "task": res, "response": msg}

        if intent == "list_tasks":
            tasks = self.planner.get_tasks(status="pending")
            if not tasks:
                msg = self.respond("You have no pending tasks.")
                return {"success": True, "tasks": [], "response": msg}
            task_titles = [f"{t['id']}. {t['title']} ({t['priority']})" for t in tasks[:5]]
            summary = f"You have {len(tasks)} pending task(s): " + ", ".join(task_titles)
            msg = self.respond(summary)
            return {"success": True, "tasks": tasks, "response": msg}

        # 7. System Status
        if intent == "system_status":
            cpu = psutil.cpu_percent(interval=0.2)
            ram = psutil.virtual_memory().percent
            battery = psutil.sensors_battery()
            bat_str = f", Battery is at {battery.percent}%" if battery else ""
            status_msg = f"System performance: CPU usage is {cpu:.1f}%, RAM usage is {ram:.1f}%{bat_str}."
            msg = self.respond(status_msg)
            return {"success": True, "response": msg}

        # 8. Sensitive / Confirmation-Gated Actions
        if intent == "delete_file":
            target = entities.get("target", "")
            risk, reason = RiskGate.evaluate("file_delete", f"delete file {target}")
            self.dialogue.set_pending_confirmation(
                action_type="delete_file",
                payload={"target": target},
                prompt_message=f"I need your confirmation to delete '{target}'. Should I proceed?"
            )
            msg = self.respond(f"Deleting '{target}' is a permanent action. Please confirm: do you want to proceed?")
            return {"success": False, "requires_confirmation": True, "response": msg}

        if intent == "kill_process":
            proc = entities.get("process", "")
            self.dialogue.set_pending_confirmation(
                action_type="kill_process",
                payload={"process": proc},
                prompt_message=f"Please confirm: terminate process '{proc}'?"
            )
            msg = self.respond(f"Terminating '{proc}' requires confirmation. Should I proceed?")
            return {"success": False, "requires_confirmation": True, "response": msg}

        # 9. Vision Actions
        if intent == "vision_camera":
            res = self.vision.see_room()
            summary = res.get("summary", "Captured room image.")
            msg = self.respond(summary)
            return {"success": res.get("success", False), "data": res, "response": msg}

        if intent == "vision_screen":
            res = self.vision.see_screen()
            summary = res.get("summary", "Screen inspected.")
            msg = self.respond(summary)
            return {"success": res.get("success", False), "data": res, "response": msg}

        # 10. Connectivity
        if intent == "connect_scan":
            dev_type = entities.get("device_type", "all")
            res = self.connect.scan_all(dev_type)
            bt_count = len(res.get("bluetooth_devices", []))
            wifi_count = len(res.get("wifi_networks", []))
            summary = f"Scan complete. Discovered {bt_count} Bluetooth device(s) and {wifi_count} Wi-Fi network(s)."
            msg = self.respond(summary)
            return {"success": True, "data": res, "response": msg}

        # 11. LLM Conversational Generation for General Queries
        if len(raw_text.strip()) >= 3 and not any(k in raw_text.lower() for k in ["delete", "remove", "kill"]):
            gen_reply = self.llm.generate(raw_text)
            if gen_reply and len(gen_reply) > 6:
                msg = self.respond(gen_reply)
                return {"success": True, "intent": "llm_chat", "response": msg}

        conf = nlu_result.get("confidence", 0.0)
        if conf < 0.40 or len(raw_text.strip()) <= 2:
            msg = self.respond("I didn't quite catch that. Could you please repeat?")
        else:
            msg = self.respond(f"I heard: '{raw_text}'. Say 'help' to see my available commands.")
        return {"success": False, "intent": intent, "response": msg}

    def _handle_compositional_write(self, app: str, doc_type: str, topic: str) -> Dict[str, Any]:
        """
        Executes: Launch App -> Focus -> Type Content.
        """
        # Step 1: Launch App
        launch_res = self.system.launch_app(app)
        if not launch_res.get("success"):
            msg = self.respond(f"Failed to open {app}: {launch_res.get('error')}")
            return {"success": False, "response": msg}

        # Step 2: Brief pause for app initialization
        time.sleep(1.0)

        # Step 3: Generate local draft text
        title = f"Title: An Analysis of {topic.title() if topic else 'the Topic'}"
        body = (
            f"\n\n1. Introduction\nThis document explores {topic if topic else 'key ideas'}.\n\n"
            f"2. Core Perspectives\nKey findings show significant opportunities for local autonomous intelligence.\n\n"
            f"3. Conclusion\nAuthored locally by Miku Assistant."
        )
        full_text = title + body

        # Step 4: Type text into active window
        type_res = self.input_sim.type_text(full_text, interval=0.01)

        msg = self.respond(f"Opened {app} and wrote your {doc_type} about {topic}.")
        return {
            "success": True,
            "app": app,
            "typed_characters": type_res.get("length", 0),
            "response": msg
        }

    def _execute_confirmed_action(self, pending: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action after user confirmation."""
        action_type = pending.get("action_type")
        payload = pending.get("payload", {})

        if action_type == "delete_file":
            target = payload.get("target")
            res = self.powershell.execute(f"Remove-Item -Path '{target}' -Force", confirmed=True)
            if res.get("success"):
                msg = self.respond(f"Confirmed. File '{target}' was deleted.")
                return {"success": True, "response": msg}
            else:
                msg = self.respond(f"Could not delete '{target}': {res.get('error')}")
                return {"success": False, "response": msg}

        if action_type == "kill_process":
            proc = payload.get("process")
            res = self.system.kill_process(proc, confirmed=True)
            msg = self.respond(res.get("message", "Process terminated."))
            return {"success": res.get("success", False), "response": msg}

        msg = self.respond(f"Action '{action_type}' executed with confirmation.")
        return {"success": True, "response": msg}
