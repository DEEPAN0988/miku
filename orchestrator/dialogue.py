"""
Dialogue Manager for Miku.
Tracks conversational state, pending confirmations, and context memory.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class DialogueManager:
    def __init__(self):
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        self.last_intent: Optional[str] = None
        self.last_response: Optional[str] = None
        self.history = []

    def set_pending_confirmation(self, action_type: str, payload: Any, prompt_message: str):
        self.pending_confirmation = {
            "action_type": action_type,
            "payload": payload,
            "prompt": prompt_message,
            "timestamp": datetime.now().isoformat()
        }

    def clear_pending_confirmation(self):
        self.pending_confirmation = None

    def has_pending_confirmation(self) -> bool:
        return self.pending_confirmation is not None

    def record_turn(self, user_text: str, intent: str, response_text: str):
        self.last_intent = intent
        self.last_response = response_text
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "intent": intent,
            "response": response_text
        })
        if len(self.history) > 50:
            self.history.pop(0)
