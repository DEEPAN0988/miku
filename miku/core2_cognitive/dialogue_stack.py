"""
Hierarchical Dialogue State Stack and In-Flight State Locker.
Core 2: Cognitive Router (Brain & Memory)
Provides:
1. Multi-turn dialogue context stack (pushes on interruption instead of wiping).
2. Activity-based expiration rather than fixed rigid timeout.
3. State lock (dead-man's switch) protecting in-flight multi-step executions.
4. Recent entity tracking (last_app, last_file, last_command, last_options) for pronoun coreference.
"""
import time
import uuid
from typing import Dict, Any, List, Optional

DEFAULT_EXPIRY_SECONDS = 900.0  # 15 minutes of idle time before expiry

class DialogueStackManager:
    def __init__(self, expiry_seconds: float = DEFAULT_EXPIRY_SECONDS):
        self.expiry_seconds = expiry_seconds
        self.stack: List[Dict[str, Any]] = []
        self.entity_memory: Dict[str, Any] = {
            "last_app": None,
            "last_file": None,
            "last_command": None,
            "last_options": [],
            "history": []
        }
        self.in_flight_lock: Optional[Dict[str, Any]] = None

    # --- In-Flight State Locking (Dead-man's switch) ---
    def acquire_lock(self, task_id: str, description: str, timeout: float = 30.0) -> bool:
        now = time.time()
        if self.in_flight_lock and (now - self.in_flight_lock["acquired_at"] < self.in_flight_lock["timeout"]):
            # Currently locked by another task
            return False
        self.in_flight_lock = {
            "task_id": task_id,
            "description": description,
            "acquired_at": now,
            "timeout": timeout
        }
        return True

    def release_lock(self, task_id: str):
        if self.in_flight_lock and self.in_flight_lock.get("task_id") == task_id:
            self.in_flight_lock = None

    def is_locked(self) -> bool:
        if not self.in_flight_lock:
            return False
        if time.time() - self.in_flight_lock["acquired_at"] >= self.in_flight_lock["timeout"]:
            # Lock timed out (dead-man switch tripped)
            self.in_flight_lock = None
            return False
        return True

    # --- Dialogue Context Stack Management ---
    def push_clarification(self, category: str, options: List[str], prompt: str, origin_query: str) -> Dict[str, Any]:
        """
        Pushes a new clarification context onto the stack.
        If an existing context is pending, it remains underneath (can be resumed).
        """
        now = time.time()
        frame = {
            "id": str(uuid.uuid4())[:8],
            "category": category,
            "options": options,
            "prompt": prompt,
            "origin_query": origin_query,
            "created_at": now,
            "last_activity": now,
            "status": "pending"
        }
        self.stack.append(frame)
        self.record_entity("last_options", options)
        return frame

    def get_active_clarification(self) -> Optional[Dict[str, Any]]:
        """
        Returns the top active, unexpired clarification context, pruning dead frames.
        """
        now = time.time()
        while self.stack:
            top = self.stack[-1]
            if now - top["last_activity"] > self.expiry_seconds:
                # Expired
                self.stack.pop()
            else:
                top["last_activity"] = now
                return top
        return None

    def pop_clarification(self) -> Optional[Dict[str, Any]]:
        if self.stack:
            return self.stack.pop()
        return None

    def clear_all_clarifications(self):
        self.stack.clear()

    # --- Entity Memory for Pronoun Resolution ---
    def record_entity(self, key: str, value: Any):
        if value:
            self.entity_memory[key] = value
            self.entity_memory["history"].append((key, value, time.time()))
            if len(self.entity_memory["history"]) > 20:
                self.entity_memory["history"].pop(0)

    def get_entity_context(self) -> Dict[str, Any]:
        return dict(self.entity_memory)
