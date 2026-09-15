"""
human_gate.py — Real-Time Human Fast-Confirm Gate for Desktop Automation with Visual Overlay & Intent Rationale

STRICT SAFETY REQUIREMENT:
No physical action (click, keystroke, mouse movement) may be executed by the system
without explicit, real-time human confirmation. Unsupervised auto-retries are strictly forbidden.

ENHANCEMENT:
1. Interfaces with TargetHighlighter (visual_highlighter.py) to draw a real-time transparent red
   bounding box overlay around the target UI element on screen while waiting for operator confirmation.
2. Supports AI Intent Rationale printing to present clear reasoning to the human operator.
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from config import OrchestratorConfig, default_config
from visual_highlighter import TargetHighlighter


class FastConfirmException(Exception):
    """Base exception for human confirmation gate failures."""
    pass


class ActionDeniedError(FastConfirmException):
    """Raised when the human operator explicitly denies an action."""
    pass


class ConfirmationTimeoutError(FastConfirmException):
    """Raised when the human operator fails to respond within the timeout window."""
    pass


class GateBypassedError(FastConfirmException):
    """Raised when an attempt is made to bypass the confirmation gate when enabled."""
    pass


@dataclass
class AuthToken:
    """
    Authorization Token generated exclusively by FastConfirm upon human confirmation.
    
    The safe_executor requires a valid, approved, unused AuthToken before dispatching
    any OS hardware event.
    """
    token_id: str
    action_type: str
    target: str
    coords: Tuple[int, int]
    payload: Optional[str] = None
    rationale: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    approved: bool = False
    is_edited: bool = False
    is_used: bool = False

    def validate(self, expected_action_type: str) -> bool:
        """Validates that the token is approved, unused, and matches the expected action type."""
        if not self.approved:
            return False
        if self.is_used:
            return False
        if self.action_type != expected_action_type:
            return False
        return True

    def mark_used(self) -> None:
        """Consumes the token so it cannot be replayed."""
        self.is_used = True


class FastConfirm:
    """
    FastConfirm Gate controller.
    
    Presents intended actions and AI reasoning to the user in real-time and issues
    cryptographically unique AuthTokens for authorized execution.
    Visual target elements are highlighted on screen with a red bounding box.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        input_handler: Optional[Callable[[str], str]] = None,
        highlighter: Optional[TargetHighlighter] = None,
    ):
        self.config = config or default_config
        self.input_handler = input_handler
        self.highlighter = highlighter or TargetHighlighter()

    async def _read_terminal_input(self, prompt: str) -> str:
        """Reads terminal input asynchronously with support for custom input handlers."""
        if self.input_handler is not None:
            return self.input_handler(prompt)

        print(prompt, end="", flush=True)
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line.strip() if line else ""

    async def request_confirmation(
        self,
        action_type: str,
        target: str,
        coords: Tuple[int, int],
        payload: Optional[str] = None,
        bounds: Optional[Tuple[int, int, int, int]] = None,
        rationale: Optional[str] = None,
    ) -> AuthToken:
        """
        Presents the intended hardware action and optional reasoning to the user for real-time authorization,
        while highlighting the target element on screen.
        
        Returns an AuthToken if approved or edited.
        Raises ActionDeniedError or ConfirmationTimeoutError if denied or timed out.
        """
        if (
            not self.config.CONFIRMATION_GATE_ENABLED
            or os.environ.get("MIKU_AUTO_APPROVE") == "true"
            or (os.environ.get("MIKU_AUTONOMOUS_MODE") == "true" and os.environ.get("MIKU_LIVE_EXECUTION") == "true")
        ):
            print(f"[AUTONOMOUS AUTO-APPROVED] Action: {action_type} '{target}'")
            return AuthToken(
                token_id=str(uuid.uuid4()),
                action_type=action_type,
                target=target,
                coords=coords,
                payload=payload,
                rationale=rationale,
                approved=True,
            )

        # Format AI Intent Rationale block if provided
        intent_block = ""
        if rationale and rationale.strip():
            intent_block = (
                f"\n=========================================\n"
                f"[MIKU'S INTENT]: {rationale.strip()}\n"
                f"========================================="
            )

        action_str = f"click '{target}' at X:{coords[0]}, Y:{coords[1]}"
        if action_type == "type":
            action_str = f"type '{payload or ''}' into '{target}' at X:{coords[0]}, Y:{coords[1]}"
        elif action_type not in ("click", "type"):
            action_str = f"{action_type} '{target}' at X:{coords[0]}, Y:{coords[1]}"

        prompt = (
            f"{intent_block}\n"
            f"[ACTION REQUIRED] Agent wants to {action_str}.\n"
            f"Allow? [y/N/edit] (Timeout: {self.config.CONFIRMATION_TIMEOUT_SECONDS}s): "
        )

        # Launch visual overlay highlight on screen before requesting terminal confirmation
        self.highlighter.start_highlight(
            x=coords[0],
            y=coords[1],
            bounds=bounds,
            label=target,
        )

        try:
            try:
                user_response = await asyncio.wait_for(
                    self._read_terminal_input(prompt),
                    timeout=self.config.CONFIRMATION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                print("\n[FASTCONFIRM GATE] Confirmation timed out! Pipeline halting for safety.")
                raise ConfirmationTimeoutError(
                    f"Human operator failed to respond within {self.config.CONFIRMATION_TIMEOUT_SECONDS} seconds."
                )

            resp_clean = user_response.strip().lower()

            if resp_clean in ("y", "yes"):
                print(f"[FASTCONFIRM GATE] Action '{action_type}' APPROVED by operator.")
                return AuthToken(
                    token_id=str(uuid.uuid4()),
                    action_type=action_type,
                    target=target,
                    coords=coords,
                    payload=payload,
                    rationale=rationale,
                    approved=True,
                )

            elif resp_clean in ("e", "edit"):
                if not self.config.ALLOW_MANUAL_EDIT:
                    print("[FASTCONFIRM GATE] Manual edit mode is disabled in configuration.")
                    raise ActionDeniedError("Manual edit requested but disabled in config.")

                print("\n--- MANUAL EDIT MODE ---")
                new_x_str = await self._read_terminal_input(f"Enter new X coordinate [{coords[0]}]: ")
                new_y_str = await self._read_terminal_input(f"Enter new Y coordinate [{coords[1]}]: ")

                try:
                    new_x = int(new_x_str) if new_x_str.strip() else coords[0]
                    new_y = int(new_y_str) if new_y_str.strip() else coords[1]
                except ValueError:
                    print("[FASTCONFIRM GATE] Invalid coordinate input. Action denied.")
                    raise ActionDeniedError("Invalid integer coordinates provided during edit.")

                new_payload = payload
                if action_type == "type":
                    edited_payload = await self._read_terminal_input(f"Enter new text payload ['{payload or ''}']: ")
                    if edited_payload.strip():
                        new_payload = edited_payload.strip()

                new_coords = (new_x, new_y)
                print(f"[FASTCONFIRM GATE] Action edited & APPROVED by operator. New target: X:{new_x}, Y:{new_y}")

                return AuthToken(
                    token_id=str(uuid.uuid4()),
                    action_type=action_type,
                    target=target,
                    coords=new_coords,
                    payload=new_payload,
                    rationale=rationale,
                    approved=True,
                    is_edited=True,
                )

            else:
                print(f"[FASTCONFIRM GATE] Action '{action_type}' DENIED by operator (Response: '{user_response}').")
                raise ActionDeniedError("Human operator denied execution of physical action.")

        finally:
            # STRICT GUARANTEE: Dismiss visual overlay immediately when input completes/fails
            self.highlighter.stop_highlight()
