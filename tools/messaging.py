"""
tools/messaging.py — Desktop Messaging Automation & Safety Gating Harness (Phase 12 Remediation)

SAFETY INCIDENT REMEDIATION SPECIFICATION:
  1. HARD-CODED SAFETY CIRCUIT BREAKER:
     REAL_SEND_ENABLED is hard-coded to False.
     All real execution pathways are completely locked in code. Real sending cannot
     occur under any circumstance without a manual edit to this file by the user.
  2. ELIMINATION OF PROGRAMMATIC CONFIRMATION:
     The boolean parameter 'hitl_confirmed=True' has been eliminated from real execution.
     Live confirmation can ONLY come from an active, interactive terminal session
     (sys.stdin.isatty()) requiring exact phrase verification. In automated scripts,
     headless sessions, or tests, it fails closed immediately.
  3. MANDATORY UI TARGET VERIFICATION:
     Before staging or sending, the system independently inspects the foreground window
     via tools.ui_verifier:
       - Window focus and process executable (WhatsApp.exe / WhatsApp.Root.exe)
       - Modal collision dialog detection (rejection of "Forward", "Attach", "Settings", etc.)
       - Chat header verification (target contact name must be visible in the chat hierarchy)
     Any mismatch or verification failure immediately ABORTS the pipeline with zero keystrokes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import os
import re
import sys
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

from tools.ui_verifier import verify_chat_ui_target

# ==============================================================================
# PERMANENT SAFETY CIRCUIT BREAKER (DEFENSE IN DEPTH)
# ==============================================================================
# REAL MESSAGING EXECUTION IS PERMANENTLY DISABLED BY DEFAULT.
# To prevent automated sends, testing incidents, or accidental execution,
# this flag must remain False. Real send testing is suspended.
REAL_SEND_ENABLED: bool = False


# Default test contact book for contact resolution and ambiguity testing
DEFAULT_CONTACT_BOOK = [
    "Alex Miller",
    "Alex Chen",
    "David Smith",
    "Sarah Jenkins",
    "Mom",
    "Dad",
    "Bob Roberts",
    "Test Channel",
]


def format_dictated_text(raw_text: str, recipient: Optional[str] = None) -> str:
    """
    Constrained rule-based dictation reformatting:
    Cleans up command preambles, capitalizes sentences and pronouns,
    and applies appropriate punctuation without hallucinating new text.
    """
    text = raw_text.strip()
    if not text:
        return ""

    t = text
    # Strip leading test command prefix if present
    if t.lower().startswith("test "):
        t = t[5:].strip()

    # Strip trailing app specification (e.g. 'in whatsapp', 'on whatsapp', 'via discord')
    app_pat = r"\s+(?:on|in|via|through)\s+(?:whatsapp|discord|telegram|teams|slack)\s*$"
    t = re.sub(app_pat, "", t, flags=re.IGNORECASE).strip()

    # 1. Strip leading command and intent verbs with specific recipient if available
    cleaned = None
    names = []
    if recipient and recipient.lower() != "unknown":
        names.append(recipient)
        parts = recipient.split()
        if len(parts) > 1:
            names.append(parts[0])

    for name in names:
        bare_recip_pat = (
            r"^(?:(?:please\s+)?(?:send|tell|message|text|ping|write)(?:\s+(?:a\s+)?message)?(?:\s+to)?)\s+"
            + re.escape(name)
            + r"\s*$"
        )
        if re.search(bare_recip_pat, t, flags=re.IGNORECASE):
            return ""

        recip_pat = (
            r"^(?:(?:please\s+)?(?:send|tell|message|text|ping|write)(?:\s+(?:a\s+)?message)?(?:\s+to)?)\s+"
            + re.escape(name)
            + r"\s+"
        )
        if re.search(recip_pat, t, flags=re.IGNORECASE):
            remainder = re.sub(recip_pat, "", t, flags=re.IGNORECASE).strip()
            # Only strip greeting preamble if followed by another word
            preamble = r"^(?:that|saying\s+that|saying|asking\s+if|asking|about|(?:hey|hi|hello|please)\s+(?=\S))\s*"
            rem_clean = re.sub(preamble, "", remainder, flags=re.IGNORECASE).strip()
            cleaned = rem_clean if rem_clean else remainder
            break

    generic_bare_pat = r"^(?:(?:please\s+)?(?:send|tell|message|text|ping|write)(?:\s+(?:a\s+)?message)?(?:\s+to)?)\s+[a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?\s*$"
    if re.search(generic_bare_pat, t, flags=re.IGNORECASE):
        if not cleaned:
            return ""

    if not cleaned or cleaned == text:
        generic_pat = r"^(?:(?:please\s+)?(?:send|tell|message|text|ping|write)(?:\s+(?:a\s+)?message)?(?:\s+to)?)\s+[a-zA-Z0-9_\-\s]+?\s+(?:that|saying\s+that|saying|asking\s+if|asking|about|(?:hey|hi|hello|please)\s+(?=\S))\s*"
        cleaned = re.sub(generic_pat, "", t, flags=re.IGNORECASE).strip()

    if not cleaned:
        cleaned = t

    # If the text was "asking if X needs Y", convert to natural direct statement/inquiry
    if cleaned.lower().startswith("if "):
        cleaned = cleaned[3:].strip()

    # 2. Fix pronoun capitalization
    cleaned = re.sub(r"\bi\b", "I", cleaned)
    cleaned = re.sub(r"\bi'm\b", "I'm", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'll\b", "I'll", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi've\b", "I've", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'd\b", "I'd", cleaned, flags=re.IGNORECASE)

    # 3. Capitalize first letter of sentences
    if len(cleaned) > 1:
        cleaned = cleaned[0].upper() + cleaned[1:].lower() if cleaned.isupper() else cleaned[0].upper() + cleaned[1:]
    elif len(cleaned) == 1:
        cleaned = cleaned.upper()

    # 4. Appropriate sentence-closing punctuation
    if not cleaned.endswith((".", "!", "?")):
        first_word = cleaned.lower().split()[0] if cleaned.split() else ""
        interrogative_words = {
            "are", "is", "can", "could", "would", "will", "do", "does", "did",
            "what", "where", "when", "why", "how", "who", "which"
        }
        if first_word in interrogative_words:
            cleaned += "?"
        else:
            cleaned += "."

    return cleaned


def parse_messaging_intent(
    query: str,
    known_contacts: Optional[List[str]] = None,
    available_apps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extracts app, recipient, and message body from user query.
    Detects recipient and app ambiguity and flags them explicitly.
    """
    contacts = known_contacts if known_contacts is not None else DEFAULT_CONTACT_BOOK
    apps = available_apps if available_apps is not None else ["whatsapp"]

    q = query.strip()
    q_lower = q.lower()

    # Strip test prefix for parsing if present
    q_parse = q
    if q_parse.lower().startswith("test "):
        q_parse = q_parse[5:].strip()
        q_lower = q_parse.lower()

    # 1. Detect target app
    target_app = "whatsapp"
    detected_apps = []
    for app in ["whatsapp", "discord", "telegram", "teams", "slack"]:
        if re.search(r"\b" + re.escape(app) + r"\b", q_lower):
            detected_apps.append(app)

    is_app_ambiguous = len(detected_apps) > 1
    if detected_apps:
        target_app = detected_apps[0]

    # 2. Detect recipient
    direct_matches = []
    for c in contacts:
        c_parts = c.lower().split()
        first_name = c_parts[0]
        # Ignore generic non-person contact names when doing partial match
        if first_name in ("test", "channel"):
            if re.search(r"\b" + re.escape(c.lower()) + r"\b", q_lower):
                direct_matches.append(c)
            continue
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", q_lower):
            direct_matches.append(c)
        elif re.search(r"\b" + re.escape(first_name) + r"\b", q_lower):
            if c not in direct_matches:
                direct_matches.append(c)

    recipient = "Unknown"
    if direct_matches:
        if len(direct_matches) == 1:
            recipient = direct_matches[0]
        else:
            recipient = direct_matches[0].split()[0]
    else:
        recipient_match = re.search(
            r"\b(?:tell|message|send\s+message\s+to|send\s+to|send|text|ping)\s+([a-zA-Z0-9_\-]+)",
            q_parse,
            flags=re.IGNORECASE
        )
        if recipient_match:
            cand = recipient_match.group(1).strip()
            if cand.lower() not in ("a", "the", "my", "this", "app", "message", "text", "someone", "him", "her"):
                recipient = cand

    # 3. Disambiguate against known contacts
    matching_contacts = []
    if direct_matches:
        matching_contacts = direct_matches
    elif recipient != "Unknown":
        r_low = recipient.lower()
        for c in contacts:
            c_low = c.lower()
            if r_low == c_low or r_low in c_low.split():
                matching_contacts.append(c)

    is_contact_ambiguous = len(matching_contacts) > 1
    resolved_contact = matching_contacts[0] if len(matching_contacts) == 1 else recipient

    # 4. Extract and reformat message body
    formatted_body = format_dictated_text(q, recipient=resolved_contact)
    is_message_empty = (len(formatted_body.strip()) == 0)

    # 5. Build clarification prompts if ambiguous or missing body
    clarification_prompt = None
    if is_contact_ambiguous:
        names_str = ", ".join(f"'{c}'" for c in matching_contacts)
        clarification_prompt = (
            f"[CONTACT AMBIGUITY DETECTED] Multiple contacts match '{recipient}': {names_str}. "
            f"Please clarify which contact you would like to message."
        )
    elif is_app_ambiguous:
        apps_str = ", ".join(f"'{a}'" for a in detected_apps)
        clarification_prompt = (
            f"[APP AMBIGUITY DETECTED] Multiple messaging applications mentioned: {apps_str}. "
            f"Please clarify which app you would like to send through."
        )
    elif is_message_empty:
        clarification_prompt = "What would you like to send?"

    return {
        "app": target_app,
        "raw_recipient": recipient,
        "resolved_recipient": resolved_contact,
        "matching_contacts": matching_contacts,
        "is_contact_ambiguous": is_contact_ambiguous,
        "is_app_ambiguous": is_app_ambiguous,
        "is_message_empty": is_message_empty,
        "is_ambiguous": (is_contact_ambiguous or is_app_ambiguous or is_message_empty),
        "clarification_prompt": clarification_prompt,
        "formatted_message": formatted_body,
    }


def request_live_human_confirmation(recipient: str, text: str) -> bool:
    """
    Strict interactive human confirmation gate:
    Must be run in an interactive console (sys.stdin.isatty()).
    Cannot be bypassed by programmatic flags or scripted arguments.
    Fails closed (returns False) in non-interactive environments, test scripts, or CI.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return False

    try:
        prompt = (
            f"\n" + "=" * 80 + "\n"
            f"[LIVE HUMAN CONFIRMATION REQUIRED]\n"
            f"Target Recipient: {recipient}\n"
            f"Exact Final Text: \"{text}\"\n"
            f"Type 'CONFIRM SEND' to transmit this message to a real person, or anything else to cancel:\n"
            + "=" * 80 + "\n"
            f"Confirmation: "
        )
        resp = input(prompt).strip()
        return resp == "CONFIRM SEND"
    except Exception:
        return False


def stage_whatsapp_message(
    recipient: str,
    text: str,
    dry_run: bool = True,
    mock_window_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Stages a message in WhatsApp Desktop with pre-execution target UI verification.
    """
    phone_digits = re.sub(r"[^\d+]", "", recipient)
    encoded_text = urllib.parse.quote(text)
    
    if phone_digits:
        uri = f"whatsapp://send?phone={phone_digits}&text={encoded_text}"
    else:
        uri = f"whatsapp://send?text={encoded_text}"

    # Target UI verification
    ui_verif = verify_chat_ui_target("whatsapp", recipient, mock_window_state=mock_window_state)

    if dry_run:
        return {
            "status": "SIMULATED_STAGE",
            "executed": False,
            "dry_run": True,
            "uri": uri,
            "recipient": recipient,
            "staged_text": text,
            "target_verification": ui_verif,
            "output": (
                f"[SIMULATED STAGING] Generated URI '{uri}' for recipient '{recipient}' with text: \"{text}\". "
                f"Target verification: {ui_verif['status']}."
            )
        }

    # If real staging is attempted, verify circuit breaker first
    if not REAL_SEND_ENABLED:
        return {
            "status": "CIRCUIT_BREAKER_BLOCKED",
            "executed": False,
            "dry_run": True,
            "recipient": recipient,
            "staged_text": text,
            "target_verification": ui_verif,
            "output": "[CIRCUIT BREAKER BLOCKED] Real execution is disabled in code (REAL_SEND_ENABLED=False)."
        }

    if not ui_verif["verified"]:
        return {
            "status": ui_verif["status"],
            "executed": False,
            "dry_run": False,
            "recipient": recipient,
            "staged_text": text,
            "target_verification": ui_verif,
            "output": f"[SAFETY ABORT] {ui_verif['reason']}"
        }

    try:
        if phone_digits:
            os.startfile(uri)
            time.sleep(1.0)
            from tools.app_launcher import focus_app
            focus_app("whatsapp")
        else:
            # Active chat is already verified in foreground by ui_verif!
            # Avoid launching whatsapp://send?text=... without a phone number,
            # as that summons WhatsApp's external Share/Forward dialog modal.
            # Instead, safely stage into the active chat input field using clipboard paste (Ctrl+V).
            import win32clipboard
            import win32con
            prior_clipboard = None
            try:
                win32clipboard.OpenClipboard()
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    prior_clipboard = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
            finally:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass

            from tools.app_launcher import focus_app
            focus_app("whatsapp")
            time.sleep(0.5)

            import ctypes
            user32 = ctypes.windll.user32
            # Ctrl+V
            user32.keybd_event(0x11, 0, 0, 0)
            user32.keybd_event(0x56, 0, 0, 0)
            user32.keybd_event(0x56, 0, 2, 0)
            user32.keybd_event(0x11, 0, 2, 0)
            time.sleep(0.5)

            # Restore prior clipboard contents to prevent overwriting user clipboard data
            if prior_clipboard is not None:
                try:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(prior_clipboard, win32con.CF_UNICODETEXT)
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass

        return {
            "status": "SUCCESS",
            "executed": True,
            "dry_run": False,
            "uri": uri,
            "recipient": recipient,
            "staged_text": text,
            "target_verification": ui_verif,
            "output": f"[REAL STAGING SUCCESS] Staged message for '{recipient}' in WhatsApp Desktop window: \"{text}\""
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "executed": False,
            "dry_run": False,
            "uri": uri,
            "recipient": recipient,
            "error": str(e),
            "output": f"[ERROR] Failed to stage message in WhatsApp Desktop: {e}"
        }


def send_whatsapp_message(
    recipient: str,
    text: str,
    dry_run: bool = True,
    mock_window_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sends a message via WhatsApp Desktop with multi-layered safety gates:
      Gate 1: Hardcoded REAL_SEND_ENABLED circuit breaker.
      Gate 2: Target UI verification (process ownership, modal dialog rejection, contact visibility).
      Gate 3: Live human confirmation (request_live_human_confirmation).
    """
    # 1. Check Circuit Breaker for any non-dry-run request
    if not dry_run:
        if not REAL_SEND_ENABLED:
            return {
                "status": "CIRCUIT_BREAKER_BLOCKED",
                "executed": False,
                "dry_run": True,
                "recipient": recipient,
                "text": text,
                "error": (
                    "REAL_SEND_DISABLED: Real messaging execution is permanently disabled in code "
                    "(REAL_SEND_ENABLED=False). Only simulation mode is permitted."
                ),
                "output": (
                    "[CIRCUIT BREAKER BLOCKED] Real messaging execution is permanently locked "
                    "(REAL_SEND_ENABLED=False). Real-send testing is suspended."
                )
            }

    # 2. Run UI Target Verification
    ui_verif = verify_chat_ui_target("whatsapp", recipient, mock_window_state=mock_window_state)

    # 3. Fail-Closed Target Verification Check (Strict Abort in both Simulation and Real)
    if not ui_verif["verified"]:
        return {
            "status": ui_verif["status"],
            "executed": False,
            "dry_run": dry_run,
            "recipient": recipient,
            "text": text,
            "target_verification": ui_verif,
            "output": f"[SAFETY ABORT] {ui_verif['reason']}"
        }

    if dry_run:
        return {
            "status": "SIMULATED_SUCCESS",
            "executed": False,
            "dry_run": True,
            "recipient": recipient,
            "text": text,
            "target_verification": ui_verif,
            "output": (
                f"[SIMULATION: MOCK SEND] Verified UI target: '{recipient}'. "
                f"Would send message to '{recipient}': \"{text}\" via WhatsApp Desktop."
            )
        }

    # 4. Target Verification Gate for Live Send
    if not ui_verif["verified"]:
        return {
            "status": ui_verif["status"],
            "executed": False,
            "dry_run": False,
            "recipient": recipient,
            "text": text,
            "target_verification": ui_verif,
            "output": f"[SAFETY ABORT] Keystrokes blocked. {ui_verif['reason']}"
        }

    # 5. Live Interactive Human Confirmation Gate (Unconditional)
    confirmed = request_live_human_confirmation(recipient, text)
    if not confirmed:
        return {
            "status": "DRY_RUN_PENDING_HITL",
            "executed": False,
            "dry_run": True,
            "recipient": recipient,
            "text": text,
            "output": (
                f"[GUARDRAIL BLOCKED] Action 'Send Message' requires explicit live human confirmation. "
                f"Planned recipient: '{recipient}' | Final text: \"{text}\""
            ),
            "confirmation_prompt": (
                f"[CONFIRMATION REQUIRED] Ready to send message to '{recipient}':\n"
                f"  Text: \"{text}\"\n"
                f"Confirm send? (yes/no)"
            )
        }

    # 6. Real Keystroke Dispatch (Only reachable if REAL_SEND_ENABLED=True, UI verified, and human confirmed)
    try:
        stage_res = stage_whatsapp_message(recipient, text, dry_run=False, mock_window_state=mock_window_state)
        if stage_res["status"] != "SUCCESS":
            return stage_res

        time.sleep(1.5)
        import ctypes
        user32 = ctypes.windll.user32
        # VK_RETURN = 0x0D
        user32.keybd_event(0x0D, 0, 0, 0)
        user32.keybd_event(0x0D, 0, 2, 0)

        return {
            "status": "SUCCESS",
            "executed": True,
            "dry_run": False,
            "recipient": recipient,
            "text": text,
            "output": f"[REAL SEND SUCCESS] Successfully sent message to '{recipient}': \"{text}\" via WhatsApp Desktop."
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "executed": False,
            "dry_run": False,
            "recipient": recipient,
            "text": text,
            "output": f"[ERROR] Failed during real message dispatch: {e}"
        }
