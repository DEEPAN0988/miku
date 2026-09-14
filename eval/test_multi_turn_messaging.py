"""
eval/test_multi_turn_messaging.py — Multi-Turn Messaging Loop & Client Interface Unit Test Suite

Tests:
  1. Interactive Clarification on missing message body ("Message Aravind")
  2. Multi-turn conversation progression (Recipient -> Message Body -> Staging -> Confirmation)
  3. Rejection / cancellation on confirmation step
  4. Multi-turn contact disambiguation ("Message Alex" -> "Alex Chen" -> "Text")
  5. Multi-client abstraction (WhatsApp, Telegram, Discord)
  6. Circuit breaker enforcement on non-dry-run execution
"""

import os
import sys
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.abspath("."))

from tools.messaging import (
    REAL_SEND_ENABLED,
    MessagingSessionState,
    MultiTurnMessagingSession,
    WhatsAppDesktopClient,
    TelegramClient,
    DiscordClient,
    get_messaging_client,
    parse_messaging_intent,
    format_dictated_text,
)


class TestMultiTurnMessaging(unittest.TestCase):

    def setUp(self):
        self.mock_whatsapp_window = {
            "hwnd": 101,
            "is_window": True,
            "process_name": "whatsapp.exe",
            "title": "WhatsApp - Aravind",
        }
        self.mock_telegram_window = {
            "hwnd": 102,
            "is_window": True,
            "process_name": "telegram.exe",
            "title": "Telegram - @aravind",
        }
        self.mock_discord_window = {
            "hwnd": 103,
            "is_window": True,
            "process_name": "discord.exe",
            "title": "Discord - Aravind",
        }

    def test_circuit_breaker_invariant(self):
        """Verify REAL_SEND_ENABLED remains strictly False."""
        self.assertFalse(REAL_SEND_ENABLED, "REAL_SEND_ENABLED circuit breaker must be False by default.")

    def test_format_dictated_text_bare_command(self):
        """Verify bare commands with only recipient return empty string."""
        self.assertEqual(format_dictated_text("Message Aravind", recipient="Aravind"), "")
        self.assertEqual(format_dictated_text("tell david smith", recipient="David Smith"), "")
        self.assertEqual(format_dictated_text("send message to Sarah", recipient="Sarah"), "")
        # With content
        self.assertEqual(
            format_dictated_text("Message Aravind I'll be there in 10 minutes", recipient="Aravind"),
            "I'll be there in 10 minutes."
        )

    def test_parse_messaging_intent_empty_body(self):
        """Verify parse_messaging_intent flags missing body and sets clarification prompt."""
        intent = parse_messaging_intent("Message Aravind")
        self.assertTrue(intent["is_message_empty"])
        self.assertEqual(intent["resolved_recipient"], "Aravind")
        self.assertEqual(intent["clarification_prompt"], "What would you like to send?")

    def test_multi_turn_flow_nominal(self):
        """Verify complete 3-turn interactive flow: 'Message Aravind' -> text -> confirmation."""
        session = MultiTurnMessagingSession()

        # Turn 1: User specifies recipient only
        turn1 = session.process_turn("Message Aravind")
        self.assertEqual(session.state, MessagingSessionState.INTERACTIVE_CLARIFICATION)
        self.assertTrue(turn1["needs_input"])
        self.assertEqual(turn1["prompt"], "What would you like to send?")
        self.assertEqual(turn1["recipient"], "Aravind")

        # Turn 2: User provides message text
        turn2 = session.process_turn(
            "i'll be there in 10 minutes",
            dry_run=True,
            mock_window_state=self.mock_whatsapp_window,
        )
        self.assertEqual(session.state, MessagingSessionState.STAGING_PENDING_CONFIRMATION)
        self.assertTrue(turn2["needs_input"])
        self.assertEqual(turn2["text"], "I'll be there in 10 minutes.")
        self.assertIn("Ready to send message to 'Aravind'", turn2["prompt"])

        # Turn 3: User confirms send
        turn3 = session.process_turn(
            "CONFIRM SEND",
            dry_run=True,
            mock_window_state=self.mock_whatsapp_window,
        )
        self.assertEqual(session.state, MessagingSessionState.EXECUTED)
        self.assertFalse(turn3["needs_input"])
        self.assertEqual(turn3["result"]["status"], "SIMULATED_SUCCESS")

    def test_multi_turn_flow_cancellation(self):
        """Verify user can cancel/abort at confirmation step."""
        session = MultiTurnMessagingSession()
        session.process_turn("Message Aravind")
        session.process_turn("Meeting starts now", dry_run=True, mock_window_state=self.mock_whatsapp_window)

        # User says no / cancel
        turn3 = session.process_turn("no")
        self.assertEqual(session.state, MessagingSessionState.ABORTED)
        self.assertFalse(turn3["needs_input"])
        self.assertIn("cancelled", turn3["output"])

    def test_multi_turn_contact_disambiguation(self):
        """Verify multi-turn resolves contact ambiguity then prompts for message text."""
        session = MultiTurnMessagingSession()

        # Turn 1: Ambiguous contact 'Alex' (Alex Miller vs Alex Chen)
        turn1 = session.process_turn("Message Alex")
        self.assertEqual(session.state, MessagingSessionState.INTERACTIVE_CLARIFICATION)
        self.assertIn("CONTACT AMBIGUITY DETECTED", turn1["prompt"])

        # Turn 2: User clarifies contact
        turn2 = session.process_turn("Alex Chen")
        self.assertEqual(session.state, MessagingSessionState.INTERACTIVE_CLARIFICATION)
        self.assertEqual(turn2["prompt"], "What would you like to send?")
        self.assertEqual(session.recipient, "Alex Chen")

        # Turn 3: User provides text
        mock_chen = {
            "hwnd": 104,
            "is_window": True,
            "process_name": "whatsapp.exe",
            "title": "WhatsApp - Alex Chen",
        }
        turn3 = session.process_turn("See you at 5", dry_run=True, mock_window_state=mock_chen)
        self.assertEqual(session.state, MessagingSessionState.STAGING_PENDING_CONFIRMATION)
        self.assertEqual(turn3["text"], "See you at 5.")

    def test_client_registry_and_polymorphism(self):
        """Verify WhatsApp, Telegram, and Discord client adapters."""
        wa_client = get_messaging_client("whatsapp")
        self.assertIsInstance(wa_client, WhatsAppDesktopClient)

        tg_client = get_messaging_client("telegram")
        self.assertIsInstance(tg_client, TelegramClient)

        dc_client = get_messaging_client("discord")
        self.assertIsInstance(dc_client, DiscordClient)

        with self.assertRaises(ValueError):
            get_messaging_client("unsupported_app")

    def test_telegram_and_discord_simulation(self):
        """Verify simulation staging and sending on Telegram and Discord clients."""
        tg = TelegramClient()
        tg_stage = tg.stage_message("aravind", "Hello Telegram", dry_run=True, mock_window_state=self.mock_telegram_window)
        self.assertEqual(tg_stage["status"], "SIMULATED_STAGE")
        self.assertIn("tg://resolve?domain=aravind", tg_stage["uri"])

        tg_send = tg.send_message("aravind", "Hello Telegram", dry_run=True, mock_window_state=self.mock_telegram_window)
        self.assertEqual(tg_send["status"], "SIMULATED_SUCCESS")

        dc = DiscordClient()
        dc_stage = dc.stage_message("aravind", "Hello Discord", dry_run=True, mock_window_state=self.mock_discord_window)
        self.assertEqual(dc_stage["status"], "SIMULATED_STAGE")

        dc_send = dc.send_message("aravind", "Hello Discord", dry_run=True, mock_window_state=self.mock_discord_window)
        self.assertEqual(dc_send["status"], "SIMULATED_SUCCESS")

    def test_circuit_breaker_fails_closed_non_dry_run(self):
        """Verify that non-dry-run calls are blocked by circuit breaker on all clients."""
        for client in [WhatsAppDesktopClient(), TelegramClient(), DiscordClient()]:
            res = client.send_message("aravind", "Test", dry_run=False)
            self.assertEqual(res["status"], "CIRCUIT_BREAKER_BLOCKED")
            self.assertFalse(res["executed"])


if __name__ == "__main__":
    unittest.main()
