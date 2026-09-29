"""
test_outcome_integrity.py

Regression test: proves that self.chat_engine.respond() output can NEVER
become a logged outcome in miku_misses.jsonl or miku_failed_utterances.json.

Strategy:
  1. Inject a sentinel-returning stub for chat_engine.respond() so any leak is
     immediately detectable (it returns a unique string, not "I am Miku...").
  2. Feed utterances that are GUARANTEED to reach the chat engine path through
     each of the three code paths that call chat_engine.respond():
       Path A (line 598-606): QA-prefix utterances ("what is..." / "how do...")
       Path B (line 700-709): low-confidence classifier fallback with QA prefix
       Path C (line 851-854): final else clause (no grammar match, no clf match)
  3. Intercept log_miss() with a spy and assert it was never called with the
     sentinel string.
  4. Separately assert that when log_miss() IS called (Path C triggers it),
     the outcome field matches the fixed status= | top_intent= pattern.
"""

import os
import re
import sys
import json
import time
import tempfile
import unittest
from queue import Queue
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Sentinel: if this string ever appears in a logged outcome, it proves leakage.
# ---------------------------------------------------------------------------
CHAT_ENGINE_SENTINEL = "__SENTINEL_LM_OUTPUT_DO_NOT_LOG_5c3a9f__"

# Expected pattern for a legitimate outcome string
REAL_OUTCOME_RE = re.compile(
    r"^status=(?:conversational_response|clarification_needed|negation_refusal)"
    r" \| (?:top_intent|question)=.+",
    re.IGNORECASE
)


class TestOutcomeIntegrity(unittest.TestCase):

    def setUp(self):
        """Instantiate the daemon with a stubbed chat engine."""
        from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
        self.daemon = CognitiveDaemon(action_queue=Queue())

        # Replace the real chat engine with a sentinel-returning stub.
        # The stub returns CHAT_ENGINE_SENTINEL for every query so any leakage
        # into a log is instantly detectable.
        self.daemon.chat_engine = MagicMock()
        self.daemon.chat_engine.respond = MagicMock(return_value=CHAT_ENGINE_SENTINEL)

        # Collect every call to log_miss() in a spy list.
        self.log_spy = []

        from miku.ipc.messages import STTTranscriptMsg
        self.STTTranscriptMsg = STTTranscriptMsg

    def _make_msg(self, text):
        return self.STTTranscriptMsg(text=text, confidence=1.0)

    # ------------------------------------------------------------------
    # Path A: QA-prefix early return (line 598-606 in cognitive_daemon.py)
    # chat_engine.respond() is called, then `return` immediately.
    # The miss logger at line 862 is NEVER reached.
    # ------------------------------------------------------------------
    def test_path_A_qa_prefix_no_log(self):
        """QA prefix triggers early return — log_miss must never be called."""
        logged = []
        with patch("miku.core2_cognitive.miss_reviewer.log_miss",
                   side_effect=lambda *a, **kw: logged.append(a)):
            result = self.daemon.handle_transcript(self._make_msg("what is the capital of france"))

        # The chat engine WAS called
        self.daemon.chat_engine.respond.assert_called()

        # log_miss was NOT called at all
        self.assertEqual(logged, [],
            "Path A: log_miss was called but the QA-prefix path returns before reaching it")

        # The reply to the user IS the sentinel (as expected — the stub returned it)
        self.assertEqual(result["message"], CHAT_ENGINE_SENTINEL)

    # ------------------------------------------------------------------
    # Path B: Low-confidence classifier fallback with QA prefix (line 700-709)
    # chat_engine.respond() is called, then `return` immediately.
    # The miss logger at line 862 is NEVER reached.
    # ------------------------------------------------------------------
    def test_path_B_low_confidence_qa_no_log(self):
        """
        Force a low-confidence result AND a QA prefix so the daemon calls
        chat_engine.respond() inside the classifier fallback block and
        returns early (line 704). log_miss must not be called.
        """
        # Stub the classifier to return very low confidence
        self.daemon.intent_classifier = MagicMock()
        self.daemon.intent_classifier.predict = MagicMock(return_value={
            "top_intent": "OPEN_APP",
            "confidence": 0.05,     # below any reasonable threshold
            "is_confident": False,
            "top3": [["OPEN_APP", 0.05], ["CLOSE_APP", 0.03], ["SEARCH_FILE", 0.02]],
            "clarification_prompt": "Did you mean to open an app?"
        })

        logged = []
        with patch("miku.core2_cognitive.miss_reviewer.log_miss",
                   side_effect=lambda *a, **kw: logged.append(a)):
            result = self.daemon.handle_transcript(
                self._make_msg("why is the sky blue")
            )

        # log_miss was NOT called (early return at line 704)
        self.assertEqual(logged, [],
            "Path B: log_miss was called but classifier-fallback QA path returns before reaching it")

        # The reply to the user is the sentinel
        self.assertEqual(result["message"], CHAT_ENGINE_SENTINEL)

    # ------------------------------------------------------------------
    # Path C: Final else clause (line 851-854)
    # chat_reply is stored in decision_result["message"], then the logger
    # is reached at line 862.  The logger must use real_outcome, not message.
    # ------------------------------------------------------------------
    def test_path_C_final_else_outcome_never_contains_sentinel(self):
        """
        Force the final else clause so chat_engine.respond() sets
        decision_result['message'] = SENTINEL.  The miss logger IS reached,
        but outcome must be the real system status string, never the sentinel.
        """
        # Use an utterance that bypasses grammar, classifier, and every
        # early-return guard: a low-confidence non-QA utterance.
        self.daemon.intent_classifier = MagicMock()
        self.daemon.intent_classifier.predict = MagicMock(return_value={
            "top_intent": "OUT_OF_SCOPE",
            "confidence": 0.04,
            "is_confident": False,
            "top3": [["OUT_OF_SCOPE", 0.04], ["OPEN_APP", 0.02], ["CLOSE_APP", 0.01]],
            "clarification_prompt": "I am not sure what you mean."
        })

        logged = []
        with patch("miku.core2_cognitive.miss_reviewer.log_miss",
                   side_effect=lambda *a, **kw: logged.append(a)):
            result = self.daemon.handle_transcript(
                self._make_msg("xyzzy frobnicate quux")   # guaranteed nonsense
            )

        # Whatever happened, the sentinel must NEVER appear in any logged outcome
        for call_args in logged:
            # call_args = (text, outcome, confidence, top3)
            outcome = call_args[1] if len(call_args) > 1 else ""
            self.assertNotIn(
                CHAT_ENGINE_SENTINEL, str(outcome),
                f"LEAK DETECTED: chat engine sentinel found in logged outcome: {outcome!r}"
            )

        # If log_miss WAS called, the outcome must match the real status pattern
        for call_args in logged:
            outcome = call_args[1] if len(call_args) > 1 else ""
            self.assertRegex(
                outcome,
                REAL_OUTCOME_RE,
                f"Logged outcome does not match status= pattern: {outcome!r}"
            )

    # ------------------------------------------------------------------
    # Structural proof: all chat_engine.respond() call sites either
    #   (a) return early before the miss logger, or
    #   (b) store the result in decision_result["message"] which the
    #       fixed logger explicitly ignores.
    # We verify this by checking the source code itself.
    # ------------------------------------------------------------------
    def test_source_code_no_chat_reply_in_log_miss_call(self):
        """
        Parse cognitive_daemon.py and assert that the argument passed to
        log_miss() is never `decision_result.get('message', ...)` or
        `chat_reply`.  It must be `real_outcome`.
        """
        import ast

        daemon_path = Path(__file__).parent.parent / "miku" / "core2_cognitive" / "cognitive_daemon.py"
        source = daemon_path.read_text(encoding="utf-8")

        # Find every call to log_miss in the source
        tree = ast.parse(source)
        log_miss_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == "log_miss":
                    log_miss_calls.append(node)

        self.assertGreater(len(log_miss_calls), 0, "No log_miss calls found in cognitive_daemon.py")

        for call in log_miss_calls:
            # The second argument (outcome) must NOT be:
            #   - decision_result.get("message", ...) — that's LM text
            #   - chat_reply — that's LM text
            # It MUST reference real_outcome
            if len(call.args) >= 2:
                outcome_arg = call.args[1]
                outcome_src = ast.unparse(outcome_arg)
                self.assertNotIn("message", outcome_src,
                    f"log_miss outcome arg references 'message': {outcome_src!r}")
                self.assertNotIn("chat_reply", outcome_src,
                    f"log_miss outcome arg references 'chat_reply': {outcome_src!r}")
                self.assertIn("real_outcome", outcome_src,
                    f"log_miss outcome arg is not 'real_outcome': {outcome_src!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
