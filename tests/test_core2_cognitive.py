"""
Test Suite: Core 2 Cognitive Router (Brain & Memory).
Covers CDSH Softmax Normalization, Time-Decay Monotonicity, Task Lock (Bottleneck #5),
Golden-Set Intent Routing, BM25 Memory Retrieval, and Adversarial Boundaries.
"""
import unittest
import numpy as np
from pathlib import Path
import tempfile
import time

from miku.core2_cognitive.cdsh_router import CDSHRouter
from miku.core2_cognitive.grammar_router import DeterministicGrammarRouter
from miku.core2_cognitive.bm25_memory import BM25Memory
from miku.core2_cognitive.calibration_daemon import CalibrationDaemon

class TestCore2Cognitive(unittest.TestCase):
    def test_cdsh_softmax_normalization(self):
        """
        Asserts sum(confidence scores) == 1.0 (within float epsilon) across randomized inputs.
        """
        router = CDSHRouter()
        np.random.seed(123)

        for _ in range(50):
            actions = [{"action": f"act_{i}"} for i in range(5)]
            s_l = {f"act_{i}": float(np.random.rand()) for i in range(5)}
            s_s = {f"act_{i}": float(np.random.rand()) for i in range(5)}
            s_v = {f"act_{i}": float(np.random.rand()) for i in range(5)}

            _, _, conf_map, _ = router.evaluate_candidates(
                candidate_actions=actions,
                s_lang_map=s_l,
                s_state_map=s_s,
                s_vision_map=s_v,
                delta_t_state=float(np.random.rand() * 10),
                delta_t_vision=float(np.random.rand() * 10)
            )

            total_prob = sum(conf_map.values())
            self.assertAlmostEqual(total_prob, 1.0, places=5, msg="Softmax confidence must sum to 1.0")

    def test_time_decay_correctness(self):
        """
        Feeds identical state at t=0 and t=Delta_t.
        Asserts confidence strictly decreases with Delta_t per the e^(-lambda*Delta_t) term.
        """
        router = CDSHRouter(w_lang=0.0, w_state=1.0, w_vision=0.0)
        candidates = [{"action": "target_act"}, {"action": "other_act"}]
        s_state = {"target_act": 1.0, "other_act": 0.0}

        # Decision at t = 0
        _, conf_t0, _, _ = router.evaluate_candidates(
            candidates, {}, s_state, {}, delta_t_state=0.0, delta_t_vision=0.0
        )

        # Decision at t = 20s
        _, conf_t20, _, _ = router.evaluate_candidates(
            candidates, {}, s_state, {}, delta_t_state=20.0, delta_t_vision=0.0
        )

        self.assertGreater(conf_t0, conf_t20, "State confidence must strictly decrease over elapsed time")

    def test_task_state_lock_and_dead_man_switch(self):
        """
        Bottleneck #5 Fix: Freezes delta_t advancement for compound tasks.
        Verifies dead-man switch unlocks if hung.
        """
        router = CDSHRouter(dead_man_timeout=0.2)  # 200ms timeout for test
        router.start_task_lock(task_id="compound_42", current_delta_t=1.0)
        self.assertTrue(router.in_flight)
        self.assertEqual(router.frozen_delta_t, 1.0)

        # Normal release
        released = router.release_task_lock("compound_42")
        self.assertTrue(released)
        self.assertFalse(router.in_flight)

        # Test dead-man switch timeout
        router.start_task_lock(task_id="stalled_task", current_delta_t=1.0)
        time.sleep(0.25)
        force_unlocked = router.check_dead_man_switch()
        self.assertTrue(force_unlocked, "Dead-man switch must force-unlock stalled task")
        self.assertFalse(router.in_flight)

    def test_golden_set_intent_routing(self):
        """
        Golden-set intent accuracy on curated (utterance -> expected action) pairs.
        """
        grammar = DeterministicGrammarRouter()
        golden_set = [
            ("open notepad", "open_app"),
            ("launch calculator", "open_app"),
            ("close window", "close_app"),
            ("volume up", "volume"),
            ("delete file secret.txt", "delete_file"),
            ("plan my day", "plan_day"),
            ("click red icon", "click_target"),
            ("take a screenshot", "screenshot"),
            ("browse to google.com", "browser_navigate"),
            ("status report", "status_report")
        ]

        misses = 0
        for utterance, expected_act in golden_set:
            intent, act, params, score = grammar.parse(utterance)
            if act != expected_act:
                misses += 1

        miss_rate = misses / len(golden_set)
        self.assertEqual(miss_rate, 0.0, f"Golden-set miss rate must be 0%, got {miss_rate}")
        self.assertEqual(grammar.coverage_ratio, 1.0)

    def test_expanded_app_and_browser_routing(self):
        """
        Verifies extended application alias matching, predictive canonical resolution, and conversational parsing.
        """
        grammar = DeterministicGrammarRouter()
        queries = [
            ("open calc", "OPEN_APP", "open_app", "calculator"),
            ("open calculator", "OPEN_APP", "open_app", "calculator"),
            ("open insta", "OPEN_APP", "open_app", "instagram"),
            ("hey miku please open insta", "OPEN_APP", "open_app", "instagram"),
            ("can you open instagram please", "OPEN_APP", "open_app", "instagram"),
            ("open up insta", "OPEN_APP", "open_app", "instagram"),
            ("open fb", "OPEN_APP", "open_app", "facebook"),
            ("launch yt", "OPEN_APP", "open_app", "youtube"),
            ("open wuthering wave", "OPEN_APP", "open_app", "wuthering waves"),
            ("open wuthering waves", "OPEN_APP", "open_app", "wuthering waves"),
            ("open wuwa", "OPEN_APP", "open_app", "wuthering waves"),
            ("open whatsapp", "OPEN_APP", "open_app", "whatsapp"),
            ("open wa", "OPEN_APP", "open_app", "whatsapp"),
            ("open wp", "OPEN_APP", "open_app", "whatsapp"),
            ("open edge", "OPEN_APP", "open_app", "edge"),
            ("open browser", "OPEN_APP", "open_app", "edge"),
            ("open chrome", "OPEN_APP", "open_app", "chrome"),
            ("open vscode", "OPEN_APP", "open_app", "vscode"),
            ("open code", "OPEN_APP", "open_app", "vscode"),
            ("open paint", "OPEN_APP", "open_app", "paint"),
            ("open terminal", "OPEN_APP", "open_app", "terminal"),
            ("open settings", "OPEN_APP", "open_app", "settings"),
            ("open the notepad app", "OPEN_APP", "open_app", "notepad"),
            ("open calcultor", "OPEN_APP", "open_app", "calculator"),
            ("open instgram", "OPEN_APP", "open_app", "instagram"),
            ("close calc", "CLOSE_APP", "close_app", "calculator"),
            ("close edge", "CLOSE_APP", "close_app", "edge"),
            ("close the calculator app", "CLOSE_APP", "close_app", "calculator"),
            ("open google.com", "BROWSER_NAVIGATE", "browser_navigate", "google.com"),
            ("browse to https://github.com", "BROWSER_NAVIGATE", "browser_navigate", "https://github.com"),
            ("open file readme.md", "OPEN_FILE", "open_file", "readme.md"),
            ("turn up the volume", "SYSTEM_VOLUME", "volume", "up"),
            ("mute audio", "SYSTEM_VOLUME", "volume", "mute")
        ]

        for text, exp_intent, exp_act, exp_target in queries:
            intent, act, params, score = grammar.parse(text)
            self.assertEqual(intent, exp_intent, f"Failed intent for '{text}': got {intent}")
            self.assertEqual(act, exp_act, f"Failed act for '{text}': got {act}")
            target = params.get("app") or params.get("target") or params.get("url") or params.get("path") or params.get("direction")
            self.assertEqual(target, exp_target, f"Failed target for '{text}': got {target}")

    def test_bm25_memory_exact_retrieval_and_graceful_not_found(self):
        """
        Verifies Okapi BM25 exact match recall and graceful 'not found' on unindexed queries.
        """
        temp_dir = Path(tempfile.mkdtemp())
        temp_db = temp_dir / "test_bm25.db"

        try:
            mem = BM25Memory(db_path=temp_db)
            mem.store("wifi_pass", "The office Wi-Fi password is superSecretPassword123")
            mem.store("birthday", "Alice birthday is September 28")

            # Exact query matches
            results = mem.search("superSecretPassword123")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["key"], "wifi_pass")

            # Graceful 'not found' on completely unrelated query
            miss_results = mem.search("quantum teleportation black hole entropy")
            self.assertEqual(len(miss_results), 0, "BM25 must return empty list without hallucination")
        finally:
            import gc, shutil
            gc.collect()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_wuthering_waves_vs_whatsapp_separation(self):
        """
        Guarantees 'wuthering wave' never falsely collides with WhatsApp 'wa' alias.
        """
        from miku.core2_cognitive.app_catalog import predict_app
        canonical_ww1, _ = predict_app("wuthering wave")
        canonical_ww2, _ = predict_app("wuthering waves")
        canonical_wuwa, _ = predict_app("wuwa")
        canonical_wa, _ = predict_app("wa")
        canonical_whatsapp, _ = predict_app("whatsapp")

        self.assertEqual(canonical_ww1, "wuthering waves")
        self.assertEqual(canonical_ww2, "wuthering waves")
        self.assertEqual(canonical_wuwa, "wuthering waves")
        self.assertEqual(canonical_wa, "whatsapp")
        self.assertEqual(canonical_whatsapp, "whatsapp")

    def test_category_understanding_clarification_and_realtime_learning(self):
        """
        Tests multi-turn clarification for category commands like 'open games':
        1. 'open games' prompts user for clarification with installed games.
        2. User selects 'wuthering wave' -> dispatches and saves preference in real time.
        3. Subsequent 'open games' automatically dispatches the learned preferred game.
        """
        from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
        from miku.ipc.messages import STTTranscriptMsg
        import queue

        action_q = queue.Queue()
        response_q = queue.Queue()
        daemon = CognitiveDaemon(action_queue=action_q, response_queue=response_q)

        # Clear any prior memory for fresh test
        daemon.learner.reset()

        # Step 1: User says 'open games'
        msg1 = STTTranscriptMsg(text="open games", confidence=1.0)
        res1 = daemon.handle_transcript(msg1)

        self.assertEqual(res1.get("status"), "clarification_needed")
        self.assertIn("Which game would you like to open?", res1.get("message", ""))
        self.assertIsNotNone(daemon.pending_clarification)
        self.assertEqual(daemon.pending_clarification["category"], "games")

        # Step 2: User responds with 'wuthering wave'
        msg2 = STTTranscriptMsg(text="wuthering wave", confidence=1.0)
        res2 = daemon.handle_transcript(msg2)

        self.assertEqual(res2.get("status"), "dispatched")
        self.assertEqual(res2.get("action_msg").target, "wuthering waves")
        self.assertIn("Wuthering Waves", res2.get("message", ""))
        self.assertIsNone(daemon.pending_clarification)

        # Step 3: User says 'open games' again -> should immediately dispatch learned preference!
        msg3 = STTTranscriptMsg(text="open games", confidence=1.0)
        res3 = daemon.handle_transcript(msg3)

        self.assertEqual(res3.get("status"), "dispatched")
        self.assertEqual(res3.get("action_msg").target, "wuthering waves")
        self.assertIn("preferred game", res3.get("message", "").lower())

        # Cleanup
        daemon.learner.reset()

    def test_realtime_teaching_intent(self):
        """
        Tests explicit real-time teaching: 'when I say games open wuthering wave'
        """
        from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
        from miku.ipc.messages import STTTranscriptMsg
        import queue

        action_q = queue.Queue()
        response_q = queue.Queue()
        daemon = CognitiveDaemon(action_queue=action_q, response_queue=response_q)
        daemon.learner.reset()

        teach_msg = STTTranscriptMsg(text="when I say games open wuthering wave", confidence=1.0)
        teach_res = daemon.handle_transcript(teach_msg)

        self.assertEqual(teach_res.get("status"), "conversational_response")
        self.assertIn("Whenever you say 'games'", teach_res.get("message", ""))

        # Now test triggering it
        cmd_msg = STTTranscriptMsg(text="games", confidence=1.0)
        cmd_res = daemon.handle_transcript(cmd_msg)
        self.assertEqual(cmd_res.get("status"), "dispatched")
        self.assertEqual(cmd_res.get("action_msg").target, "wuthering waves")

        # Query what was learned
        query_msg = STTTranscriptMsg(text="what have you learned", confidence=1.0)
        query_res = daemon.handle_transcript(query_msg)
        self.assertIn("Category Preferences", query_res.get("message", ""))

        # Reset
        daemon.learner.reset()

if __name__ == "__main__":
    unittest.main()
