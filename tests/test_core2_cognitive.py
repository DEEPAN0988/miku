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

if __name__ == "__main__":
    unittest.main()
