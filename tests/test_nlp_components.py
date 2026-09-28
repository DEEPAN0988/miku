"""
Unit Test Suite for Miku Upgraded English NLP & Cognitive Components.
Tests components 1 through 8 in isolation:
1. Normalizer and Tokenizer
2. Command Lexicon & Usable Teaching
3. Classical Intent Classifier
4. Slot Tagger & Entity Validator
5. Discourse Engine (Negation, Compounds, Pronouns)
6. Safe Speller (SymSpell & Destructive Safeguards)
7. Dialogue Stack & State Lock
8. Active Learning Manager
"""
import unittest
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from miku.core2_cognitive.normalizer import Normalizer
from miku.core2_cognitive.lexicon import CommandLexicon
from miku.core2_cognitive.intent_classifier import ClassicalIntentClassifier
from miku.core2_cognitive.slot_tagger import SlotTagger
from miku.core2_cognitive.discourse import DiscourseManager
from miku.core2_cognitive.safe_speller import SafeSpeller
from miku.core2_cognitive.dialogue_stack import DialogueStackManager
from miku.core2_cognitive.active_learning import ActiveLearningManager

class TestNLPComponents(unittest.TestCase):
    def setUp(self):
        self.normalizer = Normalizer()
        self.lexicon = CommandLexicon(storage_path=Path("test_usable_lexicon.json"))
        self.classifier = ClassicalIntentClassifier(model_path=Path("test_intent_model.pkl"))
        self.slot_tagger = SlotTagger()
        self.discourse = DiscourseManager()
        self.speller = SafeSpeller()
        self.dialogue = DialogueStackManager()
        self.active_learning = ActiveLearningManager(log_path=Path("test_failed_utterances.json"))

    def tearDown(self):
        for p in ("test_usable_lexicon.json", "test_intent_model.pkl", "test_failed_utterances.json"):
            file = Path(p)
            if file.exists():
                try:
                    file.unlink()
                except Exception:
                    pass

    # --- 1. Normalizer Tests ---
    def test_01_normalizer_contractions_and_fillers(self):
        res = self.normalizer.normalize("could you please fire up notepad for me")
        self.assertEqual(res["normalized"], "fire up notepad")

        res_neg = self.normalizer.normalize("don't open chrome right now")
        self.assertEqual(res_neg["normalized"], "do not open chrome")
        self.assertTrue(res_neg["has_negation"])

    def test_01_normalizer_preserves_quotes_and_numbers(self):
        res = self.normalizer.normalize('create file "my secret report.pdf"')
        self.assertIn("my secret report.pdf", res["normalized"])

        res_num = self.normalizer.normalize("choose option two please")
        self.assertIn("2", res_num["normalized"])

    # --- 2. Lexicon & Usable Teaching Tests ---
    def test_02_lexicon_natural_phrasings_and_usability(self):
        # Natural phrasing 1: "when I say surf I mean open edge"
        teach1 = self.lexicon.check_teaching_intent("when I say surf I mean open edge")
        self.assertIsNotNone(teach1)
        self.assertEqual(teach1["alias"], "surf")
        self.assertEqual(teach1["target"], "edge")

        # Register and verify usability
        self.lexicon.register_taught_word(teach1["alias"], teach1["target"])
        rewritten = self.lexicon.rewrite_taught_terms("please surf right now")
        self.assertIn("edge", rewritten)

        # Natural phrasing 2: "remember that diary is notepad"
        teach2 = self.lexicon.check_teaching_intent("can you please remember that diary is notepad")
        self.assertIsNotNone(teach2)
        self.assertEqual(teach2["alias"], "diary")
        self.assertEqual(teach2["target"], "notepad")

        # Natural phrasing 3: "calc is another word for calculator"
        teach3 = self.lexicon.check_teaching_intent("calc is another word for calculator")
        self.assertIsNotNone(teach3)
        self.assertEqual(teach3["alias"], "calc")

    # --- 3. Classical Intent Classifier Tests ---
    def test_03_intent_classifier_train_and_predict(self):
        texts = [
            "open notepad", "launch edge", "fire up calculator",
            "close notepad", "terminate edge", "shut down calculator",
            "volume up", "crank up audio", "sound down", "mute sound",
            "take screenshot", "capture screen"
        ]
        labels = [
            "OPEN_APP", "OPEN_APP", "OPEN_APP",
            "CLOSE_APP", "CLOSE_APP", "CLOSE_APP",
            "SYSTEM_VOLUME", "SYSTEM_VOLUME", "SYSTEM_VOLUME", "SYSTEM_VOLUME",
            "SCREENSHOT", "SCREENSHOT"
        ]
        self.classifier.fit(texts, labels)

        pred = self.classifier.predict("bring up calculator")
        self.assertEqual(pred["top_intent"], "OPEN_APP")
        self.assertGreater(pred["confidence"], 0.4)
        self.assertEqual(len(pred["top3"]), 3)

    # --- 4. Slot Tagger & Validation Tests ---
    def test_04_slot_extraction_and_validation(self):
        slots_vol = self.slot_tagger.extract_slots("crank up the audio to max", "SYSTEM_VOLUME")
        self.assertEqual(slots_vol.get("direction"), "up")

        slots_app = self.slot_tagger.extract_slots("launch notepad application", "OPEN_APP")
        self.assertEqual(slots_app.get("app"), "notepad")

        # Validate known app
        is_valid, canon, info = self.slot_tagger.validate_app_slot("notepad")
        self.assertTrue(is_valid)
        self.assertEqual(canon, "notepad")

        # Validate unknown app
        is_valid_fake, _, _ = self.slot_tagger.validate_app_slot("mysteryfakeapp123")
        self.assertFalse(is_valid_fake)

    # --- 5. Discourse Engine Tests ---
    def test_05_discourse_negation(self):
        # Pure negation refusal
        neg = self.discourse.analyze_negation("do not open notepad")
        self.assertTrue(neg["has_negation"])
        self.assertFalse(neg["is_correction"])

        # Correction negation
        corr = self.discourse.analyze_negation("not edge, open notepad instead")
        self.assertTrue(corr["has_negation"])
        self.assertTrue(corr["is_correction"])
        self.assertEqual(corr["target_command"], "open notepad")

    def test_05_discourse_compounds_and_pronouns(self):
        # Compound splitting
        parts = self.discourse.split_compound("open notepad and then turn up volume")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "open notepad")
        self.assertEqual(parts[1], "turn up volume")

        # Pronoun resolution
        context = {"last_app": "notepad", "last_options": ["Microsoft Edge", "Calculator"]}
        res = self.discourse.resolve_coreference("close it", context)
        self.assertTrue(res["is_resolved"])
        self.assertEqual(res["reconstructed"], "close notepad")

        res_opt = self.discourse.resolve_coreference("the second one", context)
        self.assertTrue(res_opt["is_resolved"])
        self.assertEqual(res_opt["reconstructed"], "open Calculator")

    # --- 6. Safe Speller Tests ---
    def test_06_safe_speller_behavior(self):
        # Typo in command verb
        res = self.speller.correct_sentence("opne calcultor")
        self.assertEqual(res["corrected"], "open calculator")

        # Must NOT correct valid English words
        res_valid = self.speller.correct_sentence("what can you do")
        self.assertEqual(res_valid["corrected"], "what can you do")

        # Never auto-correct into destructive command
        res_dest = self.speller.correct_token("delte")
        self.assertIn(res_dest[1], ("blocked_destructive", "ambiguous", "unknown"))

    # --- 7. Dialogue Stack Tests ---
    def test_07_dialogue_stack_interruption_and_locking(self):
        # Push clarification
        frame = self.dialogue.push_clarification("games", ["Wuthering Waves"], "Which game?", "open games")
        self.assertEqual(frame["category"], "games")
        self.assertIsNotNone(self.dialogue.get_active_clarification())

        # State lock
        acquired = self.dialogue.acquire_lock("task_123", "multi-step compound", timeout=2.0)
        self.assertTrue(acquired)
        self.assertTrue(self.dialogue.is_locked())
        self.dialogue.release_lock("task_123")
        self.assertFalse(self.dialogue.is_locked())

    # --- 8. Active Learning Tests ---
    def test_08_active_learning_logging_and_promotion(self):
        self.active_learning.log_failure("unknown weird command", "unrecognized_intent", 0.2, "UNKNOWN")
        self.active_learning.log_failure("unknown weird command", "unrecognized_intent", 0.2, "UNKNOWN")
        ranked = self.active_learning.get_ranked_failures(limit=5)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["count"], 2)

        summary = self.active_learning.format_review_summary()
        self.assertIn("unknown weird command", summary)

    # --- 9. Task 4: Teaching Safety & Validation Tests ---
    def test_09_teaching_validation_and_surf_i_mean_bug(self):
        # Regression: 'surf i mean' parsing bug
        res = self.lexicon.check_teaching_intent("when I say surf I mean open edge")
        self.assertIsNotNone(res)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["alias"], "surf")
        self.assertNotEqual(res["alias"], "surf i mean")
        self.assertEqual(res["target"], "edge")

    def test_09_ten_malformed_teaching_phrasings(self):
        from miku.core2_cognitive.lexicon import validate_teaching

        malformed_cases = [
            ("", "notepad", "empty alias"),
            ("one two three four five", "notepad", "alias longer than 4 words"),
            ("notepad", "notepad", "self-reference"),
            ("open", "close", "shadow core verb open"),
            ("delete", "calculator", "shadow core verb delete"),
            ("volume", "edge", "shadow core verb volume"),
            ("terminate", "notepad", "shadow core verb terminate"),
            ("surf", "", "empty target"),
            ("surf but edge", "chrome", "connector word in alias"),
            ("calc i mean", "calculator", "i mean in alias"),
        ]

        for alias, target, description in malformed_cases:
            is_valid, reason = validate_teaching(alias, target)
            self.assertFalse(is_valid, f"Failed on {description}: alias='{alias}', target='{target}' was incorrectly marked valid!")

    def test_09_taught_destructive_alias_requires_confirmation_at_execution(self):
        # Teach "cleanup" -> "delete file secret.txt"
        from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
        import queue
        q1 = queue.Queue()
        daemon = CognitiveDaemon(action_queue=q1)
        daemon.learner.reset()
        daemon.cmd_lexicon.reset()

        # Step 1: Bare statement prompts confirmation
        from miku.ipc.messages import STTTranscriptMsg
        teach_msg = STTTranscriptMsg(text="when I say cleanup I mean delete file secret.txt", confidence=1.0)
        res_teach = daemon.handle_transcript(teach_msg)
        self.assertEqual(res_teach.get("status"), "confirmation_required")
        self.assertIn("Link 'cleanup' to 'delete file secret.txt'?", res_teach.get("message", ""))

        # Step 2: Confirm teaching
        confirm_msg = STTTranscriptMsg(text="yes", confidence=1.0)
        res_conf = daemon.handle_transcript(confirm_msg)
        self.assertEqual(res_conf.get("status"), "conversational_response")
        self.assertIn("Learned!", res_conf.get("message", ""))

        # Step 3: Now user utters "cleanup" -> MUST trigger destructive confirmation gate!
        exec_msg = STTTranscriptMsg(text="cleanup", confidence=1.0)
        res_exec = daemon.handle_transcript(exec_msg)
        self.assertEqual(res_exec.get("status"), "confirmation_required")
        self.assertIn("delete file secret.txt", res_exec.get("message", ""))
        self.assertIn("destructive", res_exec.get("message", "").lower())

    def test_09_atomic_and_schema_validated_lexicon_writes(self):
        # Register a valid word
        rec = self.lexicon.register_taught_word("browserflow", "edge")
        self.assertEqual(rec["canonical_target"], "edge")
        self.assertTrue(self.lexicon.storage_path.exists())

        # Test forget word
        forgot = self.lexicon.forget_word("browserflow")
        self.assertTrue(forgot)
        self.assertNotIn("browserflow", self.lexicon.dynamic_lexicon)

    # --- 10. Task 5: Pronoun & Compound Safety Tests ---
    def test_10_pronoun_confirmation_names_resolved_target(self):
        context = {"last_app": "notepad", "last_file": "report.pdf"}

        # "close it" -> requires confirmation naming Notepad
        res_close = self.discourse.resolve_coreference("close it", context)
        self.assertTrue(res_close["is_resolved"])
        self.assertTrue(res_close["requires_confirmation"])
        self.assertEqual(res_close["confirmation_prompt"], "Close Notepad? (yes/no)")

        # "delete it" -> requires confirmation naming report.pdf
        res_del = self.discourse.resolve_coreference("delete it", context)
        self.assertTrue(res_del["is_resolved"])
        self.assertTrue(res_del["requires_confirmation"])
        self.assertEqual(res_del["confirmation_prompt"], "Delete Report.pdf? (yes/no)")

        # "maximize it" -> benign action, no confirmation required
        res_max = self.discourse.resolve_coreference("maximize it", context)
        self.assertTrue(res_max["is_resolved"])
        self.assertFalse(res_max["requires_confirmation"])

    def test_10_compound_command_sequential_halting(self):
        from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
        import queue
        q1 = queue.Queue()
        daemon = CognitiveDaemon(action_queue=q1)
        daemon.learner.reset()

        from miku.ipc.messages import STTTranscriptMsg

        # Test A: Failing first step halts subsequent steps
        # Step 1: "do not open edge" (negation refusal) -> must NOT run "open notepad"
        res_fail = daemon.handle_transcript(STTTranscriptMsg(text="do not open edge and then open notepad", confidence=1.0))
        self.assertEqual(res_fail.get("status"), "negation_refusal")
        self.assertEqual(res_fail.get("stopped_at_step"), 1)
        self.assertIn("stopped", res_fail.get("message", "").lower())
        self.assertEqual(len(res_fail.get("sub_results", [])), 1)

        # Test B: Destructive middle step halts sequence and requires confirmation
        # Step 1: open notepad (success), Step 2: delete file secret.txt (requires confirmation), Step 3: turn up volume (must not run)
        res_dest = daemon.handle_transcript(STTTranscriptMsg(text="open notepad and then delete file secret.txt and then turn up volume", confidence=1.0))
        self.assertEqual(res_dest.get("status"), "confirmation_required")
        self.assertEqual(res_dest.get("stopped_at_step"), 2)
        self.assertEqual(len(res_dest.get("sub_results", [])), 2)

        # Test C: "then" inside a quoted string is NOT split
        parts = self.discourse.split_compound("create file 'then and now.txt'")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], "create file 'then and now.txt'")

    def test_10_negation_scope(self):
        # "don't open edge but open chrome"
        neg_a = self.discourse.analyze_negation("don't open edge but open chrome")
        self.assertTrue(neg_a["has_negation"])
        self.assertTrue(neg_a["is_correction"])
        self.assertEqual(neg_a["target_command"], "open chrome")

        # "open chrome, not edge"
        neg_b = self.discourse.analyze_negation("open chrome, not edge")
        self.assertTrue(neg_b["has_negation"])
        self.assertTrue(neg_b["is_correction"])
        self.assertEqual(neg_b["target_command"], "open chrome")

        # "never mind"
        neg_c = self.discourse.analyze_negation("never mind")
        self.assertTrue(neg_c["has_negation"])
        self.assertFalse(neg_c["is_correction"])

if __name__ == "__main__":
    unittest.main()

