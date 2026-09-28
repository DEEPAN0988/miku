"""
Test Suite: Custom Open-Ended Chat Engine & Custom Deep Scene Understanding.
Verifies from-scratch neural and spatial modules operating with Zero APIs and Zero Pretrained Weights.
"""
import unittest
import numpy as np
from pathlib import Path
import tempfile
import torch

from miku.core2_cognitive.custom_chat_engine import CustomChatEngine, CustomTokenizer, CustomCausalLM
from miku.core4_vision.scene_understanding import CustomSceneUnderstanding
from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
from miku.ipc.messages import STTTranscriptMsg
from multiprocessing import Queue

class TestCustomChatAndScene(unittest.TestCase):
    def test_custom_tokenizer_and_model(self):
        """
        Verifies from-scratch subword tokenizer and causal transformer architecture.
        """
        tok = CustomTokenizer()
        texts = ["hello world", "miku is a sovereign assistant", "how are you today"]
        tok.train(texts, max_vocab=50)

        encoded = tok.encode("hello miku assistant")
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 2)

        decoded = tok.decode(encoded)
        self.assertIn("hello", decoded.lower())

        # Test model forward pass
        vocab_size = len(tok.vocab)
        model = CustomCausalLM(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=2, max_seq_len=64)
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        logits, _ = model(input_tensor)
        self.assertEqual(logits.shape, (1, len(encoded), vocab_size))

    def test_custom_chat_engine_responses(self):
        """
        Verifies open-ended dialogue generation without cloud APIs or commercial weights.
        """
        chat = CustomChatEngine()

        # Direct corpus reflex queries
        r1 = chat.respond("who are you")
        self.assertIn("miku", r1.lower())

        r2 = chat.respond("what can you do")
        self.assertIn("control", r2.lower())

        r3 = chat.respond("why zero api")
        self.assertIn("zero-api", r3.lower())

    def test_custom_scene_understanding_analysis(self):
        """
        Verifies Spatial Pyramid Matching and scene classification.
        """
        scene_engine = CustomSceneUnderstanding()

        # Synthetic screen workstation frame (bright top, high gradient contrast)
        frame = np.full((120, 160, 3), 180, dtype=np.uint8)
        frame[20:60, 20:140] = 30  # High contrast code editor window
        res = scene_engine.analyze_scene(frame)

        self.assertIn("scene_class", res)
        self.assertIn("description", res)
        self.assertIn("meta", res)
        self.assertGreater(res["confidence"], 0.0)
        self.assertIn("lighting", res["meta"])

    def test_cognitive_daemon_conversational_turn(self):
        """
        Verifies that conversational queries trigger custom conversational reflexes.
        """
        action_q = Queue()
        daemon = CognitiveDaemon(action_queue=action_q)

        msg = STTTranscriptMsg(text="who are you", confidence=1.0)
        decision = daemon.handle_transcript(msg)

        self.assertEqual(decision["status"], "conversational_response")
        self.assertIn("miku", decision["message"].lower())

if __name__ == "__main__":
    unittest.main()
