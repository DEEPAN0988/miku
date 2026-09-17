import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.tokenizer import MikuTokenizer
from llm.model import MikuTransformerLM
from llm.generator import MikuLLM
import torch


class TestMikuLLM(unittest.TestCase):
    def test_tokenizer_encode_decode(self):
        tok = MikuTokenizer()
        tok.train_from_corpus(["<user> hello miku <miku> greetings <eos>"])
        encoded = tok.encode("hello miku")
        self.assertTrue(len(encoded) > 0)
        decoded = tok.decode(encoded)
        self.assertIn("hello", decoded.lower())

    def test_transformer_forward(self):
        model = MikuTransformerLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_ff=64)
        x = torch.randint(0, 64, (2, 10))
        logits, loss = model(x, targets=x)
        self.assertEqual(logits.shape, (2, 10, 64))
        self.assertIsNotNone(loss)

    def test_llm_generation(self):
        llm = MikuLLM()
        res = llm.generate("who are you")
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 5)


if __name__ == "__main__":
    unittest.main()
