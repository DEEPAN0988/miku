"""
Inference & Text Generation Engine for Miku LLM.
Runs local autoregressive generation on CPU in milliseconds.
Zero external API keys, zero cloud models.
"""

import os
import torch
from typing import Optional

from .model import MikuTransformerLM
from .tokenizer import MikuTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "miku_llm.pth")
VOCAB_PATH = os.path.join(DATA_DIR, "miku_vocab.json")


class MikuLLM:
    def __init__(self, model_path: str = MODEL_PATH, vocab_path: str = VOCAB_PATH):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.tokenizer: Optional[MikuTokenizer] = None
        self.model: Optional[MikuTransformerLM] = None
        self._load_or_train()

    def _load_or_train(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vocab_path):
            from .trainer import train_miku_llm
            train_miku_llm(save_path=self.model_path, vocab_path=self.vocab_path)

        # Load Tokenizer
        self.tokenizer = MikuTokenizer(self.vocab_path)

        # Load Model Weights & Architecture Config
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        self.model = MikuTransformerLM(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            max_seq_len=cfg["max_seq_len"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.4,
        top_k: int = 4
    ) -> str:
        """
        Generate conversational response for a given user query.
        """
        if self.model is None or self.tokenizer is None:
            return "I am still initializing my local neural model."

        clean_prompt = prompt.strip().rstrip("?.!").lower()
        formatted = f"<user> {clean_prompt} <miku>"
        tokens = self.tokenizer.encode(formatted)
        input_tensor = torch.tensor([tokens], dtype=torch.long)

        out_tokens = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.token_to_id.get("<eos>", 2)
        )

        gen_ids = out_tokens[0][len(tokens):].tolist()
        text = self.tokenizer.decode(gen_ids)

        # Clean up special markers
        for tag in ["<eos>", "<user>", "<miku>", "<pad>", "<bos>"]:
            text = text.replace(tag, "")
        
        text = text.strip()
        if not text or len(text.split()) <= 1:
            text = f"That is an interesting question about {clean_prompt}. As your offline assistant, I approach that with thoughtful logic and clear focus."
        else:
            text = text[0].upper() + text[1:]
            if not text.endswith((".", "!", "?")):
                text += "."

        return text
