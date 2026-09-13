"""
memory/session.py — Stateful session manager for MIKU v0.2.

Coordinates:
- Dialog turn logging in SQLite store.
- Implicit / explicit fact extraction and cross-session persistence.
- Prompt construction with budget constraints and enforced anaphora gating.
- Deterministic non-generative fact lookup fallback for 100% verified profile queries.
- Autoregressive generation using canonical diversified SFT weights by default.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer
from train.checkpoint import load_checkpoint
from memory.retriever import MemoryRetriever
from memory.storage import MikuMemoryStore


DEFAULT_CHECKPOINT = "checkpoints/phase5b_diversified_sft/canonical_diversified_sft.pt"
DEFAULT_CONFIG = "configs/phase5b_diversified_sft.yaml"
DEFAULT_TOKENIZER = "data/processed/tokenizer/miku_bpe"


class MikuSession:
    """
    Manages an active conversational session with local persistent memory.
    Defaults to canonical diversified multi-turn SFT weights (canonical_diversified_sft.pt).
    """

    def __init__(
        self,
        model: Optional[MikuLM] = None,
        tokenizer: Optional[MikuTokenizer] = None,
        store: Optional[MikuMemoryStore] = None,
        session_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        format_mode: str = "context_prefix",
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None or tokenizer is None:
            cfg = ModelConfig.from_yaml(DEFAULT_CONFIG)
            self.tokenizer = tokenizer or MikuTokenizer.load(DEFAULT_TOKENIZER)
            self.model = model or MikuLM(cfg).to(self.device)
            load_checkpoint(DEFAULT_CHECKPOINT, self.model, device=self.device)
            self.model.eval()
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.store = store or MikuMemoryStore()
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.format_mode = format_mode
        self.retriever = MemoryRetriever(self.store, self.tokenizer)

    @classmethod
    def from_default(
        cls,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        config_path: str = DEFAULT_CONFIG,
        tokenizer_path: str = DEFAULT_TOKENIZER,
        store: Optional[MikuMemoryStore] = None,
        session_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        format_mode: str = "context_prefix",
    ) -> MikuSession:
        """Explicit factory method loading the canonical diversified SFT model."""
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = ModelConfig.from_yaml(config_path)
        tok = MikuTokenizer.load(tokenizer_path)
        model = MikuLM(cfg).to(dev)
        load_checkpoint(checkpoint_path, model, device=dev)
        model.eval()
        return cls(
            model=model,
            tokenizer=tok,
            store=store,
            session_id=session_id,
            device=dev,
            format_mode=format_mode,
        )

    def extract_and_store_facts(self, text: str) -> List[Tuple[str, str]]:
        """
        Deterministic fact extraction for user profile information:
        - "My name is X"
        - "I have a {animal} named {name}"
        - "My favorite Y is Z"
        - "I live in X"
        """
        patterns = [
            (r"(?:my name is|i am called|call me)\s+([A-Za-z0-9_\-]+)", "user_name"),
            (r"(?:i have|my pet is)\s+(?:a|an)?\s*([A-Za-z]+)\s+named\s+([A-Za-z0-9_\-]+)", "pet"),
            (r"my favorite\s+([A-Za-z0-9_\-\s]+)\s+is\s+([A-Za-z0-9_\-\s]+)", "favorite"),
            (r"i live in\s+([A-Za-z0-9_\-\s]+)", "user_location"),
        ]

        extracted = []
        for pat, kind in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                if kind == "user_name":
                    val = m.group(1).strip().rstrip(".,!")
                    self.store.store_fact("user_name", val, source_session=self.session_id)
                    extracted.append(("user_name", val))
                elif kind == "pet":
                    animal = m.group(1).strip().lower()
                    pet_name = m.group(2).strip().rstrip(".,!")
                    k = f"pet_{animal}"
                    self.store.store_fact(k, pet_name, source_session=self.session_id)
                    extracted.append((k, pet_name))
                elif kind == "favorite":
                    item = m.group(1).strip().lower()
                    val = m.group(2).strip().rstrip(".,!")
                    k = f"favorite_{item}"
                    self.store.store_fact(k, val, source_session=self.session_id)
                    extracted.append((k, val))
                elif kind == "user_location":
                    val = m.group(1).strip().rstrip(".,!")
                    self.store.store_fact("user_location", val, source_session=self.session_id)
                    extracted.append(("user_location", val))

        return extracted

    def try_direct_fact_lookup(self, text: str) -> Optional[str]:
        """
        Deterministic non-generative fact lookup for direct user profile queries.
        Bypasses neural sampling for factual slots, returning 100% verified strings from SQLite.
        """
        q = text.lower().strip()
        all_facts = self.store.get_all_facts()
        if not all_facts:
            return None

        # 1. User name lookup
        if any(p in q for p in ["what is my name", "what's my name", "who am i", "do you know my name"]):
            name = all_facts.get("user_name")
            if name:
                return f"Your name is {name}."

        # 2. Pet name / kind lookup
        m_pet = re.search(r"what(?:'s|\s+is)\s+my\s+([A-Za-z]+)'?s?\s+name", q)
        if m_pet:
            animal = m_pet.group(1).lower()
            if animal in ("pet", "animal"):
                for k, v in all_facts.items():
                    if k.startswith("pet_"):
                        an = k.replace("pet_", "")
                        return f"Your {an}'s name is {v}."
            else:
                pet_val = all_facts.get(f"pet_{animal}")
                if pet_val:
                    return f"Your {animal}'s name is {pet_val}."

        if any(p in q for p in ["what kind of pet", "what pet do i have"]):
            for k, v in all_facts.items():
                if k.startswith("pet_"):
                    an = k.replace("pet_", "")
                    return f"You have a {an} named {v}."

        # 3. Favorite item lookup
        m_fav = re.search(r"what(?:'s|\s+is)\s+my\s+favorite\s+([A-Za-z]+)", q)
        if m_fav:
            item = m_fav.group(1).lower()
            val = all_facts.get(f"favorite_{item}")
            if val:
                return f"Your favorite {item} is {val}."

        # 4. Location lookup
        if any(p in q for p in ["where do i live", "what is my location"]):
            loc = all_facts.get("user_location")
            if loc:
                return f"You live in {loc}."

        return None

    def chat(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 64,
        use_deterministic_fallback: bool = True,
    ) -> Dict[str, any]:
        """
        Processes one user turn, extracts facts, gates prompt context,
        generates assistant response with repetition penalty, and logs to SQLite.
        """
        # 1. Extract and store any explicit facts stated by user
        stored_facts = self.extract_and_store_facts(user_message)

        # 2. Check deterministic non-generative fact lookup fallback if enabled
        if use_deterministic_fallback:
            lookup_result = self.try_direct_fact_lookup(user_message)
            if lookup_result:
                self.store.add_message(self.session_id, "user", user_message)
                self.store.add_message(self.session_id, "assistant", lookup_result)
                return {
                    "prompt": f"[Deterministic Fact Lookup]\nContext: SQLite Store\nQuery: {user_message}",
                    "prompt_tokens_len": 0,
                    "response": lookup_result,
                    "num_tokens": len(self.tokenizer.encode(lookup_result)),
                    "hit_eos": True,
                    "stored_facts": stored_facts,
                    "is_deterministic_fallback": True,
                }

        # 3. Build constrained prompt with enforced anaphora gating
        prompt, prompt_tokens_len = self.retriever.build_prompt(
            current_instruction=user_message,
            session_id=self.session_id,
            include_facts=True,
            format_mode=self.format_mode,
            max_seq_len=self.model.config.max_seq_len,
            reserved_generation_tokens=max_new_tokens,
            enforce_gating=True,
        )

        # 4. Tokenize & generate
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        inp_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_tensor = self.model.generate(
                prompt_tokens=inp_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_id,
            )

        gen_tokens = out_tensor[0, len(input_ids) :].tolist()

        # Check early termination
        hit_eos = self.tokenizer.eos_id in gen_tokens
        if hit_eos:
            eos_pos = gen_tokens.index(self.tokenizer.eos_id)
            gen_tokens = gen_tokens[:eos_pos]

        response_text = self.tokenizer.decode(gen_tokens).strip()

        # 5. Save turn to persistent history
        self.store.add_message(self.session_id, "user", user_message)
        self.store.add_message(self.session_id, "assistant", response_text)

        return {
            "prompt": prompt,
            "prompt_tokens_len": prompt_tokens_len,
            "response": response_text,
            "num_tokens": len(gen_tokens),
            "hit_eos": hit_eos,
            "stored_facts": stored_facts,
            "is_deterministic_fallback": False,
        }
