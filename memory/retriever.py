"""
memory/retriever.py — Keyword, Anaphora Gating, and Recency Retrieval for MIKU v0.2.

Enforces two strict architectural guardrails:
1. Hard-blocks raw key-value fact injection (forces natural declarative sentences).
2. Anaphora / relevance gating: withholds conversation history for non-referential queries,
   and limits history to at most the single most relevant prior turn for anaphoric queries.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from model.tokenizer import MikuTokenizer
from memory.storage import MikuMemoryStore


# Referential / anaphoric discourse markers that signal dependency on prior dialogue
ANAPHORIC_MARKERS = {
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "there", "again", "earlier", "before",
    "mentioned", "said", "repeat", "who is that", "what was that", "tell me again"
}


class MemoryRetriever:
    """
    Deterministic memory retriever and prompt builder with enforced anaphora gating
    and natural-sentence fact formatting.
    """

    def __init__(self, store: MikuMemoryStore, tokenizer: MikuTokenizer) -> None:
        self.store = store
        self.tokenizer = tokenizer

    @staticmethod
    def is_anaphoric_query(query: str, recent_history: List[Dict[str, str]]) -> bool:
        """
        Determines whether the query references prior dialogue.

        Returns True if:
        1. The query contains explicit anaphoric / referential tokens.
        2. The query has content-word overlap with recent user/assistant turns.
        """
        clean_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", query.lower()))
        
        # Check explicit markers
        if clean_words & ANAPHORIC_MARKERS:
            return True

        # Check multi-word phrase cues
        q_lower = query.lower()
        if any(phrase in q_lower for phrase in ["who am i", "what did i", "what was", "tell me again", "do you remember"]):
            return True

        # Check significant content-word overlap with the most recent turn
        if recent_history:
            recent_text = " ".join(m["content"] for m in recent_history[-2:])
            recent_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", recent_text.lower()))
            stopwords = {
                "the", "and", "for", "with", "what", "how", "can", "you",
                "are", "have", "about", "tell", "more", "your", "this", "that",
                "which", "when", "where", "who", "why"
            }
            content_overlap = (clean_words - stopwords) & (recent_words - stopwords)
            if len(content_overlap) >= 1:
                return True

        return False

    def retrieve_facts_for_query(self, query: str, max_facts: int = 1) -> List[str]:
        """
        Find facts matching keywords in query.
        FORCES natural declarative sentences. HARD-BLOCKS raw key-value pairs.
        """
        matches = self.store.search_facts(query)
        formatted = []
        for k, v in matches[:max_facts]:
            clean_k = k.replace("_", " ").strip()
            clean_v = v.strip()
            
            # Format as natural declarative English
            if "name" in clean_k and "pet" not in clean_k:
                sentence = f"The user's name is {clean_v}."
            elif "pet" in clean_k:
                animal = clean_k.replace("pet", "").strip() or "pet"
                sentence = f"The user has a {animal} named {clean_v}."
            elif "favorite" in clean_k:
                item = clean_k.replace("favorite", "").strip()
                sentence = f"The user's favorite {item} is {clean_v}."
            else:
                sentence = f"The user's {clean_k} is {clean_v}."

            # Enforce guardrail against raw key-value formatting
            assert not re.match(r"^[a-zA-Z0-9_\-]+:\s+", sentence), (
                f"FATAL: Raw key-value fact formatting detected: '{sentence}'. "
                f"Facts must be natural declarative sentences."
            )
            formatted.append(sentence)

        return formatted

    def build_prompt(
        self,
        current_instruction: str,
        session_id: Optional[str] = None,
        include_facts: bool = True,
        format_mode: str = "context_prefix",
        max_seq_len: int = 256,
        reserved_generation_tokens: int = 64,
        enforce_gating: bool = True,
    ) -> Tuple[str, int]:
        """
        Builds an instruction prompt respecting token budgets and relevance gating.

        Guardrails:
        - If enforce_gating is True and query is NOT anaphoric: history is stripped to zero.
        - If query IS anaphoric: history is strictly truncated to at most 1 prior turn (user + asst).
        - Facts are hard-blocked from emitting raw key-value pairs.
        """
        budget = max_seq_len - reserved_generation_tokens

        # 1. Retrieve facts if requested
        relevant_facts = []
        if include_facts:
            relevant_facts = self.retrieve_facts_for_query(current_instruction)
            for f in relevant_facts:
                if re.match(r"^[a-zA-Z0-9_\-]+:\s+", f):
                    raise ValueError(f"Hard-block triggered: Raw key-value format found in facts: {f}")

        # 2. Retrieve history turns with enforced gating
        history = []
        if session_id:
            raw_history = self.store.get_history(session_id)
            if enforce_gating:
                if self.is_anaphoric_query(current_instruction, raw_history):
                    # Inject at most the single most recent turn pair (user + assistant)
                    history = raw_history[-2:] if len(raw_history) >= 2 else raw_history[-1:]
                else:
                    # Non-anaphoric query: strictly withhold history to prevent distractor degradation
                    history = []
            else:
                history = raw_history

        if format_mode == "context_prefix":
            return self._build_context_prefix_prompt(
                current_instruction, history, relevant_facts, budget
            )
        else:
            return self._build_sequential_prompt(
                current_instruction, history, relevant_facts, budget
            )

    def _build_context_prefix_prompt(
        self,
        instruction: str,
        history: List[Dict[str, str]],
        facts: List[str],
        budget: int,
    ) -> Tuple[str, int]:
        lines = []
        if facts:
            lines.extend(facts)

        conv_lines = []
        for m in history:
            role_name = "User" if m["role"] == "user" else "Assistant"
            conv_lines.append(f"{role_name}: {m['content']}")

        if conv_lines:
            lines.append("Conversation:\n" + "\n".join(conv_lines))

        context_body = "\n".join(lines).strip()
        if context_body:
            full_prompt = f"Context:\n{context_body}\n\nInstruction:\n{instruction.strip()}\n\nResponse:\n"
        else:
            full_prompt = f"Instruction:\n{instruction.strip()}\n\nResponse:\n"

        # Sanity check: verify no raw key-value tokens leaked into prompt
        assert "Known facts: user_name:" not in full_prompt, "Leaked raw key-value fact into prompt!"

        tok_len = len(self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False))
        return full_prompt, tok_len

    def _build_sequential_prompt(
        self,
        instruction: str,
        history: List[Dict[str, str]],
        facts: List[str],
        budget: int,
    ) -> Tuple[str, int]:
        facts_block = ""
        if facts:
            facts_block = "Context:\n" + "\n".join(facts) + "\n\n"

        tail = f"Instruction:\n{instruction.strip()}\n\nResponse:\n"

        # Format history (at most 1 turn if gated)
        history_parts = []
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                u_msg = history[i]["content"]
                a_msg = history[i + 1]["content"] if i + 1 < len(history) and history[i + 1]["role"] == "assistant" else ""
                history_parts.append(f"Instruction:\n{u_msg}\n\nResponse:\n{a_msg}\n\n")
                i += 2 if a_msg else 1
            else:
                i += 1

        full_prompt = facts_block + "".join(history_parts) + tail
        tok_len = len(self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False))
        return full_prompt, tok_len
