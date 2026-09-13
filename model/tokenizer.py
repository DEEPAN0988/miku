"""
model/tokenizer.py — MIKU BPE Tokenizer
STATUS: IMPLEMENTED

Wraps SentencePiece's BPE trainer. The tokenizer IS trained from
scratch on the project corpus. It is NOT a pretrained vocabulary.

Usage:
    # Training (once, on the clean corpus)
    MikuTokenizer.train(
        input_file="data/processed/corpus_train.txt",
        model_prefix="data/processed/tokenizer/miku_bpe",
        vocab_size=32000,
    )

    # Loading for use in training/inference
    tok = MikuTokenizer.load("data/processed/tokenizer/miku_bpe")
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# SentencePiece availability check
# ---------------------------------------------------------------------------

try:
    import sentencepiece as spm
    _SPM_AVAILABLE = True
except ImportError:
    _SPM_AVAILABLE = False


def _require_spm() -> None:
    if not _SPM_AVAILABLE:
        raise ImportError(
            "sentencepiece is required for the tokenizer. "
            "Install it with: pip install sentencepiece"
        )


# ---------------------------------------------------------------------------
# MikuTokenizer
# ---------------------------------------------------------------------------


class MikuTokenizer:
    """
    BPE tokenizer trained from scratch on the MIKU corpus.

    Not a pretrained tokenizer. Does not load OpenAI tiktoken or
    HuggingFace tokenizer files.

    Special tokens:
        <pad>  — padding (id=0)
        <unk>  — unknown (id=1)
        <bos>  — beginning of sequence (id=2)
        <eos>  — end of sequence (id=3)
    """

    # Special token strings and their reserved IDs in the SentencePiece model
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, sp_model: "spm.SentencePieceProcessor") -> None:
        _require_spm()
        self._sp = sp_model
        self.vocab_size: int = sp_model.get_piece_size()
        self.pad_id: int = sp_model.piece_to_id(self.PAD_TOKEN)
        self.unk_id: int = sp_model.piece_to_id(self.UNK_TOKEN)
        self.bos_id: int = sp_model.bos_id()
        self.eos_id: int = sp_model.eos_id()

    # ------------------------------------------------------------------
    # Training (call once per corpus)
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        input_file: Union[str, Path],
        model_prefix: Union[str, Path],
        vocab_size: int = 32_000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        num_threads: int = 4,
    ) -> None:
        """
        Train a BPE tokenizer from scratch on a text file.

        Args:
            input_file: path to the cleaned, concatenated training corpus
            model_prefix: output prefix — produces <prefix>.model and <prefix>.vocab
            vocab_size: BPE vocabulary size
            character_coverage: fraction of characters to cover (0.9995 for most languages)
            model_type: 'bpe' (default) or 'unigram'
            num_threads: parallelism for the trainer
        """
        _require_spm()

        model_prefix = str(model_prefix)
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

        # SentencePiece training call.
        # We explicitly define our 4 special tokens to ensure consistent IDs.
        train_args = (
            f"--input={input_file} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type={model_type} "
            f"--character_coverage={character_coverage} "
            f"--num_threads={num_threads} "
            f"--pad_id=0 --pad_piece={cls.PAD_TOKEN} "
            f"--unk_id=1 --unk_piece={cls.UNK_TOKEN} "
            f"--bos_id=2 --bos_piece={cls.BOS_TOKEN} "
            f"--eos_id=3 --eos_piece={cls.EOS_TOKEN} "
            f"--shuffle_input_sentence=true "
            f"--split_digits=true "
            f"--byte_fallback=true "
            f"--hard_vocab_limit=false"
        )

        print(f"Training BPE tokenizer on {input_file} ...")
        print(f"  vocab_size={vocab_size}, model_type={model_type}")
        spm.SentencePieceTrainer.train(train_args)
        print(f"Tokenizer saved to {model_prefix}.model / {model_prefix}.vocab")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_prefix: Union[str, Path]) -> "MikuTokenizer":
        """
        Load a previously trained tokenizer.

        Args:
            model_prefix: prefix used during training (without .model extension)

        Returns:
            MikuTokenizer instance
        """
        _require_spm()
        model_path = str(model_prefix) + ".model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at {model_path}. "
                f"Run MikuTokenizer.train(...) first."
            )
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return cls(sp)

    # ------------------------------------------------------------------
    # Core encode/decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a text string into a list of integer token ids.

        Args:
            text: input string
            add_bos: prepend BOS token
            add_eos: append EOS token

        Returns:
            list of int token ids
        """
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids back to a string.

        Args:
            ids: list of int token ids
            skip_special_tokens: if True, filter out special token ids before decoding

        Returns:
            decoded string
        """
        if skip_special_tokens:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self._sp.decode(ids)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode a batch of strings."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token id lists."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def id_to_piece(self, token_id: int) -> str:
        """Return the string piece for a token id."""
        return self._sp.id_to_piece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Return the token id for a string piece."""
        return self._sp.piece_to_id(piece)

    def __repr__(self) -> str:
        return (
            f"MikuTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}, "
            f"pad={self.pad_id})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIKU BPE Tokenizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train tokenizer from corpus")
    train_parser.add_argument("--input", required=True, help="Input corpus text file")
    train_parser.add_argument(
        "--model-prefix", default="data/processed/tokenizer/miku_bpe",
        help="Output model prefix"
    )
    train_parser.add_argument("--vocab-size", type=int, default=32_000)
    train_parser.add_argument("--character-coverage", type=float, default=0.9995)

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test tokenizer encode/decode")
    test_parser.add_argument(
        "--model-prefix", default="data/processed/tokenizer/miku_bpe"
    )
    test_parser.add_argument("--text", default="Hello, world! What is 2 + 2?")

    args = parser.parse_args()

    if args.command == "train":
        MikuTokenizer.train(
            input_file=args.input,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
        )

    elif args.command == "test":
        tok = MikuTokenizer.load(args.model_prefix)
        print(tok)
        ids = tok.encode(args.text)
        decoded = tok.decode(ids)
        print(f"Input:   {args.text!r}")
        print(f"Encoded: {ids}")
        print(f"Decoded: {decoded!r}")
        print(f"Round-trip match: {args.text.strip() == decoded.strip()}")
