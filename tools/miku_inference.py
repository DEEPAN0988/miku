"""
tools/miku_inference.py — Miku's Own Native Inference Engine (From Scratch)

STATUS: IMPLEMENTED — loads Miku's own trained checkpoint only.

This module loads the MikuLM transformer (model/architecture.py) with weights
trained from scratch by this project's own training pipeline (train/train_sft.py).
No pre-trained models. No external APIs. No HuggingFace from_pretrained.

Provides:
  - load_miku_model(): Loads the best checkpoint into MikuLM
  - generate(): Autoregressive text generation using Miku's own weights
  - get_screen_state_text(): Describes current desktop state as structured text
    for Miku to reason about (replaces screenshot-based vision)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from model.architecture import MikuLM
from model.config import ModelConfig
from model.tokenizer import MikuTokenizer

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase9_v03_2h_unified")
DEFAULT_TOKENIZER_PREFIX = os.path.join(REPO_ROOT, "data", "processed", "tokenizer", "miku_bpe")

# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_CACHED_MODEL: Optional[MikuLM] = None
_CACHED_TOKENIZER: Optional[MikuTokenizer] = None
_CACHED_DEVICE: Optional[torch.device] = None


def get_device() -> torch.device:
    """Detect best available device — CUDA > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_miku_model(
    checkpoint_dir: Optional[str] = None,
    tokenizer_prefix: Optional[str] = None,
    device: Optional[torch.device] = None,
    force_reload: bool = False,
) -> Tuple[MikuLM, MikuTokenizer, torch.device]:
    """
    Loads the MikuLM model from the project's own trained checkpoint.

    Finds the latest checkpoint (by step number), extracts the ModelConfig
    stored inside it, reconstructs the architecture, and loads the trained weights.

    Returns:
        (model, tokenizer, device)
    """
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_DEVICE

    ckpt_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    tok_prefix = tokenizer_prefix or DEFAULT_TOKENIZER_PREFIX
    dev = device or get_device()

    # Find the latest checkpoint
    ckpt_path = _find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"[Miku Inference] No checkpoint found in {ckpt_dir}. "
            "Train Miku first using train/train_sft.py."
        )

    print(f"[*] [MIKU INFERENCE] Loading checkpoint: {ckpt_path}", flush=True)
    print(f"[*] [MIKU INFERENCE] Device: {dev}", flush=True)

    # Load checkpoint (contains model_state + config)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    # Reconstruct config from the checkpoint
    config_dict = ckpt.get("config", {})
    if not config_dict:
        raise ValueError(
            f"[Miku Inference] Checkpoint {ckpt_path} has no 'config' key. "
            "Cannot reconstruct model architecture."
        )

    config = ModelConfig.from_dict(config_dict)
    step = ckpt.get("step", "?")
    train_loss = ckpt.get("train_loss", "?")
    val_loss = ckpt.get("val_loss", "?")
    n_params = ckpt.get("n_params", config.count_params())

    # Build the model from scratch and load trained weights
    model = MikuLM(config)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(dev)
    model.eval()

    print(
        f"[+] [MIKU INFERENCE] Loaded MikuLM — "
        f"step={step}, params={n_params:,}, "
        f"train_loss={train_loss}, val_loss={val_loss}, "
        f"device={dev}",
        flush=True,
    )

    # Load the project's own BPE tokenizer
    tokenizer = MikuTokenizer.load(tok_prefix)
    print(
        f"[+] [MIKU INFERENCE] Tokenizer loaded — "
        f"vocab_size={tokenizer.vocab_size}",
        flush=True,
    )

    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_DEVICE = dev

    return model, tokenizer, dev


def generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    eos_token_id: Optional[int] = None,
) -> str:
    """
    Generate text using Miku's own trained model.
    No external APIs. No pre-trained models. Pure from-scratch inference.

    Args:
        prompt: The input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-k sampling (0 = greedy)
        top_p: Nucleus sampling threshold (overrides top_k if set)
        repetition_penalty: Penalty for repeating tokens (1.0 = none)
        no_repeat_ngram_size: Block n-gram repetitions of this size
        eos_token_id: Stop on this token (None = use tokenizer's EOS)

    Returns:
        Generated text string
    """
    model, tokenizer, device = load_miku_model()

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return ""

    # Truncate if too long (leave room for generation)
    max_ctx = model.config.max_seq_len - max_new_tokens
    if max_ctx < 1:
        max_ctx = 1
    if len(token_ids) > max_ctx:
        token_ids = token_ids[-max_ctx:]

    prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Resolve EOS token
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_id

    # Generate
    output_tokens = model.generate(
        prompt_tokens=prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    # Decode only the newly generated tokens
    generated_ids = output_tokens[0, len(token_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def generate_action_json(
    objective: str,
    screen_state: str,
    action_history: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 96,
) -> str:
    """
    Generate a computer-use action JSON using Miku's own model.

    Constructs a structured prompt from the objective + screen state description
    and asks Miku to produce a JSON action.

    Args:
        objective: The user's task (e.g. "open calculator")
        screen_state: Structured text description of current desktop state
        action_history: List of previously executed actions
        max_new_tokens: Max tokens for the response

    Returns:
        Raw generated string (caller should parse for JSON)
    """
    prompt_parts = [
        "[INST] You are Miku OS Agent. Given the user objective and current screen state, "
        "determine the next action. Respond with a JSON object.\n",
        f"Objective: {objective}\n",
        f"Screen State: {screen_state}\n",
    ]

    if action_history:
        prompt_parts.append("Previous Actions:\n")
        for i, h in enumerate(action_history, 1):
            act_str = json.dumps(h.get("action", h))
            status = h.get("status", "OK")
            prompt_parts.append(f"  Step {i}: {act_str} -> {status}\n")

    prompt_parts.append(
        "Respond with ONLY a JSON object like:\n"
        '  {"action": "click", "x": 500, "y": 300}\n'
        '  {"action": "type", "text": "hello"}\n'
        '  {"action": "press_key", "key": "enter"}\n'
        '  {"action": "wait", "seconds": 1.0}\n'
        '  {"action": "terminate", "reason": "done"}\n'
        "[/INST]\n"
    )

    prompt = "".join(prompt_parts)
    return generate(prompt, max_new_tokens=max_new_tokens, temperature=0.3, top_k=20)



def predict_action(context_text: str) -> dict:
    """
    Evaluates the user command alongside structured UI context and outputs discrete action decisions.

    Strategy (fully local, no external calls):
      1. Try the local MikuLM model to generate a JSON action.
      2. If the model output is empty or unparseable (expected during early training),
         fall back to a deterministic NLP intent parser on the original prompt.

    Args:
        context_text: Unified prompt containing the user command and structured UI elements.
    Returns:
        A dictionary containing the parsed JSON action.
    """
    import re as _re
    import json as _json

    raw_output = generate(context_text, max_new_tokens=96, temperature=0.1, top_k=10)

    # 1. Try to parse JSON from model output
    if raw_output and raw_output.strip():
        match = _re.search(r'\{.*?\}', raw_output, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(0))
            except _json.JSONDecodeError:
                pass

    # 2. Fallback: deterministic NLP intent parser on the original prompt
    objective_match = _re.search(r'Objective:\s*(.+?)(?:\n|$)', context_text)
    objective = objective_match.group(1).strip() if objective_match else context_text.strip()
    return _parse_intent(objective)


def _parse_intent(objective: str) -> dict:
    """
    Deterministic intent parser. Converts natural-language desktop commands into
    discrete JSON action dictionaries without any model inference.
    """
    import re as _re
    cmd = objective.strip().lower()

    # open / launch / start / run <app>
    m = _re.match(r'^(?:open|launch|start|run)\s+(.+)$', cmd)
    if m:
        return {"action": "launch", "target": m.group(1).strip()}

    # click / tap <element>
    m = _re.match(r'^(?:click|tap|press\s+on)\s+(?:the\s+)?(.+)$', cmd)
    if m:
        target = m.group(1).strip()
        if target in ("enter", "return", "tab", "escape", "esc", "space", "backspace", "delete"):
            return {"action": "press_key", "key": target}
        return {"action": "click", "target": target}

    # type / write / input <text>
    m = _re.match(r'^(?:type|write|input|enter)\s+(.+)$', cmd)
    if m:
        return {"action": "type", "text": m.group(1).strip()}

    # close / exit / quit
    if _re.match(r'^(?:close|exit|quit|kill)\b', cmd):
        return {"action": "press_key", "key": "alt+f4"}

    # scroll
    m = _re.match(r'^scroll\s+(up|down|left|right)(?:\s+(\d+))?$', cmd)
    if m:
        return {"action": "scroll", "direction": m.group(1), "amount": int(m.group(2) or 3)}

    # minimize / maximize
    if "minimize" in cmd:
        return {"action": "press_key", "key": "win+down"}
    if "maximize" in cmd:
        return {"action": "press_key", "key": "win+up"}

    # Default: composite task
    return {"action": "task", "objective": objective}


def get_screen_state_text() -> str:
    """
    Captures the current desktop state as structured text using Win32 APIs.
    This replaces screenshot-based vision with text-based screen description.

    Returns a structured string describing:
      - Foreground window title and class
      - Desktop resolution
      - Current cursor position
      - List of visible top-level windows
    """
    import ctypes
    import win32gui

    user32 = ctypes.windll.user32

    # Current cursor position
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    cursor_x, cursor_y = pt.x, pt.y

    # Screen resolution
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # Foreground window
    fg_hwnd = user32.GetForegroundWindow()
    fg_title = win32gui.GetWindowText(fg_hwnd) if fg_hwnd else "(none)"
    fg_class = win32gui.GetClassName(fg_hwnd) if fg_hwnd else "(none)"

    # Enumerate visible top-level windows
    visible_windows = []

    def enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and title.strip():
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    visible_windows.append({
                        "title": title,
                        "rect": rect,
                        "hwnd": hwnd,
                    })
                except Exception:
                    pass
        return True

    win32gui.EnumWindows(enum_callback, None)

    # Build structured text
    lines = [
        f"Desktop: {screen_w}x{screen_h}",
        f"Cursor: ({cursor_x}, {cursor_y})",
        f"Foreground: \"{fg_title}\" [{fg_class}]",
        f"Visible Windows ({len(visible_windows)}):",
    ]
    for w in visible_windows[:15]:  # Cap at 15 to stay within context
        r = w["rect"]
        lines.append(f"  - \"{w['title']}\" at ({r[0]},{r[1]})-({r[2]},{r[3]})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the checkpoint with the highest step number."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    pt_files = sorted(ckpt_dir.glob("step_*.pt"))
    if not pt_files:
        return None

    return str(pt_files[-1])


def get_model_info() -> Dict[str, Any]:
    """Returns info about the currently loaded model, or the latest checkpoint."""
    if _CACHED_MODEL is not None:
        return {
            "loaded": True,
            "device": str(_CACHED_DEVICE),
            "params": _CACHED_MODEL.count_parameters(),
            "config": _CACHED_MODEL.config.summary(),
        }

    # Not loaded yet — check if checkpoint exists
    ckpt_path = _find_latest_checkpoint(DEFAULT_CHECKPOINT_DIR)
    return {
        "loaded": False,
        "checkpoint_available": ckpt_path is not None,
        "checkpoint_path": ckpt_path,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[*] Miku Inference Engine — Smoke Test")
    model, tok, dev = load_miku_model()
    print(f"\n[*] Model: {model.config.summary()}")
    print(f"[*] Device: {dev}")
    print(f"[*] Tokenizer vocab: {tok.vocab_size}")

    test_prompt = "[INST] What is 2 + 2? [/INST]"
    print(f"\n[*] Prompt: {test_prompt!r}")
    result = generate(test_prompt, max_new_tokens=64, temperature=0.7)
    print(f"[*] Generated: {result!r}")

    print("\n[*] Screen state:")
    print(get_screen_state_text())
""",
Description="Miku's own native inference engine — loads the project's trained checkpoint, not any pre-trained model."
"""
