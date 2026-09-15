# RULE: No External APIs or Pre-Trained Models Without Explicit Authorization

## Status: MANDATORY — Applies to every agent, every session, every file in this repo.

---

## What This Rule Covers

You MUST NOT introduce any of the following without the user explicitly typing
**"I authorize external API / pre-trained model"** in that specific conversation:

### Banned External API Calls
- Any call to `openai.*`, `anthropic.*`, `cohere.*`, `mistral.*`, `google.*`, `groq.*`
- Any import of `openai`, `anthropic`, `langchain`, `litellm`, `guidance`, `dspy`
- Any `requests.post` / `urllib` call targeting a **cloud AI endpoint** (OpenAI, Anthropic, HuggingFace Inference API, Replicate, Together AI, Fireworks, etc.)
- Any use of `os.environ.get("OPENAI_API_KEY")` to route production traffic to a cloud model
- Any hidden fallback that silently switches from a local model to a cloud API

### Banned Pre-Trained Model Imports
- Importing a **fully pre-trained foundation model** from HuggingFace and using it as the project own trained model
- Loading any checkpoint not trained by this project own training pipeline (train/train_sft.py, train/pretrain.py, etc.) and presenting it as Miku intelligence
- Using `pipeline(...)` from `transformers` as a drop-in intelligence layer
- Using `SentenceTransformer`, `clip`, `whisper`, etc. as the core reasoning model

### What IS Allowed (no authorization needed)
- PyTorch, NumPy, SciPy — infrastructure only
- HuggingFace `tokenizers` / `datasets` — data utilities, NOT model intelligence
- `vikhyatk/moondream2` via `tools/local_vision.py` — spatial grounding ONLY (visual element location, not language reasoning)
- Any checkpoint under `checkpoints/` produced by this project own training pipeline
- Test/evaluation scripts that call external APIs only when `MIKU_EVAL_EXTERNAL=true` is explicitly set

> NOTE: `llama3.2-vision` via Ollama is a Meta pre-trained model and is NOT exempt.
> It falls under the hard ban. The existing Ollama bridge in miku.py is a temporary
> scaffold ONLY — it must be replaced by Miku own trained checkpoint before any
> capability claim is made. Label it [SCAFFOLD] in all logs until then.

---

## Agent Enforcement Protocol

Before writing any code that touches model loading, HTTP requests to AI endpoints,
or transformers/diffusers imports, you MUST check:

1. Is this using an external cloud API?
   -> If yes: STOP. Print a conflict notice. Ask for authorization.

2. Is this loading a pre-trained model not produced by this project?
   -> If yes: STOP. Print a conflict notice. Ask for authorization.

3. Is the code disguising an external API call as local intelligence?
   -> If yes: STOP. This violates CONSTITUTION.md §3 (No Fake AI). Do not proceed.

4. Is there an existing local model checkpoint under checkpoints/?
   -> Always prefer that over any external model.

---

## How to Handle "I need a smarter model" Requests

If the user asks for a capability that currently requires a better model:

1. Label it honestly: [NOT YET IMPLEMENTED - requires further training]
2. Suggest the training path: "We can train Miku vX on this domain using train/train_sft.py"
3. Do NOT silently fall back to an external API or pre-trained model

---

This rule was added on 2026-09-15. It supplements CONSTITUTION.md §3 (No Fake AI) and §6 (From-Scratch Means From-Scratch).
