# MIKU: Sovereign Local Agent
## Product Requirements Document & System Architecture
**Version:** 1.1.0 (Zero-Model, Zero-API Framework — Revised)  
**Date:** September 2026

---

### Revision Notes
- **Scope change:** This revision does not include anti-detection browser automation — CAPTCHA/bot-detection evasion or "indistinguishable from human" input synthesis. Automating your own logged-in browser sessions on sites you're authorized to use is in scope; defeating a site's bot-detection is not. Everything below assumes automation respects the target site's terms of service.
- **Honesty pass:** Claims like "0% hallucination risk" and "instant latency" were marketing language, not engineering claims. This revision states the real tradeoffs of the zero-model approach directly — including where it will underperform a modern cloud AI assistant, not just where it wins.

---

### 1. Executive Summary
Miku is an autonomous, on-device assistant built without pre-trained deep learning models (LLMs/transformers) and without cloud APIs. It uses classical machine learning, deterministic pattern matching, and OS-level automation to run tasks locally. The core tradeoff: predictability and total data privacy, in exchange for narrower language understanding than a modern LLM. Miku is not a drop-in replacement for a cloud AI assistant — it's a different tool, suited to a smaller set of well-defined, privacy-sensitive tasks.

### 2. Target Audience
- **Privacy Maximalists:** Need a hard guarantee that no audio, text, or file content ever leaves the device, and accept reduced language flexibility for that guarantee.
- **Power Users & Developers:** Want deep OS automation and are willing to write or tune command grammars rather than rely purely on free-form natural language.
- **Low-Resource Environments:** Machines without the RAM/VRAM budget for a local LLM, where a rule-based system is the only viable option at all.

### 3. Core Functional Requirements
- **3.1 Deterministic Intent Routing:** Parses natural-language commands and current OS window state to select from a fixed library of actions. Because this is pattern/grammar matching rather than a language model, it fails predictably.
- **3.2 OS-Level Automation:** Interacts with the OS via UIAutomation and SendInput APIs to execute clicks, keystrokes, and UI navigation for tasks the user has explicitly configured.
- **3.3 Authorized Browser Session Automation:** Drives the user's own already-authenticated Chrome session over Chrome DevTools Protocol (CDP) to fill forms, extract page data, and run multi-step web tasks.
- **3.4 Offline Voice Processing:** Transcribes locally with a GMM-HMM acoustic model; synthesizes responses with TD-PSOLA concatenative synthesis.
- **3.5 Reflective Self-Extension:** Generates new automation routines as sandboxed Python modules (via AST construction), tests them in isolation, and persists ones that pass validation.

### 4. Non-Functional Requirements
| Requirement | Target | Reality Check |
|---|---|---|
| Latency | Audio-to-action < 150ms for in-vocabulary commands | Achievable for the fixed action set; disambiguation/clarification paths have their own budget |
| Resource Footprint | Peak RAM < 500MB, no GPU required | Realistic for GMM-HMM + BM25 + rule engine |
| Network Independence | 0 bytes external traffic for core cognitive engine | True for router/memory/voice core; CDP web automation uses local network directly to sites |

### 5. System Topology
Four isolated CPU processes communicating over IPC:
1. **Core 1: Audio Daemon (Sensory & Voice):** NLMS acoustic echo cancellation, GMM-HMM STT, TD-PSOLA TTS. Raw PCM never written to disk.
2. **Core 2: Cognitive Router (Brain & Memory):** UIAutomation state tracker, CDSH intent router, SQLite + Okapi BM25 memory, Day-Zero calibration daemon.
3. **Core 3: Execution Engine (Hands):** OS action executor, minimum-jerk cursor motion, closed-loop verification, CDP browser driver, AST sandbox.
4. **Core 4: Vision Daemon (Eyes - v1.2.0):** Viola-Jones face/object detection with Haar-like features & integral image, frame differencing + Lucas-Kanade optical flow, Kalman filter object tracking, HSV color/region segmentation. Feeds detections to Core 2.
