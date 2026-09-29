# MIKU: Sovereign Local Agent
## Product Requirements Document & System Architecture
**Version:** 1.3.0 (Zero-Model, Zero-API Framework — Custom Capabilities Extended)  
**Date:** September 2026

---

### Revision Notes
- **Custom Capabilities Added:** Open-Ended Chat is supported **without using third-party cloud APIs or pre-trained model weights**. Every model is built and trained from scratch on-device using custom causal transformers, and local visual/acoustic processing.
- **Strict Privacy Perimeter:** 0 external network bytes for the cognitive engine, 0 raw PCM audio or video frames written to disk.

---

### 1. Executive Summary
Miku is an autonomous, on-device assistant built without commercial cloud APIs and without third-party pre-trained weights. It uses custom classical and neural architectures trained locally from scratch, deterministic pattern matching, and OS-level automation to run tasks locally on Windows with total data privacy.

### 2. Target Audience
- **Privacy Maximalists:** Need a hard guarantee that no audio, text, or file content ever leaves the device.
- **Power Users & Developers:** Want deep OS automation, custom open-ended chat, scene awareness, and stealth browser capabilities without cloud subscriptions.
- **Low-Resource Environments:** Runs efficiently on consumer hardware without dedicated GPU requirements.

### 3. Core Functional Requirements
- **3.1 Deterministic Intent Routing:** Parses natural-language commands and OS window state via grammar/pattern matching + CDSH multi-modal fusion.
- **3.2 OS-Level Automation:** Interacts with the OS via UIAutomation and SendInput APIs to execute clicks, keystrokes, and UI navigation, gated by confirmation for destructive tasks.
- **3.3 Authorized Browser Session Automation:** Drives authenticated Chrome sessions over CDP, applying humanized input trajectories.
- **3.4 Offline Voice Processing:** Transcribes locally with a GMM-HMM acoustic model; synthesizes responses with TD-PSOLA concatenative synthesis. Raw PCM is never written to disk.
- **3.5 Reflective Self-Extension:** Generates new automation routines as sandboxed Python modules (via AST construction), tests them in isolation, and persists ones that pass validation.
- **3.6 Open-Ended Free-Form Chat (Custom LM from Scratch):** Features an in-house Causal Transformer language model with RMSNorm and custom subword tokenizer trained locally on dialogue corpora for conversational reflexes without commercial LLMs.


### 4. Non-Functional Requirements
| Requirement | Target | Reality Check |
|---|---|---|
| Latency | Audio-to-action < 150ms for in-vocabulary commands | Achievable for the fixed action set; conversational path has dedicated budget |
| Resource Footprint | Peak RAM < 500MB, no GPU required | Realistic for GMM-HMM + BM25 + Custom Lightweight Causal Transformer |
| Network Independence | 0 bytes external traffic for core cognitive engine | True for router, chat engine, memory, and voice core |

### 5. System Topology
Four isolated CPU processes communicating over IPC:
1. **Core 1: Audio Daemon (Sensory & Voice):** NLMS acoustic echo cancellation, GMM-HMM STT, TD-PSOLA TTS, shared memory ring buffer.
2. **Core 2: Cognitive Router (Brain & Memory):** Deterministic grammar matching, CDSH multi-modal router, Custom Causal Transformer Chat Engine, SQLite + Okapi BM25 memory, Day-Zero calibration daemon.
3. **Core 3: Execution Engine (Hands):** OS action executor, minimum-jerk cursor motion, closed-loop verification, CDP browser driver, AST sandbox.
4. **Core 4: Vision Daemon (Eyes):** Viola-Jones face/object detection with Haar-like features & integral image, Lucas-Kanade optical flow, Kalman filter object tracking, HSV segmentation.
