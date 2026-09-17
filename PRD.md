# Product Requirements Document
## Miku — Offline-First Personal AI Voice Assistant

---

## 1. Overview

Miku is a personal AI voice assistant, functionally similar to Alexa/Siri, built without relying on third-party cloud APIs or using pretrained models as opaque black boxes. It should run lightweight, work both offline and online, and eventually operate across Windows, Linux, and Android — starting with Windows.

Miku should be able to hear, speak, see, plan, connect to other devices, and control the host system (and, with consent, paired devices).

---

## 2. Goals

- Fully local-first operation: core loop (wake → hear → understand → act → speak) works with no internet connection.
- No dependency on commercial AI APIs (OpenAI, Google, Amazon, etc.).
- Every model used is either trained from scratch or fine-tuned/personalized on the user's own data — never used as an untouched frozen third-party weight file.
- Lightweight enough to run continuously on a personal Windows machine without heavy GPU requirements.
- Extensible command set that grows over time via retraining, not hardcoding alone.
- Eventually reach parity, in scope of command coverage, with commercial assistants — and in coding-assistance ability, with general-purpose LLMs (stretch goal, long-term).

## 3. Non-Goals / Explicit Constraints

- **No CAPTCHA solving.** Out of scope — breaks most services' terms regardless of intent.
- **No silent auto-pairing to new devices.** Every new device connection requires explicit user confirmation.
- **No unconfirmed destructive/system-level actions.** File deletion, account logins, purchases, and system config changes always require a spoken or typed confirmation step.
- Not attempting to out-train frontier LLMs from zero on commodity hardware — realistic scope is small, task-specific models plus personalized fine-tuning.

## 4. Target Platform & Rollout

| Phase | Platform | Status |
|---|---|---|
| 1 | Windows | Current focus |
| 2 | Linux | Planned |
| 3 | Android | Planned (separate app, AccessibilityService-based) |

## 5. Core Capabilities (Functional Requirements)

### 5.1 Hear
- Continuous local wake-word detection (custom-trained, low CPU footprint).
- Speech-to-text on command capture, tuned/fine-tuned on the user's voice and vocabulary.
- Support for external mic / hearing input devices (Bluetooth headset, USB mic).

### 5.2 Speak
- Local text-to-speech output through system speaker or connected audio device.
- Voice trained/fine-tuned on a dataset the user assembles (not a shipped, unmodified voice model).

### 5.3 See
- Camera input: recognize objects/state in the room when asked.
- Screen input: read what's currently on-screen to report status or locate UI elements to act on.

### 5.4 Understand (NLU / Intent Parsing)
- Rule-based slot filling plus a small trained classifier mapping utterances to {intent, entities}.
- Must support open-ended, non-enumerated commands (e.g., "write an essay in Notepad about X") without hardcoding every possible task.
- Retrainable as new intents/commands are added.

### 5.5 Plan
- Day/task planning based on user-stated goals ("plan my day," "how should I train for X").
- Logic/rules engine over calendar and task data — no ML strictly required here.

### 5.6 Connect
- Discover and connect to Bluetooth, Wi-Fi, and other local devices.
- **Always prompts before connecting to a new device.**
- Re-authentication flow: if a previously paired device's credentials changed, Miku asks the user for the new password/requirement rather than guessing or retrying blindly.

### 5.7 Control
- **Local system control (Windows):**
  - CLI / PowerShell command execution
  - App launching and in-app actions (open Notepad → write essay, etc.)
  - Mouse (LMB/RMB/MMB) and cursor control, keyboard/text input
  - System administration tasks (files, processes, services) — gated by confirmation for anything destructive or elevated-privilege
- **Connected device control** (once explicitly paired and authorized):
  - Download/delete/edit/search files
  - Send messages / calls (where the platform and permissions allow)
  - App installs from trusted sources (app store first, then web-based source; untrusted sources trigger a warning + risk rating before proceeding)
- **Network awareness:** before executing a network-dependent command (e.g., downloading an app), Miku checks connectivity; if offline, it can enable it itself only for its own device's known/trusted networks — for anything requiring new credentials, it asks the user.

## 6. Non-Functional Requirements

- **Lightweight:** designed to run continuously in the background on consumer Windows hardware without dedicated GPU dependency (GPU acceptable for training, not required for inference).
- **Responsiveness:** wake-to-listening latency and command-to-action latency should feel conversational, not batch-processed.
- **Robustness:** should degrade gracefully offline (reduced capability, not failure) and recover automatically when connectivity returns.
- **Privacy:** all audio/video/system data stays local unless the user explicitly sends it somewhere.

## 7. Safety & Consent Requirements

- Confirmation required before: any file deletion, any purchase/payment action, any new device pairing, any account login, any elevated/admin command.
- Miku must clearly state what it's about to do before doing it, for any action outside a short trusted allowlist (e.g., "open Notepad" doesn't need confirmation; "delete file X" does).
- Risk-rating shown to the user before installing from an untrusted source.

## 8. Success Metrics

- Wake-word false-accept / false-reject rate under a defined threshold.
- Command recognition accuracy on the user's own voice/vocabulary.
- % of the initial 15–20 target commands executed correctly end-to-end (voice in → action → voice confirmation out).
- Stable multi-hour background operation without memory/resource leaks.

## 9. Phased Roadmap

1. Wake-word detector (data collection + model)
2. ASR pipeline (base + personal fine-tune)
3. Intent parser covering 10–15 core commands
4. Windows control layer executing those intents, confirmation-gated
5. TTS for responses/confirmations
6. Expand vocabulary + add Plan module
7. Add See (camera/screen) module
8. Add Connect module (Bluetooth/Wi-Fi device management)
9. Port control layer to Linux
10. Build Android companion app

## 10. Open Risks

- Full from-scratch ASR/TTS training requires substantial data collection and compute; realistic mitigation is personalized fine-tuning of small open architectures rather than shipped frozen models.
- Broad system/device control surface increases security exposure — mitigated by the confirmation-gate requirement in Section 7.
- Android porting cannot reuse the Windows control layer at all; budget it as a near-separate project.
