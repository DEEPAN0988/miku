# MIKU: Sovereign Local Agent — System Architecture (v1.2.0)

## 1. Process Topology & Zero-Copy IPC
To bypass Python's GIL and isolate crashes between sensory and execution systems, Miku executes as four distinct processes:

```
┌─────────────────────────────────────────────────────────────┐
│                 MIKU CORE PROCESSES                         │
│               (isolated, IPC-connected)                     │
└─────────────────────────────────────────────────────────────┘
  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
  │    CORE 1     │ │    CORE 2     │ │    CORE 3     │ │    CORE 4     │
  │ Audio Daemon  │ │Cognitive Router│ │Execution Engine│ │ Vision Daemon │
  │(Sensory/Voice)│ │(Brain/Memory) │ │    (Hands)    │ │ (NEW - Eyes)  │
  ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
  │NLMS Echo Can. │─▶ UIAutomation  │─▶ OS Action     │ │ Viola-Jones   │
  │GMM-HMM STT    │ │  State Tracker│ │  Executor     │ │  Haar Cascades│
  │TD-PSOLA TTS   │◀─ CDSH Intent   │◀─ CDP Browser   │─▶ Lucas-Kanade  │
  │Shared RingBuf │ │  Router       │ │  Driver       │ │  Optical Flow │
  │(0-disk PCM)   │ │SQLite+BM25 Mem│ │Min-Jerk Motion│ │Kalman Tracker │
  └───────────────┘ └───────┬───────┘ └───────────────┘ └───────┬───────┘
                            │                                   │
                            │◀────── Detections (IPC Queue) ────┘
                            ▼
               ┌────────────────────────┐
               │    Persistence Layer   │
               │  SQLite WAL, AES-256   │
               └────────────────────────┘
```

### 1.1 IPC Shared Memory Ring Buffer (Bottleneck #1 Fix)
- High-frequency payloads (PCM audio buffers and vision frames) bypass `multiprocessing.Queue` to avoid serialization/pickle CPU burns.
- Ring buffer allocated via `multiprocessing.shared_memory`.
- Generation counter / sequence number for lock-free slot read/write.
- IPC queue only passes lightweight metadata structs: `(slot_id, timestamp, seq_num, size)`.

### 1.2 Cognitive CDSH Router & State-Lock (Bottleneck #5 Fix)
- Fuses linguistic intent, OS active window state, and vision detections:
  $$\text{Confidence}(a) = \text{softmax}(w_{\text{lang}} S_{\text{lang}} + w_{\text{state}} S_{\text{state}} e^{-\lambda \Delta t} + w_{\text{vision}} S_{\text{vision}} e^{-\lambda \Delta t_v})$$
- In-flight compound task state lock: `in_flight` flag freezes $\Delta t$ until completion, backed by a dead-man's switch timeout to prevent permanent locks.

### 1.3 Closed-Loop Execution (Bottleneck #3 Fix)
- Before any physical hardware click event fires, Core 3 re-samples the visual/UI-tree target coordinates.
- If the control changed (due to popups or notifications), it cancels the action and reports `target changed, action cancelled` instead of executing blind.

### 1.4 AST Synthesis Privilege Boundary (Bottleneck #4 Fix)
- Self-generated automation code executes in a sandboxed AST runner with default-deny file permissions and dry-run execution diffing.

### 1.5 Day-Zero Calibration & Incremental Learning (Bottleneck #2 Fix)
- Phonetically balanced voice enrollment and UI bounding-box vision enrollment with progress tracking.
