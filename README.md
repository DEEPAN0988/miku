# MIKU: Sovereign Local Agent (v1.2.0)
> **Zero-Model, Zero-API Framework — Autonomous On-Device Assistant**

Miku is an autonomous, on-device AI voice assistant built entirely without pre-trained deep learning models (LLMs/transformers) and without third-party cloud APIs. It uses classical machine learning, deterministic pattern matching, and OS-level automation to run tasks locally on Windows with total data privacy.

---

## 1. System Topology: 4 Isolated CPU Cores

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

- **Core 1: Audio Daemon (Sensory & Voice):** Normalized Least Mean Squares (NLMS) acoustic echo cancellation, local discrete/continuous GMM-HMM speech recognition, and TD-PSOLA concatenative speech synthesis. Raw PCM audio is **never written to disk**.
- **Core 2: Cognitive Router (Brain & Memory):** Deterministic grammar matching, UIAutomation live accessibility state tracking, SQLite with Okapi BM25 + trigram indexing for exact retrieval, Context-Decayed Spatiotemporal Hash (CDSH) multi-modal fusion, and Day-Zero Calibration Daemon.
- **Core 3: Execution Engine (Hands):** OS-level automation (Win32 SendInput), minimum-jerk trajectory natural cursor motion, closed-loop verify-before-click execution, CDP browser automation for authenticated sessions, and AST sandbox with default-deny permissions.
- **Core 4: Visual Perception Daemon (Eyes - v1.2.0):** Classical computer vision without deep models: Viola-Jones face/object detection with Haar-like features & integral image, frame differencing + Lucas-Kanade optical flow, Kalman filter object tracking, and HSV thresholding with connected components segmentation.

---

## 2. Bare-Metal Bottlenecks Solved

1. **IPC Serialization Choke:** High-frequency audio PCM and vision frames bypass pickle queues using `multiprocessing.shared_memory` ring buffers with generation counters for lock-free, zero-copy slot access.
2. **Cold Start Calibration Void:** Day-Zero Calibration Daemon with voice phoneme enrollment and UI bounding-box enrollment, enforcing a minimum-viable calibration floor and tracking incremental corrections.
3. **The Notification Trap (Open-Loop Execution):** Closed-loop verification re-samples the UI-tree at the exact target coordinates immediately before firing hardware clicks. If an unexpected popup or window shift occurred, it cancels execution (`target changed, action cancelled`) and re-plans.
4. **Excessive Agency in AST Synthesis:** AST synthesis sandbox enforces default-deny file permissions, inspects AST nodes to reject unsafe calls (`eval`, `exec`, arbitrary imports), and dry-runs against scratch directories before execution.
5. **Working-Memory Rot in CDSH Router:** Compound in-flight tasks state-lock the decay clock ($\Delta t$) until completion, protected by a dead-man's switch timeout to prevent permanent lockup.

---

## 3. Quick Start & CLI

### Run Interactive Assistant
```bash
python miku_cli.py
```

### Run Day-Zero Calibration
```bash
python miku_cli.py calibrate
```

### View System Status & Audits
```bash
python miku_cli.py status
```

---

## 4. Test & Debug Loop (Regression Suite)

Run the full deterministic verification suite:
```bash
python run_tests.py
```

### Metrics Tracked:
| Metric | Target | Status |
|---|---|---|
| Peak RAM | < 500 MB | **PASS (~84 MB)** |
| Network Independence | 0 external bytes | **PASS (100% Local)** |
| Audio-to-Action Latency | < 150 ms | **PASS** |
| Golden-Set Intent Coverage | 100% | **PASS** |
| NLMS Filter Convergence | Converged | **PASS** |
