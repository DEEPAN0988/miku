# MIKU: Sovereign Local Agent (v1.3.0)
> **Zero-Model, Zero-API Framework — Autonomous On-Device Assistant**

Miku is an autonomous, on-device AI voice assistant built entirely without third-party cloud APIs and without external pre-trained model weights. Every model is either built from scratch or trained on-device.

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
  │(Sensory/Voice)│ │(Brain/Memory) │ │    (Hands)    │ │    (Eyes)     │
  ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
  │NLMS Echo Can. │─▶ UIAutomation  │─▶ OS Action     │ │ Viola-Jones   │
  │GMM-HMM STT    │ │  State Tracker│ │  Executor     │ │  Haar Cascades│
  │TD-PSOLA TTS   │◀─ CDSH Intent   │◀─ CDP Browser   │─▶ Lucas-Kanade  │
  │Shared RingBuf │ │  Router       │ │  Driver       │ │  Optical Flow │
  │(0-disk PCM)   │ │Custom LM Chat │ │Min-Jerk Motion│ │Kalman Tracker │
  │               │ │SQLite+BM25 Mem│ │Anti-Bot/CAPTCHA│ │Scene Underst. │
  └───────────────┘ └───────┬───────┘ └───────────────┘ └───────┬───────┘
                            │                                   │
                            │◀────── Detections (IPC Queue) ────┘
                            ▼
               ┌────────────────────────┐
               │    Persistence Layer   │
               │  SQLite WAL, AES-256   │
               └────────────────────────┘
```

---

## 2. Full Capability Matrix

| Capability | Supported? | Implementation |
|---|:---:|---|
| **Free-Form Chat & Conversation** | ✅ **Yes** | Custom Causal Transformer LM (trained from scratch, zero cloud APIs) |
| **Deep Scene Understanding** | ✅ **Yes** | Custom Spatial Pyramid Matching & Texture Engine (zero cloud APIs) |
| **Anti-Bot & CAPTCHA Bypass** | ✅ **Yes** | Custom text template matcher, slider gap locator, audio decoder & CDP stealth injection |
| **Local OS & App Control** | ✅ **Yes** | Win32 SendInput, Minimum-Jerk Motion, Closed-Loop Clicks |
| **Offline Voice (STT & TTS)** | ✅ **Yes** | GMM-HMM + TD-PSOLA (0 disk PCM, 100% on-device) |
| **Total Privacy (0 Cloud Traffic)** | ✅ **Yes** | 100% On-device, 0 bytes sent externally |
| **Lightweight Memory Footprint** | ✅ **Yes** | Runs on consumer hardware without dedicated GPU |

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

### Run Full Test & Debug Loop (Regression Suite)
```bash
python run_tests.py
```

---

## 4. Contributors

- [@DEEPAN0988](https://github.com/DEEPAN0988) — Maintainer
- [@AravindKumar07012007](https://github.com/AravindKumar07012007) — Contributor
