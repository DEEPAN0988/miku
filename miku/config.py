"""
Global configuration for Miku Sovereign Local Agent.
Enforces resource budgets (<500MB RAM, zero external traffic).
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSISTENCE_DB = DATA_DIR / "miku_persistence.db"
MEMORY_DB = DATA_DIR / "miku_memory.db"
CALIBRATION_DIR = DATA_DIR / "calibration"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

# Resource & Latency Targets
MAX_RAM_MB = 500
LATENCY_P95_MS = 150.0

# Core 1: Audio Daemon
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FRAME_SIZE = 512           # ~32ms per chunk
AUDIO_RING_BUFFER_SLOTS = 64    # Ring buffer slots for zero-copy
NLMS_FILTER_LENGTH = 128         # Echo cancellation taps
NLMS_STEP_SIZE = 0.1             # Convergence rate (mu)
NLMS_EPSILON = 1e-6              # Regularization parameter

# Core 2: Cognitive Router & CDSH
CDSH_LAMBDA_DECAY = 0.05         # Elapsed time decay rate (per second)
CDSH_WEIGHT_LANG = 0.50          # Linguistic weight
CDSH_WEIGHT_STATE = 0.30         # OS state weight
CDSH_WEIGHT_VISION = 0.20        # Vision detection weight
CDSH_CONFIDENCE_THRESHOLD = 0.60 # Minimum confidence before triggering action
DEAD_MAN_SWITCH_SECONDS = 60.0   # Compound task in_flight lock timeout

# Core 3: Execution Engine
MIN_JERK_STEPS = 20              # Trajectory polynomial interpolation points
CLOSED_LOOP_VERIFY_RADIUS = 30   # Pixel radius to re-sample state before clicking
SANDBOX_DRY_RUN_ENABLED = True   # Dry-run before untrusted module execution

# Core 4: Vision Daemon
VISION_FRAME_WIDTH = 640
VISION_FRAME_HEIGHT = 480
VISION_RING_BUFFER_SLOTS = 8
MOTION_DIFF_THRESHOLD = 25.0     # Tau threshold for frame differencing
LUCAS_KANADE_WINDOW_SIZE = 5     # Window size for 2x2 optical flow matrix
KALMAN_MEASUREMENT_NOISE = 1e-2  # R covariance
KALMAN_PROCESS_NOISE = 1e-4      # Q covariance
