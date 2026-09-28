"""
Typed IPC message protocols between Miku cores.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import time

@dataclass
class AudioMetadataMsg:
    slot_index: int
    seq_num: int
    timestamp: float = field(default_factory=time.time)
    num_samples: int = 512

@dataclass
class STTTranscriptMsg:
    text: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class VisionDetectionMsg:
    label: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    motion_vector: Tuple[float, float] = (0.0, 0.0)  # (u, v)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ActionRequestMsg:
    action_type: str                  # e.g., 'click', 'type', 'open_app', 'browser_task'
    target: str                       # e.g., 'Notepad', 'Save Button'
    coords: Optional[Tuple[int, int]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    is_compound: bool = False
    timestamp: float = field(default_factory=time.time)

@dataclass
class ActionResultMsg:
    task_id: str
    success: bool
    status: str                       # 'completed', 'cancelled_target_changed', 'aborted_by_user'
    message: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class UserInterruptionMsg:
    reason: str = "user_input_detected"
    timestamp: float = field(default_factory=time.time)
