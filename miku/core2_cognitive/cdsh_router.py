"""
Context-Decayed Spatiotemporal Hash (CDSH) Routing Subsystem.
Core 2: Cognitive Router (Brain & Memory)
Implements:
1. Multi-modal fusion with exponential time-decay:
   Confidence(action) = softmax(w_lang*S_lang + w_state*S_state*e^(-lambda*Dt) + w_vision*S_vision*e^(-lambda*Dt_v))
2. State-lock for compound in-flight tasks with dead-man's switch (Bottleneck #5 Fix).
3. Softmax normalization (sum == 1.0) and full term breakdown logging.
"""
import time
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from miku.config import (
    CDSH_LAMBDA_DECAY,
    CDSH_WEIGHT_LANG,
    CDSH_WEIGHT_STATE,
    CDSH_WEIGHT_VISION,
    CDSH_CONFIDENCE_THRESHOLD,
    DEAD_MAN_SWITCH_SECONDS
)

class CDSHRouter:
    def __init__(
        self,
        lambda_decay: float = CDSH_LAMBDA_DECAY,
        w_lang: float = CDSH_WEIGHT_LANG,
        w_state: float = CDSH_WEIGHT_STATE,
        w_vision: float = CDSH_WEIGHT_VISION,
        confidence_threshold: float = CDSH_CONFIDENCE_THRESHOLD,
        dead_man_timeout: float = DEAD_MAN_SWITCH_SECONDS,
        temperature: float = 0.20
    ):
        self.lambda_decay = lambda_decay
        self.w_lang = w_lang
        self.w_state = w_state
        self.w_vision = w_vision
        self.threshold = confidence_threshold
        self.dead_man_timeout = dead_man_timeout
        self.temperature = max(1e-4, temperature)

        # In-flight task state-lock (Bottleneck #5)
        self.in_flight: bool = False
        self.in_flight_task_id: Optional[str] = None
        self.in_flight_locked_time: float = 0.0
        self.frozen_delta_t: float = 0.0

        # Term breakdown log history
        self.decision_logs: List[Dict[str, Any]] = []

    def start_task_lock(self, task_id: str, current_delta_t: float = 0.0):
        """
        Freezes delta_t advancement for compound task execution.
        """
        self.in_flight = True
        self.in_flight_task_id = task_id
        self.in_flight_locked_time = time.time()
        self.frozen_delta_t = current_delta_t

    def release_task_lock(self, task_id: str) -> bool:
        """
        Releases the task lock upon success or failure.
        """
        if self.in_flight and (self.in_flight_task_id == task_id or not task_id):
            self.in_flight = False
            self.in_flight_task_id = None
            self.in_flight_locked_time = 0.0
            self.frozen_delta_t = 0.0
            return True
        return False

    def check_dead_man_switch(self) -> bool:
        """
        Checks if task lock exceeded dead_man_timeout. Force-unlocks if hung.
        Returns True if a timeout force-unlock occurred.
        """
        if self.in_flight:
            elapsed = time.time() - self.in_flight_locked_time
            if elapsed > self.dead_man_timeout:
                self.in_flight = False
                self.in_flight_task_id = None
                return True
        return False

    def compute_decay(self, delta_t: float) -> float:
        """
        Computes exponential decay factor e^(-lambda * delta_t).
        """
        return math.exp(-self.lambda_decay * max(0.0, delta_t))

    def evaluate_candidates(
        self,
        candidate_actions: List[Dict[str, Any]],
        s_lang_map: Dict[str, float],
        s_state_map: Dict[str, float],
        s_vision_map: Dict[str, float],
        delta_t_state: float,
        delta_t_vision: float
    ) -> Tuple[Optional[str], float, Dict[str, float], Dict[str, Any]]:
        """
        Evaluates candidate actions using CDSH fusion formula and softmax.
        Returns: (best_action, best_confidence, all_confidences, breakdown_log)
        """
        self.check_dead_man_switch()

        # If in-flight lock active, freeze delta_t
        if self.in_flight:
            dt_s = self.frozen_delta_t
            dt_v = self.frozen_delta_t
        else:
            dt_s = delta_t_state
            dt_v = delta_t_vision

        decay_state = self.compute_decay(dt_s)
        decay_vision = self.compute_decay(dt_v)

        action_names = [c["action"] for c in candidate_actions]
        logits = []
        term_breakdown = {}

        for act in action_names:
            s_l = s_lang_map.get(act, 0.0)
            s_s = s_state_map.get(act, 0.0)
            s_v = s_vision_map.get(act, 0.0)

            # CDSH fusion formula
            weighted_lang = self.w_lang * s_l
            weighted_state = self.w_state * s_s * decay_state
            weighted_vision = self.w_vision * s_v * decay_vision
            logit = weighted_lang + weighted_state + weighted_vision

            logits.append(logit)
            term_breakdown[act] = {
                "S_lang": s_l,
                "S_state": s_s,
                "S_vision": s_v,
                "decay_state": decay_state,
                "decay_vision": decay_vision,
                "logit": logit
            }

        # Softmax normalization with temperature scaling
        logits_arr = np.array(logits, dtype=np.float64) / self.temperature
        max_l = np.max(logits_arr) if len(logits_arr) > 0 else 0.0
        exp_logits = np.exp(logits_arr - max_l)
        softmax_probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        confidences = {act: float(softmax_probs[i]) for i, act in enumerate(action_names)}

        best_action = None
        best_conf = 0.0
        if len(confidences) > 0:
            best_action = max(confidences, key=confidences.get)
            best_conf = confidences[best_action]

        log_entry = {
            "timestamp": time.time(),
            "in_flight": self.in_flight,
            "dt_state": dt_s,
            "dt_vision": dt_v,
            "terms": term_breakdown,
            "confidences": confidences,
            "best_action": best_action,
            "best_confidence": best_conf,
            "cleared_threshold": best_conf >= self.threshold
        }
        self.decision_logs.append(log_entry)

        return best_action, best_conf, confidences, log_entry
