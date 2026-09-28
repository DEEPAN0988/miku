"""
Core 2: Cognitive Router (Brain & Memory).
Integrates Grammar Parsing, OS State Tracking, Okapi BM25 Memory,
and CDSH Multi-Modal Fusion. Outputs Actions to Core 3.
"""
import time
import uuid
from multiprocessing import Queue
from typing import Dict, Any, Optional, Tuple, List

from miku.core2_cognitive.grammar_router import DeterministicGrammarRouter
from miku.core2_cognitive.state_tracker import OSStateTracker
from miku.core2_cognitive.bm25_memory import BM25Memory
from miku.core2_cognitive.cdsh_router import CDSHRouter
from miku.core2_cognitive.calibration_daemon import CalibrationDaemon
from miku.ipc.messages import (
    STTTranscriptMsg,
    VisionDetectionMsg,
    ActionRequestMsg,
    ActionResultMsg
)

class CognitiveDaemon:
    def __init__(
        self,
        action_queue: Queue,
        response_queue: Optional[Queue] = None
    ):
        self.action_queue = action_queue
        self.response_queue = response_queue

        self.grammar = DeterministicGrammarRouter()
        self.state_tracker = OSStateTracker()
        self.memory = BM25Memory()
        self.cdsh = CDSHRouter()
        self.calibration = CalibrationDaemon()

        from miku.core2_cognitive.custom_chat_engine import CustomChatEngine
        self.chat_engine = CustomChatEngine()

        self.latest_vision_detection: Optional[VisionDetectionMsg] = None
        self.latest_vision_time: float = 0.0

    def update_vision_detection(self, detection: VisionDetectionMsg):
        self.latest_vision_detection = detection
        self.latest_vision_time = detection.timestamp

    def handle_transcript(self, transcript: STTTranscriptMsg) -> Dict[str, Any]:
        """
        Processes incoming transcript:
        1. Deterministic Grammar parse.
        2. OS Accessibility state snapshot.
        3. Vision state check.
        4. CDSH Fusion & Softmax evaluation.
        5. Action dispatch or clarification generation.
        """
        now = time.time()
        text = transcript.text

        # 1. Grammar parse
        intent, action, params, s_lang = self.grammar.parse(text)

        # 2. OS State snapshot
        os_state = self.state_tracker.capture_active_state()
        dt_state = now - os_state["timestamp"]
        s_state = self.state_tracker.score_state_relevance(action or "", os_state) if action else 0.0

        # 3. Vision Detection
        s_vision = 0.0
        dt_vision = (now - self.latest_vision_time) if self.latest_vision_detection else 999.0
        if self.latest_vision_detection:
            # Check affinity if vision target matches
            target_label = params.get("color", "") or params.get("target", "")
            if target_label and target_label in self.latest_vision_detection.label.lower():
                s_vision = self.latest_vision_detection.confidence

        # Define candidate actions for CDSH
        candidates = [
            {"action": action or "noop"},
            {"action": "ask_clarification"},
            {"action": "ignore"}
        ]
        s_lang_map = {
            action or "noop": s_lang,
            "ask_clarification": 0.3 if not action else 0.1,
            "ignore": 0.05
        }
        s_state_map = {
            action or "noop": s_state,
            "ask_clarification": 0.2,
            "ignore": 0.1
        }
        s_vision_map = {
            action or "noop": s_vision,
            "ask_clarification": 0.1,
            "ignore": 0.05
        }

        best_act, confidence, conf_map, log_entry = self.cdsh.evaluate_candidates(
            candidate_actions=candidates,
            s_lang_map=s_lang_map,
            s_state_map=s_state_map,
            s_vision_map=s_vision_map,
            delta_t_state=dt_state,
            delta_t_vision=dt_vision
        )

        decision_result: Dict[str, Any] = {
            "query": text,
            "best_action": best_act,
            "confidence": confidence,
            "confidences": conf_map,
            "log": log_entry
        }

        # Check calibration floor
        permitted, reason = self.calibration.check_action_permitted(best_act or "")
        if not permitted:
            decision_result["status"] = "calibration_required"
            decision_result["message"] = reason
            return decision_result

        # Threshold decision
        if best_act == action and confidence >= self.cdsh.threshold and action is not None:
            task_id = str(uuid.uuid4())[:8]
            is_compound = action in ("browser_navigate", "plan_day")

            action_msg = ActionRequestMsg(
                action_type=action,
                target=params.get("app") or params.get("target") or params.get("query") or "",
                coords=self.latest_vision_detection.bbox[:2] if self.latest_vision_detection else None,
                params=params,
                task_id=task_id,
                is_compound=is_compound,
                timestamp=now
            )

            if is_compound:
                self.cdsh.start_task_lock(task_id, current_delta_t=dt_state)

            self.action_queue.put(action_msg)
            decision_result["status"] = "dispatched"
            decision_result["task_id"] = task_id
            decision_result["action_msg"] = action_msg
        else:
            # Open-ended conversational query: Engage custom Causal Transformer
            chat_reply = self.chat_engine.respond(text)
            decision_result["status"] = "conversational_response"
            decision_result["message"] = chat_reply

        return decision_result

    def handle_action_completion(self, result: ActionResultMsg):
        """
        Unlocks compound task decay clock.
        """
        self.cdsh.release_task_lock(result.task_id)
