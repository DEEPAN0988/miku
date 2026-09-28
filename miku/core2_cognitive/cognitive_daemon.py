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
from miku.core2_cognitive.realtime_learner import RealtimeLearner
from miku.core2_cognitive.app_discovery import AppDiscovery
import re
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
        self.learner = RealtimeLearner()
        self.app_discovery = AppDiscovery()
        self.pending_clarification: Optional[Dict[str, Any]] = None

        from miku.core2_cognitive.custom_chat_engine import CustomChatEngine
        self.chat_engine = CustomChatEngine()

        from miku.core2_cognitive.english_lexicon import EnglishLexicon
        self.lexicon = EnglishLexicon()

        self.latest_vision_detection: Optional[VisionDetectionMsg] = None
        self.latest_vision_time: float = 0.0

    def update_vision_detection(self, detection: VisionDetectionMsg):
        self.latest_vision_detection = detection
        self.latest_vision_time = detection.timestamp

    def handle_transcript(self, transcript: STTTranscriptMsg) -> Dict[str, Any]:
        """
        Processes incoming transcript:
        0. Multi-turn clarification dialogue, real-time learning, and category handling.
        1. Deterministic Grammar parse.
        2. OS Accessibility state snapshot.
        3. Vision state check.
        4. CDSH Fusion & Softmax evaluation.
        5. Action dispatch or clarification generation.
        """
        now = time.time()
        text = transcript.text.strip()

        # Step 0-Norm: English Lexicon typo normalization (e.g. 'undrestand' -> 'understand', 'opne' -> 'open')
        text = self.lexicon.normalize_sentence(text)
        clean_lower = text.lower()

        # Step 0-Train: Training Intent ("train miku to understand english words", "train english", "train words")
        if re.search(r"\b(?:train\s+miku|train\s+english|train\s+words?|train\s+vocabulary|train\s+language)\b", clean_lower):
            train_metrics = self.chat_engine.train_english_language(epochs=5)
            from miku.core2_cognitive.calibration_daemon import PHONETIC_BALANCED_SCRIPT
            for p in PHONETIC_BALANCED_SCRIPT:
                self.calibration.enroll_voice_phrase(p)
            calib_metrics = self.calibration.get_calibration_metrics()
            stats = self.lexicon.get_lexicon_stats()
            
            resp = (
                f"Training complete! I have calibrated my neural language model and English vocabulary.\n"
                f"  • Vocabulary indexed: {train_metrics['vocab_size']} tokens ({stats['total_vocabulary_count']} English words)\n"
                f"  • Training dialogues: {train_metrics['dialogue_pairs']} conversational pairs\n"
                f"  • Cross-entropy loss: {train_metrics['initial_loss']} -> {train_metrics['final_loss']} ({train_metrics['loss_reduction_pct']}% reduction)\n"
                f"  • Calibration status: {calib_metrics['display_string']}\n"
                f"I now understand English words, natural questions, and system commands."
            )
            return {
                "query": text,
                "status": "conversational_response",
                "message": resp,
                "confidence": 1.0,
                "metrics": train_metrics
            }

        # Step 0-WordQuery: Word Definition Query ("what does sovereign mean", "define autonomous", etc.)
        def_query = self.lexicon.check_definition_query(text)
        if def_query:
            return {
                "query": text,
                "status": "conversational_response",
                "message": def_query,
                "confidence": 1.0
            }

        # Step 0-WordLearn: Teaching a New English Word ("learn word serendipity means ...")
        learn_word_res = self.lexicon.check_word_learning_intent(text)
        if learn_word_res:
            return {
                "query": text,
                "status": "conversational_response",
                "message": learn_word_res,
                "confidence": 1.0
            }

        # Step 0A: Multi-Turn Clarification Resolution (e.g. user answering "wuthering waves" or "1")
        if self.pending_clarification and (now - self.pending_clarification.get("time", 0.0) < 180.0):
            cat = self.pending_clarification["category"]
            options = self.pending_clarification["options"]

            selected_app = None
            if clean_lower in ("1", "first", "the first one", "first one") and len(options) >= 1:
                selected_app = options[0]
            elif clean_lower in ("2", "second", "the second one", "second one") and len(options) >= 2:
                selected_app = options[1]
            elif clean_lower in ("3", "third", "the third one", "third one") and len(options) >= 3:
                selected_app = options[2]
            else:
                from miku.core2_cognitive.app_catalog import predict_app
                cand_canonical, _ = predict_app(clean_lower)
                for opt in options:
                    opt_canonical, _ = predict_app(opt)
                    if (clean_lower in opt.lower() or opt.lower() in clean_lower or
                        cand_canonical == opt_canonical or clean_lower == opt_canonical):
                        selected_app = opt
                        break

            if selected_app:
                self.learner.set_category_preference(cat, selected_app)
                self.pending_clarification = None

                task_id = str(uuid.uuid4())[:8]
                from miku.core2_cognitive.app_catalog import predict_app
                canonical, info = predict_app(selected_app)
                action_msg = ActionRequestMsg(
                    action_type="open_app",
                    target=canonical,
                    coords=None,
                    params={"app": canonical, "app_info": info},
                    task_id=task_id,
                    is_compound=False,
                    timestamp=now
                )
                self.action_queue.put(action_msg)
                cat_singular = cat[:-1] if cat.endswith("s") else cat
                return {
                    "query": text,
                    "status": "dispatched",
                    "task_id": task_id,
                    "best_action": "open_app",
                    "confidence": 1.0,
                    "action_msg": action_msg,
                    "message": f"Opening {selected_app}! I've learned this as your preferred {cat_singular}."
                }
            elif clean_lower in ("cancel", "nevermind", "stop", "abort", "no"):
                self.pending_clarification = None
                return {
                    "query": text,
                    "status": "conversational_response",
                    "message": "Cancelled.",
                    "confidence": 1.0
                }
            self.pending_clarification = None

        # Step 0B: Real-time Teaching Intent (e.g. "when I say games open wuthering wave")
        teach_res = self.learner.check_teaching_intent(text)
        if teach_res:
            trigger, target, confirm_msg = teach_res
            return {
                "query": text,
                "status": "conversational_response",
                "message": confirm_msg,
                "confidence": 1.0
            }

        # Step 0C: Query or Reset Learned Knowledge
        if clean_lower in ("what have you learned", "show learned preferences", "show learning", "what did you learn", "show memory"):
            return {
                "query": text,
                "status": "conversational_response",
                "message": self.learner.get_learned_summary(),
                "confidence": 1.0
            }
        if clean_lower in ("reset learning", "clear learned memory", "forget preferences", "clear memory"):
            self.learner.reset()
            return {
                "query": text,
                "status": "conversational_response",
                "message": "I've reset all learned preferences and custom aliases.",
                "confidence": 1.0
            }

        # Step 0D: Category Intent Handling (e.g. "games", "open games", "play a game", "browser")
        cat_match = None
        for pat, cat_name in [
            (r"^(?:(?:open(?:\s+up)?|launch|play|start|run)\s+)?(?:a\s+|some\s+|my\s+)?(?P<cat>games?)$", "games"),
            (r"^(?:let'?s\s+play(?:\s+a)?\s+games?)$", "games"),
            (r"^(?:(?:open(?:\s+up)?|launch|start)\s+)?(?:a\s+|my\s+)?(?P<cat>browser|web\s+browser)$", "browser"),
            (r"^(?:(?:open(?:\s+up)?|play|listen\s+to)\s+)?(?:some\s+|my\s+)?(?P<cat>music|songs)$", "music"),
            (r"^(?:(?:open(?:\s+up)?|check)\s+)?(?:my\s+)?(?P<cat>social\s+media|social)$", "social"),
            (r"^(?:(?:open(?:\s+up)?|launch)\s+)?(?:my\s+)?(?P<cat>code\s+editor|editor|ide)$", "developer")
        ]:
            if re.match(pat, clean_lower):
                cat_match = cat_name
                break

        if cat_match:
            preferred = self.learner.get_category_preference(cat_match)
            if preferred:
                from miku.core2_cognitive.app_catalog import predict_app
                canonical, info = predict_app(preferred)
                task_id = str(uuid.uuid4())[:8]
                action_msg = ActionRequestMsg(
                    action_type="open_app",
                    target=canonical,
                    coords=None,
                    params={"app": canonical, "app_info": info},
                    task_id=task_id,
                    is_compound=False,
                    timestamp=now
                )
                self.action_queue.put(action_msg)
                cat_singular = cat_match[:-1] if cat_match.endswith("s") else cat_match
                return {
                    "query": text,
                    "status": "dispatched",
                    "task_id": task_id,
                    "best_action": "open_app",
                    "confidence": 1.0,
                    "action_msg": action_msg,
                    "message": f"Opening your preferred {cat_singular}, {info.get('display_name', preferred)}."
                }
            else:
                options = self.app_discovery.get_apps_in_category(cat_match)
                if not options:
                    if cat_match == "games":
                        options = ["Wuthering Waves"]
                    elif cat_match == "browser":
                        options = ["Microsoft Edge", "Google Chrome"]
                    elif cat_match == "music":
                        options = ["Spotify"]
                    else:
                        options = []

                if options:
                    cat_singular = cat_match[:-1] if cat_match.endswith("s") else cat_match
                    options_str = ", ".join(options)
                    clarify_msg = f"Which {cat_singular} would you like to open? You have {options_str}."
                    self.pending_clarification = {
                        "category": cat_match,
                        "options": options,
                        "time": now
                    }
                    return {
                        "query": text,
                        "status": "clarification_needed",
                        "best_action": "ask_clarification",
                        "confidence": 1.0,
                        "message": clarify_msg
                    }

        # Step 0E: Apply custom user alias mappings (e.g. user taught "diary" -> "notepad")
        parsed_text = text
        for al, target in self.learner.data.get("custom_aliases", {}).items():
            if al in clean_lower:
                parsed_text = re.sub(rf"\b{re.escape(al)}\b", target, parsed_text, flags=re.I)

        # 1. Grammar parse
        intent, action, params, s_lang = self.grammar.parse(parsed_text)

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
                target=params.get("app") or params.get("target") or params.get("url") or params.get("path") or params.get("query") or "",
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
