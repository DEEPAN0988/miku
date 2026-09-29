"""
Core 2: Cognitive Router (Brain & Memory).
Integrates Grammar Parsing, OS State Tracking, Okapi BM25 Memory,
and CDSH Multi-Modal Fusion. Outputs Actions to Core 3.
Upgraded with zero-cloud, classical NLP safety, discourse, teaching confirmation,
destructive gates, pronoun verification, and calibrated intent classification.
"""
import time
import uuid
from multiprocessing import Queue
from typing import Dict, Any, Optional, Tuple, List
import re

from miku.core2_cognitive.grammar_router import DeterministicGrammarRouter
from miku.core2_cognitive.state_tracker import OSStateTracker
from miku.core2_cognitive.bm25_memory import BM25Memory
from miku.core2_cognitive.cdsh_router import CDSHRouter
from miku.core2_cognitive.calibration_daemon import CalibrationDaemon
from miku.core2_cognitive.realtime_learner import RealtimeLearner
from miku.core2_cognitive.app_discovery import AppDiscovery
from miku.core2_cognitive.app_catalog import GENERIC_APP_PLACEHOLDERS, APP_CATALOG, predict_app
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
        self.pending_teaching: Optional[Dict[str, Any]] = None
        self.pending_destructive: Optional[Dict[str, Any]] = None

        from miku.core2_cognitive.custom_chat_engine import CustomChatEngine
        self.chat_engine = CustomChatEngine()

        from miku.core2_cognitive.english_lexicon import EnglishLexicon
        self.lexicon = EnglishLexicon()

        from miku.core2_cognitive.normalizer import Normalizer
        from miku.core2_cognitive.lexicon import CommandLexicon
        from miku.core2_cognitive.intent_classifier import ClassicalIntentClassifier
        from miku.core2_cognitive.slot_tagger import SlotTagger
        from miku.core2_cognitive.discourse import DiscourseManager
        from miku.core2_cognitive.safe_speller import SafeSpeller
        from miku.core2_cognitive.dialogue_stack import DialogueStackManager
        from miku.core2_cognitive.active_learning import ActiveLearningManager
        from miku.core2_cognitive.paraphrase_rewriter import ParaphraseRewriter

        self.normalizer = Normalizer()
        self.cmd_lexicon = CommandLexicon()
        self.intent_classifier = ClassicalIntentClassifier()
        self.slot_tagger = SlotTagger()
        self.discourse = DiscourseManager()
        self.safe_speller = SafeSpeller()
        self.dialogue_stack = DialogueStackManager()
        self.active_learning = ActiveLearningManager()
        self.paraphrase_rewriter = ParaphraseRewriter()

        self.latest_vision_detection: Optional[VisionDetectionMsg] = None
        self.latest_vision_time: float = 0.0

    def update_vision_detection(self, detection: VisionDetectionMsg):
        self.latest_vision_detection = detection
        self.latest_vision_time = detection.timestamp

    def _dispatch_confirmed_command(self, cmd: str, original_query: str) -> Dict[str, Any]:
        """
        Executes a user-confirmed action directly through the action queue.
        """
        now = time.time()
        intent, action, params, s_lang = self.grammar.parse(cmd)
        if not action:
            clf_res = self.intent_classifier.predict(cmd.lower())
            action = "close_app" if any(w in cmd.lower() for w in ("close", "shut down", "terminate", "kill")) else "delete_file"
            params = self.slot_tagger.extract_slots(cmd.lower(), clf_res.get("top_intent", ""))

        task_id = str(uuid.uuid4())[:8]
        target = params.get("app") or params.get("target") or params.get("path") or ""
        action_msg = ActionRequestMsg(
            action_type=action or "dispatched_action",
            target=target,
            coords=None,
            params=params,
            task_id=task_id,
            is_compound=False,
            timestamp=now
        )
        self.action_queue.put(action_msg)
        
        # Record entities so that repeat/pronouns work on confirmed actions
        if action in ("open_app", "close_app") and target:
            self.dialogue_stack.record_entity("last_app", target)
        elif action in ("open_file", "search_file", "create_file", "delete_file") and target:
            self.dialogue_stack.record_entity("last_file", target)
        self.dialogue_stack.record_entity("last_command", cmd)
        
        return {
            "query": original_query,
            "status": "dispatched",
            "task_id": task_id,
            "best_action": action,
            "confidence": 1.0,
            "action_msg": action_msg,
            "message": f"Confirmed. Executed '{cmd}'."
        }

    def handle_transcript(self, transcript: STTTranscriptMsg) -> Dict[str, Any]:
        """
        Processes incoming transcript through Miku's complete cognitive pipeline:
        - Multi-turn confirmations (teaching, destructive actions, category clarifications)
        - Negation refusal and scope
        - Compound command decomposition with sequential verification
        - Pronoun resolution with target naming confirmation
        - Safe spelling and domain lexicon rewriting
        - Grammar routing & calibrated classifier fallback
        - Multi-modal CDSH arbitration
        """
        now = time.time()
        text = transcript.text.strip()
        clean_lower = text.lower()

        # Step -3: Pending teaching confirmation handling ("Link 'surf' to 'open edge'? (yes/no)")
        if self.pending_teaching:
            if clean_lower in ("yes", "y", "confirm", "sure", "proceed", "ok", "okay"):
                alias = self.pending_teaching["alias"]
                target = self.pending_teaching["target"]
                self.cmd_lexicon.register_taught_word(alias, target)
                self.learner.set_custom_alias(alias, target)
                self.pending_teaching = None
                return {
                    "query": text,
                    "status": "conversational_response",
                    "message": f"Learned! I've linked '{alias}' to '{target}'.",
                    "confidence": 1.0
                }
            elif clean_lower in ("no", "n", "cancel", "deny", "nevermind", "never mind", "abort"):
                self.pending_teaching = None
                return {
                    "query": text,
                    "status": "conversational_response",
                    "message": "Cancelled teaching.",
                    "confidence": 1.0
                }
            self.pending_teaching = None

        # Step -2: Pending destructive action confirmation ("Close Notepad? (yes/no)")
        if self.pending_destructive:
            if clean_lower in ("yes", "y", "confirm", "sure", "proceed", "ok", "okay"):
                cmd = self.pending_destructive["command"]
                self.pending_destructive = None
                return self._dispatch_confirmed_command(cmd, original_query=text)
            elif clean_lower in ("no", "n", "cancel", "deny", "nevermind", "never mind", "abort"):
                self.pending_destructive = None
                return {
                    "query": text,
                    "status": "conversational_response",
                    "message": "Cancelled.",
                    "confidence": 1.0
                }
            self.pending_destructive = None

        # Step -1A: Forget command ("forget <word>")
        forget_m = re.match(r"^(?:please\s+)?forget\s+[\"']?([a-zA-Z0-9_\-\. ]+?)[\"']?$", clean_lower)
        if forget_m:
            word = forget_m.group(1).strip().lower()
            # Strip optional 'word ' / 'alias ' prefix users tend to add
            for pfx in ("word ", "alias ", "phrase "):
                if word.startswith(pfx):
                    word = word[len(pfx):].strip()
                    break
            lex_forgot = self.cmd_lexicon.forget_word(word)
            learn_forgot = self.learner.forget_alias(word)
            msg = f"I have forgotten '{word}'." if (lex_forgot or learn_forgot) else f"I do not have '{word}' stored in memory."
            return {
                "query": text,
                "status": "conversational_response",
                "message": msg,
                "confidence": 1.0
            }

        # Step -1B: Show what you learned command
        if clean_lower in (
            "show what you learned", "show what you have learned", "show learned",
            "what have you learned", "what have you learned so far", "what did you learn",
            "show all learned preferences and aliases", "show all learned preferences",
            "show memory", "show learning", "what do you know", "what have i taught you",
            "list all aliases", "list learned words", "show aliases", "show all aliases"
        ):
            summary = self.cmd_lexicon.show_learned() + "\n" + self.learner.get_learned_summary()
            return {
                "query": text,
                "status": "conversational_response",
                "message": summary,
                "confidence": 1.0
            }

        # Step -1C: Active Learning Failure Review and Promotion
        if clean_lower in ("show what you didn't understand", "review failures", "show learning failures", "what did you fail to understand", "what could you not understand"):
            return {
                "query": text,
                "status": "conversational_response",
                "message": self.active_learning.format_review_summary(),
                "confidence": 1.0
            }
        promote_match = re.match(r"^learn\s+[\"']?(.+?)[\"']?\s+as\s+([a-zA-Z0-9_\-]+)$", clean_lower)
        if promote_match:
            promoted_phrase = promote_match.group(1).strip()
            promoted_intent = promote_match.group(2).strip().upper()
            self.active_learning.promote_to_training(promoted_phrase, promoted_intent, self.intent_classifier)
            return {
                "query": text,
                "status": "conversational_response",
                "message": f"Learned! Promoted '{promoted_phrase}' as intent '{promoted_intent}' and updated my local model.",
                "confidence": 1.0
            }

        # Step 0-WordLearning: Dictionary definition learning ("learn word euphoria means a state of intense happiness")
        word_learn = self.lexicon.check_word_learning_intent(text)
        if word_learn:
            return {
                "query": text,
                "status": "conversational_response",
                "message": word_learn,
                "confidence": 1.0
            }

        # Step 0-CategoryTeaching: Category preference learning ("when I say games open wuthering wave")
        real_teach = self.learner.check_teaching_intent(text)
        if real_teach and real_teach[0] in ("games", "browser", "music", "editor", "social"):
            trigger, target, msg = real_teach
            return {
                "query": text,
                "status": "conversational_response",
                "message": msg,
                "confidence": 1.0
            }

        # Step 0-Teaching: Natural & explicit alias teaching with mandatory confirmation
        usable_teach = self.cmd_lexicon.check_teaching_intent(text)
        teach_candidate = usable_teach or ({"is_teaching": True, "is_valid": True, "alias": real_teach[0], "target": real_teach[1]} if real_teach else None)

        if teach_candidate:
            if not teach_candidate.get("is_valid", True):
                return {
                    "query": text,
                    "status": "clarification_needed",
                    "best_action": "ask_clarification",
                    "message": f"I cannot learn that: {teach_candidate.get('validation_reason')}",
                    "confidence": 1.0
                }
            alias = teach_candidate["alias"]
            target = teach_candidate["target"]
            self.pending_teaching = {"alias": alias, "target": target}
            return {
                "query": text,
                "status": "confirmation_required",
                "best_action": "ask_confirmation",
                "message": f"Link '{alias}' to '{target}'? (yes/no)",
                "confidence": 1.0
            }


        # Step 0-Compound: Multi-step compound commands ("open notepad and then turn up volume")
        sub_commands = self.discourse.split_compound(text)
        if len(sub_commands) > 1:
            compound_results = []
            for step_idx, sub_cmd in enumerate(sub_commands, 1):
                sub_msg = STTTranscriptMsg(text=sub_cmd, confidence=transcript.confidence)
                sub_res = self.handle_transcript(sub_msg)
                compound_results.append(sub_res)

                # Sequential Safety: Step N+1 runs only after Step N reports verified success
                sub_status = sub_res.get("status")
                if sub_status != "dispatched":
                    return {
                        "query": text,
                        "status": sub_status or "compound_halted",
                        "best_action": "compound",
                        "confidence": 1.0,
                        "stopped_at_step": step_idx,
                        "sub_results": compound_results,
                        "message": f"Step {step_idx} ('{sub_cmd}') stopped: {sub_res.get('message', 'action halted')}. Halting remaining steps."
                    }
            return {
                "query": text,
                "status": "dispatched",
                "best_action": "compound",
                "confidence": 1.0,
                "sub_results": compound_results,
                "message": f"Executing ordered compound sequence: {', '.join(sub_commands)}."
            }

        # Step 0-Negation: Negation handling ("don't open notepad", "never mute sound", "open chrome, not edge")
        neg_res = self.discourse.analyze_negation(text)
        if neg_res["has_negation"]:
            if neg_res["is_correction"] and neg_res["target_command"]:
                text = neg_res["target_command"]
                clean_lower = text.lower()
            else:
                return {
                    "query": text,
                    "status": "negation_refusal",
                    "message": "Understood. I will not execute that action.",
                    "confidence": 1.0,
                    "action_msg": None
                }


        # Step 0-Coreference: Pronoun resolution ("close it", "maximize it", "the second one", "do that again")
        coref_res = self.discourse.resolve_coreference(text, self.dialogue_stack.get_entity_context())
        if coref_res.get("is_resolved") and coref_res.get("reconstructed"):
            if coref_res.get("requires_confirmation"):
                self.pending_destructive = {
                    "command": coref_res["reconstructed"],
                    "target": coref_res.get("resolved_entity", "")
                }
                return {
                    "query": text,
                    "status": "confirmation_required",
                    "best_action": "ask_confirmation",
                    "confidence": 1.0,
                    "message": coref_res["confirmation_prompt"]
                }
            text = coref_res["reconstructed"]
            clean_lower = text.lower()

        # Step 0-Norm: English Lexicon typo normalization
        pre_speller_text = text  # save for guard below
        norm = self.normalizer.normalize(text)
        taught_rewritten = self.cmd_lexicon.rewrite_taught_terms(norm["normalized"])
        speller_res = self.safe_speller.correct_sentence(taught_rewritten)
        text = self.cmd_lexicon.rewrite_taught_terms(speller_res["corrected"])
        clean_lower = text.lower()

        # Step 0-Paraphrase: Rewrite indirect / idiomatic phrasings to canonical command forms
        # (e.g. "get rid of notepad" -> "close notepad", "summon chrome" -> "open chrome")
        # Runs on the already-normalised clean_lower so fillers have already been stripped.
        paraphrase_input = clean_lower
        rewritten_text, did_rewrite = self.paraphrase_rewriter.rewrite(paraphrase_input)
        if did_rewrite:
            text = rewritten_text
            clean_lower = rewritten_text

        # Step 0-OutOfScope-Early: Identity and physical-world OOS check runs BEFORE the
        # destructive gate to prevent "who built you" matching "built" as destructive.
        oos_pat = r"\b(?:sing(?:\s+me)?(?:\s+a)?\s+(?:song|melody|tune)|dance\s+for\s+me|what(?:'s|\s+is)\s+the\s+weather|read\s+me\s+a\s+(?:book|story)|give\s+me\s+a\s+hug|make\s+me\s+(?:coffee|tea|food)|cook\s+for\s+me)\b"
        oos_identity = r"\b(?:who(?:\s+(?:creat\w*|made|built|wrote|designed|developed|programmed)\s+you)|what\s+are\s+you|who\s+are\s+you|are\s+you\s+(?:a\s+)?(?:ai|robot|human|bot|computer))\b"
        check_oos = (pre_speller_text + " " + clean_lower).lower()
        if re.search(oos_pat, check_oos) or re.search(oos_identity, check_oos):
            return {
                "query": text,
                "status": "conversational_response",
                "best_action": None,
                "confidence": 1.0,
                "message": "I am Miku, a computer assistant — I can open apps, manage files, and handle system tasks, but that's a bit outside what I can do!"
            }

        # Step 0-Destructive: Safeguard destructive actions.
        # IMPORTANT: Check BOTH the pre-speller text (catches 'wipe', 'shut it') AND the
        # post-speller text (catches 'close', 'delete', 'terminate' that survive spelling).
        # This prevents speller substitutions from erasing destructive intent.
        # Exempt sentences starting with question words (who/what/why/how/when/where) to avoid
        # false positives on identity and capability queries.
        destructive_pat = r"\b(?:delete|remove|format|wipe|destroy|erase|overwrite|kill|terminate|close|quit|exit|shut(?:\s+down|\s+it)?|rm\s+-rf|drop\s+table|transfer\s+\d+|send\s+an?\s+email)\b"
        guard_text = (pre_speller_text + " " + clean_lower).lower()
        if re.search(destructive_pat, guard_text) and not re.match(r"^(?:who|what|why|how|when|where|are\s+you|do\s+you|is\s+it)\b", clean_lower):
            # Parse target if it's a close command
            match = re.search(r"\b(?:close|quit|exit|shut\s+down|kill|terminate)\s+(?:the\s+)?([a-zA-Z0-9_\-\. ]+?)(?:\s+application|\s+app)?$", clean_lower)
            target = match.group(1).strip() if match else ""
            
            self.pending_destructive = {
                "command": text,
                "target": target
            }
            return {
                "query": text,
                "status": "confirmation_required",
                "best_action": "ask_confirmation",
                "confidence": 1.0,
                "message": f"This command involves a potentially destructive action ('{text}'). Explicit user confirmation is required to proceed. Do you wish to confirm? (yes/no)"
            }

        # Step 0-Train: Training Intent
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

        # Step 0-PendingClarification: Multi-turn response
        if self.pending_clarification:
            # If the user issued a brand new category command or generic intent, drop the clarification and fall through.
            if (re.match(r"^(?:(?:open(?:\s+up)?|launch|play|start|run|suggest)\s+)?(?:something|anything|whatever|stuff|an?\s+app|an?\s+application|applications?|tools?|some\s+app|a\s+program|some\s+game)(?:\s+(?:to\s+play|to\s+do|fun))?$", clean_lower) or
                re.match(r"^(?:what\s+should\s+i\s+(?:open|play|run)|suggest\s+(?:an?\s+app|a\s+game|something)|what\s+can\s+i\s+(?:open|play))$", clean_lower) or
                re.match(r"^(?:(?:open(?:\s+up)?|launch|play|start|run)\s+)?(?:a\s+|some\s+|my\s+)?(?P<cat>games?|browser|web\s+browser|music|songs|social\s+media|social|code\s+editor|editor|ide)$", clean_lower) or
                re.match(r"^(?:let'?s\s+play(?:\s+a)?\s+games?)$", clean_lower)):
                self.pending_clarification = None
            else:
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
                    cand_canonical, _ = predict_app(clean_lower)
                    for opt in options:
                        opt_canonical, _ = predict_app(opt)
                        if (clean_lower in opt.lower() or opt.lower() in clean_lower or
                            cand_canonical == opt_canonical or clean_lower == opt_canonical):
                            selected_app = opt
                            break
                    if not selected_app and cand_canonical in APP_CATALOG:
                        selected_app = cand_canonical
    
                if selected_app:
                    clarification_category = (self.pending_clarification or {}).get("category")
                    self.pending_clarification = None
    
                    task_id = str(uuid.uuid4())[:8]
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
                    # Persist preference so next "open games" dispatches directly
                    if clarification_category and clarification_category not in ("general",):
                        self.learner.set_category_preference(clarification_category, canonical)
                    msg_text = f"Opening {info.get('display_name', selected_app)}!"
                    return {
                        "query": text,
                        "status": "dispatched",
                        "task_id": task_id,
                        "best_action": "open_app",
                        "confidence": 1.0,
                        "action_msg": action_msg,
                        "message": msg_text
                    }
    
                elif clean_lower in ("cancel", "nevermind", "never mind", "stop", "abort", "no"):
                    self.pending_clarification = None
                    return {
                        "query": text,
                        "status": "conversational_response",
                        "message": "Cancelled.",
                        "confidence": 1.0
                    }
                self.pending_clarification = None

        # Step 0D-0: Generic / Ambiguous Open Intent
        if re.match(r"^(?:(?:open(?:\s+up)?|launch|play|start|run|suggest)\s+)?(?:something|anything|whatever|stuff|an?\s+app|an?\s+application|applications?|tools?|some\s+app|a\s+program|some\s+game|some\s+software|software|programs?)(?:\s+(?:to\s+play|to\s+do|fun))?$", clean_lower) or \
           re.match(r"^(?:what\s+should\s+i\s+(?:open|play|run)|suggest\s+(?:an?\s+app|a\s+game|something)|what\s+can\s+i\s+(?:open|play))$", clean_lower):
            games = self.app_discovery.get_apps_in_category("games") or ["Wuthering Waves"]
            options = [games[0], "Microsoft Edge", "Notepad", "Calculator"]
            self.pending_clarification = {
                "category": "general",
                "options": options,
                "time": now
            }
            options_str = ", ".join(options)
            return {
                "query": text,
                "status": "clarification_needed",
                "best_action": "ask_clarification",
                "confidence": 1.0,
                "message": f"What would you like me to open? You have {options_str}, or you can ask for games, browser, or tools."
            }

        # Step 0D: Category Intent Handling (e.g. "games", "open games", "play a game", "browser")
        cat_match = None
        for pat, cat_name in [
            (r"^(?:(?:open(?:\s+up)?|launch|play|start|run)\s+)?(?:a\s+|some\s+|my\s+)?(?P<cat>games?)$", "games"),
            (r"^(?:let'?s\s+play(?:\s+a)?\s+games?)$", "games"),
            (r"^(?:(?:open(?:\s+up)?|launch|start)\s+)?(?:a\s+|my\s+)?(?P<cat>browser|web\s+browser)$", "browser"),
            (r"^(?:(?:open(?:\s+up)?|play|listen\s+to)\s+)?(?:some\s+|my\s+)?(?P<cat>music|songs)$", "music"),
            (r"^(?:(?:open(?:\s+up)?|check)\s+)?(?:my\s+)?(?P<cat>social\s+media|social)$", "social"),
            (r"^(?:(?:open(?:\s+up)?|launch)\s+)?(?:my\s+)?(?P<cat>code\s+editor|editor|ide)$", "developer"),
            (r"^(?:(?:open(?:\s+up)?|launch|start|run)\s+)?(?:some\s+|a\s+|my\s+)?(?P<cat>software|apps|programs|applications?)$", "apps")
        ]:
            if re.match(pat, clean_lower):
                cat_match = cat_name
                break

        if cat_match:
            preferred = self.learner.get_category_preference(cat_match)
            if preferred:
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
                        options = ["Notepad", "Calculator", "Microsoft Edge"]

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

        # Step 0E: Apply custom user alias mappings
        parsed_text = text
        for al, target in self.learner.data.get("custom_aliases", {}).items():
            if al in clean_lower:
                parsed_text = re.sub(rf"\b{re.escape(al)}\b", target, parsed_text, flags=re.I)

        # Step 1A: Explicit Out of Scope / Conversational Intercept
        qa_chat_pattern = r"^(?:who|what|why|how|when|where|tell\s+me|do\s+you|are\s+you|can\s+you|could\s+you|would\s+you|hi|hello|hey|yo|sup|greetings|good\s+(?:morning|afternoon|evening|night)|howdy|are\s+okay|help(?:\s+me)?|thanks?|thank\s+you|okay|ok|bye|goodbye|chat|talk)\b|^\?+$"
        if re.match(qa_chat_pattern, clean_lower) and not re.match(r"^(?:what|how|where)\s+(?:is|are|can\s+i\s+find)\s+(?:my\s+)?(?:schedule|calendar|tasks?|agenda|plan)", clean_lower) and not re.match(r"^(?:what|how)\s+(?:is|about)\s+(?:the\s+)?(?:status|system|memory|cpu)", clean_lower):
            # Let chat engine handle it or return conversational response
            chat_reply = self.chat_engine.respond(text)
            return {
                "query": text,
                "status": "conversational_response",
                "message": chat_reply,
                "confidence": 1.0
            }
            
        if "weather" in clean_lower or re.match(r"^(?:search(?:\s+the\s+web)?\s+for|google|look\s+up|find\s+out)\b", clean_lower):
            return {
                "query": text,
                "status": "conversational_response",
                "best_action": "None",
                "confidence": 1.0,
                "message": "I am a local PC control agent. I do not answer general trivia, check the weather, or perform web searches."
            }

        # 1. Grammar parse (First pass)
        intent, action, params, s_lang = self.grammar.parse(parsed_text)

        # 1A-bis: Refuse any grammar-matched OOS action (CAPTCHA / anti-bot / evasion)
        if action == "refuse_oos":
            return {
                "query": text,
                "status": "conversational_response",
                "best_action": "None",
                "confidence": 1.0,
                "message": "I cannot help with CAPTCHA solving, anti-bot evasion, or stealth mode. These are out of scope for a local PC control assistant."
            }

        # 1B. Classical Intent Classifier (Second pass for what grammar misses)
        if not action:
            clf_res = self.intent_classifier.predict(clean_lower)
            if clf_res["is_confident"]:
                c_intent = clf_res["top_intent"]
                c_conf = clf_res["confidence"]
                slots = self.slot_tagger.extract_slots(clean_lower, c_intent)

                if c_intent == "OUT_OF_SCOPE":
                    chat_reply = self.chat_engine.respond(text)
                    return {
                        "query": text,
                        "status": "conversational_response",
                        "best_action": "None",
                        "confidence": c_conf,
                        "message": chat_reply
                    }
                
                if c_intent == "NEGATION_REFUSAL":
                    return {
                        "query": text,
                        "status": "negation_refusal",
                        "best_action": "None",
                        "confidence": c_conf,
                        "message": "Understood. I will not execute that action."
                    }

                if c_intent == "REFUSE_CAPTCHA_REQUEST":
                    return {
                        "query": text,
                        "status": "conversational_response",
                        "best_action": "None",
                        "confidence": c_conf,
                        "message": "I cannot help with CAPTCHA solving, anti-bot evasion, or stealth mode. These are out of scope for a local PC control assistant."
                    }

                action_map = {
                    "OPEN_APP": "open_app",
                    "CLOSE_APP": "close_app",
                    "SYSTEM_VOLUME": "volume",
                    "BROWSER_NAVIGATE": "browser_navigate",
                    "BROWSER_EXTRACT": "browser_extract",
                    "OPEN_FILE": "open_file",
                    "SEARCH_FILE": "search_file",
                    "CREATE_FILE": "create_file",
                    "PLAN_DAY": "plan_day",
                    "ADD_TASK": "add_task",
                    "SCREENSHOT": "screenshot",
                    "CLICK_TARGET": "click_target",
                    "INSPECT_CAMERA": "inspect_camera",
                    "SYSTEM_STATUS": "status_report",
                    "ABORT_AUTOMATION": "abort_automation",
                    "WINDOW_STATE": "window_state"
                    # REFUSE_CAPTCHA_REQUEST / ANTI_BOT intentionally omitted — handled above
                }
                mapped_action = action_map.get(c_intent)
                if mapped_action:
                    intent = c_intent
                    action = mapped_action
                    params = slots
                    s_lang = c_conf

                    if "app" in params:
                        c_app, c_info = predict_app(params["app"])
                        params["app"] = c_app
                        params["app_info"] = c_info
                    if "target" in params and action == "close_app":
                        c_tgt, c_info = predict_app(params["target"])
                        params["target"] = c_tgt
                        params["app_info"] = c_info
            else:
                self.active_learning.log_failure(parsed_text, "low_confidence_intent", clf_res["confidence"], clf_res["top_intent"])
                # Let the chat engine respond gracefully to any unrecognized/new command
                chat_reply = self.chat_engine.respond(text)
                return {
                    "query": text,
                    "status": "conversational_response",
                    "message": chat_reply,
                    "confidence": 1.0
                }

        # 2. OS State snapshot
        os_state = self.state_tracker.capture_active_state()
        dt_state = now - os_state["timestamp"]
        s_state = self.state_tracker.score_state_relevance(action or "", os_state) if action else 0.0

        # 3. Vision Detection
        s_vision = 0.0
        dt_vision = (now - self.latest_vision_time) if self.latest_vision_detection else 999.0
        if self.latest_vision_detection:
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
            if action == "open_app":
                target_app = (params.get("app") or params.get("target") or "").strip().lower()
                if target_app in GENERIC_APP_PLACEHOLDERS or params.get("app_info", {}).get("category") == "generic_placeholder":
                    games = self.app_discovery.get_apps_in_category("games") or ["Wuthering Waves"]
                    options = [games[0], "Microsoft Edge", "Notepad", "Calculator"]
                    self.pending_clarification = {
                        "category": "general",
                        "options": options,
                        "time": now
                    }
                    options_str = ", ".join(options)
                    return {
                        "query": text,
                        "status": "clarification_needed",
                        "best_action": "ask_clarification",
                        "confidence": 1.0,
                        "message": f"What would you like me to open? You have {options_str}, or you can ask for games, browser, or tools."
                    }

                # Unknown app safeguard
                if target_app not in APP_CATALOG:
                    found_entry = self.app_discovery.find_app(target_app)
                    if not found_entry:
                        games = self.app_discovery.get_apps_in_category("games") or ["Wuthering Waves"]
                        options = [games[0], "Microsoft Edge", "Notepad", "Calculator"]
                        self.pending_clarification = {
                            "category": "general",
                            "options": options,
                            "time": now
                        }
                        return {
                            "query": text,
                            "status": "clarification_needed",
                            "best_action": "ask_clarification",
                            "confidence": 1.0,
                            "message": f"I couldn't find an application named '{target_app}' on your computer. Would you like to open your games, browser, or a desktop app instead?"
                        }

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

            # Record recent entity in dialogue stack for pronoun resolution
            if action in ("open_app", "close_app"):
                app_target = params.get("app") or params.get("target")
                if app_target:
                    self.dialogue_stack.record_entity("last_app", app_target)
            if action in ("open_file", "search_file", "create_file", "delete_file"):
                path_target = params.get("path") or params.get("filename")
                if path_target:
                    self.dialogue_stack.record_entity("last_file", path_target)
            self.dialogue_stack.record_entity("last_command", text)

            self.action_queue.put(action_msg)
            decision_result["status"] = "dispatched"
            decision_result["task_id"] = task_id
            decision_result["action_msg"] = action_msg
        else:
            chat_reply = self.chat_engine.respond(text)
            decision_result["status"] = "conversational_response"
            decision_result["message"] = chat_reply

        # Log misses for the miss-review loop.
        # IMPORTANT: outcome must record the machine-readable system status, NOT the
        # chat engine's generated reply.  The chat engine uses stochastic sampling
        # (np.random.choice) so its text is non-deterministic and meaningless as a
        # system outcome.  We record the status code and the top predicted intent
        # instead, so the miss-review loop shows actionable signal.
        status = decision_result.get("status")
        if status in ("conversational_response", "clarification_needed", "negation_refusal"):
            from miku.core2_cognitive.miss_reviewer import log_miss
            top3 = []
            top_intent = "UNKNOWN"
            if hasattr(self, "intent_classifier"):
                res = self.intent_classifier.predict(text.strip().lower())
                top3 = res.get("top3", [])
                if top3:
                    top_intent = top3[0][0]
            # Build a deterministic, human-readable outcome string from system state only.
            # Never use decision_result["message"] here — that may be LM-generated text.
            real_outcome = f"status={status} | top_intent={top_intent}"
            if status == "clarification_needed":
                # For clarification, record the clarifying question (a fixed template string,
                # not generated text) so reviewers know what was asked.
                clarify_q = decision_result.get("clarification_question", "")
                if clarify_q:
                    real_outcome = f"status=clarification_needed | question={clarify_q[:120]}"
            log_miss(text, real_outcome, decision_result.get("confidence", 0.0), top3)

        return decision_result

    def handle_action_completion(self, result: ActionResultMsg):
        self.cdsh.release_task_lock(result.task_id)
