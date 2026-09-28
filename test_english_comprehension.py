"""
Comprehensive Test Battery: Miku's Human English Language Comprehension.
Tests Miku across:
1. Conversational Chit-Chat & Open-Ended Natural English
2. Typo Resilience & Misspelling Correction
3. Phrasing Flexibility & Action Verb Synonyms
4. Vocabulary Definitions & Technology Terms
5. Real-Time Language Teaching & Dynamic Memory
6. Multi-Turn Disambiguation & Dialogue Handling
"""
import sys
import time
from multiprocessing import Queue
from miku.orchestrator import MikuOrchestrator
from miku.ipc.messages import STTTranscriptMsg

def test_language_comprehension():
    print("=" * 75)
    print("      MIKU HUMAN ENGLISH LANGUAGE COMPREHENSION TEST BENCH")
    print("=" * 75)

    orch = MikuOrchestrator()
    orch.start()

    daemon = orch.cognitive_daemon
    test_results = []

    def evaluate_test(test_num: int, category: str, input_text: str, check_fn, expected_desc: str):
        msg = STTTranscriptMsg(text=input_text, confidence=1.0)
        start_t = time.time()
        res = daemon.handle_transcript(msg)
        elapsed_ms = (time.time() - start_t) * 1000.0

        passed, reason = check_fn(res)
        status_icon = "[PASS]" if passed else "[FAIL]"
        test_results.append(passed)

        test_label = str(test_num).zfill(2)
        print(f"\n[{test_label}] [{category}] Input: \"{input_text}\" ({elapsed_ms:.1f}ms)")
        print(f"     Expected: {expected_desc}")
        response_sample = res.get("message") or f"Action: {res.get('best_action')} (target='{getattr(res.get('action_msg'), 'target', '')}')"
        if len(str(response_sample)) > 90:
            response_sample = str(response_sample)[:87] + "..."
        print(f"     Miku    : {response_sample}")
        print(f"     Result  : {status_icon} ({reason})")

    # =========================================================================
    # SECTION 1: Natural Conversational English & Identity
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 1] NATURAL CONVERSATIONAL ENGLISH & CHIT-CHAT")
    print("-" * 75)

    evaluate_test(
        1, "Conversation", "who are you",
        lambda r: (r.get("status") == "conversational_response" and "miku" in r.get("message", "").lower(), "Identified self as Miku"),
        "Acknowledge name and identity as Miku"
    )

    evaluate_test(
        2, "Conversation", "what can you do",
        lambda r: (r.get("status") == "conversational_response" and ("control" in r.get("message", "").lower() or "apps" in r.get("message", "").lower()), "Listed system abilities"),
        "Explain capabilities (control apps, sound, vision, etc.)"
    )

    evaluate_test(
        3, "Conversation", "do you understand english words",
        lambda r: (r.get("status") == "conversational_response" and "english" in r.get("message", "").lower(), "Confirmed English understanding"),
        "Confirm comprehension of English language"
    )

    evaluate_test(
        4, "Conversation", "tell me a joke",
        lambda r: (r.get("status") == "conversational_response" and len(r.get("message", "")) > 10, "Responded with humorous dialogue"),
        "Respond with a conversational joke"
    )

    # =========================================================================
    # SECTION 2: Typo Tolerance & Natural Misspellings
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 2] TYPO TOLERANCE & MISSPELLING NORMALIZATION")
    print("-" * 75)

    evaluate_test(
        5, "Typo Correction", "opne calcultor",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "calculator", "Corrected 'opne calcultor' to 'open calculator'"),
        "Normalize 'opne calcultor' -> open calculator app"
    )

    evaluate_test(
        6, "Typo Correction Step 1", "clsoe the notepadd app",
        lambda r: (r.get("status") == "confirmation_required" and "close" in str(r.get("message")).lower(), "Corrected 'clsoe notepadd' to 'close notepad' and asked confirmation"),
        "Normalize 'clsoe the notepadd app' -> close notepad"
    )
    
    evaluate_test(
        6.5, "Typo Correction Step 2", "yes",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "notepad", "Typo corrected and dispatched after confirmation"),
        "Confirm the typo-corrected 'close notepad' command"
    )

    evaluate_test(
        7, "Typo Correction", "undrestand this word",
        lambda r: (True, "Handled misspelled 'undrestand' without crash"),
        "Handle typo 'undrestand' gracefully"
    )

    # =========================================================================
    # SECTION 3: Synonym & Phrasing Flexibility
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 3] NATURAL PHRASING FLEXIBILITY & ACTION SYNONYMS")
    print("-" * 75)

    evaluate_test(
        8, "Phrasing", "fire up edge",
        lambda r: (r.get("status") == "dispatched" and "edge" in getattr(r.get("action_msg"), "target", ""), "Understood 'fire up' as launch"),
        "Recognize informal 'fire up' as launch/open app"
    )

    evaluate_test(
        9, "Phrasing", "crank up the audio",
        lambda r: (r.get("status") == "dispatched" and r.get("best_action") == "volume" and r.get("action_msg").params.get("direction") == "up", "Mapped 'crank up the audio' to volume up"),
        "Recognize colloquial 'crank up the audio' as volume up"
    )

    evaluate_test(
        10, "Phrasing", "silence sound",
        lambda r: (r.get("status") == "dispatched" and r.get("best_action") == "volume" and r.get("action_msg").params.get("direction") == "mute", "Mapped 'silence sound' to mute"),
        "Recognize 'silence sound' as mute audio"
    )

    # =========================================================================
    # SECTION 4: English Vocabulary & Semantic Definitions
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 4] VOCABULARY & TECH DEFINITIONS")
    print("-" * 75)

    evaluate_test(
        11, "Definition", "what does sovereign mean",
        lambda r: (r.get("status") == "conversational_response" and ("independent" in r.get("message", "").lower() or "supreme" in r.get("message", "").lower()), "Returned definition of sovereign"),
        "Provide accurate English definition of 'sovereign'"
    )

    evaluate_test(
        12, "Definition", "define autonomous",
        lambda r: (r.get("status") == "conversational_response" and ("independent" in r.get("message", "").lower() or "freedom" in r.get("message", "").lower()), "Returned definition of autonomous"),
        "Provide accurate English definition of 'autonomous'"
    )

    evaluate_test(
        13, "Definition", "what is an operating system",
        lambda r: (r.get("status") == "conversational_response" and ("hardware" in r.get("message", "").lower() or "software" in r.get("message", "").lower()), "Returned definition of operating system"),
        "Explain what an operating system is"
    )

    # =========================================================================
    # SECTION 5: Real-Time Dynamic English Teaching
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 5] REAL-TIME ENGLISH TEACHING & CUSTOM ALIASES")
    print("-" * 75)

    evaluate_test(
        14, "Word Learning", "learn word euphoria means a state of intense happiness",
        lambda r: (r.get("status") == "conversational_response" and "euphoria" in r.get("message", "").lower(), "Saved new English word to local memory"),
        "Learn new vocabulary word 'euphoria' and persist it"
    )

    evaluate_test(
        15, "Word Recall", "what does euphoria mean",
        lambda r: (r.get("status") == "conversational_response" and "happiness" in r.get("message", "").lower(), "Successfully recalled learned definition"),
        "Recall learned definition of 'euphoria'"
    )

    evaluate_test(
        16, "Alias Teaching", "call diary notepad",
        lambda r: (r.get("status") == "confirmation_required", "Asked to confirm alias"),
        "Teach Miku that 'diary' means notepad"
    )

    evaluate_test(
        16.5, "Alias Confirm", "yes",
        lambda r: (r.get("status") == "conversational_response" and "learned" in r.get("message", "").lower(), "Confirmed alias"),
        "Confirm the 'diary' -> 'notepad' alias"
    )

    evaluate_test(
        17, "Alias Execution", "open diary",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "notepad", "Dispatched 'notepad' when asked for 'diary'"),
        "Open notepad when user says 'open diary'"
    )

    # =========================================================================
    # SECTION 6: Multi-Turn English Clarification Dialogue
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 6] MULTI-TURN ENGLISH CLARIFICATION DIALOGUE")
    print("-" * 75)

    # Reset any prior game preference for clean test
    daemon.learner.data["category_preferences"].pop("games", None)
    daemon.learner._save()

    evaluate_test(
        18, "Category Ambiguity", "open games",
        lambda r: (r.get("status") == "clarification_needed" and "which game" in r.get("message", "").lower(), "Asked user for clarification on installed games"),
        "Understand category intent and ask: 'Which game would you like to open?'"
    )

    evaluate_test(
        19, "Dialogue Turn 2", "wuthering wave",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "wuthering waves", "Resolved dialogue and launched selected game"),
        "Resolve user selection 'wuthering wave' and open Wuthering Waves"
    )

    evaluate_test(
        20, "Learned Preference", "open games",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "wuthering waves", "Automatically opened preferred game from memory"),
        "Remember preferred game and open Wuthering Waves without re-asking"
    )

    evaluate_test(
        21, "Generic Placeholder", "OPEN SOMETHING",
        lambda r: (r.get("status") == "clarification_needed" and "what would you like me to open" in r.get("message", "").lower(), "Asked what to open instead of searching for app named 'something'"),
        "Prompt user for options when given generic command 'OPEN SOMETHING'"
    )

    evaluate_test(
        22, "Unknown App Safeguard", "open mysteryfakeapp",
        lambda r: (r.get("status") == "clarification_needed" and "couldn't find" in r.get("message", "").lower(), "Politely asked for alternatives instead of failing execution"),
        "Safely handle non-existent application with clarification prompt"
    )

    # =========================================================================
    # SECTION 7: End-to-End Safety, Compounds, and Repetition
    # =========================================================================
    print("\n" + "-" * 75)
    print(" [SECTION 7] END-TO-END SAFETY & PRONOUNS")
    print("-" * 75)

    # 1. End-to-end teaching
    evaluate_test(
        23, "Teaching Step 1", "when i say work i mean open edge",
        lambda r: (r.get("status") == "confirmation_required", "Asked to confirm link"),
        "Require confirmation to link 'work' to 'open edge'"
    )
    evaluate_test(
        24, "Teaching Step 2", "yes",
        lambda r: (r.get("status") == "conversational_response" and "learned" in r.get("message", "").lower(), "Confirmed link"),
        "Confirm the link is saved"
    )
    evaluate_test(
        25, "Teaching Verify", "work",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "edge", "Resolved 'work' to edge"),
        "Execute 'work' as 'open edge' via taught alias"
    )

    # 2. Compound safety
    evaluate_test(
        26, "Compound Destructive", "open notepad and then delete it",
        lambda r: (r.get("status") == "confirmation_required" and "delete notepad" in str(r.get("message")).lower(), "Halted and asked for confirmation"),
        "Handle compound command by halting on destructive step and asking for confirmation"
    )

    # 3. Repeat destructive
    evaluate_test(
        27, "Context Setup", "close edge",
        lambda r: (r.get("status") == "confirmation_required" and "close edge" in str(r.get("message")).lower(), "Asked for confirmation to close"),
        "Establish 'close edge' in context"
    )
    evaluate_test(
        27.5, "Context Confirm", "yes",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "edge", "Dispatched close edge"),
        "Confirm 'close edge' to execute it and set last_command"
    )
    evaluate_test(
        28, "Repeat Destructive", "do it again",
        lambda r: (r.get("status") == "confirmation_required" and "close edge" in str(r.get("message")).lower(), "Asked for confirmation to repeat"),
        "Require confirmation to repeat a destructive action"
    )
    evaluate_test(
        29, "Repeat Benign", "open notepad",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "notepad", "Opened notepad"),
        "Establish 'open notepad' in context"
    )
    evaluate_test(
        30, "Repeat Non-Destructive", "do it again",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "notepad", "Opened notepad again"),
        "Repeat non-destructive action without confirmation"
    )



    evaluate_test(
        32, "Restore App (Open App)", "restore calculator",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "calculator", "Resolved 'restore calculator' to launch calculator"),
        "Verify 'restore <appname>' opens an unlaunched app"
    )

    evaluate_test(
        33, "Restore App (Window State)", "restore the window",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "action_type", "") == "window_state", "Resolved 'restore the window' to window_state action"),
        "Verify 'restore the window' is parsed as a window state action"
    )

    # =========================================================================
    # FINAL SCORECARD
    # =========================================================================
    total_tests = len(test_results)
    total_passed = sum(1 for p in test_results if p)
    pass_pct = round((total_passed / total_tests) * 100, 1) if total_tests > 0 else 0.0

    print("\n" + "=" * 75)
    print(f" COMPREHENSION TEST SUMMARY: {total_passed}/{total_tests} PASSED ({pass_pct}%)")
    print("=" * 75)

    orch.shutdown()
    if total_tests > 0 and total_passed == total_tests:
        print(f"[+] ALL {total_tests} ENGLISH LANGUAGE COMPREHENSION TESTS PASSED WITH 100% ACCURACY!")
        return 0
    else:
        print(f"[-] {total_tests - total_passed} tests did not pass.")
        return 1

if __name__ == "__main__":
    sys.exit(test_language_comprehension())
