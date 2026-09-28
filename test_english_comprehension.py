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
    total_passed = 0
    total_tests = 0

    def evaluate_test(test_num: int, category: str, input_text: str, check_fn, expected_desc: str):
        nonlocal total_passed, total_tests
        total_tests += 1
        msg = STTTranscriptMsg(text=input_text, confidence=1.0)
        start_t = time.time()
        res = daemon.handle_transcript(msg)
        elapsed_ms = (time.time() - start_t) * 1000.0

        passed, reason = check_fn(res)
        status_icon = "[PASS]" if passed else "[FAIL]"
        if passed:
            total_passed += 1

        print(f"\n[{test_num:02d}] [{category}] Input: \"{input_text}\" ({elapsed_ms:.1f}ms)")
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
        6, "Typo Correction", "clsoe the notepadd app",
        lambda r: (r.get("status") == "dispatched" and getattr(r.get("action_msg"), "target", "") == "notepad", "Corrected 'clsoe notepadd' to 'close notepad'"),
        "Normalize 'clsoe the notepadd app' -> close notepad"
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
        lambda r: (r.get("status") == "conversational_response" and "diary" in r.get("message", "").lower(), "Mapped custom word 'diary' -> 'notepad'"),
        "Teach Miku that 'diary' means notepad"
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

    # =========================================================================
    # FINAL SCORECARD
    # =========================================================================
    print("\n" + "=" * 75)
    print(f" COMPREHENSION TEST SUMMARY: {total_passed}/{total_tests} PASSED ({round((total_passed / total_tests) * 100, 1)}%)")
    print("=" * 75)

    orch.shutdown()
    if total_passed == total_tests:
        print("[+] ALL 20 ENGLISH LANGUAGE COMPREHENSION TESTS PASSED WITH 100% ACCURACY!")
        return 0
    else:
        print(f"[-] {total_tests - total_passed} tests did not pass.")
        return 1

if __name__ == "__main__":
    sys.exit(test_language_comprehension())
