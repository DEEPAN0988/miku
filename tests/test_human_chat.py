"""
Test Real-Time Human Conversational Responses and Memory
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.router import Orchestrator
from nlu.engine import NLUEngine

def main():
    orc = Orchestrator(tts_enabled=False)
    nlu = NLUEngine()

    test_queries = [
        "i feel sad today",
        "my name is Deepan",
        "hello",
        "tell me a joke",
        "can we be friends",
        "what is the time",
        "tell me a secret",
        "i am so tired",
        "i got the job",
        "what is your favorite food"
    ]

    for q in test_queries:
        parsed = nlu.parse(q)
        res = orc.handle_intent(parsed)
        print(f"User: {q}")
        print(f"Miku: {res.get('response')}")
        print("-" * 50)

if __name__ == "__main__":
    main()
