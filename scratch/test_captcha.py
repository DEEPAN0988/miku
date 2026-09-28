import sys
sys.path.insert(0, 'c:/miku')
from miku.core2_cognitive.cognitive_daemon import CognitiveDaemon
from multiprocessing import Queue
from miku.ipc.messages import STTTranscriptMsg

d = CognitiveDaemon(action_queue=Queue())

cases = [
    'solve this captcha',
    'bypass the captcha',
    'crack the captcha',
    'enable anti-bot mode',
    'stealth mode',
    'anti-bot',
]
print("=== CAPTCHA / ANTI-BOT HANDLING ===")
for t in cases:
    r = d.handle_transcript(STTTranscriptMsg(text=t, confidence=1.0))
    status = r.get("status")
    action = r.get("best_action")
    msg = r.get("message", "")[:80]
    print(f"  {t!r}")
    print(f"    status={status}  action={action}")
    print(f"    msg={msg!r}")
    print()
