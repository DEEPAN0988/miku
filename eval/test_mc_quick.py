"""Quick smoke test — no model loading required (calls _parse_intent directly)."""
import re, sys
sys.path.insert(0, r"c:\miku")

errors = []

# --- Test 1: split regex ---
prompt = '"open chrome", "click start", "type hello world", "close"'
quoted = re.findall(r'"([^"]+)"', prompt)
expected = ["open chrome", "click start", "type hello world", "close"]
ok = (quoted == expected)
print(f"[T1] Multi-command split: {'PASS' if ok else 'FAIL'} -> {quoted}")
if not ok:
    errors.append(f"T1 failed: {quoted}")

sub_commands = quoted if len(quoted) >= 2 else [prompt]
assert sub_commands == expected

# --- Test 2: _parse_intent maps each bare command ---
from tools.miku_inference import _parse_intent

EXPECT = {
    "open chrome":      {"action": "launch", "target": "chrome"},
    "click start":      {"action": "click",  "target": "start"},
    "type hello world": {"action": "type",   "text":   "hello world"},
    "close":            {"action": "press_key", "key": "alt+f4"},
}

print("\n[T2] _parse_intent results:")
for cmd, want in EXPECT.items():
    got = _parse_intent(cmd)
    ok = all(got.get(k) == v for k, v in want.items())
    print(f"  {'PASS' if ok else 'FAIL'}  {cmd!r:30s} -> {got}")
    if not ok:
        errors.append(f"_parse_intent({cmd!r}): want {want}, got {got}")

# --- Test 3: no "task" fallback for known commands ---
print("\n[T3] No 'task' fallback for any known sub-command:")
for cmd in EXPECT:
    got = _parse_intent(cmd)
    is_task = got.get("action") == "task"
    ok = not is_task
    print(f"  {'PASS' if ok else 'FAIL'}  {cmd!r:30s}  action={got.get('action')!r}")
    if not ok:
        errors.append(f"{cmd!r} still returns task")

# --- Test 4: single unquoted command not split ---
single = "open notepad"
q2 = re.findall(r'"([^"]+)"', single)
sub2 = q2 if len(q2) >= 2 else [single]
ok = (sub2 == ["open notepad"])
print(f"\n[T4] Single unquoted command: {'PASS' if ok else 'FAIL'} -> {sub2}")
if not ok:
    errors.append(f"single command split: {sub2}")

print()
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ! {e}")
    sys.exit(1)
else:
    print("All tests PASSED.")
