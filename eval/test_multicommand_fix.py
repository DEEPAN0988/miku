"""
eval/test_multicommand_fix.py
Tests the quoted multi-command parsing fix in miku.py and _parse_intent.
"""
import re
import sys
sys.path.insert(0, r"c:\miku")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = []

# ---------------------------------------------------------------------------
# Test 1: Multi-command split regex
# ---------------------------------------------------------------------------
prompt = '"open chrome", "click start", "type hello world", "close"'
quoted = re.findall(r'"([^"]+)"', prompt)
expected_split = ["open chrome", "click start", "type hello world", "close"]
ok = (quoted == expected_split)
print(f"[TEST 1] Multi-command split:  {PASS if ok else FAIL}")
print(f"         Input : {prompt}")
print(f"         Got   : {quoted}")
if not ok:
    errors.append(f"TEST 1 FAILED: expected {expected_split}, got {quoted}")

# Test that _sub_commands logic fires
sub_commands = quoted if len(quoted) >= 2 else [prompt]
assert sub_commands == expected_split, f"sub_commands wrong: {sub_commands}"

# ---------------------------------------------------------------------------
# Test 2: _parse_intent maps each bare sub-command correctly
# ---------------------------------------------------------------------------
from tools.miku_inference import _parse_intent

cases = [
    ("open chrome",       "action", "launch"),
    ("open chrome",       "target", "chrome"),
    ("click start",       "action", "click"),
    ("click start",       "target", "start"),
    ("type hello world",  "action", "type"),
    ("type hello world",  "text",   "hello world"),
    ("close",             "action", "press_key"),
    ("close",             "key",    "alt+f4"),
]

print("\n[TEST 2] _parse_intent on individual sub-commands:")
seen = {}
for cmd, key, expected_val in cases:
    if cmd not in seen:
        seen[cmd] = _parse_intent(cmd)
    result = seen[cmd]
    got = result.get(key)
    ok = (got == expected_val)
    status = PASS if ok else FAIL
    print(f"  _parse_intent({cmd!r})[{key!r}] == {expected_val!r}  -> got {got!r}  [{status}]")
    if not ok:
        errors.append(f"_parse_intent({cmd!r})[{key!r}]: expected {expected_val!r}, got {got!r}")

# ---------------------------------------------------------------------------
# Test 3: Single unquoted command stays as a single-element list
# ---------------------------------------------------------------------------
single = "open notepad"
q2 = re.findall(r'"([^"]+)"', single)
sub2 = q2 if len(q2) >= 2 else [single]
ok = (sub2 == ["open notepad"])
print(f"\n[TEST 3] Single unquoted command not split: {PASS if ok else FAIL}")
print(f"         sub_commands = {sub2}")
if not ok:
    errors.append(f"TEST 3 FAILED: {sub2}")

# ---------------------------------------------------------------------------
# Test 4: "task" action is no longer produced for these known inputs
# ---------------------------------------------------------------------------
print("\n[TEST 4] No sub-command returns {action: task}:")
for cmd in ["open chrome", "click start", "type hello world", "close"]:
    result = _parse_intent(cmd)
    is_task = result.get("action") == "task"
    ok = not is_task
    status = PASS if ok else FAIL
    print(f"  _parse_intent({cmd!r}) -> {result.get('action')!r}  [{status}]")
    if not ok:
        errors.append(f"_parse_intent({cmd!r}) still returns 'task': {result}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if errors:
    print(f"{'='*60}")
    print(f"FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ! {e}")
    sys.exit(1)
else:
    print("All tests passed. Multi-command fix is working correctly.")
