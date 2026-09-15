with open(r'c:\miku\tools\miku_inference.py', 'r', encoding='utf-8') as f:
    text = f.read()

OLD = '''def predict_action(context_text: str) -> dict:
    """
    Evaluates the user command alongside structured UI context and outputs discrete action decisions.
    Args:
        context_text: The unified prompt containing the user command and structured UI elements.
    Returns:
        A dictionary containing the parsed JSON action (e.g., {"action": "click", "target": "Calculator"}).
    """
    raw_output = generate(context_text, max_new_tokens=96, temperature=0.1, top_k=10)
    
    # Attempt to parse the first JSON object from the raw_output
    import re, json
    match = re.search(r\'\\{.*?\\}\', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "error", "message": "Failed to parse action", "raw_output": raw_output}'''

NEW = r'''def predict_action(context_text: str) -> dict:
    """
    Evaluates the user command alongside structured UI context and outputs discrete action decisions.

    Strategy (fully local, no external calls):
      1. Try the local MikuLM model to generate a JSON action.
      2. If the model output is empty or unparseable (expected during early training),
         fall back to a deterministic NLP intent parser on the original prompt.

    Args:
        context_text: Unified prompt containing the user command and structured UI elements.
    Returns:
        A dictionary containing the parsed JSON action.
    """
    import re as _re
    import json as _json

    raw_output = generate(context_text, max_new_tokens=96, temperature=0.1, top_k=10)

    # 1. Try to parse JSON from model output
    if raw_output and raw_output.strip():
        match = _re.search(r'\{.*?\}', raw_output, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group(0))
            except _json.JSONDecodeError:
                pass

    # 2. Fallback: deterministic NLP intent parser on the original prompt
    objective_match = _re.search(r'Objective:\s*(.+?)(?:\n|$)', context_text)
    objective = objective_match.group(1).strip() if objective_match else context_text.strip()
    return _parse_intent(objective)


def _parse_intent(objective: str) -> dict:
    """
    Deterministic intent parser. Converts natural-language desktop commands into
    discrete JSON action dictionaries without any model inference.
    """
    import re as _re
    cmd = objective.strip().lower()

    # open / launch / start / run <app>
    m = _re.match(r'^(?:open|launch|start|run)\s+(.+)$', cmd)
    if m:
        return {"action": "launch", "target": m.group(1).strip()}

    # click / tap <element>
    m = _re.match(r'^(?:click|tap|press\s+on)\s+(?:the\s+)?(.+)$', cmd)
    if m:
        target = m.group(1).strip()
        if target in ("enter", "return", "tab", "escape", "esc", "space", "backspace", "delete"):
            return {"action": "press_key", "key": target}
        return {"action": "click", "target": target}

    # type / write / input <text>
    m = _re.match(r'^(?:type|write|input|enter)\s+(.+)$', cmd)
    if m:
        return {"action": "type", "text": m.group(1).strip()}

    # close / exit / quit
    if _re.match(r'^(?:close|exit|quit|kill)\b', cmd):
        return {"action": "press_key", "key": "alt+f4"}

    # scroll
    m = _re.match(r'^scroll\s+(up|down|left|right)(?:\s+(\d+))?$', cmd)
    if m:
        return {"action": "scroll", "direction": m.group(1), "amount": int(m.group(2) or 3)}

    # minimize / maximize
    if "minimize" in cmd:
        return {"action": "press_key", "key": "win+down"}
    if "maximize" in cmd:
        return {"action": "press_key", "key": "win+up"}

    # Default: composite task
    return {"action": "task", "objective": objective}'''

if OLD in text:
    text = text.replace(OLD, NEW)
    print("Replaced OK")
else:
    print("Pattern not found — trying whitespace-normalized search")
    import re
    # Try regex fallback
    pattern = r'def predict_action\(context_text: str\) -> dict:.*?return \{"action": "error".*?\}'
    if re.search(pattern, text, re.DOTALL):
        text = re.sub(pattern, NEW, text, flags=re.DOTALL)
        print("Replaced via regex OK")
    else:
        print("ERROR: Could not find pattern")

with open(r'c:\miku\tools\miku_inference.py', 'w', encoding='utf-8') as f:
    f.write(text)
