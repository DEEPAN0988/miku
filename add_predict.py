import sys
with open(r'c:\miku\tools\miku_inference.py', 'r', encoding='utf-8') as f:
    text = f.read()

predict_func = '''
def predict_action(context_text: str) -> dict:
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
    match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "error", "message": "Failed to parse action", "raw_output": raw_output}
'''

text = text.replace('def get_screen_state_text() -> str:', predict_func + '\n\ndef get_screen_state_text() -> str:')

with open(r'c:\miku\tools\miku_inference.py', 'w', encoding='utf-8') as f:
    f.write(text)
