import re

with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix literal newline in predict_action
text = re.sub(r'predict_action\(f"Objective: \{prompt_input\}\nScreen State: \{screen_state\}"\)', r'predict_action(f"Objective: {prompt_input}\\nScreen State: {screen_state}")', text)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
