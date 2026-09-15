import re

with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace all occurrences of physical newlines inside strings in print statements that were mistakenly added
text = re.sub(r'print\((f?)"\n(.*?)"', r'print(\1"\\n\2"', text)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
