import os
import re

# Fix dangling ollama kwargs in miku.py
with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()
    
text = re.sub(r'                model=ollama_model,\s+ollama_host=ollama_host,\s+ollama_model=ollama_model,', '', text, flags=re.MULTILINE|re.DOTALL)
text = re.sub(r'            ollama_host = os\.environ\.get\("OLLAMA_HOST", "http://localhost:11434"\)\n            ollama_model = os\.environ\.get\("OLLAMA_VISION_MODEL", "llama3\.2-vision"\)\n', '', text, flags=re.MULTILINE|re.DOTALL)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)

# Delete obsolete test files
for test_file in [r'c:\miku\eval\test_orchestrator_ollama.py', r'c:\miku\eval\test_local_vision.py']:
    if os.path.exists(test_file):
        os.remove(test_file)
