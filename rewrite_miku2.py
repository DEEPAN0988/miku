with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace print_status variables
text = text.replace('    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")\n', '')
text = text.replace('    ollama_model = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")\n', '')
text = text.replace('    print(f"  Local Ollama Host   : {ollama_host} (Model: {ollama_model})", flush=True)\n', '')
text = text.replace('    print(f"  API Key Configured  : {\'YES\' if has_key else \'NO (Local Ollama bridge active)\'}", flush=True)\n', '    print(f"  API Key Configured  : {\'YES\' if has_key else \'NO (Miku inference active)\'}", flush=True)\n')

# Replace AstraVisionClient instantiation for local mode (lines 247-252)
old_client_instantiation = '''        else:
            ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            ollama_model = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")
            client = AstraVisionClient(
                model=ollama_model,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
            )'''
new_client_instantiation = '''        else:
            client = AstraVisionClient(model="miku_inference")'''
text = text.replace(old_client_instantiation, new_client_instantiation)

# Replace the comment line 237
text = text.replace('# Multi-Step Computer Use Agent Loop Dispatch (Cloud Astra or Local Ollama)', '# Multi-Step Computer Use Agent Loop Dispatch (Cloud Astra or Miku Inference)')

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
