import re

with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update banner text
content = re.sub(
    r' \[\*\] In-Process VLM      : ACTIVE \(Moondream2 / PyTorch Native Spatial Grounding\)\n \[\*\] Local Vision Bridge : \[SCAFFOLD\] Ollama llama3\.2-vision \(Meta pre-trained, NOT Miku-trained\)',
    ' [*] Native Inference    : Miku v0.3 loaded directly via PyTorch (no pre-trained models)\n [*] Vision Encoder      : [NOT YET TRAINED] using Win32 structured text parsing',
    content,
    flags=re.MULTILINE|re.DOTALL
)

# Update print_banner()
old_init = r'    if HAS_VISION_DEPS:.*?    print\(\"\\nType your command below, \'/help\' for commands, or \'exit\' / \'quit\' to quit\.\\n\", flush=True\)'

new_init = r'''    print(" [*] Local Inference     : [ACTIVE] Miku v0.3 (own checkpoint, step 98458)", flush=True)
    print(" [*] Vision Encoder      : [NOT YET TRAINED] Win32 structured text fallback", flush=True)

    # Validate Miku own checkpoint exists
    from tools.miku_inference import get_model_info
    info = get_model_info()
    if not info.get("checkpoint_available"):
        print("[!] [WARNING] Miku checkpoint not found! Run train/train_sft.py first.", flush=True)

    print("\nType your command below, '/help' for commands, or 'exit' / 'quit' to quit.\n", flush=True)'''

content = re.sub(old_init, new_init, content, flags=re.MULTILINE|re.DOTALL)

# Also remove check_ollama_health
content = re.sub(r'def check_ollama_health.*?return True\n', '', content, flags=re.MULTILINE|re.DOTALL)

# replace commands help
content = content.replace('  /ollama <host>      Set local Ollama host (default: http://localhost:11434)\n', '')
content = content.replace('  /check              Test Ollama connectivity & list available models\n', '  /check              Test Miku Inference health\n')

# replace /check command implementation
old_check = r'        if prompt_input\.startswith\(\"/check\"\):.*?        continue'
new_check = r'''        if prompt_input.startswith("/check"):
            from tools.miku_inference import get_model_info
            info = get_model_info()
            print("[MIKU SYSTEM HEALTH]")
            print(f"- Checkpoint Available: {info.get('checkpoint_available')}")
            print(f"- Checkpoint Path:      {info.get('checkpoint_path')}")
            print(f"- Weights Loaded:       {info.get('loaded')}")
            if info.get("loaded"):
                print(f"- Device:               {info.get('device')}")
                print(f"- Parameter Count:      {info.get('params')}")
                print(f"- Config:               {info.get('config')}")
            continue'''

content = re.sub(old_check, new_check, content, flags=re.MULTILINE|re.DOTALL)

# replace /ollama command implementation
content = re.sub(r'        if prompt_input\.startswith\(\"/ollama\"\):.*?continue', '', content, flags=re.MULTILINE|re.DOTALL)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(content)
