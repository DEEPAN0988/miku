with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

OLD = '    # Validate Miku own checkpoint exists\n    from tools.miku_inference import get_model_info\n    info = get_model_info()\n    if not info.get("checkpoint_available"):\n        print("[!] [WARNING] Miku checkpoint not found! Run train/train_sft.py first.", flush=True)\n\n    print("\\nType your command below, \'/help\' for commands, or \'exit\' / \'quit\' to quit.\\n", flush=True)'

NEW = '''    # Eagerly load & cache Miku model at startup to avoid cold-start on first command
    from tools.miku_inference import load_miku_model, get_model_info
    info = get_model_info()
    if info.get("checkpoint_available"):
        print(" [*] Loading Miku model (one-time startup)...", flush=True)
        try:
            _, _, dev = load_miku_model()
            print(f" [+] Model ready on {dev}.", flush=True)
        except Exception as e:
            print(f" [!] Model load failed: {e}", flush=True)
    else:
        print("[!] [WARNING] Miku checkpoint not found! Run train/train_sft.py first.", flush=True)

    print("\\nType your command below, \'/help\' for commands, or \'exit\' / \'quit\' to quit.\\n", flush=True)'''

if OLD in text:
    text = text.replace(OLD, NEW)
    print("Replaced OK")
else:
    print("Pattern not found, skipping")

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
