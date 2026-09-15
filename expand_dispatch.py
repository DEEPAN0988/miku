with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

OLD = '''                if action_type == "click":
                    target = action_res.get("target", "")
                    if target:
                        print(f"[*] [UI GROUNDING] Searching for '{target}' in accessibility tree...", flush=True)
                        bounds_info = find_element_bounds(target)
                        if bounds_info:
                            rect, center = bounds_info
                            cx, cy = center
                            print(f"[+] [FOUND] '{target}' at ({cx}, {cy}). Dispatching motor commands.", flush=True)
                            human_mouse_move(cx, cy)
                            dispatch_real_click("left")
                        else:
                            print(f"[!] [UI GROUNDING FAILED] Could not locate '{target}'.", flush=True)
                            
                elif action_type == "type":
                    text_to_type = action_res.get("text", "")
                    if text_to_type:
                        print(f"[*] [MOTOR DISPATCH] Typing '{text_to_type}'...", flush=True)
                        dispatch_human_keystrokes(text_to_type)
                        
                else:
                    print(f"[*] [UNHANDLED ACTION] {action_res}", flush=True)'''

NEW = '''                if action_type == "click":
                    target = action_res.get("target", "")
                    if target:
                        print(f"[*] [UI GROUNDING] Searching for '{target}' in accessibility tree...", flush=True)
                        bounds_info = find_element_bounds(target)
                        if bounds_info:
                            rect, center = bounds_info
                            cx, cy = center
                            print(f"[+] [FOUND] '{target}' at ({cx}, {cy}). Dispatching motor commands.", flush=True)
                            human_mouse_move(cx, cy)
                            dispatch_real_click("left")
                        else:
                            print(f"[!] [UI GROUNDING FAILED] Could not locate '{target}'.", flush=True)

                elif action_type == "type":
                    text_to_type = action_res.get("text", "")
                    if text_to_type:
                        print(f"[*] [MOTOR DISPATCH] Typing '{text_to_type}'...", flush=True)
                        dispatch_human_keystrokes(text_to_type)

                elif action_type == "launch":
                    target = action_res.get("target", "")
                    if target:
                        print(f"[*] [LAUNCH] Opening '{target}' via ShellExecuteW...", flush=True)
                        try:
                            ctypes.windll.shell32.ShellExecuteW(None, "open", target, None, None, 1)
                            print(f"[+] [LAUNCH OK] '{target}' dispatched.", flush=True)
                        except Exception as launch_err:
                            print(f"[!] ShellExecute failed ({launch_err}), trying Start Menu search...", flush=True)
                            import time as _time
                            u32 = ctypes.windll.user32
                            u32.keybd_event(0x5B, 0, 0, 0)
                            u32.keybd_event(0x5B, 0, 2, 0)
                            _time.sleep(0.4)
                            dispatch_human_keystrokes(target)
                            _time.sleep(0.3)
                            u32.keybd_event(0x0D, 0, 0, 0)
                            u32.keybd_event(0x0D, 0, 2, 0)

                elif action_type == "press_key":
                    key = action_res.get("key", "")
                    if key:
                        print(f"[*] [KEY DISPATCH] Pressing '{key}'...", flush=True)
                        _KEY_VK = {
                            "win": 0x5B, "alt": 0x12, "ctrl": 0x11, "shift": 0x10,
                            "enter": 0x0D, "return": 0x0D, "tab": 0x09, "escape": 0x1B,
                            "esc": 0x1B, "space": 0x20, "backspace": 0x08, "delete": 0x2E,
                            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
                            "f4": 0x73, "f5": 0x74,
                        }
                        parts = [p.strip() for p in key.lower().split("+")]
                        mods = [p for p in parts[:-1] if p in _KEY_VK]
                        main_key = parts[-1]
                        u32 = ctypes.windll.user32
                        for mod in mods:
                            u32.keybd_event(_KEY_VK[mod], 0, 0, 0)
                        vk = _KEY_VK.get(main_key, ord(main_key.upper()[0]) if len(main_key) == 1 else 0)
                        if vk:
                            u32.keybd_event(vk, 0, 0, 0)
                            u32.keybd_event(vk, 0, 2, 0)
                        for mod in reversed(mods):
                            u32.keybd_event(_KEY_VK[mod], 0, 2, 0)

                elif action_type == "task":
                    print(f"[*] [TASK] Delegating composite task to agent loop...", flush=True)
                    client_local = AstraVisionClient(model="miku_inference")
                    res = run_computer_use_task(
                        objective=action_res.get("objective", prompt_input),
                        client=client_local,
                        real_execution=True,
                        delay_between_steps=0.25,
                    )
                    elapsed = time.perf_counter() - t_start
                    print(f"[+] [TASK DONE] {elapsed:.2f}s | Steps: {res.total_steps} | {res.final_status}", flush=True)

                else:
                    print(f"[!] [UNHANDLED ACTION] {action_res}", flush=True)'''

if OLD in text:
    text = text.replace(OLD, NEW)
    print("Replaced OK")
else:
    print("ERROR: Could not find pattern — manual inspection needed")
    exit(1)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
