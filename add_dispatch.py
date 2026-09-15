import sys
with open(r'c:\miku\miku.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_loop = '''        # ----------------------------------------------------------------------
        # Multi-Step Computer Use Agent Loop Dispatch (Cloud Astra or Miku Inference)
        # ----------------------------------------------------------------------
        print(f"\\n[*] [TASK DISPATCHED] Objective: \\"{prompt_input}\\"", flush=True)
        t_start = time.perf_counter()

        has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
        if has_api_key:
            model = os.environ.get("ASTRA_MODEL", "openai/gpt-6-astra")
            client = AstraVisionClient(model=model)
            try:
                res: ComputerUseTaskResult = run_computer_use_task(
                    objective=prompt_input,
                    client=client,
                    real_execution=True,
                    delay_between_steps=0.25,
                )
                elapsed = time.perf_counter() - t_start
                print(f"\\n[+] [TASK COMPLETED] Duration: {elapsed:.2f}s | Steps: {res.total_steps} | Status: {res.final_status}", flush=True)
                if res.output:
                    print(f"    Summary: {res.output}", flush=True)
                if res.error:
                    print(f"    [!] Details: {res.error}", flush=True)
            except KeyboardInterrupt:
                print("\\n[!] [TASK ABORTED] Execution stopped by user via CTRL+C. Control returned to console.", flush=True)
            except Exception as exc:
                print(f"\\n[!] [TASK ERROR] {exc}", flush=True)
        else:
            # 100% Native Real-Time Motor Dispatch
            print(f"[*] [MIKU NATIVE INFERENCE] Invoking local model...", flush=True)
            from tools.miku_inference import predict_action, get_screen_state_text
            from tools.screen_inspector import find_element_bounds, human_mouse_move, dispatch_real_click
            from tools.typing_automation import dispatch_human_keystrokes
            
            try:
                screen_state = get_screen_state_text()
                action_res = predict_action(f"Objective: {prompt_input}\\nScreen State: {screen_state}")
                
                print(f"[*] [MODEL OUTPUT] {action_res}", flush=True)
                
                action_type = action_res.get("action", "")
                
                if action_type == "click":
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
                    print(f"[*] [UNHANDLED ACTION] {action_res}", flush=True)
                    
            except KeyboardInterrupt:
                print("\\n[!] [TASK ABORTED] Execution stopped by user via CTRL+C. Control returned to console.", flush=True)
            except Exception as exc:
                print(f"\\n[!] [TASK ERROR] {exc}", flush=True)

        print("-" * 75, flush=True)
'''

import re
text = re.sub(r'        # ----------------------------------------------------------------------\n        # Multi-Step Computer Use Agent Loop Dispatch.*?print\("-" \* 75, flush=True\)', new_loop, text, flags=re.MULTILINE|re.DOTALL)

with open(r'c:\miku\miku.py', 'w', encoding='utf-8') as f:
    f.write(text)
