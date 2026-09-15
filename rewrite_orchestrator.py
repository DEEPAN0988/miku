import re

with open(r'c:\miku\tools\orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove query_ollama_vision completely
content = re.sub(r'def query_ollama_vision\(.*?^SUPPORTED_ASTRA_ACTIONS =', 'SUPPORTED_ASTRA_ACTIONS =', content, flags=re.MULTILINE|re.DOTALL)

# Replace the query_action method
old_method = r'    def query_action\(.*?return res\n'
new_method = '''    def query_action(
        self,
        query: str,
        base64_image_url: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Dispatches multimodal vision request to model endpoint.
        Uses api_runner if provided, calls OpenAI if api_key is configured,
        otherwise seamlessly routes through the local Miku inference engine (text-only).
        """
        payload = self.build_vision_payload(query, base64_image_url, system_prompt=system_prompt, history=history)
        if self.api_runner is not None:
            res = self.api_runner(payload)
        elif self.api_key:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                res = data["choices"][0]["message"]["content"]
        else:
            # Offline / local execution bridge: route to Miku's native inference engine
            from tools.miku_inference import generate_action_json, get_screen_state_text
            screen_state = get_screen_state_text()
            res = generate_action_json(
                objective=query,
                screen_state=screen_state,
                action_history=history,
            )

        # Physical visual glide preview during live execution
        if os.environ.get("MIKU_LIVE_EXECUTION", "false").strip().lower() in ("true", "1", "yes"):
            try:
                self._dispatch_visual_preview(res)
            except Exception:
                pass
        return res
'''
content = re.sub(old_method, new_method, content, flags=re.MULTILINE|re.DOTALL)

with open(r'c:\miku\tools\orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)
