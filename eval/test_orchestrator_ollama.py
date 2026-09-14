"""
eval/test_orchestrator_ollama.py — Unit Tests for Local Ollama Vision Client Bridge
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from tools.orchestrator import (
    AstraVisionClient,
    query_ollama_vision,
    parse_astra_action,
)


class TestOllamaVisionBridge(unittest.TestCase):
    """Tests for query_ollama_vision and AstraVisionClient offline Ollama routing."""

    @patch("requests.post")
    def test_query_ollama_vision_payload_and_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": '{"action": "click", "x": 450, "y": 250}',
            },
            "done": True,
        }
        mock_post.return_value = mock_resp

        b64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        response = query_ollama_vision(
            prompt="Click the Submit button",
            base64_image_url=b64_img,
            host="http://localhost:11434",
            model="llama3.2-vision",
        )

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], "http://localhost:11434/api/chat")
        
        payload = call_kwargs["json"]
        self.assertEqual(payload["model"], "llama3.2-vision")
        self.assertEqual(payload["format"], "json")
        self.assertFalse(payload["stream"])
        
        # Verify image prefix is stripped
        user_msg = payload["messages"][1]
        self.assertEqual(user_msg["role"], "user")
        self.assertTrue(user_msg["images"][0].startswith("iVBORw0KGgo"))
        self.assertNotIn("data:image", user_msg["images"][0])

        # Verify parsed action
        parsed = parse_astra_action(response)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["action"], "click")
        self.assertEqual(parsed["x"], 450)
        self.assertEqual(parsed["y"], 250)

    @patch("requests.post")
    def test_query_ollama_vision_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused on port 11434")

        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            with self.assertRaises(ConnectionError) as ctx:
                query_ollama_vision(
                    prompt="Click Calculator",
                    base64_image_url="abc123==",
                    host="http://localhost:11434",
                )

        # Verify the required exact terminal warning is printed
        printed = stdout_capture.getvalue()
        self.assertIn("[!] Local Ollama server not found on port 11434. Please start Ollama.", printed)
        self.assertIn("Local Ollama server not found on port 11434", str(ctx.exception))

    @patch("tools.orchestrator.query_ollama_vision")
    def test_astra_client_routes_to_ollama_when_no_api_key(self, mock_ollama):
        mock_ollama.return_value = '{"action": "press_key", "key": "enter"}'

        # Client with no api_key and no api_runner
        client = AstraVisionClient(
            model="llama3.2-vision",
            api_key="",
            api_runner=None,
        )

        res = client.query_action(
            query="Press enter to search",
            base64_image_url="xyz==",
        )

        self.assertEqual(res, '{"action": "press_key", "key": "enter"}')
        mock_ollama.assert_called_once()
        kwargs = mock_ollama.call_args[1]
        self.assertEqual(kwargs["prompt"], "Press enter to search")
        self.assertEqual(kwargs["base64_image_url"], "xyz==")


if __name__ == "__main__":
    unittest.main()
