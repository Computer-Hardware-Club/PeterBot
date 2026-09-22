"""Optional real-Hermes compatibility test; stdlib-only, runnable in worker image.

Run this file with the pinned Hermes installed. Ordinary PeterBot tests skip it.
The HTTP fixture only binds loopback and never uses production credentials.
"""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest


@unittest.skipUnless(importlib.util.find_spec("run_agent"), "requires optional pinned Hermes runtime")
class RealHermesLoop(unittest.TestCase):
    def test_broker_and_native_dispatch_then_final_answer(self):
        from peterbot.hermes_worker import run_job
        calls = []
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tool":
                    calls.append(request)
                    response = {"result": 4}
                else:
                    requests.append(request)
                    results = [message for message in request.get("messages", []) if message.get("role") == "tool"]
                    finished = len(results) >= 2
                    message = {"role": "assistant", "content": "The result is 4." if finished else None}
                    if not results:
                        message["tool_calls"] = [{"id": "fixture-call", "type": "function", "function": {
                            "name": "calculate", "arguments": '{"expression":"2+2"}'}}]
                    elif not finished:
                        message["tool_calls"] = [{"id": "fixture-write", "type": "function", "function": {
                            "name": "write_file", "arguments": '{"path":"artifacts/result.txt","content":"4"}'}}]
                    response = {"id": "fixture", "object": "chat.completion", "created": 1,
                                "model": "fixture-qwen", "choices": [{"index": 0, "message": message,
                                "finish_reason": "stop" if finished else "tool_calls"}],
                                "usage": {"prompt_tokens": 500, "completion_tokens": 10, "total_tokens": 510}}
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = run_job({"prompt": "Calculate 2+2 using the calculate tool.", "identity": {"user_id": "fixture"},
                    "tool_service_url": f"http://127.0.0.1:{server.server_port}", "capability_token": "fixture-only",
                    "model": "fixture-qwen", "max_iterations": 5}, workspace=root / "workspace", home=root / "hermes")
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(calls, [{"tool": "calculate", "arguments": {"expression": "2+2"}}])
                self.assertEqual(len(requests), 3)
                self.assertIn("4", result["answer"])
                self.assertEqual((root / "workspace/artifacts/result.txt").read_text(), "4")
                self.assertTrue(all(not request.get("stream") for request in requests))
                self.assertTrue(all(request["chat_template_kwargs"]["enable_thinking"] for request in requests))
                self.assertEqual(result.get("diagnostics"), [
                    {"tool": "calculate", "error_type": "none", "succeeded": True},
                    {"tool": "write_file", "error_type": "none", "succeeded": True}])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
