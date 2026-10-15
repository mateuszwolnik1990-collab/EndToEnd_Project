"""Test integracyjny: pełna kampania przez prawdziwy klient HTTP vLLM (lokalny serwer atrapy).

Weryfikuje, że system działa przez protokół OpenAI-compatible (``/v1/models``,
``/v1/chat/completions``), a nie tylko przez symulator w pamięci.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app_discovery_system.config import VLLMConfig
from app_discovery_system.llm.local_simulator import LocalSimulatorClient
from app_discovery_system.llm.vllm_client import VLLMClient

_SIM = LocalSimulatorClient(config=VLLMConfig())


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    """Minimalny serwer zgodny z OpenAI API; odpowiedzi generuje lokalny symulator."""

    def log_message(self, *args) -> None:  # noqa: D102 - wyciszenie logów HTTP
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        content = _SIM.chat(messages=payload.get("messages", []))
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


@pytest.fixture()
def fake_vllm_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_healthcheck_over_real_http(fake_vllm_server):
    cfg = VLLMConfig(base_url=fake_vllm_server, model_name="fake-model")
    client = VLLMClient(cfg)
    assert client.healthcheck() is True
    assert client.chat([{"role": "user", "content": "[TASK:IDEAS]"}]).startswith("Final Answer:")


def test_full_discovery_campaign_over_http(fake_vllm_server, temp_db, monkeypatch):
    import app_discovery_system.orchestrator.campaign_runner as runner

    monkeypatch.setattr(runner, "DatabaseManager", lambda *a, **k: temp_db)
    cfg = VLLMConfig(base_url=fake_vllm_server, model_name="fake-model", max_calls_per_campaign=50)

    campaign_id = temp_db.create_campaign("HTTP campaign")
    result = runner._execute_discovery(campaign_id, cfg, use_simulator=False)  # noqa: SLF001

    assert result["status"] == "WAITING_APPROVAL"
    ideas = temp_db.get_ideas_for_campaign(campaign_id)
    assert len(ideas) == 10
    assert len([i for i in ideas if i["is_top3"]]) == 3
    assert temp_db.get_campaign(campaign_id)["vllm_calls_count"] == 5
