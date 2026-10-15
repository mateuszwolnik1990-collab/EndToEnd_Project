"""Testy odporności klienta vLLM: limity wywołań, timeouty, retry, circuit breaker, błędy połączenia."""

from __future__ import annotations

import httpx
import pytest

from app_discovery_system.config import VLLMConfig
from app_discovery_system.llm.vllm_client import (
    CallLimitExceededError,
    CampaignStoppedError,
    VLLMClient,
    VLLMConnectionError,
)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": "OK"}}]},
    )


def _make_client(handler, **overrides) -> VLLMClient:
    cfg = VLLMConfig(
        base_url="http://vllm.test/v1",
        model_name="test-model",
        request_timeout_seconds=0.5,
        max_calls_per_campaign=overrides.pop("max_calls_per_campaign", 10),
        max_calls_per_minute=overrides.pop("max_calls_per_minute", 100),
        max_retries=overrides.pop("max_retries", 1),
        retry_backoff_seconds=0.0,
        **overrides,
    )
    return VLLMClient(cfg, transport=httpx.MockTransport(handler))


def test_successful_call_increments_counter():
    client = _make_client(lambda request: _ok_response())
    assert client.chat([{"role": "user", "content": "hi"}]) == "OK"
    assert client.calls_made == 1


def test_call_limit_is_enforced():
    client = _make_client(lambda request: _ok_response(), max_calls_per_campaign=2)
    client.chat([{"role": "user", "content": "1"}])
    client.chat([{"role": "user", "content": "2"}])
    with pytest.raises(CallLimitExceededError):
        client.chat([{"role": "user", "content": "3"}])


def test_timeout_is_retried_then_raises_connection_error():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectTimeout("timeout")

    client = _make_client(handler, max_retries=1)
    with pytest.raises(VLLMConnectionError):
        client.chat([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 2  # 1 próba + 1 retry


def test_connection_refused_raises_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(handler, max_retries=0)
    with pytest.raises(VLLMConnectionError):
        client.chat([{"role": "user", "content": "hi"}])


def test_server_error_503_is_retried():
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _ok_response()

    client = _make_client(handler, max_retries=2)
    assert client.chat([{"role": "user", "content": "hi"}]) == "OK"
    assert state["n"] == 2


def test_circuit_breaker_opens_after_repeated_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _make_client(handler, max_retries=1)
    with pytest.raises(VLLMConnectionError):
        client.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(VLLMConnectionError):
        client.chat([{"role": "user", "content": "hi again"}])


def test_stop_flag_aborts_call():
    client = _make_client(lambda request: _ok_response(), max_retries=0)
    client._stop_check = lambda: True  # noqa: SLF001
    with pytest.raises(CampaignStoppedError):
        client.chat([{"role": "user", "content": "hi"}])


def test_healthcheck_detects_unavailable_server():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _make_client(handler)
    assert client.healthcheck() is False
