"""Klient lokalnego serwera vLLM (OpenAI-compatible) z limitami wywołań, timeoutami i obsługą błędów.

Zawiera:
- ``VLLMClient``: niskopoziomowy klient HTTP (limity RPM, limit wywołań na kampanię, retry, circuit breaker).
- ``CrewAIVLLM``: adapter ``crewai.llms.base_llm.BaseLLM`` używany przez agentów CrewAI.
- ``extract_json`` / ``repair_json``: prosty parser naprawiający ucięte lub zabrudzone JSON-y z małych modeli.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from app_discovery_system.config import VLLMConfig

try:  # CrewAI jest opcjonalne na poziomie importu, aby testy logiki JSON działały bez niego.
    from crewai.llms.base_llm import BaseLLM
except Exception:  # pragma: no cover - awaryjna ścieżka bez CrewAI
    BaseLLM = object  # type: ignore[assignment, misc]


class CallLimitExceededError(RuntimeError):
    """Przekroczono limit wywołań LLM na kampanię."""


class VLLMConnectionError(RuntimeError):
    """Nie udało się połączyć z lokalnym serwerem vLLM."""


class CampaignStoppedError(RuntimeError):
    """Użytkownik zatrzymał kampanię (przycisk Stop)."""


class JSONRepairError(ValueError):
    """Nie udało się naprawić ani sparsować JSON-a zwróconego przez model."""


def repair_json(raw: str) -> str:
    """Prosty naprawiacz JSON: usuwa ogrodzenia markdown, obcina ogonki i domyka nawiasy.

    Małe modele lokalne (7B-14B) często zwracają JSON ucięty przez limit tokenów lub
    owinięty w ```json ... ```. Ta funkcja odzyskuje strukturę bez zewnętrznych usług.
    """
    if raw is None:
        raise JSONRepairError("Pusty tekst wejściowy.")

    text = raw.strip()

    # 1. Usuń ogrodzenia markdown.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # 2. Wytnij najbardziej prawdopodobny fragment JSON (od pierwszego { lub [).
    start_positions = [p for p in (text.find("{"), text.find("[")) if p != -1]
    if start_positions:
        text = text[min(start_positions):]

    # 3. Usuń ewentualne śmieci po ostatnim zamykającym nawiasie.
    last_brace = max(text.rfind("}"), text.rfind("]"))
    if last_brace != -1:
        text = text[: last_brace + 1]

    # 4. Usuń wiszące przecinki przed zamknięciem.
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def _close_open_brackets(text: str) -> str:
    """Domyka niezbalansowane nawiasy klamrowe/kwadratowe (typowe dla uciętych odpowiedzi)."""
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    # Zamknij otwarty string.
    if in_string:
        text += '"'
    # Zamknij nawiasy w odwrotnej kolejności.
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


def extract_json(raw: str) -> Any:
    """Wyciąga i parsuje JSON z odpowiedzi modelu, naprawiając typowe błędy małych LLM-ów."""
    candidate = repair_json(raw)
    for attempt in (candidate, _close_open_brackets(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            # Ostatnia próba: usuń niedomknięte pary klucz: wartość na końcu.
            trimmed = re.sub(r',\s*"[^"]*"\s*:\s*[^,}\]]*$', "", attempt)
            trimmed = re.sub(r",\s*$", "", trimmed)
            try:
                return json.loads(_close_open_brackets(trimmed))
            except json.JSONDecodeError:
                continue
    raise JSONRepairError(f"Nie udało się sparsować JSON-a z odpowiedzi modelu: {raw[:400]}")


class VLLMClient:
    """Klient HTTP do lokalnego vLLM z limitami, timeoutami, retry i circuit breakerem.

    Bezpieczny wątkowo: każda instancja utrzymuje własne liczniki, a wywołania są serializowane
    przez ``threading.Lock`` na potrzeby egzekwowania limitu RPM.
    """

    def __init__(
        self,
        config: VLLMConfig,
        on_call: Optional[Callable[[], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.config = config
        self._on_call = on_call
        self._stop_check = stop_check
        self._transport = transport
        self._call_timestamps: List[float] = []
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    def healthcheck(self, timeout: float = 3.0) -> bool:
        """Sprawdza, czy serwer vLLM odpowiada. Nie liczy się do limitu wywołań."""
        url = f"{self.config.base_url.rstrip('/')}/models"
        try:
            with httpx.Client(timeout=timeout, transport=self._transport) as client:
                resp = client.get(url, headers={"Authorization": f"Bearer {self.config.api_key}"})
            return resp.status_code < 500
        except Exception:
            return False

    def _enforce_limits(self) -> None:
        if self._stop_check and self._stop_check():
            raise CampaignStoppedError("Kampania została zatrzymana przez użytkownika.")
        if self._calls_made >= self.config.max_calls_per_campaign:
            raise CallLimitExceededError(
                f"Przekroczono limit {self.config.max_calls_per_campaign} wywołań LLM na kampanię."
            )
        now = time.monotonic()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60.0]
        if len(self._call_timestamps) >= self.config.max_calls_per_minute:
            sleep_for = 60.0 - (now - self._call_timestamps[0]) + 0.05
            time.sleep(max(0.0, sleep_for))

    def _register_call(self) -> None:
        self._calls_made += 1
        self._call_timestamps.append(time.monotonic())
        if self._on_call:
            self._on_call()

    def _check_circuit(self) -> None:
        if time.monotonic() < self._circuit_open_until:
            remaining = self._circuit_open_until - time.monotonic()
            raise VLLMConnectionError(
                f"Circuit breaker otwarty po serii błędów połączenia. Ponów próbę za {remaining:.1f}s."
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.max_retries + 1:
            self._circuit_open_until = time.monotonic() + 30.0

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Wykonuje zapytanie do ``/chat/completions`` z retry, timeoutem i obsługą błędów."""
        self._check_circuit()
        self._enforce_limits()

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.config.request_timeout_seconds, transport=self._transport
                ) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code in (429, 502, 503, 504):
                    raise VLLMConnectionError(f"Serwer vLLM zwrócił {resp.status_code}.")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                self._register_call()
                self._record_success()
                return content or ""
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                VLLMConnectionError,
                httpx.HTTPStatusError,
                KeyError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                    continue
        self._record_failure()
        raise VLLMConnectionError(
            f"Nie udało się uzyskać odpowiedzi z vLLM ({self.config.base_url}) po "
            f"{self.config.max_retries + 1} próbach: {last_error}"
        )



class CrewAIVLLM(BaseLLM):
    """Adapter ``crewai.llms.base_llm.BaseLLM`` kierujący zapytania do lokalnego serwera vLLM.

    Używa trybu ReAct (tekstowy ``Action:``/``Action Input:``) zamiast natywnego function-calling,
    ponieważ lokalne modele 7B-14B są znacznie bardziej niezawodne w formacie tekstowym.
    """

    llm_type: str = "vllm_local"

    def __init__(self, client: Optional[VLLMClient] = None, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_client", client)

    @property
    def client(self) -> VLLMClient:
        client = getattr(self, "_client", None)
        if client is None:
            raise VLLMConnectionError("CrewAIVLLM nie ma podpiętego klienta VLLMClient.")
        return client

    def supports_function_calling(self) -> bool:
        """Zwraca False, aby CrewAI korzystał z tekstowego formatu ReAct."""
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def _messages_to_payload(self, messages: Any) -> List[Dict[str, str]]:
        formatted = self._format_messages(messages)
        payload: List[Dict[str, str]] = []
        for msg in formatted:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            payload.append({"role": msg.get("role", "user"), "content": content})
        return payload

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        payload = self._messages_to_payload(messages)
        text = self.client.chat(
            messages=payload,
            temperature=self.temperature,
            max_tokens=int(self.max_tokens) if self.max_tokens else None,
        )
        if self.stop:
            text = self._apply_stop_words(text)
        return text

    async def acall(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        """Asynchroniczna wersja ``call`` (deleguje do synchronicznego klienta HTTP)."""
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )
