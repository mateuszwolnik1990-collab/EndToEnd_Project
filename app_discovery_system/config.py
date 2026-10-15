"""Konfiguracja lokalnego systemu wieloagentowego (vLLM, limity, ścieżki, bezpieczeństwo)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "database" / "app_discovery.db"
MVP_WORKSPACE_DIR = BASE_DIR / "workspace_mvp"


@dataclass
class VLLMConfig:
    """Konfiguracja połączenia z lokalnym serwerem vLLM (OpenAI-compatible API) i limitów."""

    base_url: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.environ.get("VLLM_API_KEY", "EMPTY")
    model_name: str = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
    request_timeout_seconds: float = float(os.environ.get("VLLM_TIMEOUT_SECONDS", "30.0"))
    max_calls_per_campaign: int = int(os.environ.get("VLLM_MAX_CALLS_PER_CAMPAIGN", "50"))
    max_calls_per_minute: int = int(os.environ.get("VLLM_MAX_RPM", "30"))
    max_retries: int = int(os.environ.get("VLLM_MAX_RETRIES", "2"))
    retry_backoff_seconds: float = float(os.environ.get("VLLM_RETRY_BACKOFF", "0.5"))
    campaign_timeout_seconds: float = float(os.environ.get("CAMPAIGN_TIMEOUT_SECONDS", "300.0"))
    temperature: float = 0.2
    max_tokens: int = 2048
    fallback_to_local_simulator: bool = (
        os.environ.get("VLLM_FALLBACK_SIMULATOR", "true").lower() in ("1", "true", "yes")
    )


@dataclass
class SafetyConfig:
    """Bramki bezpieczeństwa (Human-in-the-Loop, zakaz publikacji/zakupów/wykonywania kodu)."""

    allow_external_publishing: bool = False
    allow_purchases: bool = False
    allow_sending_messages: bool = False
    allow_unverified_code_execution: bool = False
    require_user_approval_for_mvp: bool = True


DEFAULT_VLLM_CONFIG = VLLMConfig()
DEFAULT_SAFETY_CONFIG = SafetyConfig()
