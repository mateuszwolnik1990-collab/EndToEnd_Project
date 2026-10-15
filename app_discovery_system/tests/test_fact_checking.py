"""Testy rygoru informacyjnego: każde twierdzenie musi mieć źródło URL albo być oznaczone [HIPOTEZA]."""

from __future__ import annotations

import pytest

from app_discovery_system.llm.vllm_client import (
    JSONRepairError,
    extract_json,
    repair_json,
)
from app_discovery_system.models.schemas import (
    ClaimClassification,
    MarketClaim,
    is_valid_http_url,
)
from app_discovery_system.tools.verification_tools import validate_claims


def test_valid_url_marks_claim_as_verified_source():
    claim = MarketClaim(claim_text="Rynek SaaS rośnie", source_url="https://example.com/raport")
    assert claim.classification == ClaimClassification.VERIFIED_SOURCE
    assert "[ŹRÓDŁO: https://example.com/raport]" in claim.formatted()
    assert "[HIPOTEZA]" not in claim.formatted()


def test_missing_url_forces_hypothesis_tag():
    claim = MarketClaim(claim_text="Rynek SaaS rośnie")
    assert claim.classification == ClaimClassification.HYPOTHESIS
    assert claim.source_url is None
    assert "[HIPOTEZA]" in claim.formatted()


def test_invalid_url_forces_hypothesis_tag():
    claim = MarketClaim(claim_text="Twierdzenie", source_url="not-a-url")
    assert claim.classification == ClaimClassification.HYPOTHESIS
    assert "[HIPOTEZA]" in claim.formatted()


def test_hypothesis_tag_removed_when_real_source_present():
    claim = MarketClaim(claim_text="Twierdzenie [HIPOTEZA]", source_url="https://news.ycombinator.com/item?id=1")
    assert claim.classification == ClaimClassification.VERIFIED_SOURCE
    assert "[HIPOTEZA]" not in claim.claim_text


def test_validate_claims_normalizes_raw_dicts():
    claims = validate_claims(
        [
            {"claim_text": "Ze źródłem", "source_url": "https://example.com/a"},
            {"claim_text": "Bez źródła"},
            {"claim_text": "Zły url", "url": "ftp://nope"},
            "nie-słownik",
            {"claim_text": ""},
        ]
    )
    assert len(claims) == 3
    assert claims[0].classification == ClaimClassification.VERIFIED_SOURCE
    assert claims[1].classification == ClaimClassification.HYPOTHESIS
    assert claims[2].classification == ClaimClassification.HYPOTHESIS


def test_is_valid_http_url():
    assert is_valid_http_url("https://example.com")
    assert is_valid_http_url("http://sub.example.org/path?q=1")
    assert not is_valid_http_url("example.com")
    assert not is_valid_http_url("javascript:alert(1)")
    assert not is_valid_http_url(None)


def test_repair_json_strips_markdown_fences_and_trailing_commas():
    raw = '```json\n{"a": 1, "b": [1, 2,],}\n```'
    assert extract_json(raw) == {"a": 1, "b": [1, 2]}


def test_repair_json_closes_truncated_object():
    raw = '{"ideas": [{"title": "X", "category": "Web"'
    parsed = extract_json(raw)
    assert parsed["ideas"][0]["title"] == "X"


def test_extract_json_raises_on_garbage():
    with pytest.raises(JSONRepairError):
        extract_json("zupełnie nie JSON")
