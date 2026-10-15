"""Testy przepływu kampanii: 10 pomysłów, Top 3, ocena 0-100, kompletność raportu, MVP."""

from __future__ import annotations

import pytest

from app_discovery_system.config import VLLMConfig
from app_discovery_system.models.schemas import AppCategory, CampaignReport
from app_discovery_system.orchestrator.campaign_runner import (
    _execute_discovery,
    _execute_mvp_build,
    parse_ideas_payload,
)


@pytest.fixture()
def patched_db(temp_db, monkeypatch):
    """Podmienia globalny DatabaseManager w orchestratorze na izolowaną bazę testową."""
    import app_discovery_system.orchestrator.campaign_runner as runner

    monkeypatch.setattr(runner, "DatabaseManager", lambda *a, **k: temp_db)
    return temp_db


def test_parse_ideas_payload_returns_ten_ideas_with_scores():
    from app_discovery_system.llm.local_simulator import LocalSimulatorClient

    raw = LocalSimulatorClient(config=VLLMConfig())._ideas_json()  # noqa: SLF001
    ideas = parse_ideas_payload(raw)
    assert len(ideas) == 10
    for idea in ideas:
        assert 0 <= idea.score_total <= 100
        assert idea.category in list(AppCategory)
        assert idea.claims, "Każdy pomysł musi mieć co najmniej jedno twierdzenie rynkowe."


def test_discovery_campaign_produces_top3_and_full_report(patched_db):
    db = patched_db
    campaign_id = db.create_campaign("Test kampanii")
    result = _execute_discovery(campaign_id, VLLMConfig(), use_simulator=True)

    assert result["status"] == "WAITING_APPROVAL"
    campaign = db.get_campaign(campaign_id)
    assert campaign["status"] == "WAITING_APPROVAL"

    ideas = db.get_ideas_for_campaign(campaign_id)
    assert len(ideas) == 10
    assert {i["category"] for i in ideas} >= {"Mobile", "Web", "B2B"}

    top3 = [i for i in ideas if i["is_top3"]]
    assert len(top3) == 3

    for idea in top3:
        assert idea["competitors_analysis"], "Brak analizy konkurencji"
        assert idea["pricing_analysis"], "Brak analizy cen"
        assert idea["build_difficulty"], "Brak oceny trudności budowy"
        assert idea["sales_channels"], "Brak kanałów sprzedaży"
        assert idea["recommendation"], "Brak rekomendacji"

    report = campaign["final_report_md"]
    for section in ["Konkurencja", "Ceny i monetyzacja", "Trudność budowy", "Kanały sprzedaży", "Rekomendacja"]:
        assert section in report, f"Raport nie zawiera sekcji: {section}"
    assert "HIPOTEZA" in report or "ŹRÓDŁO" in report


def test_ranking_is_sorted_descending(patched_db):
    db = patched_db
    campaign_id = db.create_campaign("Ranking")
    _execute_discovery(campaign_id, VLLMConfig(), use_simulator=True)
    scores = [i["score_total"] for i in db.get_ideas_for_campaign(campaign_id)]
    assert scores == sorted(scores, reverse=True)


def test_campaign_report_schema_validates_categories(patched_db):
    from app_discovery_system.llm.local_simulator import LocalSimulatorClient

    ideas = parse_ideas_payload(LocalSimulatorClient(config=VLLMConfig())._ideas_json())  # noqa: SLF001
    report = CampaignReport(
        campaign_name="X",
        ideas=ideas,
        top3_titles=[i.title for i in ideas[:3]],
        executive_summary="s",
        final_recommendation="r",
    )
    assert len(report.ideas) == 10


def test_mvp_build_requires_approval_then_completes(patched_db):
    db = patched_db
    campaign_id = db.create_campaign("MVP")
    _execute_discovery(campaign_id, VLLMConfig(), use_simulator=True)
    top = [i for i in db.get_ideas_for_campaign(campaign_id) if i["is_top3"]][0]

    blocked = _execute_mvp_build(campaign_id, top["id"], VLLMConfig(), use_simulator=True)
    assert blocked["status"] == "BLOCKED"
    assert db.get_mvp_artifacts(idea_id=top["id"]) == []

    db.approve_idea_for_mvp(top["id"])
    done = _execute_mvp_build(campaign_id, top["id"], VLLMConfig(), use_simulator=True)
    assert done["status"] == "COMPLETED"
    assert done["files_written"] > 0
    assert done["code_safe"] is True

    artifacts = db.get_mvp_artifacts(idea_id=top["id"])
    assert len(artifacts) == 1
    assert artifacts[0]["generated_files"]
    assert "Audyt statyczny" in artifacts[0]["static_security_audit"]
    assert db.get_campaign(campaign_id)["status"] == "COMPLETED"
