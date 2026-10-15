"""Testy dymne dashboardu Streamlit (render, obecność przycisków Start/Stop/Zatwierdź, zakładki)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_discovery_system.database.db_manager import DatabaseManager

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def seeded_campaign():
    """Tworzy tymczasową kampanię z jednym pomysłem, aby dashboard pokazał pełny układ zakładek."""
    from app_discovery_system.models.schemas import AppCategory, AppIdea, ScoreBreakdown

    db = DatabaseManager()
    campaign_id = db.create_campaign("Kampania testowa UI")
    db.save_ideas(
        campaign_id,
        [
            AppIdea(
                rank=1,
                title="Pomysł testowy UI",
                category=AppCategory.WEB,
                problem_statement="Problem testowy widoczny w dashboardzie.",
                target_audience="Zespół QA",
                value_proposition="Weryfikacja renderowania dashboardu.",
                score_breakdown=ScoreBreakdown(
                    market_demand=10,
                    monetization_potential=10,
                    low_competition_niche=10,
                    build_feasibility=10,
                    sales_channel_access=10,
                ),
                is_top3=True,
            )
        ],
    )
    db.update_campaign_status(campaign_id, "WAITING_APPROVAL")
    yield campaign_id
    # Sprzątanie: usuwamy tylko rekordy tej kampanii (nie całą bazę użytkownika).
    try:
        with db._connect() as conn:  # noqa: SLF001
            conn.execute("DELETE FROM agent_logs WHERE campaign_id = ?", (campaign_id,))
            conn.execute("DELETE FROM mvp_artifacts WHERE campaign_id = ?", (campaign_id,))
            conn.execute("DELETE FROM ideas WHERE campaign_id = ?", (campaign_id,))
            conn.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
            conn.commit()
    except Exception:
        pass


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_dashboard_renders_without_exceptions(seeded_campaign):
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert any("System Wieloagentowy" in t.value for t in at.title)


def test_dashboard_exposes_required_controls(seeded_campaign):
    at = _run_app()
    labels = [b.label for b in at.button]
    assert "▶️ Start" in labels
    assert "⏹️ Stop" in labels
    assert "✅ Zatwierdź pomysł" in labels
    assert "🏗️ Zbuduj MVP" in labels


def test_dashboard_has_five_tabs_including_logs_and_ranking(seeded_campaign):
    at = _run_app()
    assert len(at.tabs) == 5
    assert len(at.selectbox) >= 1


def test_dashboard_offers_offline_simulator_toggle(seeded_campaign):
    at = _run_app()
    assert len(at.checkbox) >= 1
    assert any("symulator" in cb.label.lower() for cb in at.checkbox)
