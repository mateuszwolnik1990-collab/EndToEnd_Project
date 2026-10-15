"""Testy dymne dashboardu Streamlit (render, obecność przycisków Start/Stop/Zatwierdź, zakładki)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_dashboard_renders_without_exceptions():
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert any("System Wieloagentowy" in t.value for t in at.title)


def test_dashboard_exposes_required_controls():
    at = _run_app()
    labels = [b.label for b in at.button]
    assert "▶️ Start" in labels
    assert "⏹️ Stop" in labels
    assert "✅ Zatwierdź pomysł" in labels
    assert "🏗️ Zbuduj MVP" in labels


def test_dashboard_has_five_tabs_including_logs_and_ranking():
    at = _run_app()
    assert len(at.tabs) == 5
    # Zakładki: logi na żywo, ranking, raport, MVP, historia.
    assert len(at.selectbox) >= 1


def test_dashboard_offers_offline_simulator_toggle():
    at = _run_app()
    assert len(at.checkbox) >= 1
    assert any("symulator" in cb.label.lower() for cb in at.checkbox)
