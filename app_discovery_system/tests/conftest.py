"""Wspólna konfiguracja testów (ścieżki, izolowana baza danych)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_discovery_system.database.db_manager import DatabaseManager  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path) -> DatabaseManager:
    """Izolowana baza SQLite w katalogu tymczasowym (nie dotyka produkcyjnej bazy)."""
    return DatabaseManager(db_path=tmp_path / "test.db")
