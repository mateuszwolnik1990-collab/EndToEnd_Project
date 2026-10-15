"""Pakiet zadań CrewAI (Faza 1: odkrywanie, Faza 2: MVP)."""
from app_discovery_system.tasks.discovery_tasks import build_discovery_tasks
from app_discovery_system.tasks.mvp_tasks import build_mvp_tasks

__all__ = ["build_discovery_tasks", "build_mvp_tasks"]
