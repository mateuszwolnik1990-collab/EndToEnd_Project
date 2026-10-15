"""Zadania CrewAI dla Fazy 2 (Specyfikacja MVP i moduł tworzenia aplikacji).

Uruchamiane WYŁĄCZNIE po zatwierdzeniu pomysłu przez użytkownika (bramka bezpieczeństwa).
"""

from __future__ import annotations

from typing import List

from crewai import Agent, Task

from app_discovery_system.agents.crew_agents import NO_SIDE_EFFECTS

MVP_SCHEMA_HINT = """
Zwróć WYŁĄCZNIE poprawny JSON (bez komentarzy, bez markdown) w formacie:
{
  "architecture_overview": "opis architektury MVP",
  "tech_stack": ["technologia 1", "technologia 2"],
  "core_features": ["funkcja 1", "funkcja 2"],
  "database_schema_sql": "CREATE TABLE ...",
  "api_endpoints": ["GET /api/v1/...", "POST /api/v1/..."],
  "generated_files": {
    "README.md": "treść pliku",
    "app/main.py": "kod źródłowy",
    "requirements.txt": "zależności"
  }
}
"""

QA_SCHEMA_HINT = """
Zwróć WYŁĄCZNIE poprawny JSON (bez komentarzy, bez markdown) w formacie:
{
  "qa_test_plan": "plan testów akceptacyjnych",
  "static_security_audit": "wynik statycznego audytu bezpieczeństwa",
  "safe_run_instructions": "instrukcja bezpiecznego uruchomienia przez człowieka"
}
"""


def build_mvp_tasks(
    ceo: Agent,
    developer: Agent,
    tester: Agent,
    idea_title: str,
    idea_category: str,
    idea_context: str,
) -> List[Task]:
    """Buduje sekwencję zadań Crew 2 (specyfikacja MVP + audyt QA)."""
    spec_task = Task(
        description=(
            f"[TASK:MVP]\n"
            f"Zatwierdzony przez użytkownika pomysł: '{idea_title}' (kategoria: {idea_category}).\n"
            f"Kontekst analizy rynkowej:\n{idea_context}\n\n"
            "Przygotuj specyfikację techniczną MVP oraz kompletny kod źródłowy szkieletu aplikacji. "
            "Kod zostanie zapisany lokalnie do przeglądu i NIE będzie automatycznie uruchamiany. "
            "Używaj wyłącznie bibliotek standardowych i popularnych, darmowych frameworków. "
            f"{NO_SIDE_EFFECTS}\n{MVP_SCHEMA_HINT}"
        ),
        expected_output="Poprawny JSON ze specyfikacją MVP, schematem bazy, endpointami i plikami kodu.",
        agent=developer,
    )

    qa_task = Task(
        description=(
            f"[TASK:QA]\n"
            f"Dla wygenerowanego MVP '{idea_title}' przygotuj plan testów akceptacyjnych oraz "
            "statyczny audyt bezpieczeństwa kodu (bez jego uruchamiania). Wskaż, jak bezpiecznie "
            "uruchomić aplikację ręcznie. "
            f"{NO_SIDE_EFFECTS}\n{QA_SCHEMA_HINT}"
        ),
        expected_output="Poprawny JSON z planem testów, audytem bezpieczeństwa i instrukcją uruchomienia.",
        agent=tester,
        context=[spec_task],
    )

    review_task = Task(
        description=(
            "Zweryfikuj spójność specyfikacji MVP i planu testów. Potwierdź zakres MVP oraz wskaż "
            "ryzyka wdrożeniowe. Zwróć zwięzłe podsumowanie w formie tekstu (max 200 słów). "
            f"{NO_SIDE_EFFECTS}"
        ),
        expected_output="Krótkie podsumowanie zatwierdzenia zakresu MVP i ryzyk.",
        agent=ceo,
        context=[spec_task, qa_task],
    )

    return [spec_task, qa_task, review_task]
