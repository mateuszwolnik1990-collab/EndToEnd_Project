"""Zadania CrewAI dla Fazy 1 (Odkrywanie i Analiza).

Przepływ: 10 pomysłów (Mobile/Web/B2B) -> selekcja Top 3 -> analiza konkurencji,
monetyzacji i trudności budowy -> raport końcowy CEO z oceną 0-100.
"""

from __future__ import annotations

from typing import List

from crewai import Agent, Task

from app_discovery_system.agents.crew_agents import NO_SIDE_EFFECTS, SOURCE_DISCIPLINE

IDEAS_SCHEMA_HINT = """
Zwróć WYŁĄCZNIE poprawny JSON (bez komentarzy, bez markdown) w formacie:
{
  "ideas": [
    {
      "title": "krótka nazwa pomysłu",
      "category": "Mobile" | "Web" | "B2B",
      "problem_statement": "konkretny problem klienta",
      "target_audience": "kto dokładnie płaci",
      "value_proposition": "co dostaje klient",
      "score_breakdown": {
        "market_demand": 0-25,
        "monetization_potential": 0-25,
        "low_competition_niche": 0-20,
        "build_feasibility": 0-15,
        "sales_channel_access": 0-15
      },
      "claims": [
        {"claim_text": "twierdzenie", "source_url": "https://... lub null"}
      ]
    }
  ]
}
Lista MUSI mieć dokładnie 10 elementów i MUSI obejmować kategorie Mobile, Web oraz B2B.
"""

ANALYSIS_SCHEMA_HINT = """
Zwróć WYŁĄCZNIE poprawny JSON (bez komentarzy, bez markdown) w formacie:
{
  "analyses": [
    {
      "title": "nazwa pomysłu z Top 3",
      "competitors_analysis": "konkurenci, ich słabości, nasycenie rynku",
      "pricing_analysis": "konkretne modele cenowe i poziomy cen",
      "build_difficulty": "trudność X/10 + uzasadnienie technologiczne",
      "sales_channels": "konkretne kanały sprzedaży i akwizycji",
      "recommendation": "rekomendacja dla tego pomysłu"
    }
  ],
  "executive_summary": "podsumowanie kampanii",
  "final_recommendation": "rekomendacja CEO z uzasadnieniem"
}
Lista "analyses" MUSI mieć dokładnie 3 elementy (Top 3).
"""


def build_discovery_tasks(
    ceo: Agent,
    researcher: Agent,
    competitor: Agent,
    monetization: Agent,
    top_n: int = 3,
) -> List[Task]:
    """Buduje sekwencję zadań Crew 1 (odkrywanie i analiza)."""
    ideas_task = Task(
        description=(
            "[TASK:IDEAS]\n"
            "Wykorzystaj dostępne darmowe narzędzia wyszukiwania, aby zidentyfikować 10 różnych, "
            "realnych problemów rynkowych i przekuć je w pomysły na aplikacje. "
            "Zapewnij różnorodność: aplikacje mobilne, webowe oraz rozwiązania B2B. "
            "NIE narzucaj z góry żadnej niszy — szukaj szeroko.\n"
            f"{SOURCE_DISCIPLINE}\n{NO_SIDE_EFFECTS}\n"
            f"{IDEAS_SCHEMA_HINT}"
        ),
        expected_output="Poprawny JSON z listą dokładnie 10 pomysłów (Mobile, Web, B2B) i ocenami 0-100.",
        agent=researcher,
    )

    top3_task = Task(
        description=(
            "[TASK:TOP3]\n"
            "Na podstawie 10 pomysłów z poprzedniego kroku wybierz 3 najlepsze według sumy punktów "
            "oraz potencjału dochodowego. Zwróć WYŁĄCZNIE poprawny JSON:\n"
            '{"top3_titles": ["pomysł 1", "pomysł 2", "pomysł 3"]}'
        ),
        expected_output="Poprawny JSON z dokładnie 3 tytułami pomysłów.",
        agent=ceo,
        context=[ideas_task],
    )

    competitor_task = Task(
        description=(
            "Dla 3 najlepszych pomysłów zbadaj konkurencję przy pomocy darmowych narzędzi wyszukiwania: "
            "istniejący gracze, ich słabości, nasycenie rynku, bariery wejścia oraz trudność budowy MVP (skala 1-10). "
            f"{SOURCE_DISCIPLINE}\n{NO_SIDE_EFFECTS}"
        ),
        expected_output="Ustrukturyzowana analiza konkurencji i trudności budowy dla każdego z 3 pomysłów.",
        agent=competitor,
        context=[ideas_task, top3_task],
    )

    monetization_task = Task(
        description=(
            "Dla 3 najlepszych pomysłów zaprojektuj monetyzację: model biznesowy (SaaS/freemium/licencja B2B), "
            "konkretne poziomy cenowe, szacunek potencjału przychodowego oraz kanały sprzedaży i akwizycji klientów. "
            f"{SOURCE_DISCIPLINE}\n{NO_SIDE_EFFECTS}"
        ),
        expected_output="Ustrukturyzowana analiza monetyzacji, cen i kanałów sprzedaży dla każdego z 3 pomysłów.",
        agent=monetization,
        context=[ideas_task, top3_task],
    )

    final_task = Task(
        description=(
            "[TASK:ANALYSIS]\n"
            "Połącz analizę konkurencji i monetyzacji dla 3 najlepszych pomysłów. "
            "Dla każdego pomysłu uzupełnij: competitors_analysis, pricing_analysis, build_difficulty, "
            "sales_channels oraz recommendation. Dodaj executive_summary oraz final_recommendation. "
            "Wystaw końcowe oceny 0-100 zgodnie ze schematem.\n"
            f"{SOURCE_DISCIPLINE}\n{NO_SIDE_EFFECTS}\n"
            f"{ANALYSIS_SCHEMA_HINT}"
        ),
        expected_output="Poprawny JSON z analizą 3 najlepszych pomysłów oraz rekomendacją CEO.",
        agent=ceo,
        context=[ideas_task, top3_task, competitor_task, monetization_task],
    )

    return [ideas_task, top3_task, competitor_task, monetization_task, final_task]
