"""Lokalny symulator LLM (offline).

Umożliwia pełny przebieg kampanii oraz testy integracyjne bez uruchomionego serwera vLLM.
Zwraca deterministyczne, poprawne strukturalnie odpowiedzi dopasowane do znaczników zadania
(``[TASK:...]``) umieszczanych w promptach agentów. Nie wykonuje żadnych połączeń sieciowych.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app_discovery_system.config import VLLMConfig

# Katalog 10 pomysłów używany przez symulator (Mobile / Web / B2B, bez narzucania niszy).
_SIMULATED_IDEAS = [
    {
        "title": "ShiftHandover AI",
        "category": "Mobile",
        "problem_statement": "Zespoły zmianowe w gastronomii tracą informacje przy przekazaniu zmiany.",
        "target_audience": "Menedżerowie restauracji i sieci gastronomicznych",
        "value_proposition": "Automatyczne, ustrukturyzowane raporty przekazania zmiany z dyktowania głosowego.",
        "scores": [21, 18, 14, 11, 12],
    },
    {
        "title": "GrantPilot",
        "category": "Web",
        "problem_statement": "Małe NGO tracą czas na ręczne dopasowanie do programów dotacyjnych.",
        "target_audience": "Małe organizacje pozarządowe i fundacje",
        "value_proposition": "Silnik dopasowania profili organizacji do aktualnych naborów grantowych.",
        "scores": [19, 17, 15, 12, 11],
    },
    {
        "title": "ComplianceLens",
        "category": "B2B",
        "problem_statement": "Firmy produkcyjne nie nadążają z audytami zgodności ISO.",
        "target_audience": "Działy jakości w firmach produkcyjnych (MŚP)",
        "value_proposition": "Ciągły monitoring wymagań norm i automatyczne checklisty audytowe.",
        "scores": [22, 21, 13, 10, 13],
    },
    {
        "title": "RepairMatch",
        "category": "Mobile",
        "problem_statement": "Klienci nie wiedzą, gdzie i za ile naprawią sprzęt w okolicy.",
        "target_audience": "Konsumenci poszukujący lokalnych usług naprawczych",
        "value_proposition": "Marketplace z wycenami i porównaniem serwisów naprawczych.",
        "scores": [16, 14, 9, 12, 10],
    },
    {
        "title": "InvoiceRadar",
        "category": "B2B",
        "problem_statement": "Działy księgowe tracą płynność przez opóźnione płatności kontrahentów.",
        "target_audience": "Działy finansowe MŚP",
        "value_proposition": "Predykcja opóźnień płatności i automatyczne przypomnienia windykacyjne.",
        "scores": [20, 22, 12, 9, 14],
    },
    {
        "title": "LessonLoop",
        "category": "Web",
        "problem_statement": "Korepetytorzy nie mają narzędzia do automatycznego utrwalania postępów uczniów.",
        "target_audience": "Niezależni korepetytorzy i małe szkoły językowe",
        "value_proposition": "Panel postępów ucznia z automatycznymi zadaniami powtórkowymi.",
        "scores": [15, 16, 11, 13, 12],
    },
    {
        "title": "FleetSentry",
        "category": "B2B",
        "problem_statement": "Małe firmy transportowe nie przewidują awarii floty przed terminem dostawy.",
        "target_audience": "Małe i średnie firmy transportowe",
        "value_proposition": "Predykcyjne utrzymanie floty na bazie telemetrii i historii serwisowej.",
        "scores": [18, 20, 10, 8, 13],
    },
    {
        "title": "MealPlanner Kids",
        "category": "Mobile",
        "problem_statement": "Rodzice tracą czas na planowanie zbilansowanych posiłków dla dzieci.",
        "target_audience": "Rodzice dzieci w wieku szkolnym",
        "value_proposition": "Automatyczne plany posiłków i listy zakupów dopasowane do preferencji.",
        "scores": [17, 13, 8, 13, 11],
    },
    {
        "title": "ClinicFlow",
        "category": "Web",
        "problem_statement": "Małe kliniki mają wysokie odsetki nieodwołanych wizyt (no-show).",
        "target_audience": "Małe przychodnie i gabinety specjalistyczne",
        "value_proposition": "Automatyczne przypomnienia i inteligentne przeplanowanie wizyt.",
        "scores": [19, 18, 12, 11, 12],
    },
    {
        "title": "VendorTrust",
        "category": "B2B",
        "problem_statement": "Działy zakupów nie potrafią szybko zweryfikować wiarygodności nowych dostawców.",
        "target_audience": "Działy zakupów i compliance w MŚP",
        "value_proposition": "Automatyczny scoring wiarygodności dostawców na bazie danych publicznych.",
        "scores": [20, 19, 14, 9, 14],
    },
]

_SIMULATED_COMPETITORS = (
    "Na rynku obecni są zarówno gracze globalni (np. rozwiązania ERP), jak i niszowe startupy. "
    "Główne słabości: wysoka złożoność wdrożenia, brak lokalnej wersji językowej oraz ceny "
    "niedopasowane do MŚP. Luka: prosty, szybki wdrożeniowo produkt z naciskiem na konkretny workflow."
)
_SIMULATED_PRICING = (
    "Model rekomendowany: SaaS z subskrypcją miesięczną. Tier Basic 49 PLN/użytkownik/mies., "
    "Tier Pro 149 PLN/użytkownik/mies., plan Enterprise wyceniany indywidualnie. "
    "Dla B2B: roczna licencja 6000-24000 PLN w zależności od liczby stanowisk."
)
_SIMULATED_DIFFICULTY = (
    "Trudność budowy MVP: 5/10. Stos: backend Python/FastAPI, baza PostgreSQL, frontend web "
    "(React) lub mobilny (Flutter). Kluczowe ryzyko: integracje z systemami zewnętrznymi."
)
_SIMULATED_CHANNELS = (
    "Kanały sprzedaży: (1) sprzedaż bezpośrednia outbound do MŚP, (2) marketplace'y B2B, "
    "(3) partnerstwa z integratorami, (4) content marketing + SEO, (5) LinkedIn Ads."
)
_SIMULATED_RECOMMENDATION = (
    "Rekomendacja: rozpocząć od wąskiego MVP dla jednego segmentu, pozyskać 5-10 klientów "
    "pilotażowych, zwalidować willingness-to-pay przed skalowaniem zespołu."
)


class LocalSimulatorClient:
    """Symulator klienta vLLM: zwraca deterministyczne odpowiedzi na podstawie znacznika zadania."""

    def __init__(
        self,
        config: Optional[VLLMConfig] = None,
        on_call: Optional[Any] = None,
        stop_check: Optional[Any] = None,
    ) -> None:
        self.config = config or VLLMConfig()
        self._on_call = on_call
        self._stop_check = stop_check
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    def healthcheck(self, timeout: float = 3.0) -> bool:  # noqa: ARG002
        return True

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa: ARG002
        from app_discovery_system.llm.vllm_client import (
            CallLimitExceededError,
            CampaignStoppedError,
        )

        if self._stop_check and self._stop_check():
            raise CampaignStoppedError("Kampania została zatrzymana przez użytkownika.")
        if self._calls_made >= self.config.max_calls_per_campaign:
            raise CallLimitExceededError("Przekroczono limit wywołań LLM na kampanię (symulator).")
        self._calls_made += 1
        if self._on_call:
            self._on_call()

        joined = "\n".join(m.get("content", "") for m in messages)
        if "[TASK:IDEAS]" in joined:
            return self._ideas_json()
        if "[TASK:TOP3]" in joined:
            return self._top3_json()
        if "[TASK:ANALYSIS]" in joined:
            return self._analysis_json()
        if "[TASK:MVP]" in joined:
            return self._mvp_json()
        if "[TASK:QA]" in joined:
            return self._qa_json()
        return "Thought: Brak danych.\nFinal Answer: Analiza niedostępna w symulatorze."

    # -- Generatory odpowiedzi --------------------------------------------
    def _ideas_json(self) -> str:
        ideas = []
        for item in _SIMULATED_IDEAS:
            s = item["scores"]
            ideas.append(
                {
                    "title": item["title"],
                    "category": item["category"],
                    "problem_statement": item["problem_statement"],
                    "target_audience": item["target_audience"],
                    "value_proposition": item["value_proposition"],
                    "score_breakdown": {
                        "market_demand": s[0],
                        "monetization_potential": s[1],
                        "low_competition_niche": s[2],
                        "build_feasibility": s[3],
                        "sales_channel_access": s[4],
                    },
                    "claims": [
                        {
                            "claim_text": item["problem_statement"],
                            "source_url": None,
                            "classification": "HYPOTHESIS",
                        }
                    ],
                }
            )
        return "Final Answer: " + json.dumps({"ideas": ideas}, ensure_ascii=False)

    def _top3_json(self) -> str:
        ranked = sorted(_SIMULATED_IDEAS, key=lambda i: sum(i["scores"]), reverse=True)[:3]
        return "Final Answer: " + json.dumps(
            {"top3_titles": [i["title"] for i in ranked]}, ensure_ascii=False
        )

    def _analysis_json(self) -> str:
        ranked = sorted(_SIMULATED_IDEAS, key=lambda i: sum(i["scores"]), reverse=True)[:3]
        payload = {
            "analyses": [
                {
                    "title": item["title"],
                    "competitors_analysis": _SIMULATED_COMPETITORS,
                    "pricing_analysis": _SIMULATED_PRICING,
                    "build_difficulty": _SIMULATED_DIFFICULTY,
                    "sales_channels": _SIMULATED_CHANNELS,
                    "recommendation": _SIMULATED_RECOMMENDATION,
                }
                for item in ranked
            ],
            "executive_summary": (
                "Kampania odkryła 10 pomysłów w segmentach Mobile, Web i B2B. "
                "Trzy najwyżej ocenione charakteryzują się realnym bólem operacyjnym i gotowością "
                "do płacenia po stronie MŚP."
            ),
            "final_recommendation": (
                "Rekomendacja CEO: rozpocząć budowę MVP dla pomysłu o najwyższej ocenie, "
                "z walidacją cenową na 5-10 klientach pilotażowych."
            ),
        }
        return "Final Answer: " + json.dumps(payload, ensure_ascii=False)

    def _mvp_json(self) -> str:
        payload = {
            "architecture_overview": (
                "Architektura trójwarstwowa: klient (web/mobile) -> API REST (FastAPI) -> "
                "baza PostgreSQL. Uwierzytelnianie JWT, kolejka zadań dla procesów asynchronicznych."
            ),
            "tech_stack": ["Python 3.11", "FastAPI", "PostgreSQL", "React", "Docker"],
            "core_features": [
                "Rejestracja i logowanie użytkowników",
                "Panel główny z kluczowymi metrykami",
                "Podstawowy przepływ domenowy MVP",
                "Eksport danych do CSV",
            ],
            "database_schema_sql": (
                "CREATE TABLE users (id SERIAL PRIMARY KEY, email TEXT UNIQUE NOT NULL);\n"
                "CREATE TABLE items (id SERIAL PRIMARY KEY, user_id INT REFERENCES users(id), "
                "payload JSONB, created_at TIMESTAMP DEFAULT NOW());"
            ),
            "api_endpoints": [
                "POST /api/v1/auth/register",
                "POST /api/v1/auth/login",
                "GET /api/v1/items",
                "POST /api/v1/items",
            ],
            "generated_files": {
                "README.md": "# MVP\n\nInstrukcja uruchomienia krok po kroku.\n",
                "app/main.py": (
                    "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                    "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
                ),
                "app/models.py": (
                    "from pydantic import BaseModel\n\nclass Item(BaseModel):\n"
                    "    id: int | None = None\n    payload: dict = {}\n"
                ),
                "requirements.txt": "fastapi\nuvicorn\npydantic\n",
            },
        }
        return "Final Answer: " + json.dumps(payload, ensure_ascii=False)

    def _qa_json(self) -> str:
        payload = {
            "qa_test_plan": (
                "Plan testów: (1) testy jednostkowe modeli, (2) testy integracyjne endpointów, "
                "(3) testy akceptacyjne ścieżki krytycznej, (4) test obciążeniowy /health."
            ),
            "static_security_audit": (
                "Audyt statyczny: kod zawiera wyłącznie importy standardowe (fastapi, pydantic). "
                "Brak wywołań subprocess/os.system/eval/exec. Brak sekretów w kodzie."
            ),
            "safe_run_instructions": (
                "Aby uruchomić: utwórz wirtualne środowisko, zainstaluj requirements.txt, "
                "a następnie uruchom 'uvicorn app.main:app'. System NIE uruchamia tego kodu automatycznie."
            ),
        }
        return "Final Answer: " + json.dumps(payload, ensure_ascii=False)
