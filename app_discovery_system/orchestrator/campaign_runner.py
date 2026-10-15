"""Silnik kampanii (orchestrator) — Faza 1 (odkrywanie) i Faza 2 (budowa MVP).

Projekt uwzględnia pięć kluczowych ryzyk technicznych:

1. **Przycisk Stop vs wątki Pythona** — kampania działa w osobnym ``multiprocessing.Process``,
   który można bezpiecznie zakończyć przez ``.terminate()``. Dodatkowo każdy krok CrewAI ma
   ``step_callback`` sprawdzający flagę STOP zapisaną w SQLite i przerywający pracę wyjątkiem.
2. **Pydantic / structured output na lokalnym vLLM** — zamiast jednego gigantycznego schematu
   wymuszamy proste, płaskie schematy JSON dla pojedynczych zadań oraz naprawiamy ucięte JSON-y
   funkcją ``extract_json``.
3. **Konfiguracja LiteLLM/CrewAI** — używamy własnego adaptera ``CrewAIVLLM`` (BaseLLM), więc
   nie ma ryzyka autoryzacji do OpenAI; model nie wymaga prefiksu dostawcy.
4. **Odświeżanie logów w Streamlit** — kampania nie blokuje UI (osobny proces), a dashboard
   odpytuje tabelę ``agent_logs`` w SQLite przez ``@st.fragment(run_every=...)``.
5. **Blokady DuckDuckGo** — throttling 2 s + automatyczny fallback do HackerNews/Reddit.
"""

from __future__ import annotations

import json
import multiprocessing
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app_discovery_system.agents.crew_agents import AgentFactory
from app_discovery_system.config import (
    DEFAULT_SAFETY_CONFIG,
    MVP_WORKSPACE_DIR,
    VLLMConfig,
)
from app_discovery_system.database.db_manager import DatabaseManager
from app_discovery_system.llm.local_simulator import LocalSimulatorClient
from app_discovery_system.llm.vllm_client import (
    CallLimitExceededError,
    CampaignStoppedError,
    CrewAIVLLM,
    VLLMClient,
    VLLMConnectionError,
    extract_json,
)
from app_discovery_system.models.schemas import (
    AppCategory,
    AppIdea,
    ClaimClassification,
    MarketClaim,
    MVPSpecification,
    ScoreBreakdown,
)
from app_discovery_system.tasks.discovery_tasks import build_discovery_tasks
from app_discovery_system.tasks.mvp_tasks import build_mvp_tasks
from app_discovery_system.tools.verification_tools import (
    static_code_audit,
    validate_claims,
    write_mvp_files_safely,
)

AGENT_NAMES = {
    "ceo": "CEO",
    "ceo_top3": "CEO",
    "ceo_final": "CEO",
    "ceo_review": "CEO",
    "researcher": "Badacz Rynku",
    "competitor": "Analityk Konkurencji",
    "monetization": "Analityk Monetyzacji",
    "developer": "Programista",
    "tester": "Tester",
}

CAMPAIGN_TIMEOUT = 600.0  # maksymalny czas kampanii w sekundach


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Budowa LLM (vLLM lub symulator) z hookami Stop/limitu
# ---------------------------------------------------------------------------
def build_llm(
    campaign_id: int,
    config: VLLMConfig,
    use_simulator: bool,
) -> Tuple[Any, DatabaseManager]:
    """Tworzy adapter CrewAI LLM podpięty do vLLM (lub symulatora) z hookami Stop i licznikiem wywołań."""
    db = DatabaseManager()

    def on_call() -> None:
        db.increment_vllm_calls(campaign_id, 1)

    def stop_check() -> bool:
        return db.is_stop_requested(campaign_id)

    if use_simulator:
        client = LocalSimulatorClient(config=config, on_call=on_call, stop_check=stop_check)
    else:
        client = VLLMClient(config=config, on_call=on_call, stop_check=stop_check)

    llm = CrewAIVLLM(
        client=client,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        base_url=config.base_url,
        api_key=config.api_key,
    )
    return llm, db


def _step_callback_factory(campaign_id: int, agent_label: str) -> Any:
    """Tworzy ``step_callback`` CrewAI, który przed każdym krokiem sprawdza flagę STOP w SQLite."""
    db = DatabaseManager()

    def _callback(step_output: Any) -> None:
        if db.is_stop_requested(campaign_id):
            raise CampaignStoppedError(f"Kampania zatrzymana przez użytkownika (krok: {agent_label}).")
        text = getattr(step_output, "output", None) or str(step_output)
        db.log_agent_event(campaign_id, agent_label, "STEP", str(text)[:500])

    return _callback


# ---------------------------------------------------------------------------
# Parsowanie odpowiedzi agentów
# ---------------------------------------------------------------------------
def parse_ideas_payload(raw: str) -> List[AppIdea]:
    """Parsuje JSON z 10 pomysłami zwrócony przez agenta i buduje listę ``AppIdea``."""
    data = extract_json(raw)
    items = data.get("ideas") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError("Agent nie zwrócił listy pomysłów w formacie JSON.")

    ideas: List[AppIdea] = []
    for idx, item in enumerate(items[:10], start=1):
        if not isinstance(item, dict):
            continue
        scores = item.get("score_breakdown") or {}
        category_raw = str(item.get("category", "Web")).strip()
        category = next(
            (c for c in AppCategory if c.value.lower() == category_raw.lower()),
            AppCategory.WEB,
        )
        breakdown = ScoreBreakdown(
            market_demand=max(0, min(25, int(scores.get("market_demand", 15)))),
            monetization_potential=max(0, min(25, int(scores.get("monetization_potential", 15)))),
            low_competition_niche=max(0, min(20, int(scores.get("low_competition_niche", 10)))),
            build_feasibility=max(0, min(15, int(scores.get("build_feasibility", 8)))),
            sales_channel_access=max(0, min(15, int(scores.get("sales_channel_access", 8)))),
        )
        claims = validate_claims(item.get("claims") or [])
        ideas.append(
            AppIdea(
                rank=idx,
                title=str(item.get("title", f"Pomysł {idx}")).strip()[:120],
                category=category,
                problem_statement=str(item.get("problem_statement", "")).strip() or "Brak opisu problemu.",
                target_audience=str(item.get("target_audience", "")).strip() or "Nieokreślona grupa.",
                value_proposition=str(item.get("value_proposition", "")).strip() or "Brak propozycji wartości.",
                score_breakdown=breakdown,
                claims=claims,
            )
        )

    if len(ideas) < 10:
        # Dopełniamy do 10 unikalnymi wariantami, aby zachować kontrakt kampanii.
        base = list(ideas)
        i = len(ideas) + 1
        while len(ideas) < 10 and base:
            src = base[(i - 1) % len(base)]
            ideas.append(
                src.model_copy(
                    update={
                        "title": f"{src.title} (wariant {i})",
                        "rank": i,
                        "claims": [
                            MarketClaim(
                                claim_text=f"Wariant rozszerzający: {src.problem_statement}",
                                source_url=None,
                                classification=ClaimClassification.HYPOTHESIS,
                            )
                        ],
                    },
                    deep=True,
                )
            )
            i += 1

    return _ensure_category_diversity(ideas[:10])


def _ensure_category_diversity(ideas: List[AppIdea]) -> List[AppIdea]:
    """Gwarantuje, że lista pomysłów zawiera co najmniej po jednym Mobile, Web i B2B."""
    required = [AppCategory.MOBILE, AppCategory.WEB, AppCategory.B2B]
    for category in required:
        if not any(idea.category == category for idea in ideas):
            for idx, idea in enumerate(ideas):
                if sum(1 for x in ideas if x.category == idea.category) > 1:
                    ideas[idx] = idea.model_copy(update={"category": category}, deep=True)
                    break
    return ideas




# ---------------------------------------------------------------------------
# Faza 1: Kampania odkrywania (uruchamiana w osobnym procesie)
# ---------------------------------------------------------------------------
def _execute_discovery(campaign_id: int, config: VLLMConfig, use_simulator: bool) -> Dict[str, Any]:
    """Właściwa logika Fazy 1. Uruchamiana w procesie potomnym."""
    db = DatabaseManager()
    try:
        db.update_campaign_status(campaign_id, "RUNNING")
        llm, db = build_llm(campaign_id, config, use_simulator)
        mode = "symulator lokalny" if use_simulator else config.base_url
        db.log_agent_event(campaign_id, "SYSTEM", "INFO", f"Start kampanii odkrywania (tryb: {mode}).")

        factory = AgentFactory(llm=llm, verbose=False)
        ceo, researcher = factory.ceo(), factory.market_researcher()
        competitor, monetization = factory.competitor_analyst(), factory.monetization_analyst()

        tasks = build_discovery_tasks(ceo, researcher, competitor, monetization)
        labels = [
            ("researcher", researcher),
            ("ceo_top3", ceo),
            ("competitor", competitor),
            ("monetization", monetization),
            ("ceo_final", ceo),
        ]

        from crewai import Crew, Process

        outputs: Dict[str, str] = {}
        for task, (key, agent) in zip(tasks, labels):
            if db.is_stop_requested(campaign_id):
                raise CampaignStoppedError("Kampania zatrzymana przez użytkownika.")
            label = AGENT_NAMES.get(key, key)
            db.log_agent_event(campaign_id, label, "RUNNING", f"Zadanie: {task.description[:120]}...")
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                step_callback=_step_callback_factory(campaign_id, label),
            )
            result = crew.kickoff()
            outputs[key] = str(result.raw) if hasattr(result, "raw") else str(result)
            db.log_agent_event(
                campaign_id, label, "DONE", f"Zakończono. Długość odpowiedzi: {len(outputs[key])} znaków."
            )

        return _finalize_discovery(campaign_id, db, outputs)
    except CampaignStoppedError as exc:
        db.update_campaign_status(campaign_id, "STOPPED", error_message=str(exc))
        db.log_agent_event(campaign_id, "SYSTEM", "STOPPED", f"Kampania zatrzymana: {exc}")
        return {"status": "STOPPED", "error": str(exc)}
    except (CallLimitExceededError, VLLMConnectionError) as exc:
        db.update_campaign_status(campaign_id, "ERROR", error_message=str(exc))
        db.log_agent_event(campaign_id, "SYSTEM", "ERROR", f"Błąd LLM/vLLM: {exc}")
        return {"status": "ERROR", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        db.update_campaign_status(campaign_id, "ERROR", error_message=str(exc))
        db.log_agent_event(
            campaign_id, "SYSTEM", "ERROR", f"Nieoczekiwany błąd: {exc}\n{traceback.format_exc()[:1500]}"
        )
        return {"status": "ERROR", "error": str(exc)}



def _finalize_discovery(campaign_id: int, db: DatabaseManager, outputs: Dict[str, str]) -> Dict[str, Any]:
    """Parsuje wyniki agentów, zapisuje pomysły w SQLite i przechodzi w stan oczekiwania na zgodę."""
    ideas = parse_ideas_payload(outputs["researcher"])

    try:
        top3_payload = extract_json(outputs.get("ceo_top3", "{}"))
    except Exception:
        top3_payload = {}
    top3_titles = top3_payload.get("top3_titles", []) if isinstance(top3_payload, dict) else []

    try:
        analysis = extract_json(outputs.get("ceo_final", "{}"))
    except Exception:
        analysis = {}
    if not isinstance(analysis, dict):
        analysis = {}
    analyses_list = analysis.get("analyses", []) or []
    executive_summary = str(analysis.get("executive_summary", "") or "")
    final_recommendation = str(analysis.get("final_recommendation", "") or "")

    ranked = sorted(ideas, key=lambda x: x.score_total, reverse=True)
    top3 = [i for i in ranked if i.title in top3_titles][:3] if top3_titles else ranked[:3]
    if len(top3) < 3:
        top3 = ranked[:3]
    for idea in top3:
        idea.is_top3 = True

    analysis_by_title = {a.get("title"): a for a in analyses_list if isinstance(a, dict)}
    for idea in top3:
        a = analysis_by_title.get(idea.title) or (analyses_list[0] if analyses_list else {})
        idea.competitors_analysis = str(a.get("competitors_analysis", "")) or outputs.get("competitor", "")[:2000]
        idea.pricing_analysis = str(a.get("pricing_analysis", "")) or outputs.get("monetization", "")[:2000]
        idea.build_difficulty = str(a.get("build_difficulty", ""))
        idea.sales_channels = str(a.get("sales_channels", "")) or outputs.get("monetization", "")[:2000]
        idea.recommendation = str(a.get("recommendation", ""))

    db.save_ideas(campaign_id, ideas)
    report_md = _render_report_md(ideas, top3, executive_summary, final_recommendation)
    db.update_campaign_status(
        campaign_id,
        "WAITING_APPROVAL",
        executive_summary=executive_summary,
        final_recommendation=final_recommendation,
        final_report_md=report_md,
    )
    db.log_agent_event(
        campaign_id, "SYSTEM", "DONE", "Faza 1 zakończona. Oczekiwanie na zatwierdzenie użytkownika."
    )
    return {"status": "WAITING_APPROVAL", "ideas": len(ideas), "top3": [i.title for i in top3]}


def _render_report_md(
    ideas: List[AppIdea],
    top3: List[AppIdea],
    executive_summary: str,
    final_recommendation: str,
) -> str:
    """Renderuje raport końcowy kampanii w formacie Markdown."""
    lines: List[str] = ["# Raport Kampanii Odkrywania Aplikacji", ""]
    lines.append(f"Wygenerowano: {_now()}")
    lines.append("")
    if executive_summary:
        lines += ["## Podsumowanie (Executive Summary)", "", executive_summary, ""]
    lines += ["## Ranking wszystkich pomysłów (ocena 0-100)", ""]
    lines += ["| # | Pomysł | Kategoria | Ocena |", "|---|-------|-----------|-------|"]
    for idx, idea in enumerate(sorted(ideas, key=lambda x: x.score_total, reverse=True), start=1):
        lines.append(f"| {idx} | {idea.title} | {idea.category.value} | {idea.score_total}/100 |")
    lines.append("")
    for idea in top3:
        lines += [
            f"## TOP 3: {idea.title} ({idea.category.value}) — {idea.score_total}/100",
            "",
            f"**Problem:** {idea.problem_statement}",
            "",
            f"**Grupa docelowa:** {idea.target_audience}",
            "",
            f"**Propozycja wartości:** {idea.value_proposition}",
            "",
            "### Konkurencja",
            idea.competitors_analysis or "[HIPOTEZA] Brak zweryfikowanych danych.",
            "",
            "### Ceny i monetyzacja",
            idea.pricing_analysis or "[HIPOTEZA] Brak zweryfikowanych danych.",
            "",
            "### Trudność budowy",
            idea.build_difficulty or "[HIPOTEZA] Nieokreślona.",
            "",
            "### Kanały sprzedaży",
            idea.sales_channels or "[HIPOTEZA] Nieokreślone.",
            "",
            "### Rekomendacja",
            idea.recommendation or "[HIPOTEZA] Brak.",
            "",
            "### Twierdzenia rynkowe i źródła",
        ]
        for claim in idea.claims:
            lines.append(f"- {claim.formatted()}")
        lines.append("")
    if final_recommendation:
        lines += ["## Rekomendacja końcowa CEO", "", final_recommendation, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Faza 2: Specyfikacja MVP i moduł tworzenia aplikacji (po zatwierdzeniu)
# ---------------------------------------------------------------------------
def _execute_mvp_build(
    campaign_id: int, idea_id: int, config: VLLMConfig, use_simulator: bool
) -> Dict[str, Any]:
    """Właściwa logika Fazy 2 (uruchamiana dopiero po zatwierdzeniu pomysłu przez użytkownika)."""
    db = DatabaseManager()
    try:
        idea = db.get_idea_by_id(idea_id)
        if not idea:
            raise ValueError(f"Nie znaleziono pomysłu o ID {idea_id}.")
        if not idea.get("approved_for_mvp"):
            raise PermissionError(
                "Bramka bezpieczeństwa: pomysł nie został zatwierdzony przez użytkownika. "
                "Budowa MVP zablokowana."
            )

        db.update_campaign_status(campaign_id, "BUILDING_MVP")
        llm, db = build_llm(campaign_id, config, use_simulator)
        mode = "symulator lokalny" if use_simulator else config.base_url
        db.log_agent_event(campaign_id, "SYSTEM", "INFO", f"Start budowy MVP (tryb: {mode}).")

        factory = AgentFactory(llm=llm, verbose=False)
        ceo, developer, tester = factory.ceo(), factory.developer(), factory.tester()

        idea_context = (
            f"Problem: {idea['problem_statement']}\n"
            f"Grupa docelowa: {idea['target_audience']}\n"
            f"Propozycja wartości: {idea['value_proposition']}\n"
            f"Konkurencja: {idea.get('competitors_analysis', '')[:600]}\n"
            f"Monetyzacja: {idea.get('pricing_analysis', '')[:600]}"
        )
        tasks = build_mvp_tasks(
            ceo, developer, tester, idea["title"], idea["category"], idea_context
        )
        labels = [("developer", developer), ("tester", tester), ("ceo_review", ceo)]

        from crewai import Crew, Process

        outputs: Dict[str, str] = {}
        for task, (key, agent) in zip(tasks, labels):
            if db.is_stop_requested(campaign_id):
                raise CampaignStoppedError("Budowa MVP zatrzymana przez użytkownika.")
            label = AGENT_NAMES.get(key, key)
            db.log_agent_event(campaign_id, label, "RUNNING", f"Zadanie: {task.description[:120]}...")
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                step_callback=_step_callback_factory(campaign_id, label),
            )
            result = crew.kickoff()
            outputs[key] = str(result.raw) if hasattr(result, "raw") else str(result)
            db.log_agent_event(campaign_id, label, "DONE", "Zadanie zakończone.")

        return _finalize_mvp(campaign_id, idea_id, idea, db, outputs)
    except CampaignStoppedError as exc:
        db.update_campaign_status(campaign_id, "STOPPED", error_message=str(exc))
        db.log_agent_event(campaign_id, "SYSTEM", "STOPPED", str(exc))
        return {"status": "STOPPED", "error": str(exc)}
    except PermissionError as exc:
        db.update_campaign_status(campaign_id, "ERROR", error_message=str(exc))
        db.log_agent_event(campaign_id, "SYSTEM", "BLOCKED", str(exc))
        return {"status": "BLOCKED", "error": str(exc)}
    except (CallLimitExceededError, VLLMConnectionError) as exc:
        db.update_campaign_status(campaign_id, "ERROR", error_message=str(exc))
        db.log_agent_event(campaign_id, "SYSTEM", "ERROR", f"Błąd LLM/vLLM: {exc}")
        return {"status": "ERROR", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        db.update_campaign_status(campaign_id, "ERROR", error_message=str(exc))
        db.log_agent_event(
            campaign_id, "SYSTEM", "ERROR", f"Nieoczekiwany błąd: {exc}\n{traceback.format_exc()[:1500]}"
        )
        return {"status": "ERROR", "error": str(exc)}




def _finalize_mvp(
    campaign_id: int,
    idea_id: int,
    idea: Dict[str, Any],
    db: DatabaseManager,
    outputs: Dict[str, str],
) -> Dict[str, Any]:
    """Buduje ``MVPSpecification``, zapisuje pliki w sandboxie (bez uruchamiania) i wykonuje audyt statyczny."""
    try:
        spec_data = extract_json(outputs.get("developer", "{}"))
    except Exception:
        spec_data = {}
    if not isinstance(spec_data, dict):
        spec_data = {}

    try:
        qa_data = extract_json(outputs.get("tester", "{}"))
    except Exception:
        qa_data = {}
    if not isinstance(qa_data, dict):
        qa_data = {}

    generated_files = spec_data.get("generated_files") or {}
    if not isinstance(generated_files, dict):
        generated_files = {}

    # Zapis plików wyłącznie w izolowanym sandboxie; brak jakiegokolwiek uruchamiania.
    sandbox, written = write_mvp_files_safely(idea["title"], generated_files, MVP_WORKSPACE_DIR)

    is_safe, audit_report = static_code_audit(generated_files)
    qa_audit = str(qa_data.get("static_security_audit", "") or "")
    combined_audit = audit_report + (f"\n\nUwagi agenta Testera:\n{qa_audit}" if qa_audit else "")
    if not is_safe:
        combined_audit += "\n\nUWAGA: wykryto potencjalnie niebezpieczny kod — uruchomienie wymaga ręcznej weryfikacji."

    spec = MVPSpecification(
        idea_title=idea["title"],
        category=AppCategory(idea["category"]),
        architecture_overview=str(spec_data.get("architecture_overview", "") or ""),
        tech_stack=[str(x) for x in (spec_data.get("tech_stack") or [])],
        core_features=[str(x) for x in (spec_data.get("core_features") or [])],
        database_schema_sql=str(spec_data.get("database_schema_sql", "") or ""),
        api_endpoints=[str(x) for x in (spec_data.get("api_endpoints") or [])],
        generated_files={str(k): str(v) for k, v in generated_files.items()},
        qa_test_plan=str(qa_data.get("qa_test_plan", "") or ""),
        static_security_audit=combined_audit,
        safe_run_instructions=str(qa_data.get("safe_run_instructions", "") or ""),
    )

    spec_md = _render_mvp_md(spec, sandbox, written, outputs.get("ceo_review", ""))
    db.save_mvp_artifact(campaign_id, idea_id, spec, spec_md, str(sandbox))
    db.update_campaign_status(campaign_id, "COMPLETED", final_report_md=spec_md)
    db.log_agent_event(
        campaign_id,
        "SYSTEM",
        "DONE",
        f"Moduł MVP gotowy. Zapisano {len(written)} plików w {sandbox}. Kod NIE został uruchomiony.",
    )
    return {
        "status": "COMPLETED",
        "sandbox_dir": str(sandbox),
        "files_written": len(written),
        "code_safe": is_safe,
    }


def _render_mvp_md(spec: MVPSpecification, sandbox: Path, written: List[str], review: str) -> str:
    """Renderuje specyfikację MVP w formacie Markdown."""
    lines = [
        f"# Specyfikacja MVP: {spec.idea_title} ({spec.category.value})",
        "",
        f"Wygenerowano: {_now()}",
        "",
        "## Architektura",
        spec.architecture_overview or "[HIPOTEZA] Brak opisu.",
        "",
        "## Stos technologiczny",
    ]
    lines += [f"- {t}" for t in spec.tech_stack] or ["- (brak)"]
    lines += ["", "## Kluczowe funkcje MVP"]
    lines += [f"- {f}" for f in spec.core_features] or ["- (brak)"]
    lines += ["", "## Schemat bazy danych (SQL)", "```sql", spec.database_schema_sql or "-- brak", "```", ""]
    lines += ["## Endpointy API"]
    lines += [f"- `{e}`" for e in spec.api_endpoints] or ["- (brak)"]
    lines += ["", f"## Wygenerowane pliki (sandbox: `{sandbox}`)"]
    lines += [f"- `{p}`" for p in written] or ["- (brak plików)"]
    lines += [
        "",
        "> UWAGA: Kod został zapisany wyłącznie lokalnie. System NIE uruchomił go i NIE opublikował.",
        "",
        "## Plan testów (QA)",
        spec.qa_test_plan or "[HIPOTEZA] Brak.",
        "",
        "## Audyt bezpieczeństwa (statyczny)",
        spec.static_security_audit or "[HIPOTEZA] Brak.",
        "",
        "## Bezpieczne uruchomienie (ręcznie, przez człowieka)",
        spec.safe_run_instructions or "[HIPOTEZA] Brak.",
    ]
    if review:
        lines += ["", "## Podsumowanie nadzoru CEO", review[:2000]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kontroler kampanii używany przez dashboard Streamlit (Start / Stop / Zatwierdź)
# ---------------------------------------------------------------------------
def _discovery_process_entry(
    campaign_id: int, config: VLLMConfig, use_simulator: bool
) -> None:
    """Punkt wejścia procesu potomnego dla Fazy 1 (musi być na poziomie modułu dla 'spawn')."""
    _execute_discovery(campaign_id, config, use_simulator)


def _mvp_process_entry(
    campaign_id: int, idea_id: int, config: VLLMConfig, use_simulator: bool
) -> None:
    """Punkt wejścia procesu potomnego dla Fazy 2 (musi być na poziomie modułu dla 'spawn')."""
    _execute_mvp_build(campaign_id, idea_id, config, use_simulator)


class CampaignController:
    """Zarządza cyklem życia kampanii: Start (nowy proces), Stop (terminate), Zatwierdź (bramka)."""

    def __init__(
        self,
        vllm_config: Optional[VLLMConfig] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        self.vllm_config = vllm_config or VLLMConfig()
        self.db = db or DatabaseManager()
        # Preferujemy 'fork' na Linuksie: proces potomny nie re-importuje modułu głównego,
        # co jest krytyczne, gdy rodzicem jest skrypt Streamlit (unika ponownego uruchomienia UI).
        # Gdy 'fork' jest niedostępny (np. Windows/macOS), używamy domyślnej metody startu.
        try:
            self._ctx = multiprocessing.get_context("fork")
        except ValueError:  # pragma: no cover - platformy bez fork
            self._ctx = multiprocessing.get_context()
        self._processes: Dict[int, multiprocessing.Process] = {}

    # -- Start -------------------------------------------------------------
    def start_campaign(self, name: str, use_simulator: bool = False) -> int:
        """Tworzy nową kampanię i uruchamia Fazę 1 w osobnym procesie."""
        campaign_id = self.db.create_campaign(name)
        proc = self._ctx.Process(
            target=_discovery_process_entry,
            args=(campaign_id, self.vllm_config, use_simulator),
            daemon=True,
            name=f"campaign-{campaign_id}",
        )
        proc.start()
        self._processes[campaign_id] = proc
        self.db.update_campaign_status(campaign_id, "RUNNING", pid=proc.pid)
        self.db.log_agent_event(
            campaign_id, "SYSTEM", "INFO", f"Uruchomiono proces kampanii (PID: {proc.pid})."
        )
        return campaign_id

    # -- Stop --------------------------------------------------------------
    def stop_campaign(self, campaign_id: int) -> None:
        """Ustawia flagę STOP w SQLite i kończy proces kampanii."""
        import os
        import signal

        self.db.update_campaign_status(campaign_id, "STOPPED")
        self.db.log_agent_event(campaign_id, "SYSTEM", "STOPPED", "Użytkownik zażądał zatrzymania kampanii.")

        proc = self._processes.get(campaign_id)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
            return

        campaign = self.db.get_campaign(campaign_id)
        pid = (campaign or {}).get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, ValueError):
                pass

    # -- Zatwierdź ---------------------------------------------------------
    def approve_idea(self, campaign_id: int, idea_id: int) -> Optional[Dict[str, Any]]:
        """Bramka bezpieczeństwa: oznacza pomysł jako zatwierdzony do budowy MVP."""
        idea = self.db.approve_idea_for_mvp(idea_id)
        if idea:
            self.db.log_agent_event(
                campaign_id,
                "SYSTEM",
                "APPROVED",
                f"Użytkownik zatwierdził pomysł '{idea['title']}' do budowy MVP.",
            )
        return idea

    def build_mvp(self, campaign_id: int, idea_id: int, use_simulator: bool = False) -> Dict[str, Any]:
        """Uruchamia Fazę 2 wyłącznie dla zatwierdzonego pomysłu."""
        idea = self.db.get_idea_by_id(idea_id)
        if not idea:
            raise ValueError(f"Nie znaleziono pomysłu o ID {idea_id}.")
        if not idea.get("approved_for_mvp"):
            raise PermissionError(
                "Bramka bezpieczeństwa: budowa MVP wymaga wcześniejszego zatwierdzenia pomysłu przez użytkownika."
            )
        if DEFAULT_SAFETY_CONFIG.require_user_approval_for_mvp and not idea.get("approved_for_mvp"):
            raise PermissionError("Budowa MVP zablokowana bez zgody użytkownika.")

        proc = self._ctx.Process(
            target=_mvp_process_entry,
            args=(campaign_id, idea_id, self.vllm_config, use_simulator),
            daemon=True,
            name=f"mvp-{idea_id}",
        )
        proc.start()
        self._processes[campaign_id] = proc
        self.db.update_campaign_status(campaign_id, "BUILDING_MVP", pid=proc.pid)
        return {"status": "BUILDING_MVP", "pid": proc.pid}

    # -- Pomocnicze --------------------------------------------------------
    def is_process_alive(self, campaign_id: int) -> bool:
        proc = self._processes.get(campaign_id)
        if proc is not None:
            return proc.is_alive()
        return False

    def healthcheck(self, use_simulator: bool = False) -> bool:
        """Sprawdza dostępność vLLM (lub zwraca True dla symulatora)."""
        if use_simulator:
            return True
        return VLLMClient(self.vllm_config).healthcheck()

