"""Dashboard Streamlit dla systemu wieloagentowego odkrywania aplikacji.

Uruchomienie:
    streamlit run app_discovery_system/app.py

Funkcje: Start / Stop / Zatwierdź, podgląd statusu 6 agentów, logi na żywo (odpytywane
z SQLite bez przeładowania strony), ranking pomysłów 0-100, raport Top 3, historia kampanii
oraz podgląd specyfikacji MVP wygenerowanej po zatwierdzeniu.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Zapewniamy import pakietu przy uruchomieniu `streamlit run app_discovery_system/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_discovery_system.config import VLLMConfig  # noqa: E402
from app_discovery_system.database.db_manager import DatabaseManager  # noqa: E402
from app_discovery_system.orchestrator.campaign_runner import (  # noqa: E402
    CampaignController,
)

st.set_page_config(page_title="System Odkrywania Aplikacji", page_icon="🧭", layout="wide")

AGENT_ORDER = ["CEO", "Badacz Rynku", "Analityk Konkurencji", "Analityk Monetyzacji", "Programista", "Tester"]
STATUS_ICON = {
    "RUNNING": "🟡",
    "STEP": "🟡",
    "DONE": "🟢",
    "INFO": "🔵",
    "ERROR": "🔴",
    "STOPPED": "⛔",
    "BLOCKED": "🔒",
    "APPROVED": "✅",
    "IDLE": "⚪",
}
CAMPAIGN_STATUS_LABEL = {
    "IDLE": "Bezczynny",
    "RUNNING": "W trakcie (Faza 1)",
    "STOPPED": "Zatrzymany",
    "WAITING_APPROVAL": "Oczekuje na zatwierdzenie",
    "BUILDING_MVP": "Budowa MVP (Faza 2)",
    "COMPLETED": "Zakończony",
    "ERROR": "Błąd",
}


@st.cache_resource
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource
def get_controller(base_url: str, model_name: str, api_key: str, timeout: float, max_calls: int) -> CampaignController:
    cfg = VLLMConfig(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        request_timeout_seconds=timeout,
        max_calls_per_campaign=max_calls,
    )
    return CampaignController(vllm_config=cfg)


def _sidebar() -> dict:
    st.sidebar.header("⚙️ Konfiguracja vLLM (lokalny)")
    base_url = st.sidebar.text_input("Adres API vLLM", value=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    model_name = st.sidebar.text_input("Nazwa modelu", value=os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct"))
    api_key = st.sidebar.text_input("Klucz API (dowolny dla vLLM)", value=os.environ.get("VLLM_API_KEY", "EMPTY"), type="password")
    timeout = st.sidebar.number_input("Timeout zapytania [s]", min_value=5.0, max_value=300.0, value=30.0, step=5.0)
    max_calls = st.sidebar.number_input("Limit wywołań LLM / kampanię", min_value=5, max_value=500, value=50, step=5)
    use_simulator = st.sidebar.checkbox(
        "Tryb offline (lokalny symulator LLM)",
        value=False,
        help="Użyj, gdy serwer vLLM jest niedostępny. Symulator zwraca deterministyczne odpowiedzi bez sieci.",
    )
    st.sidebar.divider()
    st.sidebar.caption("Bezpieczeństwo: brak publikacji, zakupów i wysyłki wiadomości. Kod MVP nie jest uruchamiany.")
    return {
        "base_url": base_url,
        "model_name": model_name,
        "api_key": api_key,
        "timeout": timeout,
        "max_calls": max_calls,
        "use_simulator": use_simulator,
    }


def _render_agents_panel(db: DatabaseManager, campaign_id: int) -> None:
    """Panel statusu 6 agentów na podstawie ostatnich wpisów w tabeli agent_logs."""
    st.subheader("🤖 Agenci")
    logs = db.get_agent_logs(campaign_id, limit=500)
    latest: dict[str, str] = {name: "IDLE" for name in AGENT_ORDER}
    for entry in logs:
        name = entry["agent_name"]
        if name in latest:
            latest[name] = entry["status"]
    cols = st.columns(6)
    for col, name in zip(cols, AGENT_ORDER):
        status = latest.get(name, "IDLE")
        icon = STATUS_ICON.get(status, "⚪")
        col.metric(label=name, value=f"{icon} {status}")


@st.fragment(run_every="2s")
def _render_live_logs(campaign_id: int) -> None:
    """Strumień logów agentów odpytywany z SQLite bez przeładowania całej strony."""
    db = get_db()
    logs = db.get_agent_logs(campaign_id, limit=200)
    st.caption(f"Ostatnia aktualizacja: {len(logs)} wpisów (odświeżanie co 2 s)")
    if not logs:
        st.info("Brak logów dla tej kampanii.")
        return
    lines = [
        f"[{entry['timestamp']}] {STATUS_ICON.get(entry['status'], '•')} "
        f"{entry['agent_name']} :: {entry['status']} :: {entry['message']}"
        for entry in logs[-60:]
    ]
    st.code("\n".join(lines), language="log")


def _render_ranking(db: DatabaseManager, campaign_id: int) -> None:
    """Ranking pomysłów z oceną 0-100 i szczegółami Top 3."""
    st.subheader("🏆 Ranking pomysłów (0-100)")
    ideas = db.get_ideas_for_campaign(campaign_id)
    if not ideas:
        st.info("Brak pomysłów. Uruchom kampanię (przycisk Start).")
        return

    rows = [
        {
            "#": idea["rank"],
            "Pomysł": idea["title"],
            "Kategoria": idea["category"],
            "Ocena": idea["score_total"],
            "Top 3": "✅" if idea["is_top3"] else "",
            "Zatwierdzony": "✅" if idea["approved_for_mvp"] else "",
        }
        for idea in ideas
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("📌 Szczegółowa analiza Top 3")
    for idea in [i for i in ideas if i["is_top3"]]:
        with st.expander(f"TOP 3: {idea['title']} ({idea['category']}) — {idea['score_total']}/100", expanded=False):
            sb = idea.get("score_breakdown", {})
            st.markdown(
                f"**Punktacja:** popyt {sb.get('market_demand', 0)}/25 · "
                f"monetyzacja {sb.get('monetization_potential', 0)}/25 · "
                f"nisza {sb.get('low_competition_niche', 0)}/20 · "
                f"łatwość budowy {sb.get('build_feasibility', 0)}/15 · "
                f"kanały sprzedaży {sb.get('sales_channel_access', 0)}/15"
            )
            st.markdown(f"**Problem:** {idea['problem_statement']}")
            st.markdown(f"**Grupa docelowa:** {idea['target_audience']}")
            st.markdown(f"**Propozycja wartości:** {idea['value_proposition']}")
            st.markdown(f"**Konkurencja:** {idea['competitors_analysis'] or '[HIPOTEZA] brak danych'}")
            st.markdown(f"**Ceny / monetyzacja:** {idea['pricing_analysis'] or '[HIPOTEZA] brak danych'}")
            st.markdown(f"**Trudność budowy:** {idea['build_difficulty'] or '[HIPOTEZA] nieokreślona'}")
            st.markdown(f"**Kanały sprzedaży:** {idea['sales_channels'] or '[HIPOTEZA] nieokreślone'}")
            st.markdown(f"**Rekomendacja:** {idea['recommendation'] or '[HIPOTEZA] brak'}")
            st.markdown("**Twierdzenia rynkowe i źródła:**")
            for claim in idea.get("claims", []):
                url = claim.get("source_url")
                if url:
                    st.markdown(f"- {claim['claim_text']} — [źródło]({url})")
                else:
                    st.markdown(f"- {claim['claim_text']} **[HIPOTEZA]**")


def _render_mvp(db: DatabaseManager, campaign_id: int) -> None:
    """Podgląd specyfikacji MVP i wygenerowanego (nieuruchamianego) kodu."""
    st.subheader("🛠️ Specyfikacja MVP i moduł tworzenia aplikacji")
    artifacts = db.get_mvp_artifacts(campaign_id=campaign_id)
    if not artifacts:
        st.info(
            "Brak wygenerowanego MVP. Wybierz pomysł z Top 3, zatwierdź go, a następnie kliknij "
            "'Zbuduj MVP' — dopiero wtedy aktywują się agenci Programista i Tester."
        )
        return
    for art in artifacts:
        with st.expander(f"MVP: {art['idea_title']} (sandbox: {art['sandbox_dir']})", expanded=True):
            st.markdown(art["mvp_spec_md"])
            if art.get("generated_files"):
                st.markdown("**Podgląd plików kodu:**")
                for fname, content in art["generated_files"].items():
                    st.markdown(f"`{fname}`")
                    st.code(content, language="python" if fname.endswith(".py") else "text")


def _render_history(db: DatabaseManager) -> None:
    """Historia kampanii."""
    st.subheader("📜 Historia kampanii")
    campaigns = db.list_campaigns(limit=50)
    if not campaigns:
        st.info("Brak kampanii w historii.")
        return
    st.dataframe(
        [
            {
                "ID": c["id"],
                "Nazwa": c["name"],
                "Status": CAMPAIGN_STATUS_LABEL.get(c["status"], c["status"]),
                "Wywołania LLM": c["vllm_calls_count"],
                "Utworzono": c["created_at"],
                "Aktualizacja": c["updated_at"],
                "Błąd": c["error_message"] or "",
            }
            for c in campaigns
        ],
        width="stretch",
        hide_index=True,
    )


def _render_controls(db: DatabaseManager, controller: CampaignController, cfg: dict) -> int | None:
    """Pasek sterowania: Start / Stop / Zatwierdź + wybór kampanii. Zwraca wybrane campaign_id."""
    st.subheader("🎛️ Sterowanie kampanią")
    campaigns = db.list_campaigns(limit=50)
    options = {f"#{c['id']} · {c['name']} · {CAMPAIGN_STATUS_LABEL.get(c['status'], c['status'])}": c["id"] for c in campaigns}
    selected_id: int | None = None
    if options:
        label = st.selectbox("Aktywna kampania", list(options.keys()))
        selected_id = options[label]

    col1, col2, col3 = st.columns(3)
    with col1:
        campaign_name = st.text_input("Nazwa nowej kampanii", value=f"Kampania {len(campaigns) + 1}")
        if st.button("▶️ Start", type="primary", width="stretch"):
            if not cfg["use_simulator"] and not controller.healthcheck(use_simulator=False):
                st.error(
                    f"Serwer vLLM nie odpowiada pod adresem {cfg['base_url']}. "
                    "Uruchom serwer vLLM albo zaznacz tryb offline (symulator) w panelu bocznym."
                )
            else:
                new_id = controller.start_campaign(campaign_name, use_simulator=cfg["use_simulator"])
                st.success(f"Uruchomiono kampanię #{new_id} w osobnym procesie.")
                st.rerun()
    with col2:
        if st.button("⏹️ Stop", width="stretch", disabled=selected_id is None):
            if selected_id is not None:
                controller.stop_campaign(selected_id)
                st.warning(f"Zatrzymano kampanię #{selected_id}.")
                st.rerun()
    with col3:
        st.caption("Stop ustawia flagę w SQLite i kończy proces kampanii.")

    return selected_id


def _render_approval(db: DatabaseManager, controller: CampaignController, campaign_id: int, cfg: dict) -> None:
    """Bramka zatwierdzenia: wybór pomysłu i uruchomienie Fazy 2 (MVP)."""
    st.subheader("✅ Zatwierdzenie pomysłu i budowa MVP")
    ideas = db.get_ideas_for_campaign(campaign_id)
    if not ideas:
        st.info("Brak pomysłów do zatwierdzenia.")
        return
    choices = {f"#{i['rank']} {i['title']} ({i['category']}) — {i['score_total']}/100": i["id"] for i in ideas}
    label = st.selectbox("Wybierz pomysł do zatwierdzenia", list(choices.keys()))
    idea_id = choices[label]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Zatwierdź pomysł", width="stretch"):
            controller.approve_idea(campaign_id, idea_id)
            st.success("Pomysł zatwierdzony. Agenci Programista i Tester zostali odblokowani.")
            st.rerun()
    with col2:
        approved = db.get_idea_by_id(idea_id)
        is_approved = bool(approved and approved.get("approved_for_mvp"))
        if st.button("🏗️ Zbuduj MVP", width="stretch", disabled=not is_approved):
            try:
                controller.build_mvp(campaign_id, idea_id, use_simulator=cfg["use_simulator"])
                st.success("Uruchomiono Fazę 2 (specyfikacja MVP + audyt QA) w osobnym procesie.")
                st.rerun()
            except PermissionError as exc:
                st.error(str(exc))

    if not is_approved:
        st.warning(
            "🔒 Bramka bezpieczeństwa: budowa MVP jest zablokowana, dopóki nie zatwierdzisz pomysłu. "
            "System nie publikuje, nie kupuje, nie wysyła wiadomości ani nie uruchamia niezweryfikowanego kodu."
        )


def main() -> None:
    st.title("🧭 System Wieloagentowy: Odkrywanie Dochodowych Aplikacji")
    st.caption(
        "CrewAI + Streamlit + SQLite + lokalny vLLM · 6 agentów · bez płatnych usług zewnętrznych · "
        "każde twierdzenie rynkowe ma źródło lub jest oznaczone jako [HIPOTEZA]."
    )

    cfg = _sidebar()
    db = get_db()
    controller = get_controller(cfg["base_url"], cfg["model_name"], cfg["api_key"], cfg["timeout"], int(cfg["max_calls"]))

    campaign_id = _render_controls(db, controller, cfg)
    if campaign_id is None:
        st.info("Brak kampanii. Kliknij **Start**, aby uruchomić pierwszą kampanię odkrywania.")
        _render_history(db)
        return

    campaign = db.get_campaign(campaign_id) or {}
    status = campaign.get("status", "IDLE")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status kampanii", CAMPAIGN_STATUS_LABEL.get(status, status))
    c2.metric("Wywołania LLM", campaign.get("vllm_calls_count", 0))
    c3.metric("Tryb", "Symulator offline" if cfg["use_simulator"] else "vLLM")
    if campaign.get("error_message"):
        st.error(f"Ostatni błąd: {campaign['error_message']}")

    _render_agents_panel(db, campaign_id)

    tab_logs, tab_ranking, tab_report, tab_mvp, tab_history = st.tabs(
        ["📡 Logi na żywo", "🏆 Ranking", "📄 Raport", "🛠️ MVP", "📜 Historia"]
    )
    with tab_logs:
        _render_live_logs(campaign_id)
    with tab_ranking:
        _render_ranking(db, campaign_id)
        _render_approval(db, controller, campaign_id, cfg)
    with tab_report:
        st.subheader("📄 Raport kampanii")
        st.markdown(campaign.get("final_report_md") or "_Raport nie został jeszcze wygenerowany._")
    with tab_mvp:
        _render_mvp(db, campaign_id)
    with tab_history:
        _render_history(db)


if __name__ == "__main__":
    main()

