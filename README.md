# EndToEnd_Project

## System wieloagentowy: Odkrywanie Dochodowych Aplikacji

Lokalny system wieloagentowy (CrewAI + Streamlit + SQLite + lokalny vLLM) do autonomicznego
odkrywania dochodowych pomysłów na aplikacje (Mobile / Web / B2B).

* Kod: [`app_discovery_system/`](app_discovery_system/)
* Dokumentacja, architektura i instrukcja uruchomienia: [`app_discovery_system/README.md`](app_discovery_system/README.md)

```bash
pip install -r app_discovery_system/requirements.txt
streamlit run app_discovery_system/app.py          # dashboard (Start / Stop / Zatwierdź)
python -m pytest app_discovery_system/tests -q     # testy
```
