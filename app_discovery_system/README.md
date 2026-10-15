# System Wieloagentowy: Odkrywanie Dochodowych Aplikacji

Lokalny system wieloagentowy (**CrewAI + Streamlit + SQLite + lokalny vLLM**) do autonomicznego
odkrywania możliwości tworzenia dochodowych aplikacji **bez narzucania niszy z góry**.

Brak płatnych usług zewnętrznych. Brak publikacji, zakupów, wysyłki wiadomości i uruchamiania
niezweryfikowanego kodu.

---

## 1. Architektura

```
Streamlit Dashboard (Start / Stop / Zatwierdź, agenci, logi, ranking, raport, MVP, historia)
        │
        ▼
Orchestrator (multiprocessing.Process + flaga STOP w SQLite)  ──►  SQLite (WAL)
        │
        ├── Crew 1 (Faza 1: Odkrywanie i Analiza)
        │     CEO · Badacz Rynku · Analityk Konkurencji · Analityk Monetyzacji
        │     10 pomysłów (Mobile/Web/B2B) → Top 3 → raport + ocena 0-100
        │
        └── Crew 2 (Faza 2: MVP — DOPIERO po zatwierdzeniu przez użytkownika)
              Programista · Tester · CEO (nadzór)
              specyfikacja MVP + kod w sandboxie + audyt statyczny (bez uruchamiania)
        │
        ▼
Warstwa LLM: CrewAIVLLM (BaseLLM) ──► lokalny vLLM (OpenAI-compatible)   [albo tryb offline: symulator]
Warstwa narzędzi: DuckDuckGo (throttling 2 s) · HackerNews Algolia · Reddit JSON · scraper WWW
```

### Agenci (6)

| Agent | Rola | Aktywacja |
|---|---|---|
| CEO | Koordynacja, wybór Top 3, ocena końcowa, nadzór nad MVP | Faza 1 i 2 |
| Badacz Rynku | 10 pomysłów z realnych problemów (Mobile/Web/B2B) | Faza 1 |
| Analityk Konkurencji | Konkurenci, nasycenie rynku, trudność budowy | Faza 1 |
| Analityk Monetyzacji | Model biznesowy, ceny, kanały sprzedaży | Faza 1 |
| Programista | Specyfikacja MVP + kod źródłowy | **tylko po zatwierdzeniu** |
| Tester | Plan testów + statyczny audyt bezpieczeństwa | **tylko po zatwierdzeniu** |

### Ocena 0–100

| Kryterium | Punkty |
|---|---|
| Potencjał rynkowy i pilność problemu | 0–25 |
| Monetyzacja i marżowość | 0–25 |
| Nisza i przewaga konkurencyjna | 0–20 |
| Łatwość budowy MVP / time-to-market | 0–15 |
| Dostępność kanałów sprzedaży | 0–15 |

### Rygor informacyjny

Każde twierdzenie rynkowe (`MarketClaim`) ma **poprawny URL źródła** (`VERIFIED_SOURCE`)
albo jest automatycznie oznaczane jako **`[HIPOTEZA]`**. Brak/niepoprawny URL wymusza hipotezę.

---

## 2. Obsłużone ryzyka techniczne

| # | Ryzyko | Rozwiązanie w kodzie |
|---|---|---|
| 1 | Przycisk **Stop** vs wątki Pythona | Kampania w `multiprocessing.Process` (kontekst `fork`) + `terminate()`; dodatkowo `step_callback` CrewAI sprawdza flagę STOP w SQLite przed każdym krokiem i rzuca `CampaignStoppedError` |
| 2 | `output_pydantic` na lokalnym vLLM | Proste, płaskie schematy JSON per zadanie + `extract_json()`/`repair_json()` naprawiające ucięte JSON-y i ogrodzenia markdown |
| 3 | Konfiguracja LiteLLM | Własny adapter `CrewAIVLLM(BaseLLM)` — brak zależności od autoryzacji OpenAI i prefiksów dostawcy |
| 4 | Odświeżanie logów w Streamlit | Kampania w osobnym procesie (UI się nie blokuje); logi odpytywane z SQLite przez `@st.fragment(run_every="2s")` |
| 5 | Blokady DuckDuckGo (202 Ratelimit) | Globalny throttle `time.sleep(2)` + automatyczny fallback do HackerNews Algolia API i Reddit JSON |

---

## 3. Instalacja i uruchomienie

```bash
# 1. Środowisko wirtualne
python3 -m venv .venv
source .venv/bin/activate

# 2. Zależności
pip install -r app_discovery_system/requirements.txt

# 3. (Opcjonalnie) lokalny serwer vLLM zgodny z OpenAI API
#    vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000

# 4. Dashboard
streamlit run app_discovery_system/app.py
```

### Konfiguracja (panel boczny lub zmienne środowiskowe)

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Adres API vLLM |
| `VLLM_MODEL_NAME` | `Qwen/Qwen2.5-14B-Instruct` | Nazwa modelu |
| `VLLM_API_KEY` | `EMPTY` | Klucz (dowolny dla vLLM) |
| `VLLM_TIMEOUT_SECONDS` | `30` | Timeout pojedynczego zapytania |
| `VLLM_MAX_CALLS_PER_CAMPAIGN` | `50` | Limit wywołań LLM na kampanię |
| `VLLM_MAX_RPM` | `30` | Limit wywołań na minutę |
| `VLLM_MAX_RETRIES` | `2` | Liczba ponowień przy błędach sieciowych |

### Tryb offline (bez GPU/vLLM)

W panelu bocznym zaznacz **„Tryb offline (lokalny symulator LLM)”**. Symulator zwraca
deterministyczne, poprawne strukturalnie odpowiedzi, dzięki czemu cały przepływ (Faza 1 i 2)
działa bez żadnego połączenia sieciowego — używane także w testach.

---

## 4. Testy

```bash
python -m pytest app_discovery_system/tests -q
```

| Plik | Zakres |
|---|---|
| `test_vllm_resilience.py` | limity wywołań, timeouty, retry, circuit breaker, błędy połączenia, flaga STOP |
| `test_fact_checking.py` | źródło URL vs `[HIPOTEZA]`, walidacja URL, naprawa JSON |
| `test_campaign_flow.py` | 10 pomysłów, Top 3, ranking malejąco, kompletność raportu, MVP |
| `test_safety_gates.py` | bramka zatwierdzenia, blokada MVP, brak uruchamiania kodu, path traversal |

---

## 5. Bezpieczeństwo (Human-in-the-Loop)

* Faza 1 kończy się statusem `WAITING_APPROVAL` — **nic nie dzieje się dalej bez kliknięcia „Zatwierdź”**.
* Faza 2 (Programista/Tester) startuje tylko dla pomysłu z `approved_for_mvp = 1`.
* Wygenerowany kod trafia wyłącznie do `app_discovery_system/workspace_mvp/<pomysł>/`
  (blokada path traversal) i jest poddawany **statycznej** analizie AST (`eval`/`exec`/`subprocess`/…).
* System **nigdy** nie wykonuje `subprocess.run`/`exec()` na kodzie wygenerowanym przez LLM.
