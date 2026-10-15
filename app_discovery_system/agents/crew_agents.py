"""Definicje sześciu agentów CrewAI.

Agenci: CEO, Badacz Rynku, Analityk Konkurencji, Analityk Monetyzacji, Programista, Tester.
Programista i Tester są aktywowani wyłącznie po zatwierdzeniu pomysłu przez użytkownika.
"""

from __future__ import annotations

from typing import Any, List

from crewai import Agent

from app_discovery_system.tools.search_tools import (
    DuckDuckGoSearchTool,
    HackerNewsSearchTool,
    RedditSearchTool,
    WebPageFetchTool,
)

# Wspólna instrukcja rygoru informacyjnego dla wszystkich agentów.
SOURCE_DISCIPLINE = (
    "RYGOR INFORMACYJNY (OBOWIĄZKOWY): Każde twierdzenie rynkowe, liczba lub opinia o rynku MUSI "
    "mieć rzeczywiste źródło w formie pełnego adresu URL pozyskanego z narzędzi wyszukiwania. "
    "Jeżeli nie masz potwierdzenia w źródle, MUSISZ oznaczyć twierdzenie dokładnie jako [HIPOTEZA]. "
    "Nigdy nie wymyślaj adresów URL ani danych liczbowych. "
    "Nie narzucaj z góry żadnej niszy — odkrywaj możliwości obiektywnie."
)

NO_SIDE_EFFECTS = (


class AgentFactory:
    """Fabryka budująca agentów CrewAI z podpiętym lokalnym LLM i darmowymi narzędziami."""

    def __init__(self, llm: Any, verbose: bool = False) -> None:
        self.llm = llm
        self.verbose = verbose
        self._search_tools = [
            DuckDuckGoSearchTool(),
            HackerNewsSearchTool(),
            RedditSearchTool(),
            WebPageFetchTool(),
        ]

    def ceo(self) -> Agent:
        return Agent(
            role="CEO / Główny Strateg Produktowy",
            goal=(
                "Nadzorować kampanię odkrywania dochodowych aplikacji bez narzucania niszy, "
                "wybrać 3 najlepsze pomysły z 10 wygenerowanych i wystawić końcową rekomendację "
                "wraz z oceną 0-100."
            ),
            backstory=(
                "Doświadczony założyciel i inwestor, który ocenia pomysły twardymi kryteriami: "
                "wielkość bólu klienta, gotowość do płacenia, bariery wejścia i czas do pierwszego przychodu. "
                f"{SOURCE_DISCIPLINE} {NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=[],
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=3,
        )

    def market_researcher(self) -> Agent:
        return Agent(
            role="Badacz Rynku",
            goal=(
                "Odnaleźć w darmowych źródłach 10 realnych problemów rynkowych i przekuć je w pomysły "
                "na aplikacje mobilne, webowe oraz B2B. Każde twierdzenie musi mieć źródło lub [HIPOTEZA]."
            ),
            backstory=(
                "Analityk trendów i bólu użytkowników. Przeszukuje fora, Reddit, HackerNews i artykuły, "
                "szukając powtarzalnych, kosztownych problemów, za których rozwiązanie klienci już płacą. "
                f"{SOURCE_DISCIPLINE} {NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=self._search_tools,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def competitor_analyst(self) -> Agent:
        return Agent(
            role="Analityk Konkurencji",
            goal=(
                "Dla 3 najlepszych pomysłów zbadać istniejących konkurentów, ich słabości, nasycenie rynku "
                "oraz realną trudność budowy MVP."
            ),
            backstory=(
                "Specjalista od analizy konkurencji i barier wejścia. Ocenia, czy rynek jest nasycony, "
                "gdzie są luki i jak trudno zbudować przewagę technologiczną. "
                f"{SOURCE_DISCIPLINE} {NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=self._search_tools,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def monetization_analyst(self) -> Agent:
        return Agent(
            role="Analityk Monetyzacji",
            goal=(
                "Dla 3 najlepszych pomysłów zaprojektować model monetyzacji z realnymi cenami, "
                "oszacować przychody i wskazać konkretne kanały sprzedaży."
            ),
            backstory=(
                "Architekt modeli biznesowych. Zna cenniki SaaS, modele freemium, licencje B2B "
                "i potrafi wskazać, jak realnie dotrzeć do pierwszych płacących klientów. "
                f"{SOURCE_DISCIPLINE} {NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=self._search_tools,
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=5,
        )

    def developer(self) -> Agent:
        return Agent(
            role="Programista / Architekt MVP",
            goal=(
                "Przygotować specyfikację techniczną MVP oraz kompletny, gotowy do przeglądu kod źródłowy "
                "aplikacji — bez jego uruchamiania i bez publikacji."
            ),
            backstory=(
                "Senior full-stack engineer. Projektuje minimalne, wdrażalne MVP z jasną architekturą, "
                "modelem danych i endpointami. Generuje kod wyłącznie do lokalnego przeglądu. "
                f"{NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=[],
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=3,
        )

    def tester(self) -> Agent:
        return Agent(
            role="Tester / Inżynier QA i Audytor Bezpieczeństwa",
            goal=(
                "Stworzyć plan testów MVP i przeprowadzić statyczny audyt bezpieczeństwa wygenerowanego kodu "
                "bez jego uruchamiania."
            ),
            backstory=(
                "Inżynier QA specjalizujący się w testach akceptacyjnych i audycie bezpieczeństwa kodu. "
                "Weryfikuje kod statycznie i przygotowuje instrukcję bezpiecznego uruchomienia przez człowieka. "
                f"{NO_SIDE_EFFECTS}"
            ),
            llm=self.llm,
            tools=[],
            verbose=self.verbose,
            allow_delegation=False,
            max_iter=3,
        )

    def all_agents(self) -> List[Agent]:
        """Zwraca wszystkich sześciu agentów (CEO, badacz, konkurencja, monetyzacja, dev, tester)."""
        return [
            self.ceo(),
            self.market_researcher(),
            self.competitor_analyst(),
            self.monetization_analyst(),
            self.developer(),
            self.tester(),
        ]

    "ZAKAZY BEZPIECZEŃSTWA: Nie publikuj treści, nie wysyłaj wiadomości, nie dokonuj zakupów, "
    "nie uruchamiaj żadnego kodu ani nie łącz się z płatnymi usługami zewnętrznymi. "
    "Działasz wyłącznie lokalnie i tylko na potrzeby analizy."
)
