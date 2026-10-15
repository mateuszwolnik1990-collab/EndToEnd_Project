"""Darmowe narzędzia badawcze (bez płatnych API).

Zawiera:
- ``DuckDuckGoSearchTool``: wyszukiwanie webowe z obowiązkowym opóźnieniem (rate limit) i fallbackiem.
- ``HackerNewsSearchTool``: stabilne, darmowe API Algolia (HackerNews).
- ``RedditSearchTool``: publiczne JSON API Reddita.
- ``WebPageFetchTool``: pobiera i oczyszcza treść strony WWW (walidacja URL + limit rozmiaru).

Każdy wynik zawiera adres URL, aby agent mógł oznaczyć twierdzenie jako ``VERIFIED_SOURCE``.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List
from urllib.parse import quote_plus, urlparse

import httpx

try:
    from crewai.tools import BaseTool
except Exception:  # pragma: no cover
    class BaseTool:  # type: ignore[no-redef]
        """Minimalny fallback, gdy CrewAI nie jest dostępne (np. w testach jednostkowych)."""

        name: str = ""
        description: str = ""

        def run(self, *args: Any, **kwargs: Any) -> Any:
            return self._run(*args, **kwargs)


USER_AGENT = "AppDiscoveryBot/1.0 (+local research agent)"

# Globalny throttle dla DuckDuckGo (problem 5: blokady 202 Ratelimit przy szybkich zapytaniach).
_DDG_LOCK = threading.Lock()
_DDG_LAST_CALL = 0.0
_DDG_MIN_INTERVAL_SECONDS = 2.0


def _throttle_duckduckgo() -> None:
    """Wymusza co najmniej 2 sekundy przerwy między zapytaniami do DuckDuckGo."""
    global _DDG_LAST_CALL
    with _DDG_LOCK:
        now = time.monotonic()
        wait = _DDG_MIN_INTERVAL_SECONDS - (now - _DDG_LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _DDG_LAST_CALL = time.monotonic()


def _is_safe_url(url: str) -> bool:
    """Dopuszcza wyłącznie publiczne adresy http/https (blokuje localhost i sieci prywatne)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    host = parsed.hostname or ""
    blocked_prefixes = ("localhost", "127.", "10.", "192.168.", "169.254.", "0.")
    if host.startswith(blocked_prefixes):
        return False
    if re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host):
        return False
    return "." in host


class DuckDuckGoSearchTool(BaseTool):
    """Wyszukiwanie w DuckDuckGo (darmowe) z throttlingiem i fallbackiem do HackerNews."""

    name: str = "duckduckgo_search"
    description: str = (
        "Wyszukuje w internecie darmowe informacje o rynku, konkurencji i problemach użytkowników. "
        "Argument: 'query' (tekst zapytania). Zwraca listę wyników z adresami URL."
    )
    max_results: int = 5

    def _run(self, query: str) -> str:
        results = self._search_duckduckgo(query)
        source = "DuckDuckGo"
        if not results:
            results = hackernews_search(query, max_results=self.max_results)
            source = "HackerNews (fallback)"
        if not results:
            return (
                f"Brak wyników dla zapytania '{query}'. "
                "Oznacz powiązane twierdzenia jako [HIPOTEZA]."
            )
        lines = [f"Źródło: {source} | Zapytanie: {query}"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r.get('snippet', '')[:220]}")
        return "\n".join(lines)

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        try:
            _throttle_duckduckgo()
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self.max_results))
            out = []
            for item in raw:
                url = item.get("href") or item.get("url") or ""
                if _is_safe_url(url):
                    out.append(
                        {
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("body", ""),
                        }
                    )
            return out
        except Exception:
            return []


def hackernews_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict[str, str]]:
    """Darmowe wyszukiwanie w HackerNews przez API Algolia (stabilne, bez limitu)."""
    url = (
        "https://hn.algolia.com/api/v1/search"
        f"?query={quote_plus(query)}&tags=story&hitsPerPage={max_results}"
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except Exception:
        return []
    results = []
    for hit in hits:
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        if _is_safe_url(link):
            results.append(
                {
                    "title": hit.get("title") or "",
                    "url": link,
                    "snippet": (hit.get("story_text") or hit.get("title") or "")[:300],
                }
            )
    return results


def reddit_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict[str, str]]:
    """Darmowe wyszukiwanie w publicznym JSON API Reddita."""
    url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&limit={max_results}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
    except Exception:
        return []
    results = []
    for child in children:
        data = child.get("data", {})
        link = data.get("url") or ""
        if _is_safe_url(link):
            results.append(
                {
                    "title": data.get("title") or "",
                    "url": link,
                    "snippet": (data.get("selftext") or data.get("title") or "")[:300],
                }
            )
    return results


class HackerNewsSearchTool(BaseTool):
    """Stabilne, darmowe wyszukiwanie w HackerNews (API Algolia)."""

    name: str = "hackernews_search"
    description: str = (
        "Wyszukuje dyskusje techniczne i biznesowe na HackerNews. "
        "Argument: 'query' (tekst zapytania)."
    )
    max_results: int = 5

    def _run(self, query: str) -> str:
        results = hackernews_search(query, max_results=self.max_results)
        if not results:
            return f"Brak wyników HackerNews dla '{query}'. Oznacz twierdzenia jako [HIPOTEZA]."
        lines = [f"Źródło: HackerNews | Zapytanie: {query}"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r.get('snippet', '')[:200]}")
        return "\n".join(lines)


class RedditSearchTool(BaseTool):
    """Darmowe wyszukiwanie w publicznym JSON API Reddita (opinie i problemy użytkowników)."""

    name: str = "reddit_search"
    description: str = (
        "Wyszukuje realne problemy i opinie użytkowników na Reddicie. "
        "Argument: 'query' (tekst zapytania)."
    )
    max_results: int = 5

    def _run(self, query: str) -> str:
        results = reddit_search(query, max_results=self.max_results)
        if not results:
            return f"Brak wyników Reddit dla '{query}'. Oznacz twierdzenia jako [HIPOTEZA]."
        lines = [f"Źródło: Reddit | Zapytanie: {query}"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r.get('snippet', '')[:200]}")
        return "\n".join(lines)


class WebPageFetchTool(BaseTool):
    """Pobiera i oczyszcza treść strony WWW (walidacja URL, limit rozmiaru, timeout)."""

    name: str = "fetch_web_page"
    description: str = (
        "Pobiera treść publicznej strony WWW w celu weryfikacji twierdzeń rynkowych. "
        "Argument: 'url' (pełny adres http/https)."
    )
    max_chars: int = 4000

    def _run(self, url: str) -> str:
        if not _is_safe_url(url):
            return f"Odrzucono niebezpieczny lub nieprawidłowy URL: {url}"
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            html = resp.text[:200000]
        except Exception as exc:
            return f"Nie udało się pobrać {url}: {exc}"
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s{2,}", " ", text)
        return f"URL: {url}\nTreść:\n{text[: self.max_chars]}"

