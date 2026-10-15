"""Pakiet darmowych narzędzi badawczych i weryfikacyjnych."""
from app_discovery_system.tools.search_tools import (
    DuckDuckGoSearchTool,
    HackerNewsSearchTool,
    RedditSearchTool,
    WebPageFetchTool,
    hackernews_search,
    reddit_search,
)
from app_discovery_system.tools.verification_tools import (
    static_code_audit,
    validate_claims,
    write_mvp_files_safely,
)

__all__ = [
    "DuckDuckGoSearchTool",
    "HackerNewsSearchTool",
    "RedditSearchTool",
    "WebPageFetchTool",
    "hackernews_search",
    "reddit_search",
    "static_code_audit",
    "validate_claims",
    "write_mvp_files_safely",
]
