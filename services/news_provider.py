"""
News provider abstraction (Phase 4A).

    NewsProvider
    ├── AlpacaNewsProvider         services/alpaca_news_provider.py  (primary)
    ├── AlphaVantageNewsProvider   future, reference only
    └── NewsDataProvider           future, optional

A provider turns a NewsQuery into canonical NewsArticle records plus
provenance. Provider-specific details (URLs, headers, response fields)
stay inside the concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from services.news_schema import NewsArticle, NewsValidationError, ensure_utc

SORT_ORDERS = ("asc", "desc")


class NewsProviderError(RuntimeError):
    """A provider request failed (network, HTTP, malformed response)."""


class NewsAuthError(NewsProviderError):
    """Credentials missing or rejected. Messages never contain secret values."""


class NewsPaginationError(NewsProviderError):
    """Pagination did not terminate safely (repeated token or page cap)."""


@dataclass(frozen=True)
class NewsQuery:
    """
    Provider-independent historical news request.
    start/end must be timezone-aware; they are normalized to UTC.
    page_size=None means the provider's maximum.
    """

    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    sort: str = "asc"
    include_content: bool = True
    page_size: int | None = None

    def __post_init__(self):
        if isinstance(self.symbols, str) or not self.symbols:
            raise ValueError("symbols must be a non-empty sequence of tickers")
        symbols = tuple(s.strip().upper() for s in self.symbols)
        if not all(symbols):
            raise ValueError("symbols must not contain empty tickers")
        object.__setattr__(self, "symbols", symbols)

        try:
            object.__setattr__(self, "start", ensure_utc(self.start, "start"))
            object.__setattr__(self, "end", ensure_utc(self.end, "end"))
        except NewsValidationError as e:
            raise ValueError(str(e)) from e
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.sort not in SORT_ORDERS:
            raise ValueError(f"sort must be one of {SORT_ORDERS}")
        if self.page_size is not None and (not isinstance(self.page_size, int) or self.page_size < 1):
            raise ValueError("page_size must be a positive integer")

    def describe(self) -> dict[str, Any]:
        """Request parameters for provenance (never contains credentials)."""
        return {
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "sort": self.sort,
            "include_content": self.include_content,
            "page_size": self.page_size,
        }


@dataclass
class NewsFetchResult:
    articles: list[NewsArticle]
    provenance: dict[str, Any] = field(default_factory=dict)


def deduplicate(articles: list[NewsArticle]) -> tuple[list[NewsArticle], int, int]:
    """
    Keep the first occurrence of each (provider, provider_article_id).
    Returns (unique articles in original order, duplicates removed,
    duplicates whose content differed from the kept record).
    """
    kept: dict[tuple[str, str], NewsArticle] = {}
    order: list[NewsArticle] = []
    duplicates = conflicts = 0
    for article in articles:
        first = kept.get(article.key)
        if first is None:
            kept[article.key] = article
            order.append(article)
            continue
        duplicates += 1
        if (first.headline, first.content, first.updated_at) != (article.headline, article.content, article.updated_at):
            conflicts += 1
    return order, duplicates, conflicts


class NewsProvider(ABC):
    """Interface every news provider implements."""

    name: str

    @abstractmethod
    def fetch_historical(self, query: NewsQuery, *, max_pages: int | None = None) -> NewsFetchResult:
        """
        Fetch all articles matching `query`, deduplicated, as canonical
        NewsArticle records, with provenance (request parameters without
        credentials, retrieval time, pages, counts).
        """
