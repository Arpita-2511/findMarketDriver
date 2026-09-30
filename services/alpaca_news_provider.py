"""
Alpaca (Benzinga) historical news provider - Phase 4A.

All Alpaca-specific details live in this module: endpoint, auth headers,
query parameter names, page-size limit and response field names. It
returns canonical NewsArticle records only (services/news_schema.py).

API: GET https://data.alpaca.markets/v1beta1/news
    symbols          comma-separated tickers
    start / end      RFC 3339
    limit            page size, 1..50 (Alpaca's documented maximum)
    sort             asc | desc
    include_content  true | false
    page_token       from the previous page's next_page_token

Credentials come from the environment (ALPACA_API_KEY /
ALPACA_API_SECRET, loaded from .env by config/settings.py). They are sent
only as request headers and never appear in repr(), logs, errors,
provenance or snapshots.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import requests

import config.settings  # noqa: F401  (loads .env into the environment; reads no values here)
from services.news_provider import (
    NewsAuthError,
    NewsFetchResult,
    NewsPaginationError,
    NewsProvider,
    NewsProviderError,
    NewsQuery,
    deduplicate,
)
from services.news_schema import (
    HISTORICAL_AVAILABILITY_RULE,
    NEWS_SCHEMA_VERSION,
    NewsArticle,
    NewsValidationError,
    historical_information_available_at,
    parse_timestamp,
)

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ALPACA_MAX_PAGE_SIZE = 50
API_KEY_ENV = "ALPACA_API_KEY"
API_SECRET_ENV = "ALPACA_API_SECRET"

DEFAULT_TIMEOUT = (5.0, 30.0)      # (connect, read) seconds
DEFAULT_MAX_RETRIES = 3            # retries AFTER the first attempt
DEFAULT_BACKOFF_SECONDS = 1.0      # 1, 2, 4, ... seconds
MAX_RETRY_AFTER_SECONDS = 60.0
DEFAULT_MAX_PAGES = 10_000         # hard stop: 500k articles at 50/page

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Response fields mapped into the canonical schema; everything else is
# kept in provider_metadata for provenance only.
MAPPED_FIELDS = {"id", "headline", "summary", "content", "symbols", "source", "url",
                 "created_at", "updated_at"}


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def alpaca_to_canonical(raw: Mapping[str, Any], fetched_at: datetime) -> NewsArticle:
    """Convert one Alpaca news item to a canonical NewsArticle."""
    if not isinstance(raw, Mapping):
        raise NewsValidationError("Alpaca news item is not an object")

    article_id = raw.get("id")
    if article_id is None or str(article_id).strip() == "":
        raise NewsValidationError("Alpaca news item has no id")
    article_id = str(article_id)

    created_at = parse_timestamp(raw.get("created_at"), f"created_at (article {article_id})")
    updated_raw = raw.get("updated_at")
    updated_at = parse_timestamp(updated_raw, f"updated_at (article {article_id})") if updated_raw else None

    symbols = raw.get("symbols") or []
    if not isinstance(symbols, list):
        raise NewsValidationError(f"symbols must be a list (article {article_id})")

    metadata = {k: v for k, v in raw.items() if k not in MAPPED_FIELDS}
    metadata["raw_created_at"] = raw.get("created_at")
    metadata["raw_updated_at"] = updated_raw

    return NewsArticle(
        provider=AlpacaNewsProvider.name,
        provider_article_id=article_id,
        headline=raw.get("headline") or "",
        summary=raw.get("summary"),
        content=raw.get("content"),
        symbols=tuple(symbols),
        source=raw.get("source"),
        source_url=raw.get("url"),
        created_at=created_at,
        updated_at=updated_at,
        information_available_at=historical_information_available_at(created_at, updated_at),
        fetched_at=fetched_at,
        provider_metadata=metadata,
    )


class AlpacaNewsProvider(NewsProvider):
    name = "alpaca"

    def __init__(
        self,
        api_key: str | None,
        api_secret: str | None,
        *,
        session: Any = None,
        base_url: str = ALPACA_NEWS_URL,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ):
        if not api_key or not api_secret:
            raise NewsAuthError(
                f"Alpaca credentials are missing: set {API_KEY_ENV} and {API_SECRET_ENV} in .env"
            )
        self.__api_key = api_key
        self.__api_secret = api_secret
        self._session = session or requests.Session()
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_env(cls, **kwargs) -> "AlpacaNewsProvider":
        return cls(os.getenv(API_KEY_ENV), os.getenv(API_SECRET_ENV), **kwargs)

    def __repr__(self) -> str:
        return f"AlpacaNewsProvider(base_url={self.base_url!r}, credentials=<redacted>)"

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.__api_key,
            "APCA-API-SECRET-KEY": self.__api_secret,
            "Accept": "application/json",
        }

    def _redact(self, text: str) -> str:
        for secret in (self.__api_key, self.__api_secret):
            text = text.replace(secret, "<redacted>")
        return text

    def _retry_delay(self, attempt: int, response=None) -> float:
        if response is not None:
            retry_after = (getattr(response, "headers", None) or {}).get("Retry-After")
            try:
                return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
            except (TypeError, ValueError):
                pass
        return self.backoff_seconds * (2 ** attempt)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """One API call with bounded retries for 429 / 5xx / timeouts / connection errors."""
        last_problem = "no attempt made"
        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self._session.get(self.base_url, params=params, headers=self._headers(),
                                             timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_problem = f"{type(e).__name__}: {self._redact(str(e))}"
            except requests.RequestException as e:
                # Not retryable (e.g. InvalidHeader). Its message can contain
                # header values - i.e. credentials - so it is neither included
                # nor chained (`from None` keeps it out of the traceback).
                raise NewsProviderError(
                    f"Alpaca request could not be sent ({type(e).__name__}); check {API_KEY_ENV} / "
                    f"{API_SECRET_ENV} in .env for stray whitespace or invalid characters"
                ) from None
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                    except ValueError as e:
                        raise NewsProviderError("Alpaca returned a non-JSON response") from e
                    if not isinstance(payload, dict):
                        raise NewsProviderError("Alpaca returned an unexpected JSON structure")
                    return payload
                body = self._redact(str(getattr(response, "text", "") or ""))[:200]
                if status in (401, 403):
                    raise NewsAuthError(f"Alpaca rejected the credentials (HTTP {status})")
                if status not in RETRYABLE_STATUS:
                    raise NewsProviderError(f"Alpaca request failed (HTTP {status}): {body}")
                last_problem = f"HTTP {status}: {body}"

            if attempt < self.max_retries:
                self._sleep(self._retry_delay(attempt, response))

        raise NewsProviderError(
            f"Alpaca request failed after {self.max_retries + 1} attempts ({last_problem})"
        )

    # ------------------------------------------------------------------
    # Historical fetch
    # ------------------------------------------------------------------

    def _page_size(self, query: NewsQuery) -> int:
        size = ALPACA_MAX_PAGE_SIZE if query.page_size is None else query.page_size
        if not 1 <= size <= ALPACA_MAX_PAGE_SIZE:
            raise ValueError(f"Alpaca page size must be between 1 and {ALPACA_MAX_PAGE_SIZE}, got {size}")
        return size

    def fetch_historical(self, query: NewsQuery, *, max_pages: int | None = DEFAULT_MAX_PAGES) -> NewsFetchResult:
        page_size = self._page_size(query)
        max_pages = DEFAULT_MAX_PAGES if max_pages is None else max_pages
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")

        base_params = {
            "symbols": ",".join(query.symbols),
            "start": _rfc3339(query.start),
            "end": _rfc3339(query.end),
            "limit": page_size,
            "sort": query.sort,
            "include_content": "true" if query.include_content else "false",
        }

        started_at = self._clock()
        articles: list[NewsArticle] = []
        seen_tokens: set[str] = set()
        token: str | None = None
        pages = 0

        while True:
            params = dict(base_params)
            if token:
                params["page_token"] = token

            payload = self._get(params)
            fetched_at = self._clock()
            items = payload.get("news")
            if not isinstance(items, list):
                raise NewsProviderError("Alpaca response has no 'news' list")

            articles.extend(alpaca_to_canonical(item, fetched_at) for item in items)
            pages += 1

            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen_tokens:
                raise NewsPaginationError(f"Alpaca returned a repeated page token after {pages} page(s)")
            seen_tokens.add(token)
            if pages >= max_pages:
                raise NewsPaginationError(
                    f"Stopped after max_pages={max_pages} with more results remaining; "
                    "narrow the date range or raise max_pages"
                )

        unique, n_duplicates, n_conflicts = deduplicate(articles)
        requested = set(query.symbols)

        provenance = {
            "provider": self.name,
            "endpoint": self.base_url,
            "request": {**query.describe(), "page_size_used": page_size, "max_pages": max_pages},
            "schema_version": NEWS_SCHEMA_VERSION,
            "availability_rule": HISTORICAL_AVAILABILITY_RULE,
            "fetch_started_at": started_at.isoformat(),
            "fetch_finished_at": self._clock().isoformat(),
            "pages": pages,
            "n_raw_articles": len(articles),
            "n_articles": len(unique),
            "n_duplicates_removed": n_duplicates,
            "n_conflicting_duplicates": n_conflicts,
            "n_created_outside_window": sum(
                1 for a in unique if not (query.start <= a.created_at <= query.end)),
            "n_without_requested_symbol": sum(1 for a in unique if not requested & set(a.symbols)),
        }
        return NewsFetchResult(articles=unique, provenance=provenance)
