"""
Canonical, provider-independent news schema (Phase 4A, schema news_v1).

Every NewsProvider converts its response into NewsArticle. Nothing
downstream (trading-date assignment, sentiment, features) may depend on
provider-specific fields; ML code uses NewsArticle.ml_view() only.

Timestamps (ARCHITECTURE.md sections 11.2 and 12):

    created_at                provider publication time
    updated_at                provider last-update time (optional)
    fetched_at                when we retrieved the record
    information_available_at  earliest time the STORED version may be
                              used as model information

All are timezone-aware and normalized to UTC; naive datetimes are
rejected. updated_at never replaces created_at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

NEWS_SCHEMA_VERSION = "news_v1"

# Historical backfill returns the LATEST version of each article, so its
# text is only proven to exist from updated_at onwards.
HISTORICAL_AVAILABILITY_RULE = "historical_backfill_v1: information_available_at = max(created_at, updated_at)"

# The only fields ML code may consume - no provider name, no provider
# metadata, no retrieval bookkeeping.
ML_FACING_FIELDS = (
    "article_key",
    "headline",
    "summary",
    "content",
    "symbols",
    "source",
    "created_at",
    "information_available_at",
)


class NewsValidationError(ValueError):
    """A record violates the canonical news schema."""


def ensure_utc(value: datetime, name: str) -> datetime:
    """Reject naive datetimes; convert aware ones to UTC."""
    if not isinstance(value, datetime):
        raise NewsValidationError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NewsValidationError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def parse_timestamp(value: Any, name: str) -> datetime:
    """Parse an ISO-8601 / RFC 3339 string (e.g. '2021-12-31T11:08:42Z') to aware UTC."""
    if isinstance(value, datetime):
        return ensure_utc(value, name)
    if not isinstance(value, str) or not value.strip():
        raise NewsValidationError(f"{name} is missing")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as e:
        raise NewsValidationError(f"{name} is not an ISO-8601 timestamp: {value!r}") from e
    return ensure_utc(parsed, name)


def historical_information_available_at(created_at: datetime, updated_at: datetime | None) -> datetime:
    """Apply HISTORICAL_AVAILABILITY_RULE."""
    created_at = ensure_utc(created_at, "created_at")
    if updated_at is None:
        return created_at
    return max(created_at, ensure_utc(updated_at, "updated_at"))


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NewsValidationError(f"{name} must be a string or None")
    return value if value.strip() else None


@dataclass(frozen=True)
class NewsArticle:
    provider: str
    provider_article_id: str
    headline: str
    symbols: tuple[str, ...]
    created_at: datetime
    information_available_at: datetime
    fetched_at: datetime
    summary: str | None = None
    content: str | None = None
    source: str | None = None
    source_url: str | None = None
    updated_at: datetime | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise NewsValidationError("provider is required")
        if not isinstance(self.provider_article_id, str) or not self.provider_article_id.strip():
            raise NewsValidationError("provider_article_id is required")
        if not isinstance(self.headline, str) or not self.headline.strip():
            raise NewsValidationError(f"headline is required (article {self.provider_article_id})")

        if isinstance(self.symbols, str) or not all(isinstance(s, str) and s.strip() for s in self.symbols):
            raise NewsValidationError("symbols must be a sequence of non-empty strings")
        object.__setattr__(self, "symbols", tuple(sorted({s.strip().upper() for s in self.symbols})))

        for name in ("summary", "content", "source", "source_url"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))

        for name in ("created_at", "information_available_at", "fetched_at"):
            object.__setattr__(self, name, ensure_utc(getattr(self, name), name))
        if self.updated_at is not None:
            object.__setattr__(self, "updated_at", ensure_utc(self.updated_at, "updated_at"))

        if self.information_available_at < self.created_at:
            raise NewsValidationError(
                f"information_available_at precedes created_at (article {self.provider_article_id})"
            )
        if not isinstance(self.provider_metadata, Mapping):
            raise NewsValidationError("provider_metadata must be a mapping")

    @property
    def key(self) -> tuple[str, str]:
        """Deduplication key."""
        return (self.provider, self.provider_article_id)

    @property
    def article_key(self) -> str:
        """Opaque stable identifier (not a feature)."""
        return f"{self.provider}:{self.provider_article_id}"

    def ml_view(self) -> dict[str, Any]:
        """The ONLY representation ML code may consume."""
        return {name: getattr(self, name) for name in ML_FACING_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (ISO-8601 UTC timestamps)."""
        def iso(d):
            return d.isoformat() if d is not None else None

        return {
            "provider": self.provider,
            "provider_article_id": self.provider_article_id,
            "headline": self.headline,
            "summary": self.summary,
            "content": self.content,
            "symbols": list(self.symbols),
            "source": self.source,
            "source_url": self.source_url,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "information_available_at": iso(self.information_available_at),
            "fetched_at": iso(self.fetched_at),
            "provider_metadata": dict(self.provider_metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NewsArticle":
        updated = d.get("updated_at")
        return cls(
            provider=d["provider"],
            provider_article_id=d["provider_article_id"],
            headline=d["headline"],
            summary=d.get("summary"),
            content=d.get("content"),
            symbols=tuple(d.get("symbols") or ()),
            source=d.get("source"),
            source_url=d.get("source_url"),
            created_at=parse_timestamp(d["created_at"], "created_at"),
            updated_at=parse_timestamp(updated, "updated_at") if updated else None,
            information_available_at=parse_timestamp(d["information_available_at"], "information_available_at"),
            fetched_at=parse_timestamp(d["fetched_at"], "fetched_at"),
            provider_metadata=dict(d.get("provider_metadata") or {}),
        )
