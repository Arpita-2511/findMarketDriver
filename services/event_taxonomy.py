"""
Financial event taxonomy (Phase 6) - the ONLY place event types and
impacts are defined. Everything else imports from here.

taxonomy_v1 was derived from inspecting the historical AAPL news sample
(372 Benzinga headlines, 2017-01/02), not copied from a generic list:

- PRODUCT_LAUNCH / PRODUCT_UPDATE merged into PRODUCT: headlines rarely
  distinguish them (mostly rumours / reports about devices and services).
- MANAGEMENT_CHANGE widened to MANAGEMENT_GOVERNANCE: hires/departures plus
  shareholder meetings, proxy votes, board matters.
- OPERATIONS covers supply chain, suppliers, manufacturing and plants
  (a large group in the sample).
- OWNERSHIP added: 13F filings and famous-investor stakes (Berkshire,
  Einhorn, Tiger...) were frequent.
- MARKET_ACTIVITY added: option alerts, technical alerts, block trades,
  movers lists and market wraps. They describe trading/price action rather
  than a corporate event - kept separate so price-action reports are never
  mistaken for fundamental events.

Order = tie-break priority (earlier wins a tie).
"""

from __future__ import annotations

TAXONOMY_VERSION = "taxonomy_v1"

EVENT_TYPES: tuple[str, ...] = (
    "EARNINGS",
    "GUIDANCE",
    "M_AND_A",
    "LEGAL",
    "REGULATORY",
    "CAPITAL_ACTION",
    "ANALYST_RATING",
    "MANAGEMENT_GOVERNANCE",
    "PARTNERSHIP",
    "OWNERSHIP",
    "PRODUCT",
    "OPERATIONS",
    "MACRO",
    "MARKET_ACTIVITY",
    "OTHER",
)

NO_EVENT = "OTHER"                                   # no rule evidence
EVENT_DESCRIPTIONS: dict[str, str] = {
    "EARNINGS": "Reported results, EPS/revenue vs estimates, earnings previews and calls",
    "GUIDANCE": "Company outlook / forecast statements",
    "M_AND_A": "Acquisitions, mergers, takeovers, strategic investments in other companies",
    "LEGAL": "Lawsuits, litigation, patents, court rulings",
    "REGULATORY": "Regulators, antitrust, government policy/tax treatment specific to the company",
    "CAPITAL_ACTION": "Dividends, buybacks, bond/debt issuance, stock splits",
    "ANALYST_RATING": "Ratings, initiations, upgrades/downgrades, price targets, estimate changes",
    "MANAGEMENT_GOVERNANCE": "Executive hires/departures, board, shareholder meetings, proxy votes",
    "PARTNERSHIP": "Partnerships, collaborations, consortia, content/licensing deals",
    "OWNERSHIP": "Holdings of the stock by investors: 13F filings, stake changes",
    "PRODUCT": "Products and services: launches, updates, rumours, features",
    "OPERATIONS": "Supply chain, suppliers, manufacturing, plants, shipments",
    "MACRO": "Economy, politics, trade, central bank - not company-specific",
    "MARKET_ACTIVITY": "Trading/price action: option & technical alerts, block trades, movers, market wraps",
    "OTHER": "No taxonomy evidence (commentary, general features)",
}

EVENT_IMPACTS: tuple[str, ...] = ("POSITIVE", "NEGATIVE", "NEUTRAL")

# Phase 5A FinBERT label -> event impact (impact is NOT derived from the type)
IMPACT_FROM_SENTIMENT: dict[str, str] = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}

assert set(EVENT_DESCRIPTIONS) == set(EVENT_TYPES), "every event type needs a description"


class EventTaxonomyError(ValueError):
    """Unknown event type or impact."""


def validate_event_type(value: str) -> str:
    if value not in EVENT_TYPES:
        raise EventTaxonomyError(f"unknown event type {value!r}; valid: {EVENT_TYPES}")
    return value


def validate_event_impact(value: str) -> str:
    if value not in EVENT_IMPACTS:
        raise EventTaxonomyError(f"unknown event impact {value!r}; valid: {EVENT_IMPACTS}")
    return value


def impact_from_sentiment_label(label: str) -> str:
    try:
        return IMPACT_FROM_SENTIMENT[label]
    except KeyError:
        raise EventTaxonomyError(f"cannot map sentiment label {label!r} to an impact") from None


def type_feature_name(event_type: str) -> str:
    """Column name for a per-type daily count, e.g. EARNINGS -> earnings_event_count."""
    return f"{validate_event_type(event_type).lower()}_event_count"
