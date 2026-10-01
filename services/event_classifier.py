"""
Article-level financial event classification (Phase 6).

EventClassifier is the replaceable interface; KeywordEventClassifier is the
CURRENT BASELINE (rules_v1): weighted, case-insensitive regular-expression
rules per event type (services/event_taxonomy.py).

Why rules for v1: no labelled event data exists, so no model could be shown
to be better; rules are deterministic, instant on CPU, need no download or
new dependency, and every decision lists the rules that fired. A zero-shot
or fine-tuned classifier can replace it later behind the same interface.

Scoring:
    score(type)       = sum of weights of the distinct rules of that type that match
    event_type        = argmax score; ties -> earlier type in EVENT_TYPES;
                        no matching rule -> OTHER
    event_confidence  = score(event_type) / sum of all scores  (in (0, 1])
                        OTHER with no evidence -> 0.0

event_confidence is the SHARE OF MATCHED RULE EVIDENCE won by the chosen
type - a heuristic, NOT a calibrated probability and NOT accuracy.

Input text is the project text policy text_v1 (headline + summary,
services/finbert_sentiment.build_sentiment_text) - the same text FinBERT
scored, so both share input_text_hash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from services.event_taxonomy import EVENT_TYPES, NO_EVENT, validate_event_type

KEYWORD_CLASSIFIER_NAME = "keyword_event_classifier"
KEYWORD_CLASSIFIER_VERSION = "rules_v1"

# (rule id, pattern, weight). Patterns are matched case-insensitively on text_v1.
RULES: dict[str, tuple[tuple[str, str, int], ...]] = {
    "EARNINGS": (
        ("eps", r"\beps\b", 3),
        ("earnings", r"\bearnings\b", 2),
        ("quarterly_results", r"\bquarterly (results|revenue|earnings|sales)\b", 3),
        ("conference_call", r"\bconference call\b|\bconf\. call\b", 2),
        ("beat_miss", r"\b(beats?|tops?|miss(es)?|blows out)\b.{0,40}\b(estimates?|expectations|views|est\.?)", 2),
        ("quarter", r"\bq[1-4]\b", 1),
        ("results", r"\bresults\b", 1),
        ("units_reported", r"\bunits\b", 1),
    ),
    "GUIDANCE": (
        ("guidance", r"\bguidance\b", 3),
        ("sees_quarter", r"\bsees q[1-4]\b", 3),
        ("outlook", r"\boutlook\b", 2),
        ("forecast_change", r"\b(raises|cuts|lowers|reaffirms|reiterates)\b.{0,20}\b(forecast|outlook|guidance)\b", 3),
    ),
    "M_AND_A": (
        ("acquisition", r"\b(acquires?|acquired|acquisition|merger|takeover|buyout)\b", 3),
        ("could_buy", r"\b(could|to|would) buy\b", 2),
        ("strategic_investment", r"\binvestment (in|with)\b|\b(tech|vision) fund\b", 2),
    ),
    "LEGAL": (
        ("lawsuit", r"\b(lawsuits?|sues?|sued|suing|suit)\b", 3),
        ("litigation", r"\b(litigation|litigious)\b", 3),
        ("patent", r"\bpatents?\b", 2),
        ("court", r"\b(court|ruling|infringement|settlement)\b", 2),
        ("legal", r"\blegal\b", 2),
    ),
    "REGULATORY": (
        ("regulator", r"\b(antitrust|regulators?|regulatory|investigation)\b", 3),
        ("government", r"\b(government|minister|officials?|executive order)\b", 2),
        ("tax_policy", r"\b(tax (breaks|incentives|benefits)|incentives|concessions|customs duty|tariffs?)\b", 2),
    ),
    "CAPITAL_ACTION": (
        ("buyback", r"\b(buybacks?|repurchases?)\b", 3),
        ("dividend", r"\b(ex-)?dividends?\b", 3),
        ("debt", r"\b(bonds?|notes due|debt offering|borrow\w*)\b", 3),
        ("split", r"\bstock split\b", 3),
    ),
    "ANALYST_RATING": (
        ("rating_change", r"\b(upgrades?|upgraded|downgrades?|downgraded)\b", 3),
        ("initiation", r"\binitiates? coverage\b", 3),
        ("price_target", r"\bprice targets?\b|\braises pt\b", 3),
        ("rating_words", r"\b(overweight|underweight|equal weight|outperform|underperform)\b", 2),
        ("estimate_change", r"\b(raises|raising|cuts|lowers|trims?|trimmed)\b.{0,30}\bestimates\b", 2),
        ("analyst_stance", r"\bout (cautious|negative|positive)\b|\bcautious view\b|\btop pick\b", 2),
        ("analyst", r"\banalysts?\b", 1),
    ),
    "MANAGEMENT_GOVERNANCE": (
        ("personnel", r"\b(hires?|hired|appoint\w*|resign\w*|steps? down|poach\w*)\b", 3),
        ("governance", r"\b(shareholder meeting|shareholder mtg|annual meeting|proxy|board members?)\b", 3),
        ("executive", r"\b(ceo|cfo|coo|chief executive|executives?|execs)\b", 1),
    ),
    "PARTNERSHIP": (
        ("partner", r"\bpartner(s|ship|ships|ing)?\b", 3),
        ("collaboration", r"\b(teams? up|collaborat\w*|alliance|consortium|joins)\b", 2),
        ("content_deal", r"\bcontent deal\b", 2),
    ),
    "OWNERSHIP": (
        ("13f", r"\b13f\b", 3),
        ("stake", r"\bstakes?\b", 2),
        ("famous_investor", r"\b(berkshire|buffett|einhorn|icahn|tiger (global|management))\b", 2),
    ),
    "PRODUCT": (
        ("launch", r"\b(launch\w*|unveil\w*|debuts?|introduc\w*)\b", 2),
        ("device", r"\b(iphones?|ipads?|macbooks?|mac|airpods?|apple watch|apple tv|apple music|apple pay|ios|siri|itunes|app store)\b", 1),
        ("feature", r"\b(augmented reality|wireless charging|oled|next-gen|new model)\b", 1),
    ),
    "OPERATIONS": (
        ("supply_chain", r"\b(suppliers?|supply chain|supplied|supply)\b", 3),
        ("manufacturing", r"\b(manufactur\w*|assembl\w*|factory|plant|production|oem|foxconn|jabil|wistron)\b", 3),
        ("shipments", r"\b(shipments?|orders|components?|panels?)\b", 1),
    ),
    "MACRO": (
        ("macro_policy", r"\b(trump|federal reserve|fed meeting|yellen|jobless|inflation|interest rates?|trade war|currency|gdp|economic data|immigration)\b", 2),
        ("macro_broad", r"\b(tariffs?|repatriat\w*|jobs)\b", 1),
    ),
    "MARKET_ACTIVITY": (
        ("alerts", r"\b(option alert|technical alert|block trade|unusual options|(call|put) purchases)\b", 3),
        ("market_wrap", r"\b(the market in 5 minutes|market update|stocks moving|must watch stocks|stocks you should be watching|biggest mid-day|premarket prep)\b", 3),
        ("session_moves", r"\b(premarket|pre-market|after-hours|session highs|all-time high|futures)\b", 1),
    ),
}

assert set(RULES) == set(EVENT_TYPES) - {NO_EVENT}, "every event type except OTHER needs rules"

_COMPILED = {t: tuple((rid, re.compile(p, re.IGNORECASE), w) for rid, p, w in RULES[t]) for t in RULES}


@dataclass(frozen=True)
class EventPrediction:
    event_type: str
    event_confidence: float                 # evidence share, NOT a probability
    matched_rules: tuple[str, ...]          # e.g. ("EARNINGS:eps", "EARNINGS:quarter")

    def __post_init__(self):
        validate_event_type(self.event_type)
        if not (isinstance(self.event_confidence, float) and 0.0 <= self.event_confidence <= 1.0):
            raise ValueError(f"event_confidence must be a float in [0, 1], got {self.event_confidence!r}")
        if self.event_type == NO_EVENT and self.event_confidence != 0.0:
            raise ValueError("OTHER (no evidence) must have event_confidence 0.0")


class EventClassifier(Protocol):
    name: str
    version: str

    def classify_texts(self, texts: Sequence[str]) -> list[EventPrediction]:
        """One prediction per text, in input order."""


class KeywordEventClassifier:
    """CURRENT BASELINE classifier (rules_v1). Deterministic and dependency-free."""

    name = KEYWORD_CLASSIFIER_NAME
    version = KEYWORD_CLASSIFIER_VERSION

    def classify_text(self, text: str) -> EventPrediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("cannot classify empty text")
        scores: dict[str, int] = {}
        matched: list[str] = []
        for event_type in EVENT_TYPES:
            for rule_id, pattern, weight in _COMPILED.get(event_type, ()):
                if pattern.search(text):
                    scores[event_type] = scores.get(event_type, 0) + weight
                    matched.append(f"{event_type}:{rule_id}")
        if not scores:
            return EventPrediction(NO_EVENT, 0.0, ())
        best = max(EVENT_TYPES[:-1], key=lambda t: (scores.get(t, 0), -EVENT_TYPES.index(t)))
        return EventPrediction(best, scores[best] / sum(scores.values()), tuple(matched))

    def classify_texts(self, texts: Sequence[str]) -> list[EventPrediction]:
        return [self.classify_text(t) for t in texts]


class ConstantEventClassifier:
    """Majority-class style baseline: always predicts one type (for evaluation comparisons)."""

    name = "constant_event_classifier"
    version = "constant_v1"

    def __init__(self, event_type: str = NO_EVENT):
        self.event_type = validate_event_type(event_type)

    def classify_texts(self, texts: Sequence[str]) -> list[EventPrediction]:
        conf = 0.0 if self.event_type == NO_EVENT else 1.0
        return [EventPrediction(self.event_type, conf, ()) for _ in texts]
