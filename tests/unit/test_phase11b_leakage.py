"""
Phase 11B anti-leakage tests (T1-T15) with a three-gate execution schedule.

Gates (Amendment 03, data/results/research/phase11/11B/amendment_03_test_gates.json)

  Gate A  must pass before Stage 11B.1 news ingestion
          synthetic fixtures and already-frozen 11A / 11.0 / 11.1 artifacts
  Gate B  must pass before Stage 11B.2 FinBERT scoring
          runs against the newly ingested multi-symbol news dataset
  Gate C  must pass before Stage 11B.4 research matrix
          runs against scored articles, event-classified articles,
          and the 29 built 11B feature frames

Status codes
  PASS                       test ran, all assertions hold
  BLOCKED_NOT_YET_ELIGIBLE   skipped, prerequisite artifact absent;
                             treated as 'does NOT clear the gate'
  FAIL                       at least one assertion failed; pipeline stops

How to run (see end-of-file module docstring):
  Gate A alone (before news download):
      pytest tests/unit/test_phase11b_leakage.py::test_gate_a_is_clear -v
  Gate B alone (after Stage 11B.1 writes news_integrity_report.json):
      pytest tests/unit/test_phase11b_leakage.py::test_gate_b_is_clear -v
  Gate C alone (after Stage 11B.3 writes feature_verification.json):
      pytest tests/unit/test_phase11b_leakage.py::test_gate_c_is_clear -v
  Full file with status visibility (BLOCKED tests show as SKIPPED):
      pytest tests/unit/test_phase11b_leakage.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from services.event_classifier import KeywordEventClassifier
from services.event_features import (
    EVENT_FEATURE_COLUMNS,
    ClassifiedArticle,
    generate_event_features,
)
from services.finbert_sentiment import (
    INFERENCE_VERSION,
    LABELS,
    SentimentResult,
    build_sentiment_text,
    text_hash,
)
from services.market_calendar_service import (
    MARKET_TZ,
    REGULAR_SESSION_CLOSE,
    SETTLEMENT_DELAY,
    TradingCalendar,
    completion_time,
)
from services.news_alignment import is_eligible, row_prediction_timestamp
from services.news_schema import NewsArticle, historical_information_available_at
from services.sentiment_features import (
    FEATURE_COLUMNS as SF_COLUMNS,
    ScoredArticle,
    generate_sentiment_features,
)
from services.sentiment_window import (
    COLUMNS as SW_COLUMNS,
    generate_sentiment_window_features,
)

import training.phase10a_research as p10a
import training.phase11_design as p11d
import training.phase11a_research as a11

# ============================================================
# Paths and frozen constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE_11B_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "11B"
DESIGN_FREEZE_PATH = PHASE_11B_DIR / "design_freeze.json"
AMENDMENT_03_PATH = PHASE_11B_DIR / "amendment_03_test_gates.json"
NEWS_INTEGRITY_REPORT = PHASE_11B_DIR / "news_integrity_report.json"
FEATURE_VERIFICATION_PATH = PHASE_11B_DIR / "feature_verification.json"

PHASE_11_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase11"
PHASE_11A_REGISTRY_PATH = PHASE_11_DIR / "11A" / "registry.json"
AMENDMENT_01_PATH = PHASE_11_DIR / "amendment_01_reproducibility.json"
PRICE_VERIFICATION_PATH = PHASE_11_DIR / "price_verification.json"

NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"
SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
EVENT_DIR = PROJECT_ROOT / "data" / "processed" / "events"

# Frozen SHA constants (imported from the Phase 11A module to prevent drift)
FROZEN_DESIGN_SHA = a11.FROZEN_DESIGN_SHA
AMENDMENT_01_SHA = a11.AMENDMENT_SHA
PRICE_REPORT_SHA = a11.PRICE_REPORT_SHA

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
FETCHED = datetime(2026, 10, 1, tzinfo=UTC)

# Phase 11B frozen boundaries
T_DEV = datetime(2024, 9, 30, 20, 30, 0, tzinfo=UTC)
B_MERGER = datetime(2020, 4, 3, 4, 0, 0, tzinfo=UTC)

# Phase 11B matrix invariants
EXPECTED_PER_STOCK = 3828
EXPECTED_PANEL = 132
EXPECTED_PANEL_EX_AAPL = 132
EXPECTED_FINAL_PANEL_FAMILY = 198
EXPECTED_FINAL_PER_STOCK_FAMILY = 5742
EXPECTED_11A_PANEL = 66
EXPECTED_11A_PER_STOCK = 1914

ELEVEN_B_FEATURE_SETS = ("B", "C", "D", "F")
BLOCKED_REASON = "BLOCKED_NOT_YET_ELIGIBLE"

_CLASSIFIER = KeywordEventClassifier()


# ============================================================
# Builders: synthetic articles, calendars, trading dates
# ============================================================


def _make_article(
    i: int,
    created_at: datetime,
    symbols: tuple = ("AAPL",),
    updated: datetime | None = None,
    headline: str | None = None,
) -> NewsArticle:
    return NewsArticle(
        provider="alpaca",
        provider_article_id=str(i),
        headline=headline or f"Story {i}",
        symbols=symbols,
        created_at=created_at,
        updated_at=updated,
        information_available_at=historical_information_available_at(created_at, updated),
        fetched_at=FETCHED,
    )


def _make_scored(
    i: int,
    created_at: datetime,
    probs: tuple = (0.7, 0.1, 0.2),
    symbols: tuple = ("AAPL",),
    updated: datetime | None = None,
    headline: str | None = None,
) -> ScoredArticle:
    article = _make_article(i, created_at, symbols=symbols, updated=updated, headline=headline)
    pos, neg, neu = probs
    by_label = {"positive": pos, "negative": neg, "neutral": neu}
    label = max(LABELS, key=lambda lbl: (by_label[lbl], -LABELS.index(lbl)))
    sr = SentimentResult(
        inference_version=INFERENCE_VERSION,
        model_name="ProsusAI/finbert",
        model_revision="rev",
        label=label,
        positive_probability=pos,
        negative_probability=neg,
        neutral_probability=neu,
        sentiment_score=pos - neg,
        input_text_hash=text_hash(build_sentiment_text(article)),
    )
    return ScoredArticle(article, sr)


def _make_classified(
    i: int,
    created_at: datetime,
    symbols: tuple = ("AAPL",),
    updated: datetime | None = None,
    headline: str | None = None,
    probs: tuple = (0.2, 0.1, 0.7),
) -> ClassifiedArticle:
    scored = _make_scored(i, created_at, probs=probs, symbols=symbols, updated=updated, headline=headline)
    pred = _CLASSIFIER.classify_text(build_sentiment_text(scored.article))
    return ClassifiedArticle(scored, pred)


def _trading_dates(start: date, n: int) -> list[date]:
    """n consecutive weekday dates starting at or after start (weekend-only exclusion)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_calendar(dates) -> TradingCalendar:
    return TradingCalendar(list(dates))


def _ts(d: date) -> datetime:
    """Row prediction timestamp for a session date, as UTC. Mirrors row_prediction_timestamp."""
    return completion_time(d).astimezone(UTC)


# ============================================================
# RTX / UTX mapping reference (contract for phase11b_research.py)
# ============================================================


def _map_symbols_for_rtx(symbols: tuple, created_at: datetime) -> tuple:
    """
    Reference implementation of the Phase 11B UTX->RTX attribution rule.

    - UTX tag before B_MERGER   -> attribute to RTX
    - RTX tag on or after       -> attribute to RTX
    - RTX tag before            -> excluded (off-period)
    - UTX tag on or after       -> excluded (off-period)
    - RTN tag                   -> never substituted for RTX

    When both UTX and RTX tags appear on one article, the article is counted
    once for RTX (never twice). The output symbol tuple removes UTX/RTX from
    the input and re-adds RTX only when the above conditions make the article
    eligible, so downstream attribution via `article.symbols` naturally
    produces a single contribution.

    phase11b_research.py MUST implement an equivalent transformation.
    """
    s = set(symbols)
    has_utx = "UTX" in s
    has_rtx = "RTX" in s
    before = created_at < B_MERGER
    result = set(s) - {"UTX", "RTX"}
    if (has_utx and before) or (has_rtx and not before):
        result.add("RTX")
    return tuple(sorted(result))


# ============================================================
# Gate readiness
# ============================================================


def _gate_b_eligible() -> bool:
    return NEWS_INTEGRITY_REPORT.exists()


def _gate_c_eligible() -> bool:
    return FEATURE_VERIFICATION_PATH.exists()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# T1 - prefix invariance
# ============================================================


def _assert_t1_prefix_invariance_synthetic():
    """Adding articles with avail > ts(D) must not change any row <= D, for all three generators."""
    dates = _trading_dates(date(2024, 7, 8), 5)   # Mon-Fri, no US holidays
    cal = _make_calendar(dates)
    d0, d1, d2, d3, d4 = dates
    early_dates = [d0, d1, d2]

    # Articles whose avail <= ts(d2) (eligible for row d2 and earlier)
    base_scored = [
        _make_scored(1, datetime(2024, 7, 8, 10, 0, tzinfo=UTC), symbols=("AAPL",)),
        _make_scored(2, datetime(2024, 7, 9, 10, 0, tzinfo=UTC), symbols=("AAPL",)),
        _make_scored(3, _ts(d2), symbols=("AAPL",)),      # exactly at ts(d2) - eligible
    ]
    # Article with avail > ts(d2) - must not alter rows d0..d2
    future_scored = _make_scored(99, _ts(d2) + timedelta(hours=1), symbols=("AAPL",))

    # --- sentiment_features_v1 ---
    rb = generate_sentiment_features(base_scored, dates, cal, ["AAPL"])
    re = generate_sentiment_features(base_scored + [future_scored], dates, cal, ["AAPL"])
    rb_by = {r.trading_date: r for r in rb}
    re_by = {r.trading_date: r for r in re}
    for d in early_dates:
        for col in SF_COLUMNS:
            vb, ve = getattr(rb_by[d], col), getattr(re_by[d], col)
            assert vb == ve, (
                f"T1 sentiment_features_v1: col {col} on row {d} changed when "
                f"a future article was added (base={vb}, extended={ve})"
            )

    # --- sentiment_window_v1 ---
    cov_start = datetime(2024, 6, 1, tzinfo=UTC)
    cov_end = datetime(2024, 9, 30, 20, 30, 0, tzinfo=UTC)
    sb = generate_sentiment_window_features(
        base_scored, dates, cal, "AAPL",
        coverage_start=cov_start, coverage_end=cov_end,
    )
    se = generate_sentiment_window_features(
        base_scored + [future_scored], dates, cal, "AAPL",
        coverage_start=cov_start, coverage_end=cov_end,
    )
    sb_early = sb[sb["trading_date"].isin(early_dates)].sort_values("trading_date").reset_index(drop=True)
    se_early = se[se["trading_date"].isin(early_dates)].sort_values("trading_date").reset_index(drop=True)
    for col in SW_COLUMNS:
        pd.testing.assert_series_equal(
            sb_early[col], se_early[col], check_names=False,
            obj=f"T1 sentiment_window_v1: col {col} changed when a future article was added",
        )

    # --- event_features_v1 ---
    base_classified = [
        _make_classified(10, datetime(2024, 7, 8, 10, 0, tzinfo=UTC), symbols=("AAPL",)),
        _make_classified(11, datetime(2024, 7, 9, 10, 0, tzinfo=UTC), symbols=("AAPL",)),
        _make_classified(12, _ts(d2), symbols=("AAPL",)),
    ]
    future_classified = _make_classified(99, _ts(d2) + timedelta(hours=1), symbols=("AAPL",))
    eb = generate_event_features(base_classified, dates, cal, ["AAPL"])
    ee = generate_event_features(base_classified + [future_classified], dates, cal, ["AAPL"])
    eb_by = {r["trading_date"]: r for r in eb}
    ee_by = {r["trading_date"]: r for r in ee}
    for d in early_dates:
        for col in EVENT_FEATURE_COLUMNS:
            vb, ve = eb_by[d][col], ee_by[d][col]
            assert vb == ve, (
                f"T1 event_features_v1: col {col} on row {d} changed when a "
                f"future article was added (base={vb}, extended={ve})"
            )


def _assert_t1_prefix_invariance_real():
    """
    Real-data prefix invariance: Stage 11B.3 must have run and recorded the
    outcome in feature_verification.json.
    """
    data = _load_json(FEATURE_VERIFICATION_PATH)
    assert data.get("prefix_invariance_check_passed") is True, (
        "T1-real: feature_verification.json must set prefix_invariance_check_passed=true; "
        f"got {data.get('prefix_invariance_check_passed')!r}"
    )


# ============================================================
# T2 - boundary exact
# ============================================================


def _assert_t2_boundary_exact():
    """avail == ts(D) belongs to row D; avail == ts(D)+1us belongs to row D+1."""
    d = date(2024, 7, 10)      # Wed
    d1 = date(2024, 7, 11)     # Thu
    cal = _make_calendar([d, d1])
    ts_d = _ts(d)

    a_at = _make_scored(1, ts_d, symbols=("AAPL",))
    assert a_at.article.information_available_at == ts_d, "T2 setup: avail must equal ts(D) exactly"
    assert is_eligible(a_at.article, ts_d), "T2: article at ts(D) must be eligible for row D"

    rows_at = generate_sentiment_features([a_at], [d], cal, ["AAPL"])
    assert rows_at[0].news_count == 1, "T2: article at ts(D) must appear in row D (count=1)"

    a_after = _make_scored(2, ts_d + timedelta(microseconds=1), symbols=("AAPL",))
    assert not is_eligible(a_after.article, ts_d), (
        "T2: article at ts(D)+1us must NOT be eligible for row D"
    )

    rows_both = generate_sentiment_features([a_after], [d, d1], cal, ["AAPL"])
    rows_by = {r.trading_date: r for r in rows_both}
    assert rows_by[d].news_count == 0, "T2: article at ts(D)+1us must NOT appear in row D"
    assert rows_by[d1].news_count == 1, "T2: article at ts(D)+1us must appear in row D+1"


# ============================================================
# T3 - revision semantics
# ============================================================


def _assert_t3_revision_semantics():
    """created<=ts(D) with updated>ts(D) is NOT in row D; first appears at revision time."""
    d = date(2024, 7, 10)      # Wed
    d1 = date(2024, 7, 11)     # Thu
    d2 = date(2024, 7, 12)     # Fri
    cal = _make_calendar([d, d1, d2])

    created = datetime(2024, 7, 10, 9, 0, tzinfo=UTC)      # before ts(d) = 20:30 UTC
    updated = datetime(2024, 7, 11, 9, 0, tzinfo=UTC)      # after ts(d), before ts(d1)
    assert created < _ts(d), "T3 setup: created must be before ts(D)"
    assert updated > _ts(d), "T3 setup: updated must be after ts(D)"
    assert updated < _ts(d1), "T3 setup: updated must be before ts(D+1)"

    avail = historical_information_available_at(created, updated)
    assert avail == updated, "T3 setup: information_available_at must equal max(created, updated) == updated"

    a = _make_scored(1, created, updated=updated, symbols=("AAPL",))
    assert a.article.information_available_at == updated

    assert not is_eligible(a.article, _ts(d)), "T3: revised article must not be eligible for row D"
    assert is_eligible(a.article, _ts(d1)), "T3: revised article must be eligible for row D+1 (first appearance)"

    rows = generate_sentiment_features([a], [d, d1, d2], cal, ["AAPL"])
    rows_by = {r.trading_date: r for r in rows}
    assert rows_by[d].news_count == 0, "T3: row D must not contain the revised article"
    assert rows_by[d1].news_count == 1, "T3: row D+1 must contain the revised article (cumulative)"
    assert rows_by[d2].news_count == 1, "T3: row D+2 cumulative count stays at 1"


# ============================================================
# T4 - DST handling (March and November transitions)
# ============================================================


def _assert_t4_dst_handling():
    """ts(D) in UTC: 21:30Z in standard time, 20:30Z in daylight time; both DST edges covered."""
    # Spring 2024: DST begins 2024-03-10 02:00 local
    d_std = date(2024, 3, 8)       # Fri, EST (UTC-5), ts = 21:30Z
    d_dst = date(2024, 3, 11)      # Mon, EDT (UTC-4), ts = 20:30Z
    ts_std = _ts(d_std)
    ts_dst = _ts(d_dst)
    assert ts_std.hour == 21 and ts_std.minute == 30, (
        f"T4 spring standard: ts(2024-03-08) must be 21:30Z, got {ts_std.isoformat()}"
    )
    assert ts_dst.hour == 20 and ts_dst.minute == 30, (
        f"T4 spring daylight: ts(2024-03-11) must be 20:30Z, got {ts_dst.isoformat()}"
    )

    # Fall 2024: DST ends 2024-11-03 02:00 local
    d_dst_fall = date(2024, 11, 1)   # Fri, EDT, ts = 20:30Z
    d_std_fall = date(2024, 11, 4)   # Mon, EST, ts = 21:30Z
    ts_dst_fall = _ts(d_dst_fall)
    ts_std_fall = _ts(d_std_fall)
    assert ts_dst_fall.hour == 20 and ts_dst_fall.minute == 30, (
        f"T4 fall daylight: ts(2024-11-01) must be 20:30Z, got {ts_dst_fall.isoformat()}"
    )
    assert ts_std_fall.hour == 21 and ts_std_fall.minute == 30, (
        f"T4 fall standard: ts(2024-11-04) must be 21:30Z, got {ts_std_fall.isoformat()}"
    )

    # Eligibility edge: an article exactly at ts_std belongs to d_std (not the prior session)
    cal = _make_calendar([date(2024, 3, 7), d_std, d_dst])
    a_std = _make_scored(1, ts_std, symbols=("AAPL",))
    rows = generate_sentiment_features([a_std], [date(2024, 3, 7), d_std], cal, ["AAPL"])
    rows_by = {r.trading_date: r for r in rows}
    assert rows_by[date(2024, 3, 7)].news_count == 0, "T4: article at ts(d_std) not in prior row"
    assert rows_by[d_std].news_count == 1, "T4: article at ts(d_std) belongs to row d_std"

    a_dst = _make_scored(2, ts_dst, symbols=("AAPL",))
    rows_dst = generate_sentiment_features([a_dst], [d_dst], cal, ["AAPL"])
    assert rows_dst[0].news_count == 1, "T4: article at ts(d_dst) belongs to row d_dst"


# ============================================================
# T5 - sealing (synthetic + real)
# ============================================================


def _assert_t5_seal_synthetic():
    """Development subset (avail <= T_dev) and sealed subset (avail > T_dev) are disjoint."""
    articles = [
        _make_article(1, datetime(2024, 9, 28, 10, 0, tzinfo=UTC), symbols=("AAPL",)),
        _make_article(2, T_DEV, symbols=("AAPL",)),                                       # on cutoff
        _make_article(3, T_DEV + timedelta(microseconds=1), symbols=("AAPL",)),           # just after
        _make_article(4, datetime(2024, 10, 2, 0, 0, tzinfo=UTC), symbols=("AAPL",)),
        # created before cutoff but revised after - must land in SEALED (avail > T_dev)
        _make_article(
            5,
            datetime(2024, 9, 20, 10, 0, tzinfo=UTC),
            symbols=("AAPL",),
            updated=datetime(2024, 10, 5, 10, 0, tzinfo=UTC),
        ),
    ]
    dev = [a for a in articles if a.information_available_at <= T_DEV]
    sealed = [a for a in articles if a.information_available_at > T_DEV]

    dev_ids = {a.article_key for a in dev}
    sealed_ids = {a.article_key for a in sealed}
    assert dev_ids.isdisjoint(sealed_ids), "T5-synthetic: development and sealed sets must be disjoint"

    assert "alpaca:1" in dev_ids, "T5: early article must be in development"
    assert "alpaca:2" in dev_ids, "T5: article exactly at T_dev must be in development"
    assert "alpaca:3" in sealed_ids, "T5: article at T_dev+1us must be in sealed"
    assert "alpaca:4" in sealed_ids, "T5: late article must be in sealed"
    assert "alpaca:5" in sealed_ids, (
        "T5: created-before-cutoff-but-revised-after must be in sealed "
        "(sealed definition includes revised-after cases)"
    )


def _assert_t5_seal_real():
    """Stage 11B.1 must produce news_integrity_report.json confirming the sealing split."""
    report = _load_json(NEWS_INTEGRITY_REPORT)
    assert report.get("split_applied_before_scoring") is True, (
        "T5-real: news_integrity_report.json must set split_applied_before_scoring=true"
    )
    assert report.get("development_sealed_intersection_empty") is True, (
        "T5-real: news_integrity_report.json must set development_sealed_intersection_empty=true"
    )
    dev_count = report.get("development_article_count")
    sealed_count = report.get("sealed_article_count")
    assert isinstance(dev_count, int) and dev_count > 0, (
        f"T5-real: development_article_count must be a positive int, got {dev_count!r}"
    )
    assert isinstance(sealed_count, int) and sealed_count >= 0, (
        f"T5-real: sealed_article_count must be a non-negative int, got {sealed_count!r}"
    )
    for key in ("development_article_sha256", "sealed_article_id_sha256"):
        v = report.get(key)
        assert isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v), (
            f"T5-real: {key} must be a lowercase 64-char hex sha256, got {v!r}"
        )


# ============================================================
# T6 - attribution correctness
# ============================================================


def _assert_t6_attribution_correctness():
    """Only tagged stocks receive the article; multi-symbol articles reach each tag exactly once."""
    dates = [date(2024, 7, 10)]
    cal = _make_calendar(dates)
    when = datetime(2024, 7, 10, 10, 0, tzinfo=UTC)

    a_multi = _make_scored(1, when, symbols=("AAPL", "MSFT"))                               # both
    a_aapl = _make_scored(2, when + timedelta(hours=1), symbols=("AAPL",))                  # AAPL only
    a_untagged_gs = _make_scored(
        3, when + timedelta(hours=2),
        symbols=("AAPL",),
        headline="Goldman Sachs beats expectations",   # mentions GS, tagged AAPL only
    )

    rows = generate_sentiment_features([a_multi, a_aapl, a_untagged_gs], dates, cal, ["AAPL", "MSFT", "GS"])
    by_sym = {r.symbol: r for r in rows}

    assert by_sym["AAPL"].news_count == 3, (
        f"T6: AAPL must receive all 3 articles tagged with AAPL, got {by_sym['AAPL'].news_count}"
    )
    assert by_sym["MSFT"].news_count == 1, (
        f"T6: MSFT must receive only the multi-symbol article, got {by_sym['MSFT'].news_count}"
    )
    assert by_sym["GS"].news_count == 0, (
        f"T6: GS must receive 0 articles (not tagged, headline mention does not attribute), "
        f"got {by_sym['GS'].news_count}"
    )

    # Multi-symbol contributes exactly once to each tagged stock
    rows_msft = generate_sentiment_features([a_multi], dates, cal, ["MSFT"])
    assert rows_msft[0].news_count == 1, "T6: multi-symbol article must contribute exactly 1 to MSFT (no double count)"


# ============================================================
# T7 - RTX / UTX mapping exhaustive
# ============================================================


def _assert_t7_rtx_utx_mapping():
    """All 7 cases from the design freeze rtx_utx_mapping.rule hold on the reference implementation."""
    before = B_MERGER - timedelta(hours=1)
    on_boundary = B_MERGER
    after = B_MERGER + timedelta(hours=1)

    cases = [
        ("case 1: UTX before -> RTX",                 ("UTX",),        before,       True),
        ("case 2: RTX on boundary -> RTX",            ("RTX",),        on_boundary,  True),
        ("case 2: RTX after boundary -> RTX",         ("RTX",),        after,        True),
        ("case 3: RTX before -> excluded",            ("RTX",),        before,       False),
        ("case 4: UTX on boundary -> excluded",       ("UTX",),        on_boundary,  False),
        ("case 4: UTX after boundary -> excluded",    ("UTX",),        after,        False),
        ("case 5: RTN only -> excluded (before)",     ("RTN",),        before,       False),
        ("case 5: RTN only -> excluded (after)",      ("RTN",),        after,        False),
        ("case 6: RTN+UTX before -> once via UTX",    ("RTN", "UTX"),  before,       True),
        ("case 7: UTX+RTX before -> once",            ("UTX", "RTX"),  before,       True),
        ("case 7: UTX+RTX after -> once",             ("UTX", "RTX"),  after,        True),
    ]
    for desc, symbols, created_at, expected_rtx in cases:
        mapped = _map_symbols_for_rtx(symbols, created_at)
        got = "RTX" in mapped
        assert got == expected_rtx, (
            f"T7 {desc}: expected RTX_eligible={expected_rtx}, mapped={mapped}"
        )
        # Neither UTX nor the mapping artefact appears post-mapping
        assert "UTX" not in mapped, f"T7 {desc}: UTX must be stripped by mapping, got {mapped}"
        # "Counted once": RTX appears at most once in the symbol tuple
        assert mapped.count("RTX") <= 1, f"T7 {desc}: RTX must appear at most once, got {mapped}"

    # Case 5 preserves non-RTX tags that aren't UTX/RTX, so RTN survives in symbols
    mapped_rtn = _map_symbols_for_rtx(("RTN",), before)
    assert mapped_rtn == ("RTN",), f"T7: RTN-only must keep RTN and add nothing, got {mapped_rtn}"

    # Case 6 (RTN + UTX before): RTN kept, UTX replaced by RTX
    mapped_6 = _map_symbols_for_rtx(("RTN", "UTX"), before)
    assert set(mapped_6) == {"RTN", "RTX"}, f"T7 case 6: expected {{RTN,RTX}}, got {mapped_6}"

    # Generator-level "counted once": apply mapping, then run through the sentiment generator
    dates = [date(2020, 3, 2)]       # before boundary
    cal = _make_calendar(dates)
    when = datetime(2020, 3, 2, 10, 0, tzinfo=UTC)
    assert when < B_MERGER
    raw_symbols = ("AAPL", "UTX", "RTX")             # pathological: both UTX AND RTX before boundary
    mapped_syms = _map_symbols_for_rtx(raw_symbols, when)
    assert "RTX" in mapped_syms and "UTX" not in mapped_syms, (
        f"T7: UTX+RTX before boundary must map to RTX exactly once, got {mapped_syms}"
    )
    sc = _make_scored(500, when, symbols=mapped_syms)
    rows = generate_sentiment_features([sc], dates, cal, ["RTX"])
    assert rows[0].news_count == 1, "T7: generator must attribute exactly 1 article to RTX"


# ============================================================
# T8 - horizon independence (real)
# ============================================================


def _assert_t8_horizon_independence():
    """News feature columns are identical for h in {1, 3, 5} for every (stock, date)."""
    data = _load_json(FEATURE_VERIFICATION_PATH)
    assert data.get("horizon_independence_check_passed") is True, (
        "T8: feature_verification.json must set horizon_independence_check_passed=true; "
        f"got {data.get('horizon_independence_check_passed')!r}"
    )


# ============================================================
# T9 - missing data semantics
# ============================================================


def _assert_t9_missing_data_semantics():
    """Zero-news covered rows are frozen zeros; uncovered rows are NaN; no fillna."""
    dates = _trading_dates(date(2024, 7, 8), 5)
    cal = _make_calendar(dates)

    # Part 1: sentiment_features_v1 on an empty article set gives all-zero rows (never NaN)
    rows = generate_sentiment_features([], dates, cal, ["AAPL"])
    assert len(rows) == 5
    for r in rows:
        for col in SF_COLUMNS:
            v = float(getattr(r, col))
            assert v == 0.0, f"T9: sentiment zero-news col {col} on {r.trading_date} must be 0.0, got {v}"
            assert not math.isnan(v), f"T9: sentiment zero-news col {col} must not be NaN"

    # Part 2: event_features_v1 on an empty classified set gives all-zero numeric columns
    rows_ev = generate_event_features([], dates, cal, ["AAPL"])
    for r in rows_ev:
        for col in EVENT_FEATURE_COLUMNS:
            v = r[col]
            if col == "dominant_event_type":          # string column
                continue
            fv = float(v)
            assert fv == 0.0, f"T9: event zero-news col {col} must be 0.0, got {v}"
            assert not math.isnan(fv), f"T9: event zero-news col {col} must not be NaN"

    # Parts 3 & 4 need >= 20 trading sessions BEFORE each test date so
    # calendar.previous_session_before(session, 20) is not None. Otherwise
    # the row is uncovered by construction, regardless of coverage_start/end.
    # Build a 150-session calendar starting 2024-01-02 and test the last 5.
    cal_dates = _trading_dates(date(2024, 1, 2), 150)
    cal_wide = _make_calendar(cal_dates)
    test_dates = cal_dates[-5:]
    first_test_prev20 = cal_wide.previous_session_before(test_dates[0], 20)
    assert first_test_prev20 is not None, (
        "T9 setup: wide calendar must have >=20 sessions before the first test date"
    )

    # Part 3: COVERED empty window -> zero counts (not NaN); coverage_start well before prev_20
    cov_start = datetime(2023, 1, 1, tzinfo=UTC)
    cov_end = datetime(2025, 12, 31, 23, 59, tzinfo=UTC)
    sw_cov = generate_sentiment_window_features(
        [], test_dates, cal_wide, "AAPL",
        coverage_start=cov_start, coverage_end=cov_end,
    )
    covered = sw_cov[sw_cov["covered"]]
    assert len(covered) == len(test_dates), (
        f"T9 Part 3: with wide calendar and broad coverage, all {len(test_dates)} rows must be covered, "
        f"got {len(covered)}"
    )
    for col in ("sw_count_1", "sw_count_5", "sw_count_20"):
        vals = covered[col].to_numpy()
        assert (vals == 0).all(), f"T9: covered zero-news sw col {col} must be 0 (not NaN), got {vals}"
    # And the other sw columns are zero too (empty window convention) - never NaN on covered rows
    for col in SW_COLUMNS:
        vals = covered[col].to_numpy()
        assert not np.isnan(vals).any(), f"T9: covered zero-news sw col {col} must not contain NaN"

    # Part 4: UNCOVERED rows (coverage_start past prev_20) -> NaN (no fillna)
    sw_uncov = generate_sentiment_window_features(
        [], test_dates, cal_wide, "AAPL",
        coverage_start=datetime(2025, 1, 1, tzinfo=UTC),      # after any test date's prev_20
        coverage_end=datetime(2025, 12, 31, tzinfo=UTC),
    )
    uncovered = sw_uncov[~sw_uncov["covered"]]
    assert len(uncovered) == len(test_dates), (
        f"T9 Part 4: with coverage_start past the dates, all rows must be uncovered, "
        f"got {len(uncovered)} of {len(test_dates)}"
    )
    for col in SW_COLUMNS:
        assert uncovered[col].isna().all(), (
            f"T9: uncovered sw col {col} must be NaN (no fillna), got {uncovered[col].to_list()}"
        )


# ============================================================
# T10 - row identity (real)
# ============================================================


def _assert_t10_row_identity():
    """11B frames equal 11A frames on all shared columns; 1928 rows per stock."""
    data = _load_json(FEATURE_VERIFICATION_PATH)
    assert data.get("shared_columns_match_11a") is True, (
        "T10: feature_verification.json must set shared_columns_match_11a=true; "
        f"got {data.get('shared_columns_match_11a')!r}"
    )
    rpc = data.get("row_count_per_stock")
    assert rpc == 1928, f"T10: row_count_per_stock must be 1928, got {rpc!r}"
    panel_rows = data.get("panel_oos_rows_all29_h1")
    assert panel_rows == 52780, (
        f"T10: panel_oos_rows_all29_h1 must be 52780, got {panel_rows!r}"
    )


# ============================================================
# T11 - panel structure
# ============================================================


def _assert_t11_panel_structure():
    """11A functions reproduce correctly on B/C/D/F columns; symbol column rejected as feature."""
    from training.phase11a_research import (
        Phase11AError,
        date_splits,
        panel_walk_forward,
    )

    # Part 1: symbol / ticker column rejected as feature
    n = 100
    sample_dates = _trading_dates(date(2020, 1, 2), n)
    rng = np.random.default_rng(42)
    panel_df = pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "Date": pd.to_datetime(sample_dates),
        "sent_mean": rng.uniform(-1, 1, n),
        "future_return": rng.normal(0, 0.02, n),
        "direction": rng.integers(0, 2, n),
    })
    with pytest.raises(Phase11AError, match="ticker identity"):
        panel_walk_forward(panel_df, ["symbol", "sent_mean"], [], horizon=1, n_splits=5)
    with pytest.raises(Phase11AError, match="ticker identity"):
        panel_walk_forward(panel_df, ["ticker"], [], horizon=1, n_splits=5)

    # Part 2: date_splits yields n_splits folds and train/test dates never overlap
    dates_series = pd.Series(pd.to_datetime(_trading_dates(date(2020, 1, 2), 400)).date)
    splits = list(date_splits(dates_series, horizon=1, n_splits=20))
    assert len(splits) == 20, f"T11: date_splits must yield 20 folds, got {len(splits)}"
    for i, (tr, te) in enumerate(splits, start=1):
        overlap = set(tr) & set(te)
        assert not overlap, f"T11 fold {i}: train and test share date(s) {sorted(overlap)[:3]}"
        # Gap >= 1 (horizon) between last train and first test
        gap = (min(te) - max(tr)).days
        assert gap >= 1, f"T11 fold {i}: gap between last_train and first_test must be >= 1, got {gap}"

    # Part 3: date grouping - all rows for the same date land in the same fold
    panel = pd.DataFrame({
        "Date": list(pd.to_datetime(sample_dates)) * 3,              # 3 stocks per date
        "symbol": ["AAPL"] * n + ["MSFT"] * n + ["GS"] * n,
    })
    splits_panel = list(date_splits(panel["Date"], horizon=1, n_splits=5))
    for i, (tr, te) in enumerate(splits_panel, start=1):
        tr_set, te_set = set(pd.to_datetime(tr)), set(pd.to_datetime(te))
        for d in tr_set:
            same_date_rows = panel[panel["Date"] == d]
            assert len(same_date_rows) > 0
            # Every row for this date must be in train, not test
            assert d not in te_set, f"T11 fold {i}: date {d} is in both train and test"


# ============================================================
# T12 - matrix and families integrity
# ============================================================


def _assert_t12_id_subset():
    """3828 per-stock + 132 panel + 132 panel_exAAPL IDs enumerated from design; subset of registry."""
    models = [c for c in p10a.Spec10A().candidates() if not c.is_baseline]
    assert len(models) == 11, f"T12: expected 11 non-baseline models, got {len(models)}"

    universe = list(p11d.UNIVERSE)
    assert len(universe) == 29, f"T12: UNIVERSE must have 29 stocks, got {len(universe)}"
    horizons = list(p10a.HORIZONS)
    assert tuple(horizons) == (1, 3, 5), f"T12: HORIZONS must be (1, 3, 5), got {horizons}"

    per_stock_ids = [
        p11d.per_stock_id(sym, fs, h, m.name)
        for sym in universe
        for fs in ELEVEN_B_FEATURE_SETS
        for h in horizons
        for m in models
    ]
    assert len(per_stock_ids) == EXPECTED_PER_STOCK, (
        f"T12: per-stock count must be {EXPECTED_PER_STOCK}, got {len(per_stock_ids)}"
    )
    assert len(set(per_stock_ids)) == EXPECTED_PER_STOCK, "T12: per-stock IDs must be unique"

    panel_ids = [
        p11d.panel_id(fs, h, m.name)
        for fs in ELEVEN_B_FEATURE_SETS
        for h in horizons
        for m in models
    ]
    assert len(panel_ids) == EXPECTED_PANEL, (
        f"T12: panel count must be {EXPECTED_PANEL}, got {len(panel_ids)}"
    )
    assert len(set(panel_ids)) == EXPECTED_PANEL, "T12: panel IDs must be unique"

    panel_ex_aapl_ids = [
        p11d.panel_id(fs, h, m.name, sensitivity=True)
        for fs in ELEVEN_B_FEATURE_SETS
        for h in horizons
        for m in models
    ]
    assert len(panel_ex_aapl_ids) == EXPECTED_PANEL_EX_AAPL, (
        f"T12: panel_ex_aapl count must be {EXPECTED_PANEL_EX_AAPL}, got {len(panel_ex_aapl_ids)}"
    )

    # Subset check against the frozen registry
    registry = p11d.load_registry()        # verifies design_sha256 internally
    comparison_ids = registry["design"]["comparison_ids"]
    registry_per_stock = set(comparison_ids["per_stock"])
    registry_panel = set(comparison_ids["panel"])
    registry_sens = set(comparison_ids["aapl_excluded_sensitivity"])
    missing_per_stock = [i for i in per_stock_ids if i not in registry_per_stock]
    missing_panel = [i for i in panel_ids if i not in registry_panel]
    missing_sens = [i for i in panel_ex_aapl_ids if i not in registry_sens]
    assert not missing_per_stock, (
        f"T12: {len(missing_per_stock)} per-stock IDs not in registry.comparison_ids.per_stock: "
        f"{missing_per_stock[:3]}"
    )
    assert not missing_panel, (
        f"T12: {len(missing_panel)} panel IDs not in registry.comparison_ids.panel: {missing_panel[:3]}"
    )
    assert not missing_sens, (
        f"T12: {len(missing_sens)} panel_exAAPL IDs not in "
        f"registry.comparison_ids.aapl_excluded_sensitivity: {missing_sens[:3]}"
    )

    # Final family sizes
    assert EXPECTED_11A_PANEL + EXPECTED_PANEL == EXPECTED_FINAL_PANEL_FAMILY, (
        f"T12: 11A panel {EXPECTED_11A_PANEL} + 11B panel {EXPECTED_PANEL} must equal "
        f"{EXPECTED_FINAL_PANEL_FAMILY}, got {EXPECTED_11A_PANEL + EXPECTED_PANEL}"
    )
    assert EXPECTED_11A_PER_STOCK + EXPECTED_PER_STOCK == EXPECTED_FINAL_PER_STOCK_FAMILY, (
        f"T12: 11A per-stock {EXPECTED_11A_PER_STOCK} + 11B per-stock {EXPECTED_PER_STOCK} must equal "
        f"{EXPECTED_FINAL_PER_STOCK_FAMILY}, got {EXPECTED_11A_PER_STOCK + EXPECTED_PER_STOCK}"
    )


def _assert_t12_holm_families():
    """Holm correction runs over 198 panel and 5742 per-stock, with 11A p-values loaded by verified hash."""
    data = _load_json(FEATURE_VERIFICATION_PATH)
    panel_fam = data.get("final_panel_family_size")
    per_stock_fam = data.get("final_per_stock_family_size")
    assert panel_fam == EXPECTED_FINAL_PANEL_FAMILY, (
        f"T12-holm: final_panel_family_size must be {EXPECTED_FINAL_PANEL_FAMILY}, got {panel_fam!r}"
    )
    assert per_stock_fam == EXPECTED_FINAL_PER_STOCK_FAMILY, (
        f"T12-holm: final_per_stock_family_size must be {EXPECTED_FINAL_PER_STOCK_FAMILY}, got {per_stock_fam!r}"
    )
    assert data.get("phase_11a_p_values_hash_verified") is True, (
        "T12-holm: feature_verification.json must set phase_11a_p_values_hash_verified=true"
    )


# ============================================================
# T13 - F duplicate check (real)
# ============================================================


def _assert_t13_f_duplicate():
    """sw_count_1 == event__article_count for every stock and every row (F feature set drops sw_count_1)."""
    data = _load_json(FEATURE_VERIFICATION_PATH)
    assert data.get("f_duplicate_check_passed") is True, (
        "T13: feature_verification.json must set f_duplicate_check_passed=true; "
        f"got {data.get('f_duplicate_check_passed')!r}"
    )


# ============================================================
# T14 - input hash verification
# ============================================================


def _assert_t14_existing_artifacts():
    """Phase 11.0 design, amendment 01, Phase 11.1 price verification, and Phase 11A matrix unchanged."""
    # Design registry: load_registry() verifies design_sha256 against FROZEN_DESIGN_SHA internally
    registry = p11d.load_registry()
    assert registry["design_sha256"] == FROZEN_DESIGN_SHA, (
        f"T14: design_registry.json design_sha256 drift: "
        f"expected {FROZEN_DESIGN_SHA}, got {registry['design_sha256']}"
    )

    # Amendment 01 file SHA
    assert AMENDMENT_01_PATH.exists(), f"T14: amendment_01 missing at {AMENDMENT_01_PATH}"
    got_amend = _sha256_file(AMENDMENT_01_PATH)
    assert got_amend == AMENDMENT_01_SHA, (
        f"T14: amendment_01 file SHA drift: expected {AMENDMENT_01_SHA}, got {got_amend}"
    )

    # Price verification file SHA
    assert PRICE_VERIFICATION_PATH.exists(), f"T14: price_verification missing at {PRICE_VERIFICATION_PATH}"
    got_price = _sha256_file(PRICE_VERIFICATION_PATH)
    assert got_price == PRICE_REPORT_SHA, (
        f"T14: price_verification file SHA drift: expected {PRICE_REPORT_SHA}, got {got_price}"
    )

    # Phase 11A matrix_sha256 (inside registry) matches design_freeze.builds_on
    assert PHASE_11A_REGISTRY_PATH.exists(), f"T14: 11A registry missing at {PHASE_11A_REGISTRY_PATH}"
    a_reg = _load_json(PHASE_11A_REGISTRY_PATH)
    expected_matrix = _load_json(DESIGN_FREEZE_PATH)["builds_on"]["phase_11a_matrix_sha256"]
    got_matrix = a_reg.get("matrix_sha256")
    assert got_matrix == expected_matrix, (
        f"T14: 11A registry matrix_sha256 drift: expected {expected_matrix}, got {got_matrix}"
    )


def _assert_t14_full():
    """News, sentiment, and event file hashes match their records."""
    # Re-verify Gate A existing-artifact portion
    _assert_t14_existing_artifacts()

    # News / sentiment / event hashes recorded by Stages 11B.1 - 11B.3
    data = _load_json(FEATURE_VERIFICATION_PATH)
    assert data.get("news_file_hashes_match_records") is True, (
        "T14-full: feature_verification.json must set news_file_hashes_match_records=true"
    )
    assert data.get("sentiment_file_hashes_match_records") is True, (
        "T14-full: feature_verification.json must set sentiment_file_hashes_match_records=true"
    )
    assert data.get("event_file_hashes_match_records") is True, (
        "T14-full: feature_verification.json must set event_file_hashes_match_records=true"
    )


# ============================================================
# T15 - regression suite
# ============================================================


def _assert_t15_regression_suite():
    """
    python -m compileall and git diff --check pass cleanly.

    The 'every earlier phase test still passes' assertion is enforced by the user
    running `pytest tests/ -x` separately before authorizing Stage 11B.1; this
    test enforces the hygiene checks that can run in isolation.
    """
    # python -m compileall on the project source directories (NOT the venv or caches)
    compile_targets = [
        str(PROJECT_ROOT / d) for d in ("services", "training", "models", "tests", "config")
        if (PROJECT_ROOT / d).exists()
    ]
    assert compile_targets, "T15: no source directories found to compile"
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", *compile_targets],
        capture_output=True, text=True, timeout=180,
    )
    assert compile_result.returncode == 0, (
        f"T15: python -m compileall failed (rc={compile_result.returncode}):\n"
        f"stdout:\n{compile_result.stdout}\nstderr:\n{compile_result.stderr}"
    )

    # git diff --check (whitespace / conflict markers)
    gdc = subprocess.run(
        ["git", "diff", "--check"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    assert gdc.returncode == 0, (
        f"T15: git diff --check failed (rc={gdc.returncode}):\n"
        f"stdout:\n{gdc.stdout}\nstderr:\n{gdc.stderr}"
    )


# ============================================================
# Gate validators (pytest tests - gate authorization)
# ============================================================


def _assert_ingest_closeout_contract_passes():
    """
    Run every synthetic contract test in tests/unit/test_phase11b_ingest_closeout.py
    inline. The gate must only PASS if every assertion executed; a missing
    module or an empty discovery is a FAIL, not a silent pass.
    """
    import inspect
    import tempfile
    import tests.unit.test_phase11b_ingest_closeout as closeout_tests

    test_fns = [
        getattr(closeout_tests, name) for name in sorted(dir(closeout_tests))
        if name.startswith("test_") and callable(getattr(closeout_tests, name))
    ]
    if len(test_fns) == 0:
        raise AssertionError(
            "Gate A: no closeout contract tests discovered in test_phase11b_ingest_closeout.py"
        )
    for fn in test_fns:
        sig = inspect.signature(fn)
        if "tmp_path" in sig.parameters:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
        elif len(sig.parameters) == 0:
            fn()
        else:
            raise AssertionError(
                f"Gate A: closeout test {fn.__name__} has unexpected parameters {list(sig.parameters)}; "
                "add handling here before accepting it as a Gate A gate assertion"
            )


def test_gate_a_is_clear():
    """Gate A: every pre-ingestion assertion must PASS. No skips allowed."""
    _assert_t1_prefix_invariance_synthetic()
    _assert_t2_boundary_exact()
    _assert_t3_revision_semantics()
    _assert_t4_dst_handling()
    _assert_t5_seal_synthetic()
    _assert_t6_attribution_correctness()
    _assert_t7_rtx_utx_mapping()
    _assert_t9_missing_data_semantics()
    _assert_t11_panel_structure()
    _assert_t12_id_subset()
    _assert_t14_existing_artifacts()
    _assert_t15_regression_suite()
    _assert_ingest_closeout_contract_passes()


# Gate B / Gate C authorization checks are explicit commands that FAIL by design
# when future-stage artifacts are absent. The `gate_authorization` marker excludes
# them from routine `pytest tests/` runs so they do not fail the regression suite
# before ingestion. Amendment 03's "fail with BLOCKED message" semantics are
# preserved: when invoked explicitly, the behaviour is unchanged.
@pytest.mark.gate_authorization
def test_gate_b_is_clear():
    """Gate B: T5-real must PASS. Fails explicitly if Stage 11B.1 has not run."""
    if not _gate_b_eligible():
        pytest.fail(
            "GATE B DENIED: news_integrity_report.json does not exist at "
            f"{NEWS_INTEGRITY_REPORT}. Stage 11B.1 (news ingestion) has not completed. "
            "Run Stage 11B.1 first, then re-run this test."
        )
    _assert_t5_seal_real()


@pytest.mark.gate_authorization
def test_gate_c_is_clear():
    """Gate C: all 15 T-tests and all assertions must PASS. Fails explicitly on missing prerequisites."""
    # Prerequisites
    if not _gate_b_eligible():
        pytest.fail(
            "GATE C DENIED: Gate B prerequisite missing. news_integrity_report.json does not exist at "
            f"{NEWS_INTEGRITY_REPORT}. Complete Stage 11B.1 first."
        )
    if not _gate_c_eligible():
        pytest.fail(
            "GATE C DENIED: feature_verification.json does not exist at "
            f"{FEATURE_VERIFICATION_PATH}. Stages 11B.2 / 11B.3 (scoring and feature build) have not completed. "
            "Complete those stages first, then re-run this test."
        )
    # Re-verify Gate A
    _assert_t1_prefix_invariance_synthetic()
    _assert_t2_boundary_exact()
    _assert_t3_revision_semantics()
    _assert_t4_dst_handling()
    _assert_t5_seal_synthetic()
    _assert_t6_attribution_correctness()
    _assert_t7_rtx_utx_mapping()
    _assert_t9_missing_data_semantics()
    _assert_t11_panel_structure()
    _assert_t12_id_subset()
    _assert_t14_existing_artifacts()
    _assert_t15_regression_suite()
    _assert_ingest_closeout_contract_passes()
    # Re-verify Gate B
    _assert_t5_seal_real()
    # Gate C real-data assertions
    _assert_t1_prefix_invariance_real()
    _assert_t8_horizon_independence()
    _assert_t10_row_identity()
    _assert_t12_holm_families()
    _assert_t13_f_duplicate()
    _assert_t14_full()


# ============================================================
# Individual T-tests (pytest wrappers - status visibility)
# ============================================================


def test_t1_prefix_invariance_synthetic():
    _assert_t1_prefix_invariance_synthetic()


def test_t1_prefix_invariance_real():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T1-real]: feature_verification.json not found")
    _assert_t1_prefix_invariance_real()


def test_t2_boundary_exact():
    _assert_t2_boundary_exact()


def test_t3_revision_semantics():
    _assert_t3_revision_semantics()


def test_t4_dst_handling():
    _assert_t4_dst_handling()


def test_t5_seal_synthetic():
    _assert_t5_seal_synthetic()


def test_t5_seal_real():
    if not _gate_b_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T5-real]: news_integrity_report.json not found")
    _assert_t5_seal_real()


def test_t6_attribution_correctness():
    _assert_t6_attribution_correctness()


def test_t7_rtx_utx_mapping():
    _assert_t7_rtx_utx_mapping()


def test_t8_horizon_independence():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T8]: feature_verification.json not found")
    _assert_t8_horizon_independence()


def test_t9_missing_data_semantics():
    _assert_t9_missing_data_semantics()


def test_t10_row_identity():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T10]: feature_verification.json not found")
    _assert_t10_row_identity()


def test_t11_panel_structure():
    _assert_t11_panel_structure()


def test_t12_id_subset():
    _assert_t12_id_subset()


def test_t12_holm_families():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T12-holm]: feature_verification.json not found")
    _assert_t12_holm_families()


def test_t13_f_duplicate():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T13]: feature_verification.json not found")
    _assert_t13_f_duplicate()


def test_t14_existing_artifacts():
    _assert_t14_existing_artifacts()


def test_t14_full():
    if not _gate_c_eligible():
        pytest.skip(f"{BLOCKED_REASON}[T14-full]: feature_verification.json not found")
    _assert_t14_full()


def test_t15_regression_suite():
    _assert_t15_regression_suite()


# ============================================================
# Design freeze and Amendment 03 integrity
# ============================================================


def test_design_freeze_and_amendment_03_present():
    """Both the Phase 11B design freeze and Amendment 03 must exist and parse as JSON."""
    assert DESIGN_FREEZE_PATH.exists(), f"design_freeze.json missing at {DESIGN_FREEZE_PATH}"
    assert AMENDMENT_03_PATH.exists(), f"amendment_03_test_gates.json missing at {AMENDMENT_03_PATH}"
    df = _load_json(DESIGN_FREEZE_PATH)
    am = _load_json(AMENDMENT_03_PATH)
    assert df.get("document_type") == "phase_11b_design_freeze"
    assert am.get("document_type") == "phase_11b_amendment"
    assert am.get("amendment_id") == "03"
    assert am.get("preserves_design_freeze") is True
    assert am.get("weakens_assertions") is False
    recorded_sha = am["applies_to"]["design_freeze_sha256_at_adoption"]
    live_sha = _sha256_file(DESIGN_FREEZE_PATH)
    assert recorded_sha == live_sha, (
        f"design_freeze.json has drifted since Amendment 03 was adopted. "
        f"Amendment records {recorded_sha}, live file is {live_sha}. "
        "Either the design freeze has been modified (prohibited) or Amendment 03's recorded SHA is wrong."
    )
