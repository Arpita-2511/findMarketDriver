"""
Market-context features (Phase 9, market_context_v1).

Built from hash-verified raw daily snapshots (training/build_dataset.py
format) of SPY and QQQ, plus the AAPL snapshot that defines the trading
calendar. Definitions were fixed before any Phase 9 result was observed.

For an AAPL row dated D (S = SPY adjusted Close, Q = QQQ, A = AAPL; index
positions on each series' own sessions; "k sessions back" = k-th previous
session):

    spy_return_1 / _5 / _20     S[D] / S[D-k] - 1
    qqq_return_1 / _5           Q[D] / Q[D-k] - 1
    aapl_minus_spy_return_1/_5  (A[D] / A[D-k] - 1) - (S[D] / S[D-k] - 1)
    spy_volatility_20           sample std (ddof = 1) of the 20 SPY daily log
                                returns ln(S[j] / S[j-1]), j = D-19 .. D
    spy_close_to_sma_50         S[D] / mean(S[D-49] .. S[D])
                                (ratio, same convention as Close_to_SMA_*)

Information timestamp: every input is a daily bar dated <= D. A bar dated D is
complete at D 16:00 + 30 min = D 16:30 America/New_York (completed-bar
policy, market_calendar_service), which is exactly the prediction timestamp
row_prediction_timestamp(D). No later bar is ever read.

Calendar alignment (strict): over every session the features need (the
earliest lookback through the last row), SPY and QQQ must have exactly the
AAPL sessions. A missing or extra session raises MarketContextError - no
forward fill, no dropped rows. Too little history before the first row
also raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from services.market_calendar_service import TradingCalendar
from services.market_data import MarketDataError
from services.news_alignment import row_prediction_timestamp
from training.build_dataset import load_raw_snapshot

MARKET_CONTEXT_VERSION = "market_context_v1"
CONTEXT_TICKERS = ("SPY", "QQQ")

COLUMNS = (
    "spy_return_1", "spy_return_5", "spy_return_20",
    "qqq_return_1", "qqq_return_5",
    "aapl_minus_spy_return_1", "aapl_minus_spy_return_5",
    "spy_volatility_20",
    "spy_close_to_sma_50",
)
MAX_LOOKBACK = 49            # spy_close_to_sma_50 uses S[D-49] .. S[D]

DEFINITIONS = {
    "spy_return_k": "S[D] / S[D-k] - 1, S = SPY adjusted Close, k in (1, 5, 20)",
    "qqq_return_k": "Q[D] / Q[D-k] - 1, Q = QQQ adjusted Close, k in (1, 5)",
    "aapl_minus_spy_return_k": "(A[D] / A[D-k] - 1) - (S[D] / S[D-k] - 1), A = AAPL adjusted Close, k in (1, 5)",
    "spy_volatility_20": "sample std (ddof=1) of ln(S[j] / S[j-1]) for j = D-19 .. D",
    "spy_close_to_sma_50": "S[D] / mean(S[D-49] .. S[D])",
    "lookback": "k sessions back = k-th previous trading session of the same series",
    "information_timestamp": ("bars dated <= D only; bar D complete at D 16:30 America/New_York "
                              "= row_prediction_timestamp(D)"),
    "alignment": "SPY/QQQ sessions must equal AAPL sessions over the needed range; missing/extra -> error",
    "source": "raw daily snapshots (yfinance, auto_adjust) verified by sha256",
}


class MarketContextError(MarketDataError):
    """Market-context data is missing, misaligned or insufficient."""


def load_context_snapshot(path: Path, expected_ticker: str) -> tuple[pd.DataFrame, dict]:
    """Hash-verified raw snapshot whose metadata ticker must equal `expected_ticker`."""
    bars, meta = load_raw_snapshot(path)
    if str(meta.get("ticker", "")).upper() != expected_ticker.upper():
        raise MarketContextError(f"{Path(path).name}: snapshot ticker {meta.get('ticker')!r}, "
                                 f"expected {expected_ticker!r}")
    return bars, meta


def _closes(bars: pd.DataFrame) -> pd.Series:
    s = pd.Series(bars["Close"].to_numpy(dtype="float64"), index=list(bars["Date"].dt.date))
    if not s.index.is_monotonic_increasing or s.index.has_duplicates:
        raise MarketContextError("bars must be sorted with unique dates")
    return s


def _check_alignment(name: str, series: pd.Series, sessions: list) -> None:
    lo, hi = sessions[0], sessions[-1]
    have = {d for d in series.index if lo <= d <= hi}
    want = set(sessions)
    missing, extra = sorted(want - have), sorted(have - want)
    if missing:
        raise MarketContextError(f"{name}: {len(missing)} AAPL session(s) missing, e.g. "
                                 f"{[d.isoformat() for d in missing[:5]]} (no forward fill)")
    if extra:
        raise MarketContextError(f"{name}: {len(extra)} session(s) that are not AAPL sessions, e.g. "
                                 f"{[d.isoformat() for d in extra[:5]]}")


def compute_market_context(aapl_bars: pd.DataFrame, spy_bars: pd.DataFrame, qqq_bars: pd.DataFrame,
                           trading_dates: Iterable) -> pd.DataFrame:
    """One row per trading date: trading_date, prediction_timestamp (UTC), COLUMNS."""
    calendar = TradingCalendar.from_bars(aapl_bars)
    dates = sorted(pd.Timestamp(d).date() for d in trading_dates)
    if not dates:
        raise MarketContextError("no trading dates given")
    if len(set(dates)) != len(dates):
        raise MarketContextError("trading_dates contains duplicates")
    for d in dates:
        if not calendar.is_session(d):
            raise MarketContextError(f"{d} is not an AAPL trading session")

    first_needed = calendar.previous_session_before(dates[0], MAX_LOOKBACK)
    if first_needed is None:
        raise MarketContextError(f"AAPL calendar has fewer than {MAX_LOOKBACK} sessions before {dates[0]}")
    a, s, q = _closes(aapl_bars), _closes(spy_bars), _closes(qqq_bars)
    needed = [d for d in a.index if first_needed <= d <= dates[-1]]
    _check_alignment("SPY", s, needed)
    _check_alignment("QQQ", q, needed)
    if s.index[0] > first_needed or q.index[0] > first_needed:
        raise MarketContextError(f"context history must start on or before {first_needed}")

    # positional (session) lookbacks on each series' own sessions; inside `needed`
    # they coincide with AAPL sessions because the alignment check passed
    feats = pd.DataFrame(index=s.index)
    for k in (1, 5, 20):
        feats[f"spy_return_{k}"] = s / s.shift(k) - 1
    feats["spy_volatility_20"] = np.log(s / s.shift(1)).rolling(20).std(ddof=1)
    feats["spy_close_to_sma_50"] = s / s.rolling(MAX_LOOKBACK + 1).mean()
    qf = pd.DataFrame({f"qqq_return_{k}": q / q.shift(k) - 1 for k in (1, 5)})
    af = pd.DataFrame({f"aapl_return_{k}": a / a.shift(k) - 1 for k in (1, 5)})

    out = pd.DataFrame({"trading_date": dates})
    out["prediction_timestamp"] = [row_prediction_timestamp(d, calendar) for d in dates]
    out = out.join(feats, on="trading_date").join(qf, on="trading_date").join(af, on="trading_date")
    for k in (1, 5):
        out[f"aapl_minus_spy_return_{k}"] = out[f"aapl_return_{k}"] - out[f"spy_return_{k}"]
    out = out[["trading_date", "prediction_timestamp", *COLUMNS]]

    values = out[list(COLUMNS)].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = out.loc[~np.isfinite(values).all(axis=1), "trading_date"]
        raise MarketContextError(f"{len(bad)} row(s) without complete context history, e.g. {bad.iloc[0]}")
    return out.reset_index(drop=True)
