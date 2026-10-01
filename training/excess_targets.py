"""
Phase 10A excess-return targets (AAPL relative to SPY), h = 1 / 3 / 5.

For a row dated t (features use information available at t 16:30 New York):

    future_return_AAPL_h(t) = Close[t+h] / Close[t] - 1            (training/targets.future_return)
    future_return_SPY_h(t)  = SPY_Close[t+h] / SPY_Close[t] - 1    (same function)
    future_excess_return_h(t) = future_return_AAPL_h(t) - future_return_SPY_h(t)
    excess_direction_h(t)     = 1 if future_excess_return_h(t) > 0 else 0
                                (training/targets.direction_label - direction_v1
                                 rule: exactly zero -> 0, no dead zone)

Both legs are simple total returns from split- and dividend-adjusted closes
of one retrieval each (AAPL: the verified feature store / AAPL snapshot;
SPY: the verified SPY snapshot). No log returns, no normalization.

The last h rows of the frame have no future close and receive NO label
(removed, never imputed). The frame passed in defines the information
boundary: for development it is cut at the development end BEFORE this
function runs, so no later AAPL or SPY close enters any target.

build_excess_targets returns the layout the central harness's walk_forward
consumes: `future_return` (regression target) and `direction`
(classification target) hold the EXCESS targets. Future SPY prices are used
only here - never as features (TARGET_COLUMNS must never be feature columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.targets import DIRECTION_RULE, direction_label, future_return

EXCESS_TARGET_VERSION = "excess_return_v1"
HORIZONS = (1, 3, 5)
SPY_CLOSE = "SPY_Close"

# columns that carry future information or raw prices; never allowed as features
TARGET_COLUMNS = ("future_return", "direction", "aapl_future_return", "spy_future_return",
                  "Close", SPY_CLOSE, "Target", "target_return_1d", "target_direction_1d")


class ExcessTargetError(ValueError):
    """Invalid input to excess-target construction."""


def future_excess_return(aapl_close: pd.Series, spy_close: pd.Series, horizon: int) -> pd.Series:
    """future_return(AAPL, h) - future_return(SPY, h); NaN for the last h rows."""
    return future_return(aapl_close, horizon) - future_return(spy_close, horizon)


def _check_prices(frame: pd.DataFrame) -> None:
    for col in ("Close", SPY_CLOSE):
        if col not in frame.columns:
            raise ExcessTargetError(f"frame has no '{col}' column")
        values = frame[col].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ExcessTargetError(f"'{col}' must be finite and positive")


def build_excess_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Frame (chronological, one row per session) -> labelled rows with
    future_return = future_excess_return_h, direction = excess_direction_h,
    plus the two legs (aapl_future_return, spy_future_return) for diagnostics.
    """
    if horizon not in HORIZONS:
        raise ExcessTargetError(f"Phase 10A horizons are {HORIZONS}, got {horizon}")
    if len(frame) <= horizon:
        raise ExcessTargetError(f"need more than {horizon} rows to build {horizon}-day targets")
    _check_prices(frame)
    out = frame.copy()
    out["aapl_future_return"] = future_return(out["Close"], horizon)
    out["spy_future_return"] = future_return(out[SPY_CLOSE], horizon)
    out["future_return"] = out["aapl_future_return"] - out["spy_future_return"]
    out = out.iloc[:-horizon].reset_index(drop=True)
    legs = out[["aapl_future_return", "spy_future_return", "future_return"]].to_numpy(dtype="float64")
    if not np.isfinite(legs).all():
        raise ExcessTargetError("non-finite excess target inside the labelled rows")
    out["direction"] = direction_label(out["future_return"])
    return out


def target_definition(horizon: int) -> dict:
    """Machine-readable Phase 10A target definition (persisted in the registry)."""
    if horizon not in HORIZONS:
        raise ExcessTargetError(f"Phase 10A horizons are {HORIZONS}, got {horizon}")
    h = horizon
    return {
        "target_version": EXCESS_TARGET_VERSION,
        "horizon_trading_days": h,
        "regression": {"name": f"future_excess_return_{h}d",
                       "formula": f"(Close[t+{h}] / Close[t] - 1) - (SPY_Close[t+{h}] / SPY_Close[t] - 1)"},
        "classification": {"name": f"excess_direction_{h}d",
                           "formula": f"1 if future_excess_return_{h}d > 0 else 0",
                           "rule": DIRECTION_RULE.replace("future_return", "future_excess_return")},
        "legs": {"aapl": f"future_return_AAPL_{h} = Close[t+{h}] / Close[t] - 1",
                 "spy": f"future_return_SPY_{h} = SPY_Close[t+{h}] / SPY_Close[t] - 1"},
        "price": "split- and dividend-adjusted closes (total returns); simple returns, no log, no normalization",
        "prediction_timestamp": "D 16:30 America/New_York (row_prediction_timestamp)",
        "label_available_at": f"session(D+{h}) 16:30 America/New_York (both closes complete)",
        "last_rows_without_label": h,
        "walk_forward_gap": h,
        "future_spy_use": "target construction only; never a feature",
    }
