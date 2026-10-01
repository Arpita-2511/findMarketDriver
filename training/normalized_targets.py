"""
Phase 10B volatility-normalized excess-return targets (AAPL relative to SPY), h = 1 / 3 / 5.

Numerator (Phase 10A, unchanged - training/excess_targets.py):
    future_excess_return_h(D) = (Close[D+h]/Close[D] - 1) - (SPY_Close[D+h]/SPY_Close[D] - 1)

Denominator - prediction-time realized volatility, fixed before any Phase 10B result:
    x_j          = (Close[j]/Close[j-1] - 1) - (SPY_Close[j]/SPY_Close[j-1] - 1)
                   daily EXCESS simple return of session j (AAPL sessions)
    volatility_D = sample std (ddof = 1) of x_j for the 20 sessions j = D-19 .. D
    Conventions kept from the project: simple returns (Phase 10A target,
    Return_1); 20-session window with ddof = 1 (technical `Volatility`,
    market_context spy_volatility_20). Only closes dated <= D enter it, so it
    is known at D 16:30 New York. No horizon scaling: a constant factor per
    horizon cannot change any gate_v1 decision (MSE ratios, the DM statistic
    and direction labels are scale-invariant).

Targets:
    normalized_excess_return_h(D)    = future_excess_return_h(D) / volatility_D
    normalized_excess_direction_h(D) = 1 if normalized_excess_return_h(D) > 0 else 0
                                       (direction_label; volatility > 0, so it equals the
                                        Phase 10A excess direction - built explicitly anyway)

Insufficient history (fewer than 20 daily excess returns ending at D) or a
non-positive / non-finite volatility is an ERROR - rows are never dropped
or filled. The volatility is a target denominator only, never a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.excess_targets import HORIZONS, SPY_CLOSE, TARGET_COLUMNS, build_excess_targets
from training.excess_targets import target_definition as excess_target_definition
from training.targets import DIRECTION_RULE, direction_label

NORMALIZED_TARGET_VERSION = "normalized_excess_return_v1"
VOL_WINDOW = 20
VOL_COLUMN = "excess_vol_20"
NORMALIZED_TARGET_COLUMNS = (*TARGET_COLUMNS, "future_excess_return", VOL_COLUMN)

VOLATILITY_DEFINITION = {
    "name": VOL_COLUMN,
    "daily_excess_return": "x_j = (Close[j]/Close[j-1] - 1) - (SPY_Close[j]/SPY_Close[j-1] - 1)",
    "estimator": f"sample standard deviation (ddof = 1) of x_j over the {VOL_WINDOW} sessions j = D-{VOL_WINDOW - 1} .. D",
    "window_sessions": VOL_WINDOW, "ddof": 1, "returns": "simple (as Phase 10A target and Return_1)",
    "information_cutoff": "closes dated <= D only (known at D 16:30 America/New_York)",
    "horizon_scaling": "none (constant per-horizon factors cannot change gate_v1 decisions)",
    "insufficient_history": "error - rows are never dropped or filled",
    "role": "target denominator only; never a feature",
    "source": "hash-verified AAPL and SPY raw snapshots on AAPL trading sessions",
}


class NormalizedTargetError(ValueError):
    """Invalid input to normalized-target construction."""


def daily_excess_returns(aapl_close: pd.Series, spy_close: pd.Series) -> pd.Series:
    """x_j = AAPL simple return - SPY simple return of session j (NaN for the first row)."""
    return (aapl_close / aapl_close.shift(1) - 1) - (spy_close / spy_close.shift(1) - 1)


def prediction_time_volatility(aapl_close: pd.Series, spy_close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """volatility at row D from x_{D-window+1} .. x_D only (rolling, trailing); NaN while history is short."""
    return daily_excess_returns(aapl_close, spy_close).rolling(window=window, min_periods=window).std(ddof=1)


def attach_volatility(frame: pd.DataFrame, aapl_bars: pd.DataFrame, spy_bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add VOL_COLUMN to `frame` (rows = AAPL sessions), computed on the full AAPL
    session history of the snapshots so early rows have their 20 prior returns.
    SPY must have every AAPL session over the needed range (no fill).
    """
    a = pd.Series(aapl_bars["Close"].to_numpy(dtype="float64"), index=list(aapl_bars["Date"].dt.date))
    s_all = dict(zip(spy_bars["Date"].dt.date, spy_bars["Close"].astype("float64")))
    dates = [ts.date() for ts in frame["Date"]]
    if not dates:
        raise NormalizedTargetError("empty frame")
    sessions = list(a.index)
    try:
        first = sessions.index(dates[0]) - VOL_WINDOW
    except ValueError:
        raise NormalizedTargetError(f"{dates[0]} is not an AAPL session") from None
    if first < 0:
        raise NormalizedTargetError(f"fewer than {VOL_WINDOW} AAPL returns before {dates[0]} (insufficient history)")
    needed = [d for d in sessions[first:] if d <= dates[-1]]
    missing = [d for d in needed if d not in s_all]
    if missing:
        raise NormalizedTargetError(f"SPY close missing for {len(missing)} needed session(s), e.g. {missing[:3]}")
    a_n = a.loc[needed]
    s_n = pd.Series([s_all[d] for d in needed], index=needed)
    vol = prediction_time_volatility(a_n, s_n)
    out = frame.copy()
    out[VOL_COLUMN] = [vol.get(d, np.nan) for d in dates]
    values = out[VOL_COLUMN].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise NormalizedTargetError("prediction-time volatility missing, non-finite or non-positive for some rows")
    return out


def build_normalized_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Labelled rows: future_return = normalized excess return, direction = normalized excess direction."""
    if VOL_COLUMN not in frame.columns:
        raise NormalizedTargetError(f"frame has no '{VOL_COLUMN}' column")
    vol = frame[VOL_COLUMN].to_numpy(dtype="float64")
    if not np.isfinite(vol).all() or (vol <= 0).any():
        raise NormalizedTargetError("volatility must be finite and positive on every row")
    out = build_excess_targets(frame, horizon)               # drops the last h rows; vol stays aligned to D
    out["future_excess_return"] = out["future_return"]
    out["future_return"] = out["future_excess_return"] / out[VOL_COLUMN]
    if not np.isfinite(out["future_return"].to_numpy(dtype="float64")).all():
        raise NormalizedTargetError("non-finite normalized target")
    out["direction"] = direction_label(out["future_return"])
    return out


def target_definition(horizon: int) -> dict:
    base = excess_target_definition(horizon)
    h = horizon
    return {
        "target_version": NORMALIZED_TARGET_VERSION,
        "horizon_trading_days": h,
        "regression": {"name": f"normalized_excess_return_{h}d",
                       "formula": f"future_excess_return_{h}d / {VOL_COLUMN}(D)"},
        "classification": {"name": f"normalized_excess_direction_{h}d",
                           "formula": f"1 if normalized_excess_return_{h}d > 0 else 0",
                           "rule": DIRECTION_RULE.replace("future_return", "normalized_excess_return")},
        "numerator": base["regression"],
        "volatility": VOLATILITY_DEFINITION,
        "prediction_timestamp": base["prediction_timestamp"],
        "label_available_at": base["label_available_at"],
        "last_rows_without_label": h, "walk_forward_gap": h,
        "future_spy_use": base["future_spy_use"],
    }


__all__ = ["HORIZONS", "SPY_CLOSE", "VOL_COLUMN", "VOL_WINDOW", "VOLATILITY_DEFINITION", "NORMALIZED_TARGET_COLUMNS",
           "attach_volatility", "build_normalized_targets", "daily_excess_returns", "prediction_time_volatility",
           "target_definition", "NormalizedTargetError"]
