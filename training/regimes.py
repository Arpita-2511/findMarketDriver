"""
Phase 10C market regimes - pre-registered, prediction-time safe.

Regime VARIABLES (value at row D uses only bars dated <= D, i.e. information
available at D 16:30 America/New_York):

    regime_aapl_vol_20  20-session sample std (ddof = 1) of AAPL simple daily
                        returns ending at D (same convention as Phase 10B; equals
                        the technical `Volatility` feature - checked at freeze)
    regime_spy_vol_20   20-session sample std (ddof = 1) of SPY simple daily
                        returns ending at D, on AAPL sessions (NOT the log-return
                        market_context spy_volatility_20 feature)
    spy_close_to_sma_50 existing market_context_v1 feature: SPY close / mean of
                        the last 50 SPY closes (incl. D)
    spy_return_20       existing market_context_v1 feature: SPY[D]/SPY[D-20] - 1

Regime FAMILIES and states:

    R1 aapl_vol        LOW_VOL  if regime_aapl_vol_20 <= training-fold median else HIGH_VOL
    R2 spy_vol         LOW_SPY_VOL / HIGH_SPY_VOL, training-fold median of regime_spy_vol_20
    R3 spy_trend       POSITIVE_TREND if spy_close_to_sma_50 >= 1.0 else NEGATIVE_TREND (fixed)
    R4 spy_return_20   POSITIVE_20D_RETURN if spy_return_20 >= 0 else NEGATIVE_20D_RETURN (fixed)
    R5 spy_vol_x_trend R2 state x R3 state (4 states; R2 threshold fitted per fold)

Fold-fitted thresholds: the median is computed over the TRAINING rows of
each walk-forward fold only; training and validation rows are then labelled
with that frozen threshold. No validation / holdout row and no target value
enters any threshold or label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AAPL_VOL = "regime_aapl_vol_20"
SPY_VOL = "regime_spy_vol_20"
SPY_TREND = "spy_close_to_sma_50"
SPY_RET20 = "spy_return_20"
VOL_WINDOW = 20
REGIME_VARIABLE_COLUMNS = (AAPL_VOL, SPY_VOL)          # computed here; never features
REGIME_INPUT_COLUMNS = (AAPL_VOL, SPY_VOL, SPY_TREND, SPY_RET20)

FAMILIES = {
    "R1_aapl_vol": {"variable": AAPL_VOL, "kind": "fold_median",
                    "states": ["LOW_VOL", "HIGH_VOL"],
                    "rule": "LOW_VOL if regime_aapl_vol_20 <= training-fold median else HIGH_VOL"},
    "R2_spy_vol": {"variable": SPY_VOL, "kind": "fold_median",
                   "states": ["LOW_SPY_VOL", "HIGH_SPY_VOL"],
                   "rule": "LOW_SPY_VOL if regime_spy_vol_20 <= training-fold median else HIGH_SPY_VOL"},
    "R3_spy_trend": {"variable": SPY_TREND, "kind": "fixed", "threshold": 1.0,
                     "states": ["POSITIVE_TREND", "NEGATIVE_TREND"],
                     "rule": "POSITIVE_TREND if spy_close_to_sma_50 >= 1.0 else NEGATIVE_TREND"},
    "R4_spy_return_20": {"variable": SPY_RET20, "kind": "fixed", "threshold": 0.0,
                         "states": ["POSITIVE_20D_RETURN", "NEGATIVE_20D_RETURN"],
                         "rule": "POSITIVE_20D_RETURN if spy_return_20 >= 0 else NEGATIVE_20D_RETURN"},
    "R5_spy_vol_x_trend": {"components": ["R2_spy_vol", "R3_spy_trend"], "kind": "combined",
                           "states": ["LOW_SPY_VOL_POSITIVE_TREND", "LOW_SPY_VOL_NEGATIVE_TREND",
                                      "HIGH_SPY_VOL_POSITIVE_TREND", "HIGH_SPY_VOL_NEGATIVE_TREND"],
                           "rule": "R2 state (fold-median SPY volatility) x R3 state (SPY trend)"},
}
STATE_COUNT = sum(len(f["states"]) for f in FAMILIES.values())          # 12

VARIABLE_DEFINITIONS = {
    AAPL_VOL: "sample std (ddof=1) of AAPL simple daily returns Close[j]/Close[j-1]-1, j = D-19..D",
    SPY_VOL: "sample std (ddof=1) of SPY simple daily returns SPY[j]/SPY[j-1]-1, j = D-19..D (AAPL sessions)",
    SPY_TREND: "market_context_v1: SPY[D] / mean(SPY[D-49..D])",
    SPY_RET20: "market_context_v1: SPY[D] / SPY[D-20] - 1",
    "information_cutoff": "bars dated <= D only (complete at D 16:30 America/New_York)",
    "role": "regime assignment only; never model features",
}


class RegimeError(ValueError):
    """Invalid regime input."""


def realized_volatility(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Trailing sample std (ddof=1) of simple daily returns; value at D uses returns D-window+1..D."""
    return (close / close.shift(1) - 1).rolling(window=window, min_periods=window).std(ddof=1)


def attach_regime_variables(frame: pd.DataFrame, aapl_bars: pd.DataFrame, spy_bars: pd.DataFrame) -> pd.DataFrame:
    """Add AAPL_VOL and SPY_VOL computed on the full AAPL-session history of the verified snapshots."""
    a = pd.Series(aapl_bars["Close"].to_numpy(dtype="float64"), index=list(aapl_bars["Date"].dt.date))
    s_all = dict(zip(spy_bars["Date"].dt.date, spy_bars["Close"].astype("float64")))
    dates = [ts.date() for ts in frame["Date"]]
    sessions = list(a.index)
    try:
        first = sessions.index(dates[0]) - VOL_WINDOW
    except ValueError:
        raise RegimeError(f"{dates[0]} is not an AAPL session") from None
    if first < 0:
        raise RegimeError(f"fewer than {VOL_WINDOW} returns before {dates[0]} (insufficient history)")
    needed = [d for d in sessions[first:] if d <= dates[-1]]
    missing = [d for d in needed if d not in s_all]
    if missing:
        raise RegimeError(f"SPY close missing for {len(missing)} needed session(s), e.g. {missing[:3]}")
    a_vol = realized_volatility(a.loc[needed])
    s_vol = realized_volatility(pd.Series([s_all[d] for d in needed], index=needed))
    out = frame.copy()
    out[AAPL_VOL] = [a_vol.get(d, np.nan) for d in dates]
    out[SPY_VOL] = [s_vol.get(d, np.nan) for d in dates]
    vals = out[list(REGIME_INPUT_COLUMNS)].to_numpy(dtype="float64")
    if not np.isfinite(vals).all():
        raise RegimeError("regime variables missing or non-finite for some rows")
    return out


def _binary(values: np.ndarray, threshold: float, low_or_pos: str, high_or_neg: str, *, fitted: bool) -> np.ndarray:
    if fitted:   # <= median -> LOW, else HIGH
        return np.where(values <= threshold, low_or_pos, high_or_neg)
    return np.where(values >= threshold, low_or_pos, high_or_neg)


def fold_thresholds(frame_values: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, float]:
    """Training-fold medians for the fold-fitted variables (training rows ONLY)."""
    out = {}
    for name, fam in FAMILIES.items():
        if fam["kind"] == "fold_median":
            v = frame_values[fam["variable"]][train_idx]
            if len(v) == 0 or not np.isfinite(v).all():
                raise RegimeError(f"{name}: no finite training values for the threshold")
            out[name] = float(np.median(v))
    return out


def label_rows(frame_values: dict[str, np.ndarray], rows: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    """State label of each given row for every family, using frozen thresholds only."""
    labels = {}
    for name, fam in FAMILIES.items():
        if fam["kind"] == "fold_median":
            labels[name] = _binary(frame_values[fam["variable"]][rows], thresholds[name], *fam["states"], fitted=True)
        elif fam["kind"] == "fixed":
            labels[name] = _binary(frame_values[fam["variable"]][rows], fam["threshold"], *fam["states"], fitted=False)
    r2, r3 = labels["R2_spy_vol"], labels["R3_spy_trend"]
    labels["R5_spy_vol_x_trend"] = np.array([f"{a}_{b}" for a, b in zip(r2, r3)], dtype=object)
    for name, fam in FAMILIES.items():
        if not set(labels[name]) <= set(fam["states"]):
            raise RegimeError(f"{name}: unexpected state labels {set(labels[name]) - set(fam['states'])}")
    return labels


def assign_oos_states(frame_values: dict[str, np.ndarray], splits) -> tuple[dict[str, dict[int, str]], list[dict]]:
    """
    For each (train_idx, test_idx) split: thresholds from train_idx, labels for test_idx.
    Returns ({family: {row: state}}, per-fold records with thresholds and state counts).
    """
    states = {name: {} for name in FAMILIES}
    folds = []
    for k, (train_idx, test_idx) in enumerate(splits, start=1):
        thr = fold_thresholds(frame_values, np.asarray(train_idx))
        lab = label_rows(frame_values, np.asarray(test_idx), thr)
        for name in FAMILIES:
            for row, s in zip(test_idx, lab[name]):
                states[name][int(row)] = str(s)
        folds.append({"fold": k, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                      "thresholds": thr,
                      "state_counts": {name: {s: int((lab[name] == s).sum()) for s in FAMILIES[name]["states"]}
                                       for name in FAMILIES}})
    return states, folds
