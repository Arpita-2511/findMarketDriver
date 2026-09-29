"""Adjustment consistency of live outcome scoring (models/evaluate_live_tracking.py)."""

import numpy as np
import pandas as pd
import pytest

from models.evaluate_live_tracking import compute_outcome, fill_outcomes


def closes(values, start="2026-07-28"):
    dates = pd.bdate_range(start, periods=len(values)).strftime("%Y-%m-%d")
    return pd.Series(values, index=dates, dtype=float)


def test_outcome_uses_one_price_series():
    # Logged on 07-29 at 200.00; true next-day move +1%.
    at_log_time = closes([198.0, 200.0, 202.0])
    # A later dividend back-adjusts ALL earlier prices by 0.995
    refetched = at_log_time * 0.995

    next_close, ret = compute_outcome(refetched, "2026-07-29", logged_close=200.0)

    assert ret == pytest.approx(0.01)
    assert next_close == pytest.approx(202.0)          # expressed in the logged price basis
    # the pre-Phase-2 formula mixed bases and was biased by the adjustment
    old_formula = refetched["2026-07-30"] / 200.0 - 1
    assert not np.isclose(old_formula, 0.01)


def test_no_next_day_yet():
    assert compute_outcome(closes([198.0, 200.0]), "2026-07-29", 200.0) is None


def test_log_date_outside_fetched_window():
    assert compute_outcome(closes([198.0, 200.0, 202.0]), "2026-06-01", 200.0) is None


def test_fill_outcomes_skips_scored_rows():
    log = pd.DataFrame({
        "log_date": ["2026-07-29", "2026-07-29"],
        "symbol": ["AAPL", "AAPL"],
        "model": ["ridge", "lasso"],
        "current_close": [200.0, 200.0],
        "predicted_return": [0.001, 0.002],
        "predicted_close": [200.2, 200.4],
        "actual_next_close": [np.nan, 205.0],
        "actual_return": [np.nan, 0.025],
    })
    updated = fill_outcomes(log, {"AAPL": closes([198.0, 200.0, 202.0])})

    assert updated == 1
    assert log.loc[0, "actual_return"] == pytest.approx(0.01)
    assert log.loc[1, "actual_return"] == 0.025        # untouched
