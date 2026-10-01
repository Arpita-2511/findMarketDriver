"""Phase 9 targets: existing formulas at h = 1 / 3 / 5, boundaries, gap and label timing."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from services.market_calendar_service import TradingCalendar
from training.evaluation_harness import HarnessConfig
from training.research_targets import HORIZONS, build_research_targets, label_available_at, target_definition
from training.targets import DIRECTION_RULE, TARGET_VERSION


def frame(closes):
    return pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=len(closes)), "Close": closes})


@pytest.mark.parametrize("h", HORIZONS)
def test_targets_use_existing_formula_and_drop_last_h_rows(h):
    closes = np.array([100.0, 101.0, 99.0, 99.0, 102.0, 103.0, 101.0, 104.0, 104.0, 105.0])
    out = build_research_targets(frame(closes), h)
    assert len(out) == len(closes) - h                                   # no label without a future close
    expected = closes[h:] / closes[:-h] - 1
    np.testing.assert_allclose(out["future_return"].to_numpy(), expected, rtol=0, atol=0)
    assert (out["direction"] == (expected > 0).astype(int)).all()


def test_zero_return_is_down_no_dead_zone():
    out = build_research_targets(frame([100.0, 100.0, 100.0000001, 99.0]), 1)
    assert list(out["direction"]) == [0, 1, 0]                          # exactly unchanged -> DOWN


@pytest.mark.parametrize("h", HORIZONS)
def test_gap_equals_horizon(h):
    assert HarnessConfig(horizon=h).gap == h
    assert target_definition(h)["walk_forward_gap"] == h


@pytest.mark.parametrize("h", HORIZONS)
def test_target_definition_is_the_existing_contract(h):
    d = target_definition(h)
    assert d["regression"]["formula"] == f"future_return = Close[t+{h}] / Close[t] - 1"
    assert d["classification"]["formula"] == DIRECTION_RULE
    assert d["target_version"] == TARGET_VERSION
    assert d["last_rows_without_label"] == h


def test_only_phase9_horizons():
    with pytest.raises(ValueError):
        target_definition(2)
    with pytest.raises(ValueError):
        build_research_targets(frame([1.0, 2.0, 3.0]), 10)


def test_label_available_at_is_completion_of_the_h_th_next_session():
    sessions = [d for d in pd.bdate_range("2024-06-24", "2024-07-12").date if d != date(2024, 7, 4)]
    cal = TradingCalendar(sessions)
    # Friday 06-28, h=1 -> Monday 07-01 16:30 EDT = 20:30 UTC
    assert label_available_at(date(2024, 6, 28), 1, cal) == datetime(2024, 7, 1, 20, 30, tzinfo=timezone.utc)
    # Wednesday 07-03, h=1 skips the 07-04 holiday -> Friday 07-05
    assert label_available_at(date(2024, 7, 3), 1, cal) == datetime(2024, 7, 5, 20, 30, tzinfo=timezone.utc)
    # 07-03, h=3 -> 07-05, 07-08, 07-09
    assert label_available_at(date(2024, 7, 3), 3, cal) == datetime(2024, 7, 9, 20, 30, tzinfo=timezone.utc)
    # the label is always known strictly after the row's prediction time
    assert label_available_at(date(2024, 7, 3), 1, cal) > datetime(2024, 7, 3, 20, 30, tzinfo=timezone.utc)
