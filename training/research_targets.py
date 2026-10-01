"""
Phase 9 research targets - h = 1 / 3 / 5 trading days.

No new target formula. This module only documents how the EXISTING
implementation is used at the Phase 9 horizons:

    future_return[t] = Close[t+h] / Close[t] - 1         training/targets.future_return
    direction[t]     = 1 if future_return[t] > 0 else 0  training/targets.direction_label (direction_v1)

The harness builds both from Close inside the evaluated frame
(evaluation_harness.build_targets) and drops the last h rows (no future
close -> no label). Walk-forward gap = horizon (HarnessConfig.gap), so a
training row's label window never reaches the first test row.

Timing for a row dated D:
    prediction timestamp  D 16:30 America/New_York (row_prediction_timestamp)
    label available       completion of the h-th session after D, i.e.
                          session(D+h) 16:30 America/New_York
Labels are used only as training/evaluation targets, never as features.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from services.market_calendar_service import TradingCalendar, completion_time
from services.news_alignment import row_prediction_timestamp
from training.evaluation_harness import HarnessConfig, build_targets
from training.targets import DIRECTION_RULE, TARGET_VERSION

HORIZONS = (1, 3, 5)


def target_definition(horizon: int) -> dict:
    """Machine-readable definition of the Phase 9 targets at `horizon` (persisted in the registry)."""
    if horizon not in HORIZONS:
        raise ValueError(f"Phase 9 horizons are {HORIZONS}, got {horizon}")
    return {
        "target_version": TARGET_VERSION,
        "horizon_trading_days": horizon,
        "regression": {"name": f"future_return_{horizon}d",
                       "formula": f"future_return = Close[t+{horizon}] / Close[t] - 1"},
        "classification": {"name": f"direction_{horizon}d", "formula": DIRECTION_RULE},
        "price": "split- and dividend-adjusted Close (total return)",
        "prediction_timestamp": "D 16:30 America/New_York (row_prediction_timestamp)",
        "label_available_at": f"session(D+{horizon}) 16:30 America/New_York (completion of the {horizon}-th next bar)",
        "last_rows_without_label": horizon,
        "walk_forward_gap": HarnessConfig(horizon=horizon).gap,
    }


def label_available_at(trading_date, horizon: int, calendar: TradingCalendar) -> datetime:
    """Moment the h-day label of row D becomes known: completion of the h-th session after D (UTC)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    row_prediction_timestamp(trading_date, calendar)              # validates D is a session
    d = trading_date
    for _ in range(horizon):
        d = calendar.next_session_after(d)
    return completion_time(d).astimezone(timezone.utc)


def build_research_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The harness's own target construction (unchanged) at a Phase 9 horizon."""
    if horizon not in HORIZONS:
        raise ValueError(f"Phase 9 horizons are {HORIZONS}, got {horizon}")
    return build_targets(frame, horizon)
