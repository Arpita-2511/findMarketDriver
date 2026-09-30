"""
Prediction targets - single source of truth (Phase 3).

For a feature row dated t (features use bars up to and including t):

    future_return[t] = Close[t+h] / Close[t] - 1          (h = horizon, trading days)
    direction[t]     = 1 (UP)   if future_return[t] > 0
                       0 (DOWN) otherwise

Close is split- and dividend-adjusted (ARCHITECTURE.md 6.2), so
future_return is a total return.

Boundary decision (documented in ARCHITECTURE.md section 8):
    - An exactly-zero future return is DOWN (0). "UP" asserts the price
      rose; unchanged is not a rise. This is the rule already written in
      ARCHITECTURE.md section 8 / REQ-TARGET-001 and used by the Phase 1-2
      baselines, so results stay comparable.
    - There is NO dead zone around zero. Dropping or relabelling "small"
      moves would select evaluation rows using the future outcome, which
      is unknown at prediction time, and would make offline metrics
      unrepresentative of live use.
    - No epsilon: the ratio of two identical adjusted closes is exactly
      0.0 in floating point, and any non-zero total return (e.g. a
      dividend on a flat day) is a genuine move.
    - The last h rows have no future close and receive NO label (they
      are removed, never imputed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VERSION = "direction_v1"

UP = 1
DOWN = 0

DIRECTION_RULE = "direction = 1 (UP) if future_return > 0 else 0 (DOWN); zero -> DOWN; no dead zone"


def future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Close[t+h] / Close[t] - 1, NaN for the last `horizon` rows."""
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be an integer >= 1")
    return close.shift(-horizon) / close - 1


def direction_label(returns: pd.Series) -> pd.Series:
    """Apply DIRECTION_RULE. Unlabelable (NaN/inf) returns are an error, not DOWN."""
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("direction_label received NaN/inf returns; drop rows without a future close first")
    return pd.Series(np.where(values > 0, UP, DOWN), index=returns.index, dtype="int64")


def target_summary(returns: pd.Series, direction: pd.Series) -> dict:
    """Label counts for reports, including how many returns hit the zero boundary."""
    return {
        "target_version": TARGET_VERSION,
        "rule": DIRECTION_RULE,
        "n_rows": int(len(direction)),
        "n_up": int((direction == UP).sum()),
        "n_down": int((direction == DOWN).sum()),
        "n_zero_return_labelled_down": int((returns == 0).sum()),
        "up_share": float(direction.mean()) if len(direction) else float("nan"),
    }
