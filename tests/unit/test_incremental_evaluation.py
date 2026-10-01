"""Phase 7 incremental A/B/C evaluation through the central harness (synthetic store, offline)."""

import json
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from services.event_taxonomy import EVENT_TYPES
from training.feature_store import FEATURE_FAMILIES, OUTPUT_COLUMNS
from training.incremental_evaluation import (
    EXPERIMENTS,
    SMALL_SAMPLE_NOTE,
    IncrementalEvaluationError,
    evaluation_frame,
    experiment_columns,
    run_incremental_evaluation,
    save_report,
)


def make_store(n=320, covered_from=20, symbol="AAPL", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n).date
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, n + 1))
    df = pd.DataFrame({"symbol": symbol, "trading_date": list(dates),
                       "prediction_timestamp": pd.Timestamp("2020-01-01", tz=timezone.utc), "Close": closes[:-1]})
    for c in FEATURE_FAMILIES["technical"]:
        df[c] = rng.normal(0, 1, n)
    covered = np.arange(n) >= covered_from
    for c in (*FEATURE_FAMILIES["sentiment"], *FEATURE_FAMILIES["events"]):
        df[c] = np.where(covered, rng.poisson(2, n).astype(float), np.nan)
    df["event__dominant_event_type"] = np.where(covered, EVENT_TYPES[0], None)
    df["sentiment__covered"] = covered
    df["event__covered"] = covered
    df["target_return_1d"] = closes[1:] / closes[:-1] - 1
    df["target_direction_1d"] = (df["target_return_1d"] > 0).astype(int)
    return df[list(OUTPUT_COLUMNS)]


def test_experiment_feature_sets_are_nested():
    a, b, c = (experiment_columns(f) for f in EXPERIMENTS.values())
    assert (len(a), len(b), len(c)) == (20, 35, 58)
    assert set(a) < set(b) < set(c)
    assert not any(col.startswith(("sentiment__", "event__")) for col in a)


def test_only_rows_with_every_family_are_evaluated():
    frame = evaluation_frame(make_store(100, covered_from=30))
    assert len(frame) == 70 and not frame.isna().any().any()
    assert frame["Date"].is_monotonic_increasing


def test_insufficient_data_runs_nothing():
    report = run_incremental_evaluation(make_store(60, covered_from=37))           # 23 rows, like the sample
    assert report["status"] == "INSUFFICIENT_DATA" and report["evaluable_rows"] == 23
    assert all(e["status"] == "NOT_RUN" and "models" not in e for e in report["experiments"].values())
    assert report["note"] == SMALL_SAMPLE_NOTE and "no metric is reported" in report["reason"].lower()


def test_three_experiments_are_apples_to_apples(tmp_path):
    report = run_incremental_evaluation(make_store(320, covered_from=20))
    assert report["status"] == "RUN" and report["evaluable_rows"] == 300
    exps = report["experiments"]
    assert list(exps) == list(EXPERIMENTS)
    assert [e["n_features"] for e in exps.values()] == [20, 35, 58]
    periods = {json.dumps(e["evaluation_period"], sort_keys=True) for e in exps.values()}
    assert len(periods) == 1                                                       # identical OOS rows
    for e in exps.values():
        assert set(e["qualification_result"]) == {"regression", "classification"}
        assert {m["task"] for m in e["models"].values()} == {"regression", "classification"}
    assert set(report["comparison"]) == {"regression", "classification"}

    saved = save_report(report, tmp_path)
    json.loads(saved.read_text(encoding="utf-8"),
               parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_multi_symbol_store_rejected():
    store = pd.concat([make_store(60), make_store(60, symbol="MSFT")], ignore_index=True)
    with pytest.raises(IncrementalEvaluationError, match="one series"):
        evaluation_frame(store)
