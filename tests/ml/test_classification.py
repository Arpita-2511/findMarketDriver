"""
Phase 3 - UP/DOWN target and classification benchmark.

Covers: target construction, alignment, horizon handling, the zero-return
boundary, no future leakage, chronological ordering, compatibility with
the central evaluation harness, and reproducibility.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.evaluation_harness import (
    CLASSIFICATION,
    NO_QUALIFIED_MODEL,
    QUALIFIED,
    REGRESSION,
    Candidate,
    DatasetValidationError,
    HarnessConfig,
    build_targets,
    default_candidates,
    make_splitter,
    print_report,
    reliability_table,
    run_evaluation,
    save_results,
    walk_forward,
)
from training.targets import DOWN, UP, direction_label, future_return, target_summary
from training.train_classification import classification_candidates

FEATURES = ["f1", "f2", "f3"]


def frame(n: int = 500, signal: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Close + 3 features; f1[t] carries `signal` about the next-day direction."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    next_return = np.append(returns[1:], 0.0)
    return pd.DataFrame({
        "Date": pd.bdate_range("2016-01-04", periods=n, tz="UTC"),
        "Close": close,
        "f1": signal * np.sign(next_return) + rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })


# ==========================================
# Target construction & boundary
# ==========================================


def test_future_return_formula_and_trailing_nan():
    close = pd.Series([100.0, 110.0, 99.0, 99.0, 120.0])
    r1 = future_return(close, 1)
    assert np.allclose(r1[:4], [0.1, 99 / 110 - 1, 0.0, 120 / 99 - 1])
    assert r1.iloc[-1:].isna().all()
    r2 = future_return(close, 2)
    assert np.isclose(r2.iloc[0], 99 / 100 - 1)
    assert r2.iloc[-2:].isna().all()


@pytest.mark.parametrize("ret, label", [
    (0.02, UP),
    (1e-12, UP),        # any genuine rise is UP - no dead zone
    (0.0, DOWN),        # exactly unchanged is not UP
    (-0.0, DOWN),
    (-1e-12, DOWN),
    (-0.03, DOWN),
])
def test_direction_boundary(ret, label):
    assert direction_label(pd.Series([ret])).iloc[0] == label


def test_identical_adjusted_closes_give_exact_zero_and_down():
    close = pd.Series([187.123456789, 187.123456789, 190.0])
    r = future_return(close, 1)
    assert r.iloc[0] == 0.0                      # no floating-point residue
    assert direction_label(r.iloc[:2]).tolist() == [DOWN, UP]


def test_unlabelable_returns_are_an_error_not_down():
    with pytest.raises(ValueError, match="NaN/inf"):
        direction_label(pd.Series([0.01, np.nan]))


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_horizon_rejected(bad):
    with pytest.raises(ValueError, match="horizon"):
        future_return(pd.Series([1.0, 2.0, 3.0]), bad)


def test_target_summary_counts_zero_boundary():
    r = pd.Series([0.01, 0.0, -0.02, 0.0])
    s = target_summary(r, direction_label(r))
    assert (s["n_up"], s["n_down"], s["n_zero_return_labelled_down"]) == (1, 3, 2)
    assert s["up_share"] == 0.25


# ==========================================
# Alignment, horizon, no future leakage in labels
# ==========================================


@pytest.mark.parametrize("h", [1, 5])
def test_label_at_t_compares_close_t_plus_h_with_close_t(h):
    df = frame(120)
    out = build_targets(df, horizon=h)
    assert len(out) == len(df) - h
    expected = (df["Close"].shift(-h) > df["Close"]).astype(int).iloc[:-h].to_numpy()
    assert np.array_equal(out["direction"].to_numpy(), expected)
    # features stay on their own row - labels are attached, rows never shifted
    assert np.array_equal(out["f1"].to_numpy(), df["f1"].iloc[:-h].to_numpy())
    assert (out["Date"] == df["Date"].iloc[:-h].reset_index(drop=True)).all()


@pytest.mark.parametrize("h", [1, 5])
def test_label_ignores_prices_after_t_plus_h(h):
    df = frame(120)
    t = 40
    base = build_targets(df, horizon=h)

    later = df.copy()
    later.loc[t + h + 1:, "Close"] *= 3.0            # everything after t+h changes
    assert build_targets(later, horizon=h)["direction"].iloc[t] == base["direction"].iloc[t]

    moved = df.copy()                                 # sanity: Close[t+h] itself matters
    moved.loc[t + h, "Close"] = df.loc[t, "Close"] * (0.5 if base["direction"].iloc[t] else 2.0)
    assert build_targets(moved, horizon=h)["direction"].iloc[t] != base["direction"].iloc[t]


def test_zero_return_row_in_harness_is_down():
    df = frame(120)
    df.loc[51, "Close"] = df.loc[50, "Close"]         # flat day after row 50
    out = build_targets(df, horizon=1)
    assert out.loc[50, "future_return"] == 0.0
    assert out.loc[50, "direction"] == DOWN


@pytest.mark.parametrize("h", [1, 5])
def test_gap_equals_horizon_so_train_labels_end_before_test(h):
    df = build_targets(frame(600), horizon=h)
    for train_idx, test_idx in make_splitter(HarnessConfig(horizon=h)).split(df[FEATURES]):
        last_label_close = train_idx.max() + h        # latest close any training label uses
        assert last_label_close < test_idx.min()      # known before the first test prediction


def test_unsorted_data_rejected_before_labelling():
    df = frame(200).iloc[::-1].reset_index(drop=True)
    with pytest.raises(DatasetValidationError, match="chronological"):
        run_evaluation(data=df, feature_columns=FEATURES, candidates=classification_candidates())


# ==========================================
# No leakage inside the classification pipeline
# ==========================================


def test_future_rows_cannot_change_past_classifier_predictions():
    config = HarnessConfig(horizon=1)
    data = build_targets(frame(600, signal=0.5), horizon=1)
    selecting = Candidate("Select+Logistic", CLASSIFICATION, lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=1)),
        ("model", LogisticRegression(max_iter=1000)),
    ]))
    before, _, _ = walk_forward(data, FEATURES, [selecting], config)

    cutoff = list(make_splitter(config).split(data[FEATURES]))[9][1].max()
    poisoned = data.copy()
    future = poisoned.index > cutoff
    rng = np.random.default_rng(1)
    for col in FEATURES:
        poisoned.loc[future, col] = rng.normal(0, 50, future.sum())
    poisoned.loc[future, "direction"] = 1 - poisoned.loc[future, "direction"]

    after, _, _ = walk_forward(poisoned, FEATURES, [selecting], config)
    early = lambda p: p[p["fold"] <= 10]["prediction"].to_numpy()
    assert np.array_equal(early(before), early(after))


# ==========================================
# Phase 3 benchmark through the central harness
# ==========================================


def test_candidate_set_matches_req_ml_001():
    cands = classification_candidates()
    assert all(c.task == CLASSIFICATION for c in cands)
    assert [c.name for c in cands if c.is_baseline] == ["Always UP", "Base Rate"]
    assert [c.name for c in cands if not c.is_baseline] == [
        "Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


@pytest.fixture(scope="module")
def noise_run():
    return run_evaluation(data=frame(500, signal=0.0, seed=11), feature_columns=FEATURES,
                          feature_version="synthetic", candidates=classification_candidates(),
                          experiment="test_classification_noise")


def test_classification_only_run_reports_only_classification(noise_run):
    assert set(noise_run.report["qualification_result"]) == {CLASSIFICATION}
    assert REGRESSION not in set(noise_run.summary["task"])


def test_every_model_scores_every_oos_row_once_with_valid_probabilities(noise_run):
    n_oos = noise_run.report["evaluation_period"]["n_oos_rows"]
    for model, p in noise_run.predictions.groupby("model"):
        assert len(p) == n_oos and p["row"].is_unique
        assert p["prediction"].between(0, 1).all()


def test_no_model_qualifies_on_noise(noise_run):
    assert noise_run.report["qualification_result"][CLASSIFICATION] == NO_QUALIFIED_MODEL


def test_report_contains_target_summary_and_reliability(noise_run, tmp_path):
    r = noise_run.report
    oos = r["target_summary"]["pooled_oos_rows"]
    assert oos["n_rows"] == r["evaluation_period"]["n_oos_rows"]
    assert oos["n_up"] + oos["n_down"] == oos["n_rows"]
    for m in r["models"]:
        assert sum(b["n"] for b in m["reliability"]) == oos["n_rows"]

    paths = save_results(noise_run, tmp_path)
    json.loads(paths["report"].read_text(encoding="utf-8"),
               parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_print_report_classification_only_run(noise_run, capsys):
    """Regression test: print_report() crashed with KeyError ['MAE','RMSE','R2'] here."""
    assert not {"MAE", "RMSE", "R2"} & set(noise_run.summary.columns)   # the crash scenario

    print_report(noise_run)
    out = capsys.readouterr().out

    assert "-- Classification (pooled OOS) --" in out
    assert "-- Regression" not in out and "MAE" not in out
    for name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
        assert name in out
    assert f"{CLASSIFICATION:15s}: {NO_QUALIFIED_MODEL}" in out
    assert REGRESSION not in out.split("Result:")[1]


def test_print_report_full_run_still_prints_both_tracks(capsys):
    res = run_evaluation(data=frame(300, seed=4), feature_columns=FEATURES, feature_version="synthetic",
                         candidates=default_candidates(), config=HarnessConfig(n_splits=5),
                         experiment="test_full_print")
    print_report(res)
    out = capsys.readouterr().out

    assert "-- Regression (pooled OOS) --" in out
    assert "-- Classification (pooled OOS) --" in out
    assert "MAE" in out and "Log_Loss" in out
    result_block = out.split("Result:")[1]
    assert REGRESSION in result_block and CLASSIFICATION in result_block


def test_logistic_qualifies_when_signal_exists():
    res = run_evaluation(data=frame(800, signal=1.0, seed=5), feature_columns=FEATURES,
                         feature_version="synthetic", candidates=classification_candidates()[:3],
                         experiment="test_classification_signal")
    status = {m["model"]: m["qualification"]["status"] for m in res.report["models"]}
    assert status["Logistic Regression"] == QUALIFIED


def test_benchmark_is_reproducible():
    kwargs = dict(data=frame(300, seed=9), feature_columns=FEATURES, feature_version="synthetic",
                  candidates=classification_candidates(), config=HarnessConfig(n_splits=5),
                  experiment="test_repro")
    a, b = run_evaluation(**kwargs), run_evaluation(**kwargs)
    pd.testing.assert_frame_equal(a.predictions, b.predictions, check_exact=True)


# ==========================================
# Reliability table
# ==========================================


def test_reliability_table_bins():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.95, 1.0, 0.55])
    table = reliability_table(y, p, n_bins=10)
    assert len(table) == 10 and sum(b["n"] for b in table) == 5
    assert table[0]["n"] == 1 and table[1]["n"] == 1 and table[5]["n"] == 1
    assert table[9]["n"] == 2 and table[9]["observed_up_rate"] == 0.5   # 1.0 lands in the last bin
    assert table[3]["mean_predicted"] is None
