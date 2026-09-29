"""
Tests for training/evaluation_harness.py and the technical_v2 feature schema.

Covers REQ-TEST-002: no future leakage, chronological ordering, target
alignment, fold isolation, feature schema, baseline calculation.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.feature_engineering import (
    NON_PREDICTIVE_COLUMNS,
    TECHNICAL_FEATURES,
    engineer_features,
)
from training.evaluation_harness import (
    CLASSIFICATION,
    DEFAULT_DATASET,
    NO_QUALIFIED_MODEL,
    NOT_QUALIFIED,
    QUALIFIED,
    REGRESSION,
    Candidate,
    DatasetValidationError,
    HarnessConfig,
    build_targets,
    classification_metrics,
    default_candidates,
    diebold_mariano,
    load_dataset,
    make_splitter,
    regression_metrics,
    run_evaluation,
    save_results,
    validate_dataset,
    walk_forward,
)

FEATURES = ["f1", "f2", "f3"]


# ==========================================
# Helpers
# ==========================================


def synthetic_frame(n: int = 600, signal: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """
    Daily frame with Close and three features. f1[t] carries `signal`
    about the NEXT day's return; f2/f3 are pure noise. signal=0 -> no
    learnable relationship at all.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.cumprod(1 + returns)
    next_return = np.append(returns[1:], 0.0)  # Close[t+1]/Close[t]-1
    f1 = signal * next_return / 0.02 + rng.normal(0, 1, n)
    return pd.DataFrame({
        "Date": pd.bdate_range("2015-01-01", periods=n, tz="UTC"),
        "Close": close,
        "f1": f1,
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })


def synthetic_ohlcv(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "Dividends": 0.0,
        "Stock Splits": 0.0,
    })


def regression_only(*cands: Candidate) -> list[Candidate]:
    base = [c for c in default_candidates() if c.task == REGRESSION and c.is_baseline]
    return base + list(cands)


# ==========================================
# Real dataset: data contract
# ==========================================


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    if not DEFAULT_DATASET.is_file():
        pytest.skip("data/final_stock_dataset.csv not present")
    return load_dataset(DEFAULT_DATASET)


def test_real_dataset_passes_contract(real_df):
    summary = validate_dataset(real_df, TECHNICAL_FEATURES)
    assert summary["rows"] > 2000
    assert real_df["Date"].is_monotonic_increasing
    assert not real_df["Date"].duplicated().any()
    values = real_df[["Close", *TECHNICAL_FEATURES]]
    assert not values.isna().any().any()
    assert not np.isinf(values).any().any()


def test_real_dataset_uses_clean_schema(real_df):
    forbidden = set(NON_PREDICTIVE_COLUMNS) | {"Daily_Return"}
    assert not forbidden & set(real_df.columns)
    assert not [c for c in real_df.columns if "_Lag_" in c]
    assert set(TECHNICAL_FEATURES) <= set(real_df.columns)


def test_real_dataset_target_alignment(real_df):
    # raises DatasetValidationError if stored Target != Close[t+1]/Close[t]-1
    out = build_targets(real_df, horizon=1)
    expected = real_df["Close"].shift(-1) / real_df["Close"] - 1
    assert np.allclose(out["future_return"], expected.iloc[:-1])


# ==========================================
# Validation catches bad data
# ==========================================


@pytest.mark.parametrize("corruption, message", [
    (lambda d: d.assign(f1=d["f1"].where(d.index != 5, np.nan)), "NaN"),
    (lambda d: d.assign(f2=d["f2"].where(d.index != 7, np.inf)), "Infinite"),
    (lambda d: d.assign(Date=d["Date"].where(d.index != 10, d["Date"].iloc[9])), "duplicate"),
    (lambda d: d.iloc[::-1].reset_index(drop=True), "chronological"),
    (lambda d: d.drop(columns=["f3"]), "Missing required columns"),
])
def test_validation_rejects_bad_data(corruption, message):
    bad = corruption(synthetic_frame(100))
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(bad, FEATURES)


# ==========================================
# Target construction
# ==========================================


def test_build_targets_alignment_and_drop():
    df = synthetic_frame(50)
    out = build_targets(df, horizon=1)
    assert len(out) == 49
    assert np.allclose(out["future_return"], (df["Close"].shift(-1) / df["Close"] - 1).iloc[:-1])
    assert (out["direction"] == (out["future_return"] > 0).astype(int)).all()


def test_build_targets_multi_day_horizon():
    df = synthetic_frame(50)
    out = build_targets(df, horizon=5)
    assert len(out) == 45
    assert np.isclose(out["future_return"].iloc[0], df["Close"].iloc[5] / df["Close"].iloc[0] - 1)


def test_build_targets_rejects_misaligned_stored_target():
    df = synthetic_frame(50)
    df["Target"] = df["Close"] / df["Close"].shift(1) - 1  # PAST return: misaligned
    with pytest.raises(DatasetValidationError, match="Stored Target"):
        build_targets(df, horizon=1)


# ==========================================
# Feature engineering schema & no look-ahead
# ==========================================


def test_engineer_features_schema():
    out = engineer_features(synthetic_ohlcv())
    assert set(TECHNICAL_FEATURES) <= set(out.columns)
    assert not (set(NON_PREDICTIVE_COLUMNS) | {"Daily_Return"}) & set(out.columns)
    assert not [c for c in out.columns if "_Lag_" in c]


def test_return_features_are_k_day_changes():
    raw = synthetic_ohlcv()
    out = engineer_features(raw.copy())
    expected = raw.set_index("Date")["Close"]
    for k in [1, 2, 3, 5, 10]:
        exp = expected.pct_change(k).loc[out["Date"]].to_numpy()
        assert np.allclose(out[f"Return_{k}"], exp)


def test_features_do_not_use_future_rows():
    """Features for day t must be identical whether or not later days exist."""
    raw = synthetic_ohlcv(300)
    full = engineer_features(raw.copy()).set_index("Date")
    prefix = engineer_features(raw.iloc[:200].copy()).set_index("Date")
    overlap = prefix.index
    pd.testing.assert_frame_equal(prefix[TECHNICAL_FEATURES], full.loc[overlap, TECHNICAL_FEATURES])


# ==========================================
# Fold structure
# ==========================================


@pytest.mark.parametrize("horizon", [1, 5])
def test_twenty_folds_with_gap(horizon):
    config = HarnessConfig(horizon=horizon)
    splits = list(make_splitter(config).split(np.zeros((1000, 1))))
    assert len(splits) == 20
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() - train_idx.max() - 1 == horizon  # exactly `gap` rows skipped


# ==========================================
# Leakage: scaling / selection / fitting see training rows only
# ==========================================

FIT_LOG: list[np.ndarray] = []


class RecordingTransformer(TransformerMixin, BaseEstimator):
    """Records which rows (by index label) it was fitted on."""

    def fit(self, X, y=None):
        FIT_LOG.append(np.asarray(X.index))
        return self

    def transform(self, X):
        return X


def test_pipeline_steps_fit_on_training_rows_only():
    FIT_LOG.clear()
    data = build_targets(synthetic_frame(400), horizon=1)
    config = HarnessConfig(horizon=1)
    spy = Candidate("Spy", REGRESSION, lambda: Pipeline([
        ("record", RecordingTransformer()),
        ("model", LinearRegression()),
    ]))
    walk_forward(data, FEATURES, [spy], config)

    splits = list(make_splitter(config).split(data[FEATURES]))
    assert len(FIT_LOG) == len(splits)
    for fitted_rows, (train_idx, test_idx) in zip(FIT_LOG, splits):
        assert np.array_equal(np.sort(fitted_rows), train_idx)
        assert not set(fitted_rows) & set(test_idx)


def test_future_rows_cannot_change_past_predictions():
    """
    Scramble features AND targets of every row after fold 10's test
    window. Predictions for folds 1-10 must be bit-identical, even with
    target-driven feature selection and scaling inside the pipeline.
    """
    config = HarnessConfig(horizon=1)
    data = build_targets(synthetic_frame(600, signal=0.5), horizon=1)
    selecting = Candidate("Select+Linear", REGRESSION, lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_regression, k=1)),
        ("model", LinearRegression()),
    ]))

    before, _, folds = walk_forward(data, FEATURES, [selecting], config)

    cutoff = list(make_splitter(config).split(data[FEATURES]))[9][1].max()
    poisoned = data.copy()
    future = poisoned.index > cutoff
    rng = np.random.default_rng(99)
    for col in FEATURES + ["future_return"]:
        poisoned.loc[future, col] = rng.normal(0, 100, future.sum())

    after, _, _ = walk_forward(poisoned, FEATURES, [selecting], config)
    b = before[before["fold"] <= 10]["prediction"].to_numpy()
    a = after[after["fold"] <= 10]["prediction"].to_numpy()
    assert np.array_equal(a, b)
    # sanity: later folds DID change, so the poisoning had an effect
    assert not np.allclose(before[before["fold"] > 10]["prediction"], after[after["fold"] > 10]["prediction"])


# ==========================================
# Pooled OOS predictions & baselines
# ==========================================


@pytest.fixture(scope="module")
def noise_result():
    return run_evaluation(data=synthetic_frame(800, signal=0.0, seed=3), feature_columns=FEATURES,
                          feature_version="synthetic", experiment="test_noise")


def test_pooled_predictions_cover_every_test_row_once(noise_result):
    preds = noise_result.predictions
    n_oos = noise_result.report["evaluation_period"]["n_oos_rows"]
    for model, p in preds.groupby("model"):
        assert len(p) == n_oos
        assert p["row"].is_unique
    assert noise_result.report["n_splits"] == 20
    assert noise_result.report["gap"] == 1
    assert len(noise_result.report["folds"]) == 20
    assert len(noise_result.fold_metrics) == 20 * len(default_candidates())


def test_baseline_predictions(noise_result):
    data = build_targets(synthetic_frame(800, signal=0.0, seed=3), horizon=1)
    preds = noise_result.predictions
    splits = list(make_splitter(HarnessConfig()).split(data[FEATURES]))
    for fold, (train_idx, _) in enumerate(splits, start=1):
        in_fold = preds["fold"] == fold
        mean_pred = preds[in_fold & (preds["model"] == "Mean Return")]["prediction"]
        assert np.allclose(mean_pred, data["future_return"].iloc[train_idx].mean())
        base_rate = preds[in_fold & (preds["model"] == "Base Rate")]["prediction"]
        assert np.allclose(base_rate, data["direction"].iloc[train_idx].mean())
    assert (preds[preds["model"] == "Zero Return"]["prediction"] == 0).all()
    assert (preds[preds["model"] == "Always UP"]["prediction"] == 1).all()


# ==========================================
# Metrics
# ==========================================


def test_regression_metrics_values():
    m = regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    assert np.isclose(m["MAE"], 2 / 3)
    assert np.isclose(m["RMSE"], np.sqrt(4 / 3))
    assert np.isclose(m["R2"], 1 - 4 / 2)


def test_classification_metrics_values():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.6, 0.4])
    m = classification_metrics(y, p)
    assert m["Accuracy"] == 1.0
    assert m["Balanced_Accuracy"] == 1.0
    assert m["ROC_AUC"] == 1.0
    assert np.isclose(m["Brier"], (0.04 + 0.04 + 0.16 + 0.16) / 4)
    assert np.isclose(m["Log_Loss"], -np.mean(np.log([0.8, 0.8, 0.6, 0.6])))


def test_diebold_mariano():
    loss = np.random.default_rng(0).random(500)
    assert diebold_mariano(loss, loss, horizon=1)["p_value"] == 1.0
    assert diebold_mariano(loss * 0.5, loss, horizon=1)["p_value"] < 0.001  # model clearly better
    assert diebold_mariano(loss * 2.0, loss, horizon=1)["p_value"] > 0.999  # model clearly worse


# ==========================================
# Qualification gate
# ==========================================


def test_gate_rejects_models_on_pure_noise(noise_result):
    result = noise_result.report["qualification_result"]
    assert result[REGRESSION] == NO_QUALIFIED_MODEL
    assert result[CLASSIFICATION] == NO_QUALIFIED_MODEL
    for m in noise_result.report["models"]:
        if not m["is_baseline"]:
            assert m["qualification"]["status"] == NOT_QUALIFIED


def test_gate_accepts_models_with_real_signal():
    res = run_evaluation(data=synthetic_frame(800, signal=0.6, seed=4), feature_columns=FEATURES,
                         feature_version="synthetic", experiment="test_signal")
    status = {m["model"]: m["qualification"]["status"] for m in res.report["models"]}
    assert status["Linear Regression"] == QUALIFIED
    assert status["Ridge"] == QUALIFIED
    assert status["Logistic Regression"] == QUALIFIED


def test_gate_does_not_crown_highest_r2_when_all_fail(noise_result):
    """Even if one model has the best R2 among candidates, it must not qualify on noise."""
    s = noise_result.summary
    best_model = s[(s["task"] == REGRESSION) & ~s["is_baseline"]].sort_values("R2").iloc[-1]
    assert best_model["status"] == NOT_QUALIFIED


# ==========================================
# Results file
# ==========================================


def test_report_fields_and_strict_json(noise_result, tmp_path):
    paths = save_results(noise_result, tmp_path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"),
                        parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    for key in ["experiment", "dataset_version", "feature_version", "target", "models",
                "validation_method", "n_splits", "gap", "baseline_metrics",
                "qualification_result", "training_period", "evaluation_period", "timestamp"]:
        assert key in report
    assert paths["summary"].is_file() and paths["predictions"].is_file()
