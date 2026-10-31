"""
Phase 12 Stage 2: stock movement magnitude prediction (development period only).

Question: can the approved Stage 1 features predict |future excess return over SPY| more accurately
than a volatility-only baseline?

    target   y_h = |(Close[t+h]/Close[t] - 1) - (SPY[t+h]/SPY[t] - 1)|    h = 1 (primary), 5 (secondary)
    features Volatility, spy_volatility_20, Volume_Change_1, Volume_Change_5, event__earnings_event_count,
             event__event_count, event__negative_event_count, sw_count_surprise_1_20 (frozen frames, unchanged)
    data     the 29 hash-verified Phase 11B development frames (2017-02-01 .. 2024-09-30); nothing after the
             development cutoff is read

The protocol (PROTOCOL, written to stage2/protocol.json before any model is fitted) fixes the models,
hyperparameters, walk-forward design, metrics, statistics, comparison families and qualification rule.
`run` refuses unless that file exists and equals the protocol in code.

    python -m training.phase12_magnitude protocol   # stage2/protocol.json (write-once, no computation)
    python -m training.phase12_magnitude run        # stage2/results/ (write-once)
    python -m training.phase12_magnitude status
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11b_research as r  # noqa: E402
import training.phase12_signal_diagnostics as s1  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.phase9_research import ResearchError, _write_new_json, canonical_json, environment, holm_adjust  # noqa: E402

PHASE, STAGE = "12", "12.S2"
PROTOCOL_VERSION = "phase12_stage2_magnitude_protocol_v1"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage2"
SEED = 42
FEATURES = ("Volatility", "spy_volatility_20", "Volume_Change_1", "Volume_Change_5", "event__earnings_event_count",
            "event__event_count", "event__negative_event_count", "sw_count_surprise_1_20")
VOL_FEATURES = ("Volatility", "spy_volatility_20")
STOCK_SCALED = tuple(f for f in FEATURES if f not in VOL_FEATURES)     # per-stock z-scored in pooled models
REFERENCE = "baseline_ridge_vol"
CANDIDATES = ("ridge", "random_forest", "lightgbm")
MODEL_ROLES = {"median": "benchmark", "scaled_vol": "benchmark", REFERENCE: "reference",
               "ridge": "candidate", "random_forest": "candidate", "lightgbm": "candidate", "lightgbm_vol": "control"}
CODE_FILES = ("training/phase12_magnitude.py", "training/phase12_signal_diagnostics.py",
              "training/excess_targets.py", "training/phase11b_research.py")


class Phase12S2Error(ResearchError):
    """Phase 12 Stage 2 protocol violation; the run stops."""


@dataclass(frozen=True)
class Config:
    horizons: tuple = (1, 5)
    min_train: int = 504                  # sessions (~2 years) before the first test row
    test_len: int = 63                    # sessions per walk-forward fold (~1 quarter)
    min_last_fold: int = 21               # a shorter final fold is merged into the previous one
    clip_quantiles: tuple = (0.01, 0.99)  # ridge inputs clipped to training-fold quantiles
    ridge_alpha: float = 1.0
    rf_trees: int = 200
    rf_depth: int = 4
    rf_max_features: float = 0.5
    lgb_trees: int = 300
    lgb_lr: float = 0.03
    lgb_leaves: int = 7
    lgb_depth: int = 3
    min_leaf_pooled: int = 200
    min_leaf_stock: int = 50
    alpha: float = 0.05
    min_rel_mse_improvement: float = 0.01
    min_rel_mae_improvement: float = 0.0
    n_blocks: int = 4
    min_blocks_improved: int = 3
    min_share_stocks_improved: float = 0.6
    n_repro_stocks: int = 5


FROZEN_CONFIG = Config()


def lag_for(n_dates: int, h: int) -> int:
    return max(h, math.ceil(n_dates ** (1 / 3)))


def protocol(cfg: Config = FROZEN_CONFIG) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION, "phase": PHASE, "stage": STAGE,
        "status": "PRE-REGISTERED before any model fit",
        "target": {"definition": "|future_excess_return_h|, future_excess_return_h from excess_return_v1 "
                                 "(training.excess_targets.build_excess_targets); last h rows unlabelled",
                   "horizons": {"1": "primary", "5": "secondary"}},
        "features": list(FEATURES), "volatility_features": list(VOL_FEATURES),
        "excluded": "direction prediction; any other feature; stock identity (never a feature)",
        "data": "29 hash-verified Phase 11B development frames, 2017-02-01 .. 2024-09-30; sealed period never read",
        "models": {
            "median": "benchmark: training-fold median of y",
            "scaled_vol": "benchmark: c * Volatility, c = least squares through the origin on the training fold",
            REFERENCE: f"REFERENCE BASELINE: Ridge(alpha={cfg.ridge_alpha}) on Volatility, spy_volatility_20",
            "ridge": f"candidate: Ridge(alpha={cfg.ridge_alpha}) on the 8 features",
            "random_forest": (f"candidate: RandomForestRegressor(n_estimators={cfg.rf_trees}, max_depth={cfg.rf_depth}, "
                              f"max_features={cfg.rf_max_features}, min_samples_leaf={cfg.min_leaf_pooled} pooled / "
                              f"{cfg.min_leaf_stock} per stock, random_state={SEED}, n_jobs=1)"),
            "lightgbm": (f"candidate: LGBMRegressor(n_estimators={cfg.lgb_trees}, learning_rate={cfg.lgb_lr}, "
                         f"num_leaves={cfg.lgb_leaves}, max_depth={cfg.lgb_depth}, min_child_samples="
                         f"{cfg.min_leaf_pooled} pooled / {cfg.min_leaf_stock} per stock, subsample=0.8, "
                         f"subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state={SEED}, "
                         f"n_jobs=1, deterministic=True)"),
            "lightgbm_vol": "descriptive control: the lightgbm candidate on the 2 volatility features only",
            "no_tuning": True},
        "preprocessing": {
            "pooled": ("the 6 non-volatility features are z-scored per stock with that stock's training-fold mean "
                       "and std (std 0 -> 1); volatility features stay raw; no stock identity feature"),
            "ridge": (f"inputs clipped to training-fold quantiles {list(cfg.clip_quantiles)}, then StandardScaler fit "
                      "on the training fold"),
            "trees": "no scaling", "missing_values": "none expected; any non-finite value stops the run (no imputation)",
            "predictions": "floored at 0 for every model"},
        "validation": {
            "type": "expanding-window walk-forward over the common session index, identical folds for all models",
            "min_train_sessions": cfg.min_train, "test_sessions_per_fold": cfg.test_len,
            "gap": "h: a training row i satisfies i + h < first test row (its label is complete before prediction)",
            "last_fold": f"a final fold shorter than {cfg.min_last_fold} sessions is merged into the previous fold",
            "scopes": {"pooled": "one model on all 29 stocks (primary)",
                       "per_stock": "one model per stock (secondary)"}},
        "metrics": ["MSE", "RMSE", "MAE", "R2_oos vs reference = 1 - MSE/MSE_reference",
                    "relative MAE improvement", "R2 vs training-median benchmark",
                    "prediction deciles (mean predicted vs actual)", "top-5% actual moves: mean predicted / mean actual",
                    "per-block and per-stock relative MSE improvement"],
        "statistics": {
            "test": ("one-sided Diebold-Mariano, H0: candidate squared error not lower than the reference's; "
                     "pooled: per-date cross-sectional mean loss differential; per stock: that stock's series"),
            "variance": "Newey-West, Bartlett kernel, lag L = max(h, ceil(T^(1/3))), T = number of test dates; 2L sensitivity",
            "families": {"primary": "pooled candidates x horizons = 6 comparisons, Holm",
                         "secondary": "per-stock candidates x 29 stocks x horizons = 174 comparisons, Holm"},
            "alpha": cfg.alpha},
        "qualification": {
            "applies_to": "each candidate x horizon (pooled: primary family; per stock: secondary family)",
            "Q1_improvement": (f"relative MSE improvement over the reference >= {cfg.min_rel_mse_improvement} AND "
                               f"relative MAE improvement >= {cfg.min_rel_mae_improvement}"),
            "Q2_statistics": f"Holm-adjusted one-sided DM p < {cfg.alpha} within the family AND raw p with lag 2L < {cfg.alpha}",
            "Q3_periods": (f"MSE better than the reference in >= {cfg.min_blocks_improved} of {cfg.n_blocks} equal "
                           "consecutive blocks of test dates"),
            "Q4_leakage": "every fold satisfies the gap check; the automated leakage tests pass",
            "Q5_breadth": (f"pooled only: MSE improved for >= {cfg.min_share_stocks_improved:.0%} of stocks, and the "
                           "improvement stays > 0 without the single best stock and without the single best fold"),
            "Q6_reproducibility": (f"a second in-process run gives bit-identical predictions (pooled: every model; "
                                   f"per stock: {cfg.n_repro_stocks} hash-chosen stocks)"),
            "rule": "QUALIFIED only if Q1-Q6 all pass; thresholds are fixed here and never changed after results"},
        "finalists": "pooled qualifiers are refit on all development rows and saved; nothing else is saved",
        "holdout": "sealed confirmation period untouched; evaluated at most once, only for finalists, only with owner approval",
        "config": dataclasses.asdict(cfg),
    }


def protocol_sha256(cfg: Config = FROZEN_CONFIG) -> str:
    return hashlib.sha256(canonical_json(protocol(cfg)).encode("utf-8")).hexdigest()


# ============================================================
# Data and folds
# ============================================================


def build_panel(frames: dict[str, pd.DataFrame], h: int) -> pd.DataFrame:
    """Labelled rows of every stock: symbol, pos (common session index), Date, the 8 features, y."""
    s1.check_development_only(frames)
    parts, dates = [], None
    for sym in sorted(frames):
        lab = s1.build_targets(frames[sym], h)
        d = list(pd.to_datetime(lab["Date"]).dt.date)
        if dates is not None and d != dates:
            raise Phase12S2Error("stocks do not share the same labelled sessions")
        dates = d
        part = pd.DataFrame({"symbol": sym, "pos": np.arange(len(lab)), "Date": lab["Date"].to_numpy()})
        for f in FEATURES:
            part[f] = lab[f].to_numpy(dtype="float64")
        part["y"] = lab["t_magnitude"].to_numpy(dtype="float64")
        parts.append(part)
    panel = pd.concat(parts, ignore_index=True)
    if not np.isfinite(panel[list(FEATURES) + ["y"]].to_numpy()).all():
        raise Phase12S2Error("non-finite feature or target value (no imputation by protocol)")
    return panel


def make_folds(n_labelled: int, h: int, cfg: Config) -> list[tuple[int, int, int]]:
    """(train_end, test_start, test_end) on the session index; training rows i < train_end = test_start - h."""
    folds, start = [], cfg.min_train + h
    while start < n_labelled:
        end = min(start + cfg.test_len, n_labelled)
        folds.append((start - h, start, end))
        start = end
    if len(folds) > 1 and folds[-1][2] - folds[-1][1] < cfg.min_last_fold:
        last = folds.pop()
        folds[-1] = (folds[-1][0], folds[-1][1], last[2])
    if not folds:
        raise Phase12S2Error("not enough sessions for one fold")
    for tr_end, st, _ in folds:
        if (tr_end - 1) + h >= st:
            raise Phase12S2Error("fold violates the horizon gap")
    return folds


# ============================================================
# Models
# ============================================================


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, q=(0.01, 0.99)):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype="float64")
        self.lo_, self.hi_ = np.quantile(X, self.q[0], axis=0), np.quantile(X, self.q[1], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.lo_, self.hi_)


class Median:
    def fit(self, X, y):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class ScaledVol:
    def fit(self, X, y):
        v = np.asarray(X)[:, 0]
        self.c_ = float(np.dot(v, y) / np.dot(v, v))
        return self

    def predict(self, X):
        return self.c_ * np.asarray(X)[:, 0]


def model_columns(name: str) -> tuple:
    return {"median": VOL_FEATURES[:1], "scaled_vol": ("Volatility",), REFERENCE: VOL_FEATURES,
            "lightgbm_vol": VOL_FEATURES}.get(name, FEATURES)


def make_model(name: str, scope: str, cfg: Config):
    leaf = cfg.min_leaf_pooled if scope == "pooled" else cfg.min_leaf_stock
    if name == "median":
        return Median()
    if name == "scaled_vol":
        return ScaledVol()
    if name in (REFERENCE, "ridge"):
        return make_pipeline(QuantileClipper(cfg.clip_quantiles), StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth, max_features=cfg.rf_max_features,
                                     min_samples_leaf=leaf, random_state=SEED, n_jobs=1)
    if name in ("lightgbm", "lightgbm_vol"):
        return LGBMRegressor(n_estimators=cfg.lgb_trees, learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_leaves,
                             max_depth=cfg.lgb_depth, min_child_samples=leaf, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=1, deterministic=True,
                             verbose=-1)
    raise Phase12S2Error(f"unknown model {name}")


def pooled_scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """z-score STOCK_SCALED per stock with training-fold statistics only."""
    stats_ = {}
    tr, te = train.copy(), test.copy()
    for sym, g in train.groupby("symbol", sort=True):
        mu, sd = g[list(STOCK_SCALED)].mean(), g[list(STOCK_SCALED)].std(ddof=0).replace(0.0, 1.0)
        stats_[sym] = (mu, sd)
    for df in (tr, te):
        for sym, idx in df.groupby("symbol", sort=True).groups.items():
            if sym not in stats_:
                raise Phase12S2Error(f"{sym} has test rows but no training rows")
            mu, sd = stats_[sym]
            df.loc[idx, list(STOCK_SCALED)] = (df.loc[idx, list(STOCK_SCALED)] - mu) / sd
    return tr, te, stats_


def walk_forward(panel: pd.DataFrame, h: int, cfg: Config, scope: str, models=tuple(MODEL_ROLES)) -> pd.DataFrame:
    """Out-of-sample predictions of every model on identical rows. `panel` may hold one stock (per_stock)."""
    n_lab = int(panel["pos"].max()) + 1
    out = []
    for k, (tr_end, st, en) in enumerate(make_folds(n_lab, h, cfg)):
        train = panel[panel["pos"] < tr_end]
        test = panel[(panel["pos"] >= st) & (panel["pos"] < en)]
        if int(train["pos"].max()) + h >= int(test["pos"].min()):
            raise Phase12S2Error("leakage: a training label overlaps the test period")
        if scope == "pooled":
            train, test, _ = pooled_scale(train, test)
        res = test[["symbol", "pos", "Date", "y"]].copy()
        res["fold"] = k
        for name in models:
            cols = list(model_columns(name))
            m = make_model(name, scope, cfg).fit(train[cols].to_numpy(), train["y"].to_numpy())
            res[name] = np.maximum(0.0, np.asarray(m.predict(test[cols].to_numpy()), dtype="float64"))
        out.append(res)
    return pd.concat(out, ignore_index=True)


# ============================================================
# Evaluation
# ============================================================


def dm_one_sided(d: np.ndarray, lag: int) -> dict:
    """H0: mean(d) <= 0, d = loss_reference - loss_candidate; Newey-West (Bartlett) variance with `lag`."""
    d = np.asarray(d, dtype="float64")
    n = len(d)
    c = d - d.mean()
    lrv = float(s1.bartlett_sum(c, lag)[0]) / n
    if lrv <= 0:
        return {"mean_diff": float(d.mean()), "t": 0.0, "p": 1.0}
    t = d.mean() / math.sqrt(lrv / n)
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(stats.norm.sf(t))}


def basic_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    mse = float(np.mean(e ** 2))
    return {"n": int(len(y)), "mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(e))),
            "mean_pred": float(np.mean(p)), "mean_actual": float(np.mean(y)), "bias_actual_minus_pred": float(np.mean(e))}


def _rel(a: float, b: float) -> float:
    return 1.0 - a / b if b > 0 else float("nan")


def compare(preds: pd.DataFrame, model: str, h: int, cfg: Config) -> dict:
    """Candidate vs reference on identical rows: metrics, DM (date-mean loss differential), blocks, breadth."""
    y, pc, pb = preds["y"].to_numpy(), preds[model].to_numpy(), preds[REFERENCE].to_numpy()
    lc, lb = (y - pc) ** 2, (y - pb) ** 2
    mc, mb = basic_metrics(y, pc), basic_metrics(y, pb)
    by_date = pd.DataFrame({"pos": preds["pos"].to_numpy(), "d": lb - lc}).groupby("pos", sort=True)["d"].mean()
    T = len(by_date)
    L = lag_for(T, h)
    dm, dm2 = dm_one_sided(by_date.to_numpy(), L), dm_one_sided(by_date.to_numpy(), 2 * L)
    dates = np.array(sorted(by_date.index))
    blocks = np.array_split(dates, cfg.n_blocks)
    block_rel = []
    for b in blocks:
        m = preds["pos"].isin(b).to_numpy()
        block_rel.append(_rel(lc[m].mean(), lb[m].mean()))
    stock_rel = {sym: _rel(lc[g].mean(), lb[g].mean())
                 for sym, g in preds.groupby("symbol", sort=True).indices.items()}
    best_stock = max(stock_rel, key=stock_rel.get) if len(stock_rel) > 1 else None
    keep = (preds["symbol"] != best_stock).to_numpy()
    without_best_stock = _rel(lc[keep].mean(), lb[keep].mean()) if best_stock is not None else None
    fold_rel = {int(f): _rel(lc[g].mean(), lb[g].mean()) for f, g in preds.groupby("fold", sort=True).indices.items()}
    best_fold = max(fold_rel, key=fold_rel.get)
    keep_f = (preds["fold"] != best_fold).to_numpy()
    return {
        "model": model, "horizon": h, "metrics": mc, "reference_metrics": mb,
        "rel_mse_improvement": _rel(mc["mse"], mb["mse"]), "rel_mae_improvement": _rel(mc["mae"], mb["mae"]),
        "r2_oos_vs_reference": _rel(mc["mse"], mb["mse"]),
        "dm": {**dm, "lag": L, "t_2L": dm2["t"], "p_2L": dm2["p"], "n_dates": T},
        "blocks_rel_mse": block_rel, "blocks_improved": int(sum(x > 0 for x in block_rel)),
        "stocks_rel_mse": stock_rel, "share_stocks_improved": float(np.mean([v > 0 for v in stock_rel.values()])),
        "best_stock": best_stock, "rel_mse_without_best_stock": without_best_stock,
        "best_fold": best_fold, "rel_mse_without_best_fold": _rel(lc[keep_f].mean(), lb[keep_f].mean()),
    }


def calibration(preds: pd.DataFrame, model: str) -> dict:
    """Prediction deciles and behaviour on the largest actual moves."""
    y, p = preds["y"].to_numpy(), preds[model].to_numpy()
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    deciles = [{"decile": int(k + 1), "mean_pred": float(p[q == k].mean()), "mean_actual": float(y[q == k].mean()),
                "n": int((q == k).sum())} for k in range(10)]
    top = y >= np.quantile(y, 0.95)
    small = y <= np.quantile(y, 0.5)
    return {"deciles": deciles,
            "top5pct_actual": {"mean_actual": float(y[top].mean()), "mean_pred": float(p[top].mean()),
                               "ratio_pred_to_actual": float(p[top].mean() / y[top].mean())},
            "bottom50pct_actual": {"mean_actual": float(y[small].mean()), "mean_pred": float(p[small].mean())},
            "spearman_pred_actual": float(stats.spearmanr(p, y).statistic),
            "residual_skew": float(stats.skew(y - p))}


def qualify(c: dict, p_holm: float, leakage_ok: bool, reproducible: bool, cfg: Config, pooled: bool) -> dict:
    q = {"Q1_improvement": c["rel_mse_improvement"] >= cfg.min_rel_mse_improvement
         and c["rel_mae_improvement"] >= cfg.min_rel_mae_improvement,
         "Q2_statistics": p_holm < cfg.alpha and c["dm"]["p_2L"] < cfg.alpha,
         "Q3_periods": c["blocks_improved"] >= cfg.min_blocks_improved,
         "Q4_leakage": bool(leakage_ok),
         "Q5_breadth": (c["share_stocks_improved"] >= cfg.min_share_stocks_improved
                        and c["rel_mse_without_best_stock"] > 0 and c["rel_mse_without_best_fold"] > 0) if pooled else None,
         "Q6_reproducibility": bool(reproducible)}
    failed = [k for k, v in q.items() if v is False]
    q["qualified"] = not failed
    q["failed"] = failed
    return q


def repro_stocks(universe, n: int) -> list[str]:
    return sorted(universe, key=lambda s: hashlib.sha256(f"phase12s2|repro|{s}".encode()).hexdigest())[:n]


def pred_sha(preds: pd.DataFrame) -> str:
    cols = ["symbol", "pos", "y"] + [m for m in MODEL_ROLES if m in preds.columns]
    return hashlib.sha256(preds[cols].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


# ============================================================
# Experiment
# ============================================================


def experiment(frames: dict[str, pd.DataFrame], cfg: Config = FROZEN_CONFIG, log=print) -> dict:
    """Pure computation (no I/O): every horizon, both scopes, comparisons, Holm, qualification."""
    universe = sorted(frames)
    out = {"horizons": {}, "per_stock_rows": []}
    pooled_cmp, stock_cmp = [], []
    for h in cfg.horizons:
        panel = build_panel(frames, h)
        folds = make_folds(int(panel["pos"].max()) + 1, h, cfg)
        pooled = walk_forward(panel, h, cfg, "pooled")
        log(f"[12.S2] h={h} pooled done ({len(pooled)} predictions)")
        again = walk_forward(panel, h, cfg, "pooled")
        pooled_repro = pred_sha(pooled) == pred_sha(again)
        per = {}
        for sym in universe:
            per[sym] = walk_forward(panel[panel["symbol"] == sym], h, cfg, "per_stock")
        rs = repro_stocks(universe, cfg.n_repro_stocks)
        stock_repro = all(pred_sha(per[s]) == pred_sha(walk_forward(panel[panel["symbol"] == s], h, cfg, "per_stock"))
                          for s in rs)
        log(f"[12.S2] h={h} per-stock done")
        hres = {"folds": [list(f) for f in folds], "n_folds": len(folds),
                "test_dates": [str(pd.Timestamp(pooled["Date"].min()).date()), str(pd.Timestamp(pooled["Date"].max()).date())],
                "n_pooled_predictions": int(len(pooled)),
                "reproducibility": {"pooled_identical": pooled_repro, "per_stock_identical": stock_repro,
                                    "per_stock_checked": rs, "pooled_sha256": pred_sha(pooled)},
                "pooled": {}, "pooled_calibration": {}, "per_stock_models": {}}
        for m in MODEL_ROLES:
            hres["pooled"][m] = basic_metrics(pooled["y"].to_numpy(), pooled[m].to_numpy())
            hres["pooled"][m]["r2_vs_median"] = _rel(hres["pooled"][m]["mse"], hres["pooled"]["median"]["mse"]) \
                if "median" in hres["pooled"] else None
            hres["pooled_calibration"][m] = calibration(pooled, m)
        for m in [x for x in MODEL_ROLES if x != REFERENCE]:
            c = compare(pooled, m, h, cfg)
            hres["pooled"][m]["vs_reference"] = c
            if MODEL_ROLES[m] == "candidate":
                pooled_cmp.append(c)
        allper = pd.concat(per.values(), ignore_index=True)
        for m in MODEL_ROLES:
            hres["per_stock_models"][m] = basic_metrics(allper["y"].to_numpy(), allper[m].to_numpy())
        for sym in universe:
            for m in CANDIDATES:
                c = compare(per[sym], m, h, cfg)
                c["symbol"] = sym
                stock_cmp.append(c)
            pc = pooled[pooled["symbol"] == sym]
            for m in (REFERENCE,) + CANDIDATES:
                out["per_stock_rows"].append({"scope": "per_stock_model", "symbol": sym, "horizon": h, "model": m,
                                              **basic_metrics(per[sym]["y"].to_numpy(), per[sym][m].to_numpy())})
                out["per_stock_rows"].append({"scope": "pooled_model", "symbol": sym, "horizon": h, "model": m,
                                              **basic_metrics(pc["y"].to_numpy(), pc[m].to_numpy())})
        hres["leakage_ok"] = True                     # make_folds / walk_forward raise on any gap violation
        out["horizons"][h] = hres
    for fam, cmps in (("primary", pooled_cmp), ("secondary", stock_cmp)):
        for c, p in zip(cmps, holm_adjust([c["dm"]["p"] for c in cmps])):
            c["dm"]["p_holm"] = p
            c["family"] = fam
    quals = []
    for c in pooled_cmp:
        rep = out["horizons"][c["horizon"]]["reproducibility"]["pooled_identical"]
        quals.append({"scope": "pooled", "model": c["model"], "horizon": c["horizon"],
                      **qualify(c, c["dm"]["p_holm"], True, rep, cfg, pooled=True)})
    for c in stock_cmp:
        rep = out["horizons"][c["horizon"]]["reproducibility"]["per_stock_identical"]
        quals.append({"scope": "per_stock", "symbol": c["symbol"], "model": c["model"], "horizon": c["horizon"],
                      **qualify(c, c["dm"]["p_holm"], True, rep, cfg, pooled=False)})
    out["primary_comparisons"] = pooled_cmp
    out["secondary_comparisons"] = stock_cmp
    out["qualification"] = quals
    out["family_sizes"] = {"primary": len(pooled_cmp), "secondary": len(stock_cmp)}
    return out


def fit_finalist(frames: dict[str, pd.DataFrame], model: str, h: int, cfg: Config) -> dict:
    """Refit a pooled qualifier on all development rows; returns the artifact payload."""
    panel = build_panel(frames, h)
    scaled, _, st = pooled_scale(panel, panel.iloc[:0])
    cols = list(model_columns(model))
    m = make_model(model, "pooled", cfg).fit(scaled[cols].to_numpy(), scaled["y"].to_numpy())
    return {"model": m, "features": cols, "horizon": h,
            "per_stock_scaling": {s: {"mean": mu.to_dict(), "std": sd.to_dict()} for s, (mu, sd) in st.items()},
            "trained_on": [str(pd.Timestamp(panel["Date"].min()).date()), str(pd.Timestamp(panel["Date"].max()).date())],
            "protocol_sha256": protocol_sha256(cfg)}


# ============================================================
# I/O
# ============================================================


def write_protocol(out_dir: Path = OUT_DIR, cfg: Config = FROZEN_CONFIG) -> dict:
    path = Path(out_dir) / "protocol.json"
    if path.exists():
        raise Phase12S2Error("protocol.json already exists; the Stage 2 protocol is written once")
    doc = {**protocol(cfg), "protocol_sha256": protocol_sha256(cfg),
           "code_sha256": {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES},
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _write_new_json(path, doc)
    return doc


def load_protocol(out_dir: Path, cfg: Config) -> dict:
    path = Path(out_dir) / "protocol.json"
    if not path.is_file():
        raise Phase12S2Error("protocol.json not found (run `protocol` first; it must exist before any model is fit)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    body = {k: doc.get(k) for k in protocol(cfg)}
    if body != json.loads(canonical_json(protocol(cfg))) or doc.get("protocol_sha256") != protocol_sha256(cfg):
        raise Phase12S2Error("protocol.json differs from the protocol in code; the pre-registration cannot change")
    return doc


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def run(out_dir: Path = OUT_DIR, cfg: Config = FROZEN_CONFIG, log=print, frames: dict | None = None,
        inputs_sha: str | None = None) -> dict:
    out_dir = Path(out_dir)
    proto = load_protocol(out_dir, cfg)
    final, staging = out_dir / "results", out_dir / "results.incomplete"
    if final.exists() or staging.exists():
        raise Phase12S2Error("results (or results.incomplete) already exist; Stage 2 results are write-once")
    started = time.time()
    if frames is None:
        inputs = r.verified_inputs(r.FROZEN_SPEC)
        frames = r.load_frames(inputs, r.FROZEN_SPEC)
        inputs_sha = r.inputs_sha256(inputs)
    exp = experiment(frames, cfg, log)
    staging.mkdir(parents=True)
    rows = pd.DataFrame(exp.pop("per_stock_rows"))
    sec = pd.DataFrame([{"symbol": c["symbol"], "horizon": c["horizon"], "model": c["model"],
                         "rel_mse_improvement": c["rel_mse_improvement"], "rel_mae_improvement": c["rel_mae_improvement"],
                         "dm_t": c["dm"]["t"], "dm_p": c["dm"]["p"], "dm_p_2L": c["dm"]["p_2L"],
                         "dm_p_holm": c["dm"]["p_holm"], "blocks_improved": c["blocks_improved"]}
                        for c in exp["secondary_comparisons"]])
    with open(staging / "per_stock_metrics.csv", "w", encoding="utf-8", newline="\n") as fh:
        rows.to_csv(fh, index=False, float_format="%.10g", lineterminator="\n")
    with open(staging / "per_stock_comparisons.csv", "w", encoding="utf-8", newline="\n") as fh:
        sec.to_csv(fh, index=False, float_format="%.10g", lineterminator="\n")
    finalists = []
    for q in exp["qualification"]:
        if q["scope"] == "pooled" and q["qualified"]:
            art = fit_finalist(frames, q["model"], q["horizon"], cfg)
            p = staging / "finalists" / f"pooled_{q['model']}_h{q['horizon']}.joblib"
            p.parent.mkdir(exist_ok=True)
            joblib.dump(art, p)
            finalists.append({"model": q["model"], "horizon": q["horizon"], "file": p.name, "sha256": file_sha256(p)})
    pooled_q = [q for q in exp["qualification"] if q["scope"] == "pooled"]
    doc = {"document_type": "phase12_stage2_magnitude_results", "phase": PHASE, "stage": STAGE,
           "protocol_sha256": proto["protocol_sha256"], "protocol_file_sha256": file_sha256(out_dir / "protocol.json"),
           "inputs_sha256": inputs_sha, "status": "COMPLETE (development period only)",
           "summary": {"pooled_qualified": [f"{q['model']} h={q['horizon']}" for q in pooled_q if q["qualified"]],
                       "per_stock_qualified": [f"{q['symbol']} {q['model']} h={q['horizon']}"
                                               for q in exp["qualification"] if q["scope"] == "per_stock" and q["qualified"]]},
           **_jsonable(exp), "finalists": finalists, "sealed_data_read": False,
           "code_sha256": {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES}, "environment": environment(),
           "git": git_info(), "runtime_seconds": round(time.time() - started, 1),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_new_json(staging / "results.json", doc)
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    _write_new_json(staging / "manifest.json", {"files": {f: file_sha256(staging / f) for f in files},
                                                "protocol_sha256": proto["protocol_sha256"], "inputs_sha256": inputs_sha})
    os.replace(staging, final)
    return doc


def status_lines(out_dir: Path = OUT_DIR) -> list[str]:
    out_dir = Path(out_dir)
    lines = [f"protocol : {'written' if (out_dir / 'protocol.json').is_file() else 'not written'}"]
    final = out_dir / "results"
    if final.is_dir():
        m = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        bad = [f for f, h in m["files"].items() if file_sha256(final / f) != h]
        res = json.loads((final / "results.json").read_text(encoding="utf-8"))
        lines += [f"results  : COMPLETE; manifest {'verified' if not bad else 'MISMATCH ' + str(bad)}",
                  f"pooled qualified   : {res['summary']['pooled_qualified'] or 'none'}",
                  f"per-stock qualified: {len(res['summary']['per_stock_qualified'])}"]
    else:
        lines.append("results  : not run" + (" (incomplete run present)" if (out_dir / "results.incomplete").exists() else ""))
    lines.append("holdout  : sealed, NOT_EVALUATED")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Stage 2 magnitude prediction (development period)")
    parser.add_argument("command", choices=["protocol", "run", "status"])
    args = parser.parse_args(argv)
    if args.command == "protocol":
        doc = write_protocol()
        print(f"protocol.json written; protocol sha256 {doc['protocol_sha256']}")
    elif args.command == "run":
        run()
    print("\n".join(status_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
