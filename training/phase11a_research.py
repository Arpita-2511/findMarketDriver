"""
Phase 11A - cross-stock PRICE-ONLY signal research (development only).

Frozen inputs: Phase 11.0 design (sha 266daadd...), amendment 01 (reproducibility
tolerance), canonical Phase 11.1 price batch (data/processed/phase11/market/,
hashes recorded in price_verification.json). No news, sentiment or events.

Matrix (11A stage of the frozen design):
    per-stock  29 stocks x {A, E} x h {1, 3, 5} x 11 fixed models = 1,914
    panel      {A, E} x h x 11                                      = 66   (primary)
    sensitivity: panel without AAPL                                = 66   (descriptive only)

Per stock: the Phase 10A pipeline unchanged (training.phase10a_research.evaluate_excess:
excess-return target vs SPY, harness walk_forward TimeSeriesSplit(20) gap = h, harness
baselines, pooled_metrics, qualify / gate_v1, chronological DM).

Panel (training-only everything, one model per fold, no ticker identity feature):
    folds  TimeSeriesSplit(20, gap = h) over the UNIQUE development dates; every stock's
           rows of a date belong to the same fold
    baselines  Zero, Mean (pooled training mean), Per-Stock Mean (each stock's training mean);
           Always UP, Base Rate (pooled training share), Per-Stock Base Rate
    reference  regression: lowest pooled OOS MSE among the three; classification: lower pooled
           OOS log loss of the two base rates; Always UP = accuracy floor
    DM     per-date cross-sectional MEAN loss of model and reference, then the existing
           one-sided diebold_mariano (Newey-West h - 1 lags) on that per-date series
    gate   gate_v1 decision logic on the pooled OOS population

Development frame: bars are cut at 2024-09-30 BEFORE features and targets are built, so no
confirmation-period price enters any computation.

    python -m training.phase11a_research freeze
    python -m training.phase11a_research validate --what aapl|stock|panel
    python -m training.phase11a_research develop [--workers 12]
    python -m training.phase11a_research status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase10a_research as p10a  # noqa: E402
import training.phase11_design as p11d  # noqa: E402
import training.phase11_reproducibility as rp  # noqa: E402
from features.feature_engineering import TECHNICAL_FEATURES, engineer_features  # noqa: E402
from services import market_context as mc  # noqa: E402
from services.market_data import OHLCV_COLUMNS  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    GATE_VERSION,
    HARNESS_VERSION,
    NOT_QUALIFIED,
    QUALIFIED,
    REGRESSION,
    Candidate,
    HarnessConfig,
    diebold_mariano,
    git_info,
    per_observation_loss,
    pooled_metrics,
    predict_output,
)
from training.excess_targets import SPY_CLOSE, build_excess_targets, target_definition  # noqa: E402
from training.phase9_research import (  # noqa: E402
    ResearchError,
    _slug,
    _write_json_atomic,
    _write_new_json,
    canonical_json,
    environment,
    holm_adjust,
    phase9_candidates,
)

PHASE = "11A"
REGISTRY_VERSION = "phase11a_registry_v1"
FROZEN_DESIGN_SHA = "266daaddba0e8cb18163b4ec9db7eadabe2d5ab574bf410e043a7d28d3247650"
AMENDMENT_SHA = "94fadaa6135546b6ca1b422d1a5ebbb4e1a82442a3ca867ee3493c55bcc0485d"
PRICE_REPORT_SHA = "c6afc03fec249e146c37c0b4f5c512835c1cda9d344229b8ac3f0ce7bd1b15b0"
PRICE_REPORT = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "price_verification.json"
AMENDMENT = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "amendment_01_reproducibility.json"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "11A"
REGISTRY_PATH = OUT_DIR / "registry.json"

UNIVERSE = tuple(p11d.UNIVERSE)
STAGE_FEATURE_SETS = ("A", "E")
FEATURE_SETS = {k: p11d.FEATURE_SETS[k] for k in STAGE_FEATURE_SETS}
HORIZONS = (1, 3, 5)
N_SPLITS = 20
DEV_START, DEV_END = date(2017, 2, 1), date(2024, 9, 30)
ALPHA = HarnessConfig().significance_level
HOLM_ALPHA = 0.05
RENAMES = p11d.COLUMN_RENAMES                       # aapl_minus_spy_return_k -> stock_minus_spy_return_k
PER_STOCK_MEAN = Candidate("Per-Stock Mean Return", REGRESSION, lambda: None, is_baseline=True,
                           description="training-fold mean of each stock's own target")
PER_STOCK_RATE = Candidate("Per-Stock Base Rate", CLASSIFICATION, lambda: None, is_baseline=True,
                           description="training-fold share of UP of each stock")
PANEL_SPECIAL = {PER_STOCK_MEAN.name, PER_STOCK_RATE.name}


class Phase11AError(ResearchError):
    """Phase 11A protocol violation."""


def candidates() -> list[Candidate]:
    return phase9_candidates()


def panel_candidates() -> list[Candidate]:
    return candidates() + [PER_STOCK_MEAN, PER_STOCK_RATE]


# ==========================================
# Matrix / registry
# ==========================================


def matrix() -> dict:
    d = p11d.design()
    if p11d.STAGES["11A"]["feature_sets"] != list(STAGE_FEATURE_SETS):
        raise Phase11AError("11A feature sets differ from the frozen staging")
    models = [c for c in candidates() if not c.is_baseline]
    per_stock = [p11d.per_stock_id(s, f, h, c.name) for s in UNIVERSE for f in STAGE_FEATURE_SETS
                 for h in HORIZONS for c in models]
    panel = [p11d.panel_id(f, h, c.name) for f in STAGE_FEATURE_SETS for h in HORIZONS for c in models]
    sens = [p11d.panel_id(f, h, c.name, True) for f in STAGE_FEATURE_SETS for h in HORIZONS for c in models]
    for mine, key in ((per_stock, "per_stock"), (panel, "panel"), (sens, "aapl_excluded_sensitivity")):
        if not set(mine) <= set(d["comparison_ids"][key]) or len(set(mine)) != len(mine):
            raise Phase11AError(f"11A {key} ids are not a subset of the frozen design")
    if not (len(per_stock) == 29 * 2 * 3 * 11 == 1914 and len(panel) == 66 and len(sens) == 66):
        raise Phase11AError("11A matrix counts differ from the frozen design subset")
    return {
        "phase": PHASE, "stage": "11A (price-only: feature sets A, E)",
        "design_sha256": FROZEN_DESIGN_SHA, "amendment_sha256": AMENDMENT_SHA,
        "universe": list(UNIVERSE), "feature_sets": FEATURE_SETS, "horizons": list(HORIZONS),
        "targets": {str(h): target_definition(h) for h in HORIZONS},
        "models": [{"name": c.name, "task": c.task} for c in models],
        "baselines": {"per_stock": ["Mean Return", "Zero Return", "Always UP", "Base Rate"],
                      "panel": ["Mean Return", "Zero Return", PER_STOCK_MEAN.name, "Always UP", "Base Rate",
                                PER_STOCK_RATE.name]},
        "evaluation": {"harness_version": HARNESS_VERSION, "gate_version": GATE_VERSION, "n_splits": N_SPLITS,
                       "gap": "horizon", "significance_level": ALPHA,
                       "per_stock": "training.phase10a_research.evaluate_excess (Phase 10A pipeline, chronological DM)",
                       "panel": d["evaluation"]["panel"],
                       "development": [DEV_START.isoformat(), DEV_END.isoformat()],
                       "confirmation": "NOT evaluated in 11A",
                       "dev_cut": "price bars cut at the development end BEFORE features and targets are built"},
        "families": {"panel": {"size": len(panel), "method": "Holm", "alpha": HOLM_ALPHA},
                     "per_stock": {"size": len(per_stock), "method": "Holm", "alpha": HOLM_ALPHA},
                     "aapl_excluded_sensitivity": {"size": len(sens), "role": "descriptive only"},
                     "note": ("stage-level Holm (66 / 1,914) as instructed for 11A; the frozen Phase 11 design fixes "
                              "the final families at 198 / 5,742 across 11A + 11B, so any 11A qualifier is "
                              "PROVISIONAL until 11B completes the frozen families")},
        "qualification": "gate_v1 QUALIFIED AND raw DM p < 0.05 AND Holm-adjusted p < 0.05 within the family",
        "aapl_reproducibility": "amendment 01 tier 2 on the 66 AAPL A/E comparisons vs Phase 10A (diagnostic only)",
        "comparison_ids": {"per_stock": per_stock, "panel": panel, "aapl_excluded_sensitivity": sens},
        "counts": {"per_stock": len(per_stock), "panel": len(panel), "total": len(per_stock) + len(panel),
                   "aapl_excluded_sensitivity": len(sens)},
    }


def matrix_sha256(m: dict) -> str:
    return hashlib.sha256(canonical_json(m).encode("utf-8")).hexdigest()


def verified_inputs() -> dict:
    if file_sha256(PRICE_REPORT) != PRICE_REPORT_SHA:
        raise Phase11AError("Phase 11.1 price verification report changed")
    if file_sha256(AMENDMENT) != AMENDMENT_SHA:
        raise Phase11AError("amendment_01 changed")
    if p11d.load_registry()["design_sha256"] != FROZEN_DESIGN_SHA:
        raise Phase11AError("frozen Phase 11 design changed")
    rep = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))
    out = {}
    for sym, a in rep["artifacts"].items():
        h = file_sha256(PROJECT_ROOT / a["path"])
        if h != a["sha256"]:
            raise Phase11AError(f"canonical Phase 11.1 artifact for {sym} does not match its recorded hash")
        out[sym] = {"path": a["path"], "sha256": h}
    if set(out) != set(UNIVERSE) | {"SPY", "QQQ"}:
        raise Phase11AError("canonical batch does not contain exactly the frozen universe + SPY + QQQ")
    return {"artifacts": out, "price_report_sha256": PRICE_REPORT_SHA, "amendment_sha256": AMENDMENT_SHA,
            "design_sha256": FROZEN_DESIGN_SHA}


def freeze(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if path.exists():
        raise Phase11AError(f"{path} already exists; the 11A registry is frozen once")
    m = matrix()
    reg = {"registry_version": REGISTRY_VERSION, "phase": PHASE, "matrix": m, "matrix_sha256": matrix_sha256(m),
           "inputs": verified_inputs(), "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "frozen_git": git_info(), "environment": environment(), "status": "FROZEN", "development": None,
           "confirmation": "NOT_EVALUATED"}
    _write_json_atomic(path, reg)
    return reg


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or matrix_sha256(reg["matrix"]) != reg["matrix_sha256"]:
        raise Phase11AError("11A registry invalid or edited after freezing")
    if matrix_sha256(matrix()) != reg["matrix_sha256"]:
        raise Phase11AError("the 11A matrix in code differs from the frozen registry")
    return reg


# ==========================================
# Development frames (dev cut BEFORE features/targets)
# ==========================================


def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / path, dtype={"Date": str}, float_precision="round_trip")
    bars = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    bars["Date"] = pd.to_datetime(bars["Date"], format="%Y-%m-%d").astype("datetime64[ns]")
    return bars[OHLCV_COLUMNS]


def dev_cut(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[bars["Date"].dt.date <= DEV_END].reset_index(drop=True)


def stock_frame(stock: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    """Development frame for one stock from development-cut bars: Date, Close, SPY_Close, A/E features."""
    for b in (stock, spy, qqq):
        if b["Date"].dt.date.max() > DEV_END:
            raise Phase11AError("bars must be cut at the development end before features are built")
    tech = engineer_features(stock.copy())
    tech = tech[(tech["Date"].dt.date >= DEV_START) & (tech["Date"].dt.date <= DEV_END)].reset_index(drop=True)
    dates = list(tech["Date"].dt.date)
    ctx = mc.compute_market_context(stock, spy, qqq, dates).rename(columns=RENAMES)
    spy_close = dict(zip(spy["Date"].dt.date, spy["Close"].astype("float64")))
    frame = pd.DataFrame({"Date": tech["Date"], "Close": tech["Close"].astype("float64"),
                          SPY_CLOSE: [spy_close[d] for d in dates]})
    for c in TECHNICAL_FEATURES:
        frame[c] = tech[c].to_numpy(dtype="float64")
    for c in ctx.columns:
        if c not in ("trading_date", "prediction_timestamp"):
            frame[c] = ctx[c].to_numpy(dtype="float64")
    cols = sorted({c for fs in FEATURE_SETS.values() for c in fs["columns"]})
    if not np.isfinite(frame[cols + ["Close", SPY_CLOSE]].to_numpy(dtype="float64")).all():
        raise Phase11AError("non-finite development features")
    return frame


def build_frames(inputs: dict) -> dict[str, pd.DataFrame]:
    a = inputs["artifacts"]
    spy, qqq = dev_cut(load_bars(a["SPY"]["path"])), dev_cut(load_bars(a["QQQ"]["path"]))
    frames = {s: stock_frame(dev_cut(load_bars(a[s]["path"])), spy, qqq) for s in UNIVERSE}
    dates = {s: tuple(f["Date"]) for s, f in frames.items()}
    if len(set(dates.values())) != 1:
        raise Phase11AError("stocks do not share the same development sessions")
    return frames


# ==========================================
# Panel evaluation
# ==========================================


def panel_rows(frames: dict[str, pd.DataFrame], cols: list[str], horizon: int) -> pd.DataFrame:
    parts = []
    for sym in sorted(frames):
        ev = build_excess_targets(frames[sym][["Date", "Close", SPY_CLOSE, *cols]], horizon)
        ev.insert(0, "symbol", sym)
        parts.append(ev[["symbol", "Date", *cols, "future_return", "direction"]])
    return pd.concat(parts, ignore_index=True).sort_values(["Date", "symbol"], kind="mergesort").reset_index(drop=True)


def date_splits(dates: pd.Series, horizon: int, n_splits: int):
    uniq = np.array(sorted(dates.unique()))
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(np.zeros((len(uniq), 1))):
        yield uniq[tr], uniq[te]


def _per_date_mean_loss(pred: pd.DataFrame, task: str) -> pd.Series:
    loss = per_observation_loss(task, pred["y_true"].to_numpy(), pred["prediction"].to_numpy())
    return pd.Series(loss, index=pred.index).groupby(pred["Date"]).mean().sort_index()


def panel_qualify(predictions: pd.DataFrame, cands: list[Candidate], metrics: dict, horizon: int) -> tuple[dict, dict]:
    """gate_v1 decision logic on the pooled panel; DM on per-date cross-sectional mean losses."""
    gate, series = {}, {}
    for task in (REGRESSION, CLASSIFICATION):
        base = [c for c in cands if c.task == task and c.is_baseline]
        models = [c for c in cands if c.task == task and not c.is_baseline]
        if task == REGRESSION:
            ref = min(base, key=lambda c: metrics[c.name]["MSE"])
            primary, always_up = "MSE", None
        else:
            ref = min([c for c in base if c.name in ("Base Rate", PER_STOCK_RATE.name)],
                      key=lambda c: metrics[c.name]["Log_Loss"])
            primary = "Log_Loss"
            always_up = next(c for c in base if c.name == "Always UP")
        ref_pred = predictions[predictions["model"] == ref.name]
        ref_series = _per_date_mean_loss(ref_pred, task)
        series[f"{task}:{ref.name}"] = ref_series
        for c in base:
            gate[c.name] = {"status": "BASELINE", "task": task}
        for c in models:
            cp = predictions[predictions["model"] == c.name]
            if not np.array_equal(cp["row"].to_numpy(), ref_pred["row"].to_numpy()):
                raise Phase11AError(f"{c.name} and {ref.name} were scored on different rows")
            s = _per_date_mean_loss(cp, task)
            if not s.index.equals(ref_series.index):
                raise Phase11AError("per-date series misaligned")
            series[f"{task}:{c.name}"] = s
            dm = diebold_mariano(s.to_numpy(), ref_series.to_numpy(), horizon)
            better = metrics[c.name][primary] < metrics[ref.name][primary]
            reasons = []
            if not better:
                reasons.append(f"{primary} not better than {ref.name}")
            elif not dm["p_value"] < ALPHA:
                reasons.append(f"improvement over {ref.name} not significant (p={dm['p_value']:.3f})")
            if always_up is not None and metrics[c.name]["Accuracy"] < metrics[always_up.name]["Accuracy"]:
                reasons.append("Accuracy below Always UP")
            gate[c.name] = {"status": QUALIFIED if not reasons else NOT_QUALIFIED, "task": task,
                            "reference_baseline": ref.name, "primary_metric": primary,
                            "model_value": metrics[c.name][primary], "baseline_value": metrics[ref.name][primary],
                            "diebold_mariano": dm, "reasons": reasons,
                            "dm_series": "per-date cross-sectional mean loss", "n_dates": int(len(s))}
    return gate, series


def panel_walk_forward(data: pd.DataFrame, cols: list[str], cands: list[Candidate], horizon: int,
                       n_splits: int) -> tuple[pd.DataFrame, list[dict]]:
    """Date-aligned walk-forward on stacked rows; one fresh model per fold; per-stock baselines from training rows."""
    if set(cols) & {"symbol", "ticker"}:
        raise Phase11AError("ticker identity must not be a feature")
    X = data[cols]
    y = {REGRESSION: data["future_return"].to_numpy(), CLASSIFICATION: data["direction"].to_numpy()}
    rows, fold_info = [], []
    for k, (tr_dates, te_dates) in enumerate(date_splits(data["Date"], horizon, n_splits), start=1):
        tr = np.flatnonzero(data["Date"].isin(tr_dates).to_numpy())
        te = np.flatnonzero(data["Date"].isin(te_dates).to_numpy())
        if set(data["Date"].iloc[tr]) & set(data["Date"].iloc[te]):
            raise Phase11AError("a date appears in both training and test rows")
        fold_info.append({"fold": k, "train_start": str(tr_dates[0])[:10], "train_end": str(tr_dates[-1])[:10],
                          "test_start": str(te_dates[0])[:10], "test_end": str(te_dates[-1])[:10],
                          "n_train_dates": int(len(tr_dates)), "n_test_dates": int(len(te_dates)),
                          "n_train_rows": int(len(tr)), "n_test_rows": int(len(te))})
        sym_tr = data["symbol"].iloc[tr]
        for c in cands:
            yy = y[c.task]
            if c.name in PANEL_SPECIAL:
                means = pd.Series(yy[tr].astype("float64")).groupby(sym_tr.to_numpy()).mean()
                test_syms = data["symbol"].iloc[te]
                if not set(test_syms) <= set(means.index):
                    raise Phase11AError("a test stock has no training rows for its per-stock baseline")
                pred = means.reindex(test_syms.to_numpy()).to_numpy(dtype="float64")
            else:
                est = clone(c.factory())
                est.fit(X.iloc[tr], yy[tr])
                pred = predict_output(est, c.task, X.iloc[te])
            rows.append(pd.DataFrame({"row": te, "Date": data["Date"].iloc[te].to_numpy(),
                                      "symbol": data["symbol"].iloc[te].to_numpy(), "fold": k, "model": c.name,
                                      "task": c.task, "y_true": yy[te], "prediction": pred}))
    predictions = pd.concat(rows, ignore_index=True)
    counts = predictions.groupby("model")["row"].agg(["count", "nunique"])
    if not (counts["count"] == counts["nunique"]).all():
        raise Phase11AError("a panel row was predicted more than once")
    return predictions, fold_info


def evaluate_panel(frames: dict[str, pd.DataFrame], cols: list[str], cands: list[Candidate], *, horizon: int,
                   n_splits: int = N_SPLITS) -> dict:
    data = panel_rows(frames, cols, horizon)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions, fold_info = panel_walk_forward(data, cols, cands, horizon, n_splits)
    metrics = pooled_metrics(predictions, cands)
    gate, series = panel_qualify(predictions, cands, metrics, horizon)
    return {"horizon_days": horizon, "n_splits": n_splits, "gap": horizon, "features": cols,
            "n_symbols": int(data["symbol"].nunique()), "n_rows": int(len(data)),
            "n_dates": int(data["Date"].nunique()), "n_oos_rows": int(predictions["row"].nunique()),
            "oos_period": [fold_info[0]["test_start"], fold_info[-1]["test_end"]], "folds": fold_info,
            "models": [{"model": c.name, "task": c.task, "is_baseline": c.is_baseline, "metrics": metrics[c.name],
                        "qualification": gate[c.name]} for c in cands],
            "per_date_losses": {k: {"dates": [str(d)[:10] for d in v.index], "mean_loss": v.tolist()}
                                for k, v in series.items()},
            "warnings": {"total": len(caught),
                         "by_message": dict(Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught))}}


# ==========================================
# Workers (process-parallel; deterministic: fixed seeds, n_jobs=1 models)
# ==========================================


_THREAD_LIMIT = None


def _init_worker() -> None:
    """One BLAS/OpenMP thread per worker process (no oversubscription; models already use n_jobs=1)."""
    global _THREAD_LIMIT
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(1)


def _stock_job(sym: str, frame: pd.DataFrame) -> dict:
    out = {}
    for fs in STAGE_FEATURE_SETS:
        for h in HORIZONS:
            out[f"{fs}_h{h}"] = p10a.evaluate_excess(frame, list(FEATURE_SETS[fs]["columns"]), candidates(),
                                                     horizon=h, n_splits=N_SPLITS)
    return {"symbol": sym, "runs": out}


def _panel_job(kind: str, fs: str, h: int, frames: dict[str, pd.DataFrame]) -> dict:
    use = {s: f for s, f in frames.items() if kind == "panel" or s != "AAPL"}
    return {"kind": kind, "fs": fs, "h": h,
            "result": evaluate_panel(use, list(FEATURE_SETS[fs]["columns"]), panel_candidates(), horizon=h)}


def per_stock_id(sym, fs, h, model):
    return p11d.per_stock_id(sym, fs, h, model)


def panel_id(fs, h, model, sensitivity=False):
    return p11d.panel_id(fs, h, model, sensitivity)


def qualify_family(entries: list[dict], size: int) -> None:
    """Holm within one frozen family; qualified = gate_v1 QUALIFIED AND raw p < 0.05 AND Holm p < 0.05."""
    if len(entries) != size:
        raise Phase11AError(f"Holm family has {len(entries)} comparisons, expected {size}")
    for e, p in zip(entries, holm_adjust([e["p_value_raw"] for e in entries])):
        e["p_value_holm"] = p
        e["qualified"] = e["gate_status"] == QUALIFIED and e["p_value_raw"] < ALPHA and p < HOLM_ALPHA


def aapl_reproducibility(per_stock: list[dict], r10a_results: dict, tier1: dict) -> dict:
    """Amendment 01 tier 2: the 66 AAPL A/E comparisons vs Phase 10A development. Diagnostic only."""
    p11, ref = {}, {}
    for e in per_stock:
        if e["symbol"] != "AAPL":
            continue
        key = f"{e['feature_set']}_h{e['horizon']}_{_slug(e['model'])}"
        p11[key] = {"status": e["gate_status"], "primary_value": e["gate"]["model_value"], "p_value": e["p_value_raw"]}
        d = r10a_results[p10a.experiment_id(e["feature_set"], e["horizon"], e["model"])]["development"]
        ref[key] = {"status": d["status"], "primary_value": d["gate"]["model_value"], "p_value": d["p_value_raw"]}
    if len(ref) != 66:
        raise Phase11AError(f"AAPL reproducibility needs 66 comparisons, got {len(ref)}")
    tier2 = rp.tier2_compare(p11, ref)
    for k, v in tier2["comparisons"].items():
        v.update({"primary_p11": p11[k]["primary_value"], "primary_p10a": ref[k]["primary_value"],
                  "p_p11": p11[k]["p_value"], "p_p10a": ref[k]["p_value"]})
    return {"tolerance": rp.TOLERANCE, "tier1": tier1, "tier2": tier2, "outcome": rp.outcome(tier1, tier2),
            "failed": sorted(k for k, v in tier2["comparisons"].items() if not v["pass"]),
            "max_rel_primary_diff": max(v["rel_primary_diff"] for v in tier2["comparisons"].values()),
            "max_abs_p_diff": max(v["abs_p_diff"] for v in tier2["comparisons"].values()),
            "role": "diagnostic only; never alters Phase 11A results"}


# ==========================================
# Development
# ==========================================


def develop(registry_path: Path = REGISTRY_PATH, workers: int = 12, log=print) -> dict:
    reg = load_registry(registry_path)
    if reg["status"] != "FROZEN" or reg["development"] is not None:
        raise Phase11AError("11A development has already been run")
    inputs = verified_inputs()
    if inputs != reg["inputs"]:
        raise Phase11AError("inputs differ from the frozen 11A registry")
    out_dir = Path(registry_path).parent
    started = time.time()
    frames = build_frames(inputs)
    log(f"[11A] frames built: {len(frames)} stocks x {len(next(iter(frames.values())))} development sessions")
    stock_res, panel_res = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futs = {pool.submit(_panel_job, k, fs, h, frames): ("panel", k, fs, h)
                for k in ("panel", "panel_exAAPL") for fs in STAGE_FEATURE_SETS for h in HORIZONS}
        futs.update({pool.submit(_stock_job, s, frames[s]): ("stock", s) for s in UNIVERSE})
        for f in futs:
            key = futs[f]
            r = f.result()
            if key[0] == "stock":
                stock_res[key[1]] = r["runs"]
                log(f"[11A] stock {key[1]} done")
            else:
                panel_res[(key[1], key[2], key[3])] = r["result"]
                log(f"[11A] {key[1]} {key[2]} h{key[3]} done")
    # ---- per-stock entries + Holm (1,914) ----
    per_stock, art = [], {}
    for s in UNIVERSE:
        a = _write_new_json(out_dir / "experiments" / f"stock_{s}.json",
                            {"symbol": s, "phase": PHASE, "matrix_sha256": reg["matrix_sha256"],
                             "input_sha256": inputs["artifacts"][s]["sha256"], "runs": stock_res[s]})
        art[f"stock_{s}"] = a
        for fs in STAGE_FEATURE_SETS:
            for h in HORIZONS:
                rep = stock_res[s][f"{fs}_h{h}"]
                base = {m["model"]: m["metrics"] for m in rep["models"] if m["is_baseline"]}
                for m in rep["models"]:
                    if m["is_baseline"]:
                        continue
                    q = m["qualification"]
                    per_stock.append({"id": per_stock_id(s, fs, h, m["model"]), "symbol": s, "feature_set": fs,
                                      "horizon": h, "model": m["model"], "task": m["task"],
                                      "n_oos": rep["evaluation_period"]["n_oos_rows"], "n_folds": len(rep["folds"]),
                                      "metrics": m["metrics"],
                                      "baseline_metrics": {k: v for k, v in base.items()
                                                           if (k in ("Mean Return", "Zero Return")) == (m["task"] == REGRESSION)},
                                      "gate": q, "gate_status": q["status"],
                                      "dm_stat": q["diebold_mariano"]["dm_stat"],
                                      "loss_diff_mean": q["diebold_mariano"]["mean_loss_improvement"],
                                      "p_value_raw": q["diebold_mariano"]["p_value"], "artifact": a["path"]})
    # ---- panel + sensitivity ----
    panel, sens = [], []
    for (kind, fs, h), r in sorted(panel_res.items()):
        a = _write_new_json(out_dir / ("panel" if kind == "panel" else "sensitivity") / f"{kind}_{fs}_h{h}.json",
                            {"kind": kind, "feature_set": fs, "horizon": h, "phase": PHASE,
                             "matrix_sha256": reg["matrix_sha256"], **r})
        art[f"{kind}_{fs}_h{h}"] = a
        base = {m["model"]: m["metrics"] for m in r["models"] if m["is_baseline"]}
        for m in r["models"]:
            if m["is_baseline"]:
                continue
            q = m["qualification"]
            e = {"id": panel_id(fs, h, m["model"], kind != "panel"), "feature_set": fs, "horizon": h,
                 "model": m["model"], "task": m["task"], "n_oos": r["n_oos_rows"], "n_dates": q["n_dates"],
                 "n_symbols": r["n_symbols"], "n_folds": len(r["folds"]), "metrics": m["metrics"],
                 "baseline_metrics": {k: v for k, v in base.items()
                                      if ("Return" in k) == (m["task"] == REGRESSION)},
                 "gate": {k: v for k, v in q.items()}, "gate_status": q["status"],
                 "dm_stat": q["diebold_mariano"]["dm_stat"], "loss_diff_mean": q["diebold_mariano"]["mean_loss_improvement"],
                 "p_value_raw": q["diebold_mariano"]["p_value"], "artifact": a["path"]}
            (panel if kind == "panel" else sens).append(e)
    planned = reg["matrix"]["comparison_ids"]
    if sorted(e["id"] for e in per_stock) != sorted(planned["per_stock"]) or \
            sorted(e["id"] for e in panel) != sorted(planned["panel"]) or \
            sorted(e["id"] for e in sens) != sorted(planned["aapl_excluded_sensitivity"]):
        raise Phase11AError("results do not cover exactly the frozen 11A matrix")
    qualify_family(per_stock, 1914)
    qualify_family(panel, 66)
    for e in sens:
        e["p_value_holm"] = None
        e["qualified"] = None
    # ---- AAPL reproducibility (diagnostic only) ----
    repro = aapl_reproducibility(per_stock, json.loads(p10a.REGISTRY_PATH.read_text(encoding="utf-8"))["results"],
                                 json.loads(AMENDMENT.read_text(encoding="utf-8"))["tier1_observed"])
    art["aapl_reproducibility"] = _write_new_json(out_dir / "aapl_reproducibility.json", repro)
    qualified = {"panel": [e["id"] for e in panel if e["qualified"]],
                 "per_stock": [e["id"] for e in per_stock if e["qualified"]]}
    reg["development"] = {
        "status": "COMPLETE", "runtime_seconds": round(time.time() - started, 1), "workers": workers,
        "execution": ("ProcessPoolExecutor (one job per stock, one per panel/sensitivity run); threadpool_limits(1) "
                      "per worker; models n_jobs=1, fixed seed 42; results assembled in frozen matrix order"),
        "git": git_info(), "environment": environment(),
        "counts": {"per_stock": len(per_stock), "panel": len(panel), "sensitivity": len(sens)},
        "families": {"panel": len(panel), "per_stock": len(per_stock)},
        "summary": {fam: {"gate_passes": sum(e["gate_status"] == QUALIFIED for e in es),
                          "raw_p_below_alpha": sum(e["p_value_raw"] < ALPHA for e in es),
                          "holm_p_below_alpha": (sum(e["p_value_holm"] < HOLM_ALPHA for e in es)
                                                 if fam != "sensitivity" else None),
                          "qualified": (sum(bool(e["qualified"]) for e in es) if fam != "sensitivity" else None)}
                    for fam, es in (("panel", panel), ("per_stock", per_stock), ("sensitivity", sens))},
        "qualified": qualified,
        "result": ("PROVISIONAL DEVELOPMENT QUALIFIERS (pending 11B final families)" if any(qualified.values())
                   else "NO DEVELOPMENT QUALIFIERS"),
        "aapl_reproducibility": repro["outcome"], "artifacts": art,
        "warnings_total": sum(r["warnings"]["total"] for runs in stock_res.values() for r in runs.values())
        + sum(r["warnings"]["total"] for r in panel_res.values()),
        "confirmation": "NOT_EVALUATED",
    }
    reg["results"] = {"per_stock": per_stock, "panel": panel, "aapl_excluded_sensitivity": sens}
    reg["status"] = "DEVELOPMENT_COMPLETE"
    _write_json_atomic(registry_path, reg)
    return reg


# ==========================================
# Validation (pre-run checks; results are not recorded)
# ==========================================


def validate(what: str) -> None:
    inputs = verified_inputs()
    frames = build_frames(inputs)
    if what in ("aapl", "stock"):
        sym = "AAPL" if what == "aapl" else "AXP"
        rep = p10a.evaluate_excess(frames[sym], list(FEATURE_SETS["A"]["columns"]), candidates(), horizon=1,
                                   n_splits=N_SPLITS)
        print(f"{sym} A h1: rows {rep['frame']['rows']} OOS {rep['evaluation_period']} warnings {rep['warnings']['total']}")
        r10a = json.loads(p10a.REGISTRY_PATH.read_text(encoding="utf-8"))["results"] if sym == "AAPL" else {}
        for m in rep["models"]:
            if m["is_baseline"]:
                continue
            q = m["qualification"]
            line = f"  {m['model']:32s} {q['status']:14s} {q['primary_metric']} {q['model_value']:.6g} p {q['diebold_mariano']['p_value']:.4f}"
            if r10a:
                d = r10a[p10a.experiment_id("A", 1, m["model"])]["development"]
                line += (f" | 10A {d['status']} {d['gate']['model_value']:.6g} p {d['p_value_raw']:.4f} "
                         f"rel {abs(q['model_value'] / d['gate']['model_value'] - 1):.2e}")
            print(line)
    else:
        r = evaluate_panel(frames, list(FEATURE_SETS["A"]["columns"]), panel_candidates(), horizon=1)
        print(f"panel A h1: symbols {r['n_symbols']} rows {r['n_rows']} dates {r['n_dates']} OOS rows {r['n_oos_rows']} "
              f"{r['oos_period']} warnings {r['warnings']['total']}")
        for m in r["models"]:
            q = m["qualification"]
            if q["status"] != "BASELINE":
                print(f"  {m['model']:32s} {q['status']:14s} ref {q['reference_baseline']} p {q['diebold_mariano']['p_value']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11A price-only cross-stock research")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    v = sub.add_parser("validate")
    v.add_argument("--what", choices=["aapl", "stock", "panel"], required=True)
    d = sub.add_parser("develop")
    d.add_argument("--workers", type=int, default=12)
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        reg = freeze()
        print(f"Frozen {REGISTRY_PATH}  matrix sha256 {reg['matrix_sha256']}  counts {reg['matrix']['counts']}")
    elif args.cmd == "validate":
        validate(args.what)
    elif args.cmd == "develop":
        reg = develop(workers=args.workers)
        print(json.dumps({k: reg["development"][k] for k in ("result", "summary", "aapl_reproducibility",
                                                               "runtime_seconds", "warnings_total")}, indent=1))
    else:
        reg = load_registry()
        print(reg["status"], reg["matrix_sha256"], reg["matrix"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
