"""
Phase 12 Stage 3A: finalist freeze and the sealed-confirmation amendment (Amendment 07).

One finalist goes to the sealed confirmation period, once:

    finalist  pooled LightGBM, h = 1, the exact Stage 2 artifact (never refit)
    baseline  pooled Ridge on Volatility, spy_volatility_20, refit on all development rows with the Stage 2
              finalist procedure (Stage 2 saved only candidates) and frozen here before any sealed access
    window    prediction dates 2024-10-01 .. the session before 2026-09-25 (targets end 2026-09-25)

`freeze` uses development data only: it verifies the Stage 2 evidence, fits and saves the baseline, fingerprints
both models on the development rows and writes the amendment (write-once). `confirm` refuses unless the amendment
matches its pinned hash, every pin re-verifies on development data and the owner's authorization record names this
amendment and the exact runner and sealed feature builder code. No sealed path is opened before all of that.

    python -m training.phase12_confirmation freeze   # stage3/: baseline artifact + amendment (write-once)
    python -m training.phase12_confirmation verify   # every pin re-checked on development data
    python -m training.phase12_confirmation status
    python -m training.phase12_confirmation confirm  # one-time sealed confirmation (owner authorization required)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11b_research as r  # noqa: E402
import training.phase12_magnitude as m2  # noqa: E402
import training.phase12_signal_diagnostics as s1  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.excess_targets import ExcessTargetError  # noqa: E402
from training.phase9_research import _write_new_json, canonical_json, environment, holm_adjust  # noqa: E402

PHASE, STAGE = "12", "12.S3A"
AMENDMENT_ID = "amendment_07_phase12_sealed_confirmation"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage3"
AMENDMENT_FILE = "amendment_07_confirmation.json"
BASELINE_FILE = "baseline_ridge_vol_h1.joblib"
AUTHORIZATION_FILE = "authorization.json"
CONFIRMATION_DIR = "confirmation"

FINALIST_MODEL, HORIZON = "lightgbm", 1
STAGE2_RESULTS = m2.OUT_DIR / "results"
FINALIST_FILE = "pooled_lightgbm_h1.joblib"
FINALIST_SHA256 = "01f3c7e47a7b065998e1273f04eb198235c3648a3fec1151ba9ee770aa14e06d"
STAGE2_PINS = {                                    # Stage 2 evidence, write-once (uncommitted at freeze time)
    "stage2/protocol.json": "19d0ace0c64e409d95fd120907f1406a82576ea34fb42828fe1f9b5105463511",
    "stage2/results/manifest.json": "fd1f4a53b345293c92324e8611e89d9a9818c5909503969f7acb78a20d054926",
    "stage2/results/results.json": "c2cfd82bfda47032e5c686422c9e78769e2a3efd7faeeeeaccbe3d489418933f",
    "stage2/results/finalists/pooled_lightgbm_h1.joblib": FINALIST_SHA256,
}
STAGE2_PROTOCOL_SHA256 = "fb9e9e38a37dcafdc5d2dbde96d14a9c0efcd70ed176169083b1f203d04e9339"
INPUTS_SHA256 = "0fb601fc6abb98ae30ac7edee57aa6e45c519726d63f4b3e962f127dc310124a"
AMENDMENT_SHA256 = "54aa14a84de9f76aa79f579171e6908471fbf63ddd9056cabc5ac2c16894e0d7"   # written by `freeze`

DEV_END = date(2024, 9, 30)
CONFIRMATION_START, CONFIRMATION_END = date(2024, 10, 1), date(2026, 9, 25)
QUARTER_SESSIONS, MIN_LAST_QUARTER = 63, 21        # Stage 2 fold length and merge rule
MIN_SESSIONS = 400                                 # fewer common sealed sessions -> INVALID
MAX_EXCLUDED_SHARE, MAX_EXCLUDED_SHARE_STOCK = 0.01, 0.05
CODE_FILES = ("training/phase12_magnitude.py", "training/phase12_signal_diagnostics.py",
              "training/excess_targets.py", "training/phase11b_research.py")
GENERATOR_FILES = ("features/feature_engineering.py", "services/market_context.py", "services/sentiment_window.py",
                   "services/event_features.py", "services/event_classifier.py", "services/historical_events.py",
                   "services/news_alignment.py", "services/market_calendar_service.py",
                   "training/phase11a_research.py", "training/phase11b_feature_build.py")
RULES_V1_SPEC_SHA256 = "974016388088b7eb50c8ed2659c738bc464d10a34c7b068471fd6a24d7cd44ad"
SEALED_BUILDER_FILE = "training/phase12_sealed_features.py"   # not written yet; reviewed and approved separately
AUTHORIZATION_STATEMENT = ("I authorize the one-time Phase 12 sealed confirmation under Amendment 07, "
                           "with the runner and sealed feature builder code named in this record.")


class Phase12S3Error(m2.Phase12S2Error):
    """Phase 12 Stage 3 protocol violation; the run stops."""


# ============================================================
# Artifacts
# ============================================================


def load_artifact(path: Path, expected_sha256: str) -> dict:
    """joblib.load only after the file hash matches (unpickling runs code)."""
    path = Path(path)
    if not path.is_file():
        raise Phase12S3Error(f"{path.name} not found")
    if file_sha256(path) != expected_sha256:
        raise Phase12S3Error(f"{path.name} differs from its pinned SHA256; artifacts are never replaced")
    return joblib.load(path)


def check_artifact(art: dict, model: str) -> None:
    """Structure of a pooled artifact: feature order, horizon, scaling keys, exact Stage 2 hyperparameters."""
    if list(art.get("features", [])) != list(m2.model_columns(model)):
        raise Phase12S3Error(f"{model}: feature order differs from {list(m2.model_columns(model))}")
    if art.get("horizon") != HORIZON or art.get("protocol_sha256") != STAGE2_PROTOCOL_SHA256:
        raise Phase12S3Error(f"{model}: horizon or Stage 2 protocol hash differs")
    if date.fromisoformat(art["trained_on"][1]) > DEV_END:
        raise Phase12S3Error(f"{model}: trained on rows after the development end")
    scaling = art.get("per_stock_scaling", {})
    if not scaling:
        raise Phase12S3Error(f"{model}: no per-stock scaling")
    for sym, st in scaling.items():
        if list(st["mean"]) != list(m2.STOCK_SCALED) or list(st["std"]) != list(m2.STOCK_SCALED):
            raise Phase12S3Error(f"{sym}: scaling keys differ from {list(m2.STOCK_SCALED)}")
    want = m2.make_model(model, "pooled", m2.FROZEN_CONFIG)
    est = art["model"]
    if type(est) is not type(want) or repr(est.get_params()) != repr(want.get_params()):
        raise Phase12S3Error(f"{model}: estimator type or hyperparameters differ from the Stage 2 protocol")
    n_in = getattr(est.steps[-1][1] if hasattr(est, "steps") else est, "n_features_in_", None)
    if n_in != len(art["features"]):
        raise Phase12S3Error(f"{model}: fitted on {n_in} features")


def design_matrix(art: dict, panel: pd.DataFrame) -> np.ndarray:
    """Artifact feature order; STOCK_SCALED columns z-scored with the artifact's per-stock development stats."""
    cols = list(art["features"])
    x = panel[cols].to_numpy(dtype="float64").copy()
    scaled = [i for i, c in enumerate(cols) if c in m2.STOCK_SCALED]
    if scaled:
        sym = panel["symbol"].to_numpy()
        unknown = sorted(set(sym) - set(art["per_stock_scaling"]))
        if unknown:
            raise Phase12S3Error(f"no development scaling for {unknown}")
        for s in np.unique(sym):
            rows = sym == s
            st = art["per_stock_scaling"][s]
            mu = np.array([st["mean"][cols[i]] for i in scaled])
            sd = np.array([st["std"][cols[i]] for i in scaled])
            x[np.ix_(rows, scaled)] = (x[np.ix_(rows, scaled)] - mu) / sd
    return x


def predict(art: dict, panel: pd.DataFrame) -> np.ndarray:
    """Frozen inference: design matrix, the saved estimator's predict, floor at 0. Nothing is fitted."""
    return np.maximum(0.0, np.asarray(art["model"].predict(design_matrix(art, panel)), dtype="float64"))


def model_sha256(art: dict) -> str:
    """Library-level fingerprint: LightGBM model text, or the exact Ridge pipeline numbers (float.hex)."""
    est = art["model"]
    if isinstance(est, LGBMRegressor):
        return hashlib.sha256(est.booster_.model_to_string().encode("utf-8")).hexdigest()
    clip, scaler, ridge = (est.named_steps[k] for k in ("quantileclipper", "standardscaler", "ridge"))
    nums = {k: [float(v).hex() for v in np.ravel(a)] for k, a in (
        ("clip_lo", clip.lo_), ("clip_hi", clip.hi_), ("mean", scaler.mean_), ("scale", scaler.scale_),
        ("coef", ridge.coef_), ("intercept", ridge.intercept_))}
    return hashlib.sha256(canonical_json(nums).encode("utf-8")).hexdigest()


def predictions_sha256(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<f8").tobytes()).hexdigest()


def fit_baseline(frames: dict[str, pd.DataFrame]) -> dict:
    """The reference baseline refit on all development rows, exactly as Stage 2 refits its finalists."""
    return m2.fit_finalist(frames, m2.REFERENCE, HORIZON, m2.FROZEN_CONFIG)


# ============================================================
# Amendment
# ============================================================


def rules() -> dict:
    """Every frozen rule of the sealed confirmation (no hashes; those are in `pins`)."""
    cfg = m2.FROZEN_CONFIG
    return {
        "amendment_id": AMENDMENT_ID, "phase": PHASE, "stage": STAGE,
        "status": "FROZEN before any sealed observation is read",
        "evidence_levels": {
            "development_qualification": ("Stage 2, 2019-02 .. 2024-09 walk-forward on development data. Exploratory "
                                          "and possibly optimistic: the 8 features were chosen in Stage 1 on the same "
                                          "2017-2024 data."),
            "sealed_confirmation": ("This amendment: one evaluation of the frozen finalist on sessions never used for "
                                    "feature selection, model fitting, scaling or threshold setting. Only this result "
                                    "can confirm the Stage 2 finding.")},
        "finalist": {"model": "pooled LightGBM", "stage2_name": FINALIST_MODEL, "horizon": HORIZON,
                     "artifact": f"phase12/stage2/results/finalists/{FINALIST_FILE}",
                     "features_in_order": list(m2.FEATURES),
                     "trained_on": "all development rows, labels complete by 2024-09-30 (Stage 2 fit_finalist)"},
        "excluded": ["pooled Ridge h=1", "pooled Random Forest h=1", "every h=5 model", "every per-stock model",
                     "direction prediction"],
        "target": ("y = |(Close[t+1]/Close[t] - 1) - (SPY_Close[t+1]/SPY_Close[t] - 1)|: excess_return_v1 at h=1 "
                   "(training.excess_targets.build_excess_targets), absolute value; Close = the frozen snapshot close"),
        "baseline": {"name": m2.REFERENCE, "artifact": f"phase12/stage3/{BASELINE_FILE}",
                     "definition": (f"Ridge(alpha={cfg.ridge_alpha}) on Volatility, spy_volatility_20 after "
                                    f"training-quantile clipping {list(cfg.clip_quantiles)} and StandardScaler; "
                                    "the Stage 2 reference model"),
                     "fit": ("all development rows, Stage 2 fit_finalist procedure; fitted in Stage 3A from development "
                             "data only because Stage 2 saved candidates only; deterministic (two fits identical)")},
        "inference": [
            "verify each artifact's SHA256 before unpickling",
            "build the 8 features for every (stock, sealed session) with the frozen generators (sealed_feature_build)",
            "Volatility and spy_volatility_20 enter raw",
            ("the other 6 features are z-scored per stock with that stock's stored DEVELOPMENT mean and std "
             "(ddof 0, std 0 -> 1); no sealed statistic is ever computed for scaling"),
            "columns in features_in_order; estimator.predict; predictions floored at 0",
            "the baseline is applied the same way to its 2 features",
            "no fit, partial_fit, refit, recalibration or threshold choice on any sealed row"],
        "sealed_window": {
            "prediction_dates": f"common sessions t with {CONFIRMATION_START} <= t and t+1 <= {CONFIRMATION_END}",
            "target_dates": f"t+1 in {CONFIRMATION_START} .. {CONFIRMATION_END}",
            "lookback": ("rolling features at t may use sessions before 2024-10-01 (as in development); no feature "
                         "uses information after the session-t prediction timestamp")},
        "sealed_feature_build": {
            "status": f"NOT YET WRITTEN ({SEALED_BUILDER_FILE}); reviewed and approved separately before authorization",
            "requirements": [
                "calls the generator files in pins.generator_code_sha256 unchanged (hash-checked at run time)",
                "event labels from rules_v1 (spec hash pinned); no FinBERT or any other model is run",
                ("sentiment scores are not an input of any of the 8 features; where a generator needs one a constant "
                 "placeholder is passed"),
                "UTX->RTX mapping and the news alignment and prediction-timestamp rules exactly as in 11B.3",
                ("before any sealed row is emitted, the builder rebuilds the development period and must reproduce "
                 "the 8 columns of the frozen 11B development frames exactly (bit-identical); otherwise it stops"),
                "outputs only Date, Close, SPY_Close and the 8 features for sealed-window sessions"]},
        "eligibility_and_missing_data": {
            "universe": "the 29 Phase 11B stocks; a missing stock or SPY series -> INVALID",
            "sessions": (f"all stocks must share the same sealed sessions; fewer than {MIN_SESSIONS} -> INVALID"),
            "targets": "a non-finite Close or SPY_Close at t or t+1 -> INVALID (prices are verified snapshots)",
            "features": ("a row is excluded when any of the 8 features is non-finite or the generator marks it "
                         "uncovered (e.g. no covered 20-session news window); excluded rows are dropped for finalist "
                         "and baseline alike; no imputation, forward-fill or zero-fill"),
            "zero_counts": "zero news or event counts are values, not missing data",
            "limits": (f"excluded rows > {MAX_EXCLUDED_SHARE:.0%} of all rows or > {MAX_EXCLUDED_SHARE_STOCK:.0%} "
                       "of any stock's rows -> INVALID"),
            "order": "validity is decided before any prediction is compared with a target"},
        "metrics": ["MSE", "RMSE", "MAE", "relative MSE and MAE improvement over the baseline",
                    "per-stock and per-block relative MSE improvement",
                    "ranking (descriptive): pooled Spearman(prediction, actual); mean per-date cross-sectional Spearman",
                    "calibration (descriptive): prediction deciles; top-5% moves mean predicted / mean actual; "
                    "bottom-50% moves; residual skew"],
        "statistics": {"test": ("one-sided Diebold-Mariano, H0: finalist squared error not lower than the baseline's, "
                                "on the per-date cross-sectional mean loss differential"),
                       "variance": "Newey-West, Bartlett, lag L = max(1, ceil(T^(1/3))), T = sealed dates; 2L sensitivity",
                       "family": "1 comparison (Holm-adjusted p = raw p)", "alpha": cfg.alpha},
        "qualification": {
            "Q1_improvement": f"relative MSE improvement >= {cfg.min_rel_mse_improvement} AND relative MAE improvement "
                              f">= {cfg.min_rel_mae_improvement}",
            "Q2_statistics": f"DM p < {cfg.alpha} AND p with lag 2L < {cfg.alpha}",
            "Q3_periods": f"MSE better in >= {cfg.min_blocks_improved} of {cfg.n_blocks} equal consecutive blocks of "
                          "sealed dates",
            "Q4_separation": ("artifacts trained only on rows with labels <= 2024-09-30; every evaluated row has "
                              "t >= 2024-10-01 and t+1 <= 2026-09-25; no estimator is fitted during the run"),
            "Q5_breadth": (f"MSE improved for >= {cfg.min_share_stocks_improved:.0%} of stocks, and still > 0 without "
                           f"the best stock and without the best quarter ({QUARTER_SESSIONS}-session chunks of sealed "
                           f"dates; a final chunk < {MIN_LAST_QUARTER} merged into the previous one)"),
            "Q6_reproducibility": ("artifact and model fingerprints and development prediction fingerprints match the "
                                   "pins, and a second in-process inference on the sealed rows is bit-identical"),
            "rule": "CONFIRMED only if Q1-Q6 all pass"},
        "criteria_review": {
            "Q1": "kept unchanged",
            "Q2": "kept; the family is 1 comparison, so Holm leaves p unchanged",
            "Q3": "kept; 4 blocks of about 6 months",
            "Q4": "adapted: there is no walk-forward, so the gap check becomes the train/sealed separation check",
            "Q5": ("adapted: 'best fold' becomes 'best quarter', the same 63-session length as a Stage 2 fold, since "
                   "the sealed period has no folds"),
            "Q6": ("adapted: no refit happens, so reproducibility is checked on the frozen artifacts and on "
                   "repeated inference")},
        "outcomes": {"CONFIRMED": "valid run, Q1-Q6 pass",
                     "NOT_CONFIRMED": "valid run, any of Q1-Q6 fails; final, never re-run with changes",
                     "INVALID": ("an integrity or eligibility rule fails before any prediction is compared with a "
                                 "target; no metric is computed or written; a repeat needs a written addendum whose fix "
                                 "does not depend on sealed outcomes")},
        "calibration_boundary": ("the finalist is evaluated uncalibrated; any product calibration must be developed on "
                                 "Stage 2 development out-of-fold predictions, frozen separately, and never tuned on "
                                 "sealed results"),
        "prohibitions": ["retraining or refitting any model on sealed rows", "recalibration with sealed rows",
                         "feature, preprocessing, horizon or baseline changes", "threshold or criterion changes",
                         "a second confirmation run after a valid one", "evaluating any excluded model"],
        "authorization": {"file": f"phase12/stage3/{AUTHORIZATION_FILE}", "written_by": "the project owner, by hand",
                          "fields": ["amendment_sha256", "runner_code_sha256", "sealed_builder_code_sha256",
                                     "authorized_by", "authorized_at", "statement"],
                          "statement": AUTHORIZATION_STATEMENT},
        "config": {"min_sessions": MIN_SESSIONS, "max_excluded_share": MAX_EXCLUDED_SHARE,
                   "max_excluded_share_stock": MAX_EXCLUDED_SHARE_STOCK, "quarter_sessions": QUARTER_SESSIONS,
                   "min_last_quarter": MIN_LAST_QUARTER, "stage2_config": dataclasses.asdict(cfg)},
    }


def dev_fingerprints(finalist: dict, baseline: dict, frames: dict[str, pd.DataFrame]) -> dict:
    panel = m2.build_panel(frames, HORIZON)
    return {"rows": int(len(panel)),
            "finalist_predictions_sha256": predictions_sha256(predict(finalist, panel)),
            "baseline_predictions_sha256": predictions_sha256(predict(baseline, panel))}


def current_pins(finalist: dict, baseline: dict, baseline_sha: str, frames: dict, inputs_sha: str | None) -> dict:
    return {"stage2_files_sha256": dict(STAGE2_PINS), "stage2_protocol_sha256": STAGE2_PROTOCOL_SHA256,
            "inputs_sha256": inputs_sha,
            "finalist": {"file_sha256": FINALIST_SHA256, "model_sha256": model_sha256(finalist)},
            "baseline": {"file_sha256": baseline_sha, "model_sha256": model_sha256(baseline)},
            "development_fingerprints": dev_fingerprints(finalist, baseline, frames),
            "code_sha256": {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES},
            "generator_code_sha256": {f: file_sha256(PROJECT_ROOT / f) for f in GENERATOR_FILES},
            "rules_v1_spec_sha256": RULES_V1_SPEC_SHA256}


def verify_stage2(stage2_dir: Path = m2.OUT_DIR) -> None:
    for rel, sha in STAGE2_PINS.items():
        p = Path(stage2_dir) / Path(rel).relative_to("stage2")
        if not p.is_file() or file_sha256(p) != sha:
            raise Phase12S3Error(f"Stage 2 evidence {rel} is missing or changed")


def _dev_frames() -> tuple[dict, str]:
    inputs = r.verified_inputs(r.FROZEN_SPEC)
    return r.load_frames(inputs, r.FROZEN_SPEC), r.inputs_sha256(inputs)


def freeze(out_dir: Path = OUT_DIR, stage2_dir: Path = m2.OUT_DIR, frames: dict | None = None,
           inputs_sha: str | None = None) -> dict:
    """Development data only. Writes the baseline artifact and the amendment once, atomically."""
    out_dir = Path(out_dir)
    staging = out_dir.parent / (out_dir.name + ".incomplete")
    if out_dir.exists() or staging.exists():
        raise Phase12S3Error(f"{out_dir.name} already exists; the amendment is written once")
    verify_stage2(stage2_dir)
    if frames is None:
        frames, inputs_sha = _dev_frames()
        if inputs_sha != INPUTS_SHA256:
            raise Phase12S3Error("development inputs differ from the Stage 2 inputs")
    s1.check_development_only(frames)
    finalist = load_artifact(Path(stage2_dir) / "results" / "finalists" / FINALIST_FILE, FINALIST_SHA256)
    check_artifact(finalist, FINALIST_MODEL)
    baseline, again = fit_baseline(frames), fit_baseline(frames)
    if model_sha256(baseline) != model_sha256(again):
        raise Phase12S3Error("baseline refit is not deterministic")
    check_artifact(baseline, m2.REFERENCE)
    staging.mkdir(parents=True)
    joblib.dump(baseline, staging / BASELINE_FILE)
    baseline_sha = file_sha256(staging / BASELINE_FILE)
    if model_sha256(load_artifact(staging / BASELINE_FILE, baseline_sha)) != model_sha256(baseline):
        raise Phase12S3Error("saved baseline does not reload identically")
    doc = {**rules(), "pins": current_pins(finalist, baseline, baseline_sha, frames, inputs_sha),
           "environment": environment(), "git": git_info(), "sealed_data_read": False,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_new_json(staging / AMENDMENT_FILE, doc)
    os.replace(staging, out_dir)
    return doc


def load_amendment(out_dir: Path = OUT_DIR, pinned_sha: str | None = None) -> dict:
    """The frozen amendment: its file hash equals the pin in code and its rules equal `rules()`."""
    pinned_sha = AMENDMENT_SHA256 if pinned_sha is None else pinned_sha
    path = Path(out_dir) / AMENDMENT_FILE
    if not path.is_file():
        raise Phase12S3Error("amendment not found (run `freeze` first)")
    if pinned_sha is None:
        raise Phase12S3Error("amendment hash is not pinned in code (AMENDMENT_SHA256); confirmation is not set up")
    if file_sha256(path) != pinned_sha:
        raise Phase12S3Error("amendment differs from its pinned SHA256; the amendment cannot change")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if {k: doc.get(k) for k in rules()} != json.loads(canonical_json(rules())):
        raise Phase12S3Error("amendment rules differ from the rules in code")
    return doc


def verify(out_dir: Path = OUT_DIR, stage2_dir: Path = m2.OUT_DIR, frames: dict | None = None,
           pinned_sha: str | None = None) -> tuple[dict, dict, dict]:
    """Every pin re-checked on development data. Returns (amendment, finalist, baseline)."""
    doc = load_amendment(out_dir, pinned_sha)
    pins = doc["pins"]
    verify_stage2(stage2_dir)
    inputs_sha = pins["inputs_sha256"]
    if frames is None:
        frames, inputs_sha = _dev_frames()
    if inputs_sha != pins["inputs_sha256"]:
        raise Phase12S3Error("development inputs differ from the amendment")
    finalist = load_artifact(Path(stage2_dir) / "results" / "finalists" / FINALIST_FILE, pins["finalist"]["file_sha256"])
    baseline = load_artifact(Path(out_dir) / BASELINE_FILE, pins["baseline"]["file_sha256"])
    check_artifact(finalist, FINALIST_MODEL)
    check_artifact(baseline, m2.REFERENCE)
    live = current_pins(finalist, baseline, pins["baseline"]["file_sha256"], frames, inputs_sha)
    bad = [k for k in live if live[k] != pins.get(k)]
    if bad:
        raise Phase12S3Error(f"pins no longer reproduce: {bad}")
    return doc, finalist, baseline


# ============================================================
# Sealed evaluation (pure computation; tested on synthetic frames only)
# ============================================================


def quarters(n_dates: int) -> np.ndarray:
    """Chunk index per date: 63-session chunks, a final chunk < 21 merged into the previous one."""
    q = np.arange(n_dates) // QUARTER_SESSIONS
    if q.max() > 0 and (q == q.max()).sum() < MIN_LAST_QUARTER:
        q[q == q.max()] = q.max() - 1
    return q


def sealed_panel(frames: dict[str, pd.DataFrame], universe=None) -> tuple[pd.DataFrame | None, dict]:
    """Eligible labelled rows and the validity record; panel is None when the run is INVALID."""
    universe = sorted(r.FROZEN_SPEC.universe if universe is None else universe)
    reasons, parts, dates = [], [], None
    if sorted(frames) != universe:
        return None, {"valid": False, "reasons": [f"universe differs: {sorted(set(universe) ^ set(frames))}"]}
    for sym in universe:
        f = frames[sym]
        d = list(pd.to_datetime(f["Date"]).dt.date)
        if d != sorted(set(d)) or d[0] < CONFIRMATION_START or d[-1] > CONFIRMATION_END:
            raise Phase12S3Error(f"{sym}: sealed frame dates are unordered or outside {CONFIRMATION_START} .. "
                                 f"{CONFIRMATION_END}")
        if dates is not None and d != dates:
            reasons.append(f"{sym}: sessions differ from the other stocks")
        dates = d if dates is None else dates
        try:
            lab = s1.build_targets(f.reset_index(drop=True), HORIZON)
        except ExcessTargetError as e:                            # non-finite or invalid prices
            reasons.append(f"{sym}: target cannot be built ({e})")
            continue
        part = pd.DataFrame({"symbol": sym, "pos": np.arange(len(lab)), "Date": lab["Date"].to_numpy()})
        for c in m2.FEATURES:
            part[c] = lab[c].to_numpy(dtype="float64")
        part["y"] = lab["t_magnitude"].to_numpy(dtype="float64")
        parts.append(part)
    if dates is not None and len(dates) < MIN_SESSIONS:
        reasons.append(f"{len(dates)} common sealed sessions < {MIN_SESSIONS}")
    if reasons:
        return None, {"valid": False, "reasons": reasons}
    panel = pd.concat(parts, ignore_index=True)
    ok = np.isfinite(panel[list(m2.FEATURES)].to_numpy()).all(axis=1)
    per_stock = (~pd.Series(ok)).groupby(panel["symbol"].to_numpy()).mean()
    rec = {"rows": int(len(panel)), "excluded_rows": int((~ok).sum()), "excluded_share": float((~ok).mean()),
           "excluded_share_max_stock": float(per_stock.max()), "sessions": len(dates),
           "prediction_dates": [str(panel["Date"].min())[:10], str(panel["Date"].max())[:10]]}
    if rec["excluded_share"] > MAX_EXCLUDED_SHARE:
        reasons.append(f"excluded share {rec['excluded_share']:.4f} > {MAX_EXCLUDED_SHARE}")
    if rec["excluded_share_max_stock"] > MAX_EXCLUDED_SHARE_STOCK:
        reasons.append(f"{per_stock.idxmax()} excluded share {rec['excluded_share_max_stock']:.4f} > "
                       f"{MAX_EXCLUDED_SHARE_STOCK}")
    if reasons:
        return None, {"valid": False, "reasons": reasons, **rec}
    panel = panel[ok].reset_index(drop=True)
    panel["fold"] = quarters(len(dates))[panel["pos"].to_numpy()]
    return panel, {"valid": True, "reasons": [], **rec}


def _cross_sectional_spearman(preds: pd.DataFrame, col: str) -> float:
    vals = [stats.spearmanr(g[col], g["y"]).statistic for _, g in preds.groupby("pos", sort=True) if len(g) > 2]
    return float(np.nanmean(vals))


def evaluate(finalist: dict, baseline: dict, frames: dict[str, pd.DataFrame], universe=None) -> dict:
    """One evaluation of the frozen models on sealed-window frames. Validity is decided before any comparison."""
    cfg = m2.FROZEN_CONFIG
    panel, validity = sealed_panel(frames, universe)
    if panel is None:
        return {"outcome": "INVALID", "validity": validity}
    preds = panel[["symbol", "pos", "Date", "y", "fold"]].copy()
    preds[FINALIST_MODEL] = predict(finalist, panel)
    preds[m2.REFERENCE] = predict(baseline, panel)
    repro = (predictions_sha256(predict(finalist, panel)) == predictions_sha256(preds[FINALIST_MODEL].to_numpy())
             and predictions_sha256(predict(baseline, panel)) == predictions_sha256(preds[m2.REFERENCE].to_numpy()))
    separation = (all(date.fromisoformat(a["trained_on"][1]) <= DEV_END for a in (finalist, baseline))
                  and pd.Timestamp(panel["Date"].min()).date() >= CONFIRMATION_START)
    c = m2.compare(preds, FINALIST_MODEL, HORIZON, cfg)
    c["dm"]["p_holm"] = holm_adjust([c["dm"]["p"]])[0]
    q = m2.qualify(c, c["dm"]["p_holm"], separation, repro, cfg, pooled=True)
    q["Q4_separation"] = q.pop("Q4_leakage")
    ranking = {m: {"spearman_pooled": float(stats.spearmanr(preds[m], preds["y"]).statistic),
                   "spearman_cross_sectional_mean": _cross_sectional_spearman(preds, m)}
               for m in (FINALIST_MODEL, m2.REFERENCE)}
    return {"outcome": "CONFIRMED" if q["qualified"] else "NOT_CONFIRMED", "validity": validity,
            "comparison": c, "qualification": q, "ranking": ranking,
            "calibration": {m: m2.calibration(preds, m) for m in (FINALIST_MODEL, m2.REFERENCE)},
            "predictions_sha256": {m: predictions_sha256(preds[m].to_numpy()) for m in (FINALIST_MODEL, m2.REFERENCE)}}


# ============================================================
# Authorization and the one-time run
# ============================================================


def load_authorization(out_dir: Path, amendment_sha: str) -> dict:
    """The owner's hand-written record: this amendment, this runner, this sealed builder, the exact statement."""
    path = Path(out_dir) / AUTHORIZATION_FILE
    if not path.is_file():
        raise Phase12S3Error("no owner authorization record; the sealed confirmation is not authorized")
    a = json.loads(path.read_text(encoding="utf-8"))
    builder = PROJECT_ROOT / SEALED_BUILDER_FILE
    checks = {"amendment_sha256": a.get("amendment_sha256") == amendment_sha,
              "runner_code_sha256": a.get("runner_code_sha256") == file_sha256(Path(__file__)),
              "sealed_builder_code_sha256": builder.is_file() and a.get("sealed_builder_code_sha256") == file_sha256(builder),
              "authorized_by": bool(str(a.get("authorized_by") or "").strip()),
              "authorized_at": bool(str(a.get("authorized_at") or "").strip()),
              "statement": a.get("statement") == AUTHORIZATION_STATEMENT}
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise Phase12S3Error(f"authorization record does not authorize this run: {failed}")
    return a


def sealed_feature_build_unavailable() -> dict:
    raise Phase12S3Error(f"the sealed feature builder ({SEALED_BUILDER_FILE}) has not been written and approved")


def _refuse_existing_attempt(out_dir: Path) -> None:
    if (out_dir / CONFIRMATION_DIR).exists() or (out_dir / (CONFIRMATION_DIR + ".incomplete")).exists():
        raise Phase12S3Error("a confirmation attempt already exists; the sealed confirmation runs once")


def register_attempt(out_dir: Path, amendment_sha: str, auth: dict, preflight: dict | None = None) -> Path:
    """
    The one-time registration. mkdir of confirmation.incomplete/ is atomic and exclusive, so of any concurrent
    invocations exactly one succeeds; a run that already completed (confirmation/ exists) is refused as well.
    """
    out_dir = Path(out_dir)
    final, staging = out_dir / CONFIRMATION_DIR, out_dir / (CONFIRMATION_DIR + ".incomplete")
    _refuse_existing_attempt(out_dir)
    try:
        staging.mkdir()
    except FileExistsError:
        raise Phase12S3Error("a confirmation attempt already exists; the sealed confirmation runs once") from None
    if final.exists():                                   # completed between the check and mkdir: not ours to run
        staging.rmdir()
        raise Phase12S3Error("a confirmation attempt already exists; the sealed confirmation runs once")
    _write_new_json(staging / "attempt.json", {"amendment_sha256": amendment_sha, "authorization": auth,
                                               "preflight": preflight,
                                               "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return staging / "attempt.json"


def confirm(out_dir: Path = OUT_DIR, stage2_dir: Path = m2.OUT_DIR, frames: dict | None = None,
            sealed_loader: Callable[[], dict] = sealed_feature_build_unavailable, pinned_sha: str | None = None,
            universe=None) -> dict:
    """
    A  preflight, no sealed access, no attempt consumed: amendment, Stage 2 evidence, artifacts and fingerprints,
       authorization (runner and builder hashes), no earlier attempt, then `sealed_loader.preflight()` if the
       loader has one (the sealed builder: erratum, output directory, offline flags, FinBERT, development
       reproduction, 500/500 labels)
    B  register_attempt: confirmation.incomplete/attempt.json, exactly once
    C  sealed_loader() -> sealed frames, then the unchanged evaluation
    """
    out_dir = Path(out_dir)
    doc, finalist, baseline = verify(out_dir, stage2_dir, frames, pinned_sha)
    amendment_sha = file_sha256(out_dir / AMENDMENT_FILE)
    auth = load_authorization(out_dir, amendment_sha)
    _refuse_existing_attempt(out_dir)
    preflight = getattr(sealed_loader, "preflight", None)
    summary = preflight() if preflight is not None else None
    staging = register_attempt(out_dir, amendment_sha, auth, summary).parent
    final = out_dir / CONFIRMATION_DIR
    started = time.time()
    res = evaluate(finalist, baseline, sealed_loader(), universe)
    _write_new_json(staging / "confirmation.json", {
        "document_type": "phase12_sealed_confirmation", "amendment_sha256": amendment_sha, **m2._jsonable(res),
        "sealed_data_read": True, "environment": environment(), "git": git_info(),
        "runtime_seconds": round(time.time() - started, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    os.replace(staging, final)
    return res


def status_lines(out_dir: Path = OUT_DIR) -> list[str]:
    out_dir = Path(out_dir)
    path = out_dir / AMENDMENT_FILE
    lines = [f"amendment     : {'written, sha256 ' + file_sha256(path) if path.is_file() else 'not written'}",
             f"pinned in code: {AMENDMENT_SHA256 or 'no'}",
             f"authorization : {'present' if (out_dir / AUTHORIZATION_FILE).is_file() else 'absent'}",
             f"sealed builder: {'present' if (PROJECT_ROOT / SEALED_BUILDER_FILE).is_file() else 'not written'}"]
    if (out_dir / CONFIRMATION_DIR).is_dir():
        res = json.loads((out_dir / CONFIRMATION_DIR / "confirmation.json").read_text(encoding="utf-8"))
        lines.append(f"confirmation  : {res['outcome']}")
    elif (out_dir / (CONFIRMATION_DIR + ".incomplete")).exists():
        lines.append("confirmation  : ATTEMPTED, INCOMPLETE (report it; no silent retry)")
    else:
        lines.append("confirmation  : NOT_EVALUATED (sealed data untouched)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Stage 3A finalist freeze and sealed confirmation")
    parser.add_argument("command", choices=["freeze", "verify", "status", "confirm"])
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze()
        print(f"amendment written; sha256 {file_sha256(OUT_DIR / AMENDMENT_FILE)}")
    elif args.command == "verify":
        verify()
        print("verify: every pin reproduces on development data")
    elif args.command == "confirm":
        from training.phase12_sealed_features import SealedBuild   # hash-bound by the authorization record
        print(f"outcome: {confirm(sealed_loader=SealedBuild())['outcome']}")
    print("\n".join(status_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
