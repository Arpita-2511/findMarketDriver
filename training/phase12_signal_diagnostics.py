"""
Phase 12 Stage 1: development-data signal diagnostics (no models, no qualification).

Question: which existing feature families show a reproducible association with future excess
returns, their direction, or their magnitude - before any new model is pre-registered.

Data
    the 29 hash-verified Phase 11B development frames (training.phase11b_research.verified_inputs /
    load_frames; 2017-02-01 .. 2024-09-30). Nothing after the development cutoff is read; the sealed
    confirmation period is never touched. No feature is recomputed or redefined.

Targets (training.excess_targets.build_excess_targets, unchanged; last h rows of each frame unlabelled)
    return     future_excess_return_h = (Close[t+h]/Close[t] - 1) - (SPY[t+h]/SPY[t] - 1)
    direction  1 if future_excess_return_h > 0 else 0
    magnitude  |future_excess_return_h|                      (derived here; the only new definition)
    horizons   h = 1, 5 trading days

Feature families (columns of the frozen frames)
    sentiment_window (13 sw_*), technical (20, set A), market_context (9, set E minus A),
    event (23 event__*), cumulative_sentiment_control (15 sentiment__*; known-bad control, never a candidate)

Statistic: Spearman rank correlation with HAC uncertainty
    Within each sample, the feature and the target are converted to average ranks and standardized
    (mean 0, variance 1). The OLS slope of the standardized target ranks on the standardized feature
    ranks equals the Spearman correlation; its standard error is Newey-West (Bartlett kernel) on the
    score u_x * e, e = u_y - rho * u_x. This treats the ranks as given (no correction for rank
    estimation) and is an asymptotic approximation. Lag L = max(2h, ceil(n^(1/3))): at least the
    target overlap (h - 1) plus slack for persistent features; t with 2L is reported as a sensitivity.
    Pooled: within-stock standardized ranks (removes stock identity and scale), estimate = average
    within-stock correlation weighted by observations; Driscoll-Kraay standard error (scores summed
    per date, then Newey-West over dates) - robust to arbitrary cross-stock correlation on a date and
    to serial correlation up to L.

Samples: all labelled days, and news days (event__article_count > 0 at the row date; >= 250 rows).

Placebo: circular time shift of the target relative to the features by 100 fixed offsets
    (10% .. 90% of the series length), the same offset for every stock. Preserves each series'
    own autocorrelation and the cross-stock dependence on a date; breaks feature-target alignment.
    Placebo rejection rates measure how well the HAC p-values are calibrated.

Calibration: HAC with a short lag under-covers when both series are very persistent (volatility
    regimes, trending counts): a first run showed placebo rejection rates of 24% (pooled magnitude)
    and 50-70% (cumulative controls) at nominal 5%. Each test is therefore also judged against its
    own placebo distribution:
        placebo_rms_t   root mean square of the 100 placebo t statistics of that test
        t_calibrated    t / max(1, placebo_rms_t)      (never inflates a statistic)
        p_calibrated    two-sided normal p of t_calibrated
        p_empirical     (1 + #{|t_placebo| >= |t|}) / (1 + #placebo)
    Holm on p_calibrated is the primary multiplicity result; Holm on the raw HAC p is reported too.

Multiple testing: Holm within families fixed in code before any result (FAMILY_RULE). Every result
    here is EXPLORATORY; nothing is a qualified or confirmed predictor.

    python -m training.phase12_signal_diagnostics run      # writes data/results/research/phase12/stage1/
    python -m training.phase12_signal_diagnostics status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11_design as p11d  # noqa: E402
import training.phase11b_research as r  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.excess_targets import build_excess_targets  # noqa: E402
from training.phase11b_feature_build import FRAME_COLUMNS  # noqa: E402
from training.phase9_research import ResearchError, _write_new_json, canonical_json, environment, holm_adjust  # noqa: E402

PHASE = "12"
STAGE = "12.S1"
DIAGNOSTIC_VERSION = "phase12_signal_diagnostics_v1"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage1"
DEV_END = date(2024, 9, 30)
HORIZONS = (1, 5)
TARGETS = ("return", "direction", "magnitude")
SCOPES = ("all_days", "news_days")
NEWS_COLUMN = "event__article_count"
MIN_N = 250
ALPHA = 0.05
N_PLACEBO = 100
PLACEBO_FRACTIONS = tuple(float(x) for x in np.round(np.linspace(0.1, 0.9, N_PLACEBO), 6))
CONTROL = "cumulative_sentiment_control"
CODE_FILES = ("training/phase12_signal_diagnostics.py", "training/excess_targets.py",
              "training/phase11b_research.py")
FAMILY_RULE = ("Holm within each (level, scope, target, kind) over all eligible tests; level in {pooled, "
               "per_stock}, scope in {all_days, news_days}, kind in {candidate, control}; a family spans both "
               "horizons (and all stocks for per_stock). Pooled magnitude also has a 'candidate_partial_vol' "
               "family (association net of the stock's own Volatility feature). Eligibility depends only on "
               "sample size and feature variation, never on outcomes.")


class Phase12Error(ResearchError):
    """Phase 12 Stage 1 protocol violation; the run stops."""


# ============================================================
# Feature families and targets
# ============================================================


def feature_families(columns=FRAME_COLUMNS) -> dict[str, list[str]]:
    technical = list(p11d.FEATURE_SETS["A"]["columns"])
    market = [c for c in p11d.FEATURE_SETS["E"]["columns"] if c not in technical]
    fams = {
        "sentiment_window": [c for c in columns if c.startswith("sw_")],
        "technical": technical,
        "market_context": market,
        "event": [c for c in columns if c.startswith("event__")],
        CONTROL: [c for c in columns if c.startswith("sentiment__")],
    }
    expected = {"sentiment_window": 13, "technical": 20, "market_context": 9, "event": 23, CONTROL: 15}
    if {k: len(v) for k, v in fams.items()} != expected:
        raise Phase12Error(f"feature family sizes differ from the frozen frames: {({k: len(v) for k, v in fams.items()})}")
    missing = [c for v in fams.values() for c in v if c not in columns]
    if missing:
        raise Phase12Error(f"feature columns missing from the frames: {missing}")
    return fams


def build_targets(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """Frozen excess-return targets (last h rows dropped) plus magnitude = |future excess return|."""
    out = build_excess_targets(frame, h)
    return out.assign(t_return=out["future_return"], t_direction=out["direction"].astype("float64"),
                      t_magnitude=out["future_return"].abs())


def check_development_only(frames: dict[str, pd.DataFrame]) -> None:
    for sym, f in frames.items():
        last = pd.Timestamp(f["Date"].max()).date()
        if last > DEV_END:
            raise Phase12Error(f"{sym}: row dated {last} is after the development end {DEV_END}; sealed data is never read")


# ============================================================
# Statistics
# ============================================================


def rank_standardize(a: np.ndarray) -> np.ndarray:
    """Column-wise average ranks, standardized to mean 0 / variance 1 (ddof 0); constant columns -> NaN."""
    a = np.asarray(a, dtype="float64")
    one = a.ndim == 1
    a = a[:, None] if one else a
    rk = stats.rankdata(a, axis=0, method="average")
    sd = rk.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (rk - rk.mean(axis=0)) / sd
    z[:, sd == 0] = np.nan
    return z[:, 0] if one else z


def lag_for(n: int, h: int) -> int:
    return max(2 * h, math.ceil(n ** (1 / 3)))


def bartlett_sum(s: np.ndarray, lag: int) -> np.ndarray:
    """sum_t s_t^2 + 2 sum_{l=1..L} (1 - l/(L+1)) sum_t s_t s_{t-l}, column-wise (not divided by T)."""
    s = np.asarray(s, dtype="float64")
    s = s[:, None] if s.ndim == 1 else s
    total = (s * s).sum(axis=0)
    for l in range(1, min(lag, len(s) - 1) + 1):
        total = total + 2 * (1 - l / (lag + 1)) * (s[l:] * s[:-l]).sum(axis=0)
    return total


def _p_two_sided(t):
    return 2 * stats.norm.sf(np.abs(t))


def hac_spearman(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """Spearman rho of each column of x with y, Newey-West SE (lags `lag` and 2*lag). NaN for constant inputs."""
    ux, uy = rank_standardize(x), rank_standardize(y)
    ux = ux[:, None] if ux.ndim == 1 else ux
    return _slope_hac([(ux, uy, np.arange(len(uy)))], len(uy), lag)


def _slope_hac(units, n_dates: int, lag: int) -> dict:
    """
    Driscoll-Kraay slope of standardized y ranks on standardized x ranks over units (one unit = one stock;
    with one unit this is plain Newey-West). Columns of x that are NaN (constant) in a unit are excluded
    from that unit for that column.
    """
    p = units[0][0].shape[1]
    num, den, nobs = np.zeros(p), np.zeros(p), np.zeros(p)
    for ux, uy, _ in units:
        if np.isnan(uy).all():
            continue
        ok = ~np.isnan(ux).any(axis=0)
        x0 = np.where(ok, ux, 0.0)
        num += (x0 * uy[:, None]).sum(axis=0)
        den += (x0 * x0).sum(axis=0)
        nobs += ok * len(uy)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    H = np.zeros((n_dates, p))
    for ux, uy, pos in units:
        if np.isnan(uy).all():
            continue
        ok = ~np.isnan(ux).any(axis=0)
        x0 = np.where(ok, ux, 0.0)
        e = uy[:, None] - x0 * np.nan_to_num(rho)
        np.add.at(H, pos, x0 * e)
    out = {"rho": rho, "n": nobs}
    for tag, L in (("", lag), ("_2L", 2 * lag)):
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(bartlett_sum(H, L)) / den
            t = rho / se
        out["se" + tag], out["t" + tag] = se, t
    out["p"] = _p_two_sided(out["t"])
    out["lag"] = lag
    return out


def residualize(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove the projection on v (standardized) from each column of u; columns stay NaN if NaN."""
    v = v[:, None]
    beta = (u * v).sum(axis=0) / (v * v).sum()
    res = u - v * beta
    with np.errstate(invalid="ignore", divide="ignore"):
        res = (res - res.mean(axis=0)) / res.std(axis=0)
    return res


# ============================================================
# Samples
# ============================================================


@dataclass
class Sample:
    """One stock, one horizon, one scope: standardized feature ranks, raw targets, date positions."""
    symbol: str
    h: int
    scope: str
    pos: np.ndarray          # positions in the common date index
    x: np.ndarray            # raw features (n, p)
    ux: np.ndarray           # standardized feature ranks (n, p), NaN columns where constant
    y_full: dict             # target -> full-length target array (all labelled rows, before the scope filter)
    mask: np.ndarray         # scope filter over the labelled rows
    vol_u: np.ndarray        # standardized ranks of the stock's own Volatility feature on this sample


def build_samples(frames: dict[str, pd.DataFrame], features: list[str]) -> tuple[list[Sample], list, dict]:
    """Labelled samples per stock / horizon / scope; returns (samples, common dates, sample-size table)."""
    dates = None
    for f in frames.values():
        d = list(pd.to_datetime(f["Date"]).dt.date)
        if dates is not None and d != dates:
            raise Phase12Error("stocks do not share the same development sessions")
        dates = d
    samples, sizes = [], {}
    for sym in sorted(frames):
        for h in HORIZONS:
            lab = build_targets(frames[sym], h)
            if list(pd.to_datetime(lab["Date"]).dt.date) != dates[:len(dates) - h]:
                raise Phase12Error(f"{sym} h={h}: labelled rows are not the first n-h sessions")
            x_all = lab[features].to_numpy(dtype="float64")
            if not np.isfinite(x_all).all():
                raise Phase12Error(f"{sym}: non-finite feature values")
            y_full = {t: lab[f"t_{t}"].to_numpy(dtype="float64") for t in TARGETS}
            news = lab[NEWS_COLUMN].to_numpy() > 0
            vol = lab["Volatility"].to_numpy(dtype="float64")
            for scope, mask in (("all_days", np.ones(len(lab), dtype=bool)), ("news_days", news)):
                sizes[f"{sym}|h{h}|{scope}"] = int(mask.sum())
                x = x_all[mask]
                samples.append(Sample(sym, h, scope, np.flatnonzero(mask), x, rank_standardize(x), y_full, mask,
                                      rank_standardize(vol[mask])))
    return samples, dates, sizes


# ============================================================
# Tests, families, placebo
# ============================================================


def _family(level: str, scope: str, target: str, kind: str) -> str:
    return f"{level}|{scope}|{target}|{kind}"


def run_tests(samples: list[Sample], features: list[str], kinds: list[str], n_dates: int,
              shift: float | None = None) -> pd.DataFrame:
    """Per-stock and pooled tests for every target / horizon / scope. `shift` = placebo fraction (or None)."""
    rows = []

    def y_of(s: Sample, t: str) -> np.ndarray:
        y = s.y_full[t]
        if shift is not None:
            y = np.roll(y, int(round(shift * len(y))))
        return y[s.mask]

    groups: dict = {}
    for s in samples:
        groups.setdefault((s.h, s.scope), []).append(s)
    for (h, scope), group in sorted(groups.items()):
        for t in TARGETS:
            units, units_pv = [], []
            for s in group:
                n = len(s.pos)
                y = y_of(s, t)
                uy = rank_standardize(y)
                eligible_n = n >= MIN_N
                if eligible_n:
                    res = _slope_hac([(s.ux, uy, np.arange(n))], n, lag_for(n, h))
                    units.append((s.ux, uy, s.pos))
                    if t == "magnitude":
                        if not np.isnan(s.vol_u).all():
                            units_pv.append((residualize(s.ux, s.vol_u), residualize(uy[:, None], s.vol_u)[:, 0], s.pos))
                for j, feat in enumerate(features):
                    rows.append({"level": "per_stock", "scope": scope, "symbol": s.symbol, "target": t, "horizon": h,
                                 "feature": feat, "kind": kinds[j], "n": n,
                                 **({k: float(res[k][j]) for k in ("rho", "se", "t", "p", "se_2L", "t_2L")}
                                    if eligible_n else {k: float("nan") for k in ("rho", "se", "t", "p", "se_2L", "t_2L")}),
                                 "lag": lag_for(n, h),
                                 "eligible": bool(eligible_n and not np.isnan(s.ux[:, j]).any()
                                                  and np.isfinite(res["p"][j]))})
            for tag, us in (("", units), ("_partial_vol", units_pv)):
                if not us:
                    continue
                L = lag_for(n_dates, h)
                res = _slope_hac(us, n_dates, L)
                for j, feat in enumerate(features):
                    if tag and feat == "Volatility":
                        continue
                    rows.append({"level": "pooled", "scope": scope, "symbol": None, "target": t, "horizon": h,
                                 "feature": feat, "kind": kinds[j] + tag, "n": int(res["n"][j]),
                                 **{k: float(res[k][j]) for k in ("rho", "se", "t", "p", "se_2L", "t_2L")},
                                 "lag": L, "eligible": bool(res["n"][j] >= MIN_N and np.isfinite(res["t"][j]))})
    df = pd.DataFrame(rows)
    df["family"] = [_family(a, b, c, d) for a, b, c, d in zip(df["level"], df["scope"], df["target"], df["kind"])]
    return df


def apply_holm(df: pd.DataFrame, p_col: str = "p", out_col: str = "p_holm") -> pd.DataFrame:
    """Holm within each family over eligible rows with a finite p, in a deterministic order; others get NaN."""
    df = df.copy()
    df[out_col] = np.nan
    order_cols = ["family", "symbol", "horizon", "feature"]
    use = df[df["eligible"] & np.isfinite(df[p_col])].sort_values(order_cols, na_position="first")
    for fam, idx in use.groupby("family", sort=True).groups.items():
        idx = list(idx)
        df.loc[idx, out_col] = holm_adjust(df.loc[idx, p_col].tolist())
    return df


def _test_keys(df: pd.DataFrame) -> np.ndarray:
    return np.array([f"{a}|{b}|{c}|{e}" for a, b, c, e in zip(df["family"], df["symbol"], df["horizon"], df["feature"])])


class PlaceboAccumulator:
    """Per-test placebo statistics and per-family rejection summaries, accumulated shift by shift."""

    def __init__(self, real: pd.DataFrame):
        self.key = _test_keys(real)
        self.abs_t = real["t"].abs().to_numpy()
        n = len(real)
        self.sum_sq, self.cnt = np.zeros(n), np.zeros(n)
        self.max_abs, self.exceed = np.zeros(n), np.zeros(n)
        self.fam: dict = {}

    def add(self, pl: pd.DataFrame) -> None:
        if not np.array_equal(_test_keys(pl), self.key):
            raise Phase12Error("placebo tests are not aligned with the real tests")
        t = pl["t"].to_numpy()
        ok = pl["eligible"].to_numpy() & np.isfinite(t)
        a = np.where(ok, np.abs(t), 0.0)
        self.sum_sq += a * a
        self.cnt += ok
        self.max_abs = np.maximum(self.max_abs, a)
        self.exceed += ok & (a >= self.abs_t)
        el = pl[ok]
        for fam, g in el.groupby("family", sort=True):
            f = self.fam.setdefault(fam, {"n_tests": 0, "rej": 0, "rej_2L": 0, "abs_t": [], "survivors": []})
            f["n_tests"] += len(g)
            f["rej"] += int((g["p"] < ALPHA).sum())
            f["rej_2L"] += int((_p_two_sided(g["t_2L"]) < ALPHA).sum())
            f["abs_t"].append(g["t"].abs().to_numpy())
            f["survivors"].append(int((np.array(holm_adjust(g["p"].tolist())) < ALPHA).sum()))

    def calibrate(self, real: pd.DataFrame) -> pd.DataFrame:
        df = real.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(self.sum_sq / self.cnt)
        df["placebo_rms_t"] = rms
        df["placebo_max_abs_t"] = np.where(self.cnt > 0, self.max_abs, np.nan)
        df["p_empirical"] = np.where(self.cnt > 0, (1 + self.exceed) / (1 + self.cnt), np.nan)
        df["t_calibrated"] = df["t"] / np.maximum(1.0, rms)
        df["p_calibrated"] = _p_two_sided(df["t_calibrated"])
        return df

    def summary(self) -> list[dict]:
        out = []
        for fam, f in sorted(self.fam.items()):
            abs_t = np.concatenate(f["abs_t"])
            out.append({"family": fam, "n_tests": f["n_tests"], "n_shifts": len(f["survivors"]),
                        "reject_rate_raw_0_05": f["rej"] / f["n_tests"],
                        "reject_rate_raw_0_05_2L": f["rej_2L"] / f["n_tests"],
                        "mean_abs_t": float(abs_t.mean()), "p99_abs_t": float(np.quantile(abs_t, 0.99)),
                        "holm_survivors_mean_per_shift": float(np.mean(f["survivors"])),
                        "holm_survivors_max_over_shifts": int(max(f["survivors"]))})
        return out


# ============================================================
# Summary
# ============================================================


def summarize(df: pd.DataFrame, families: dict, placebo: list[dict], sizes: dict) -> dict:
    fam_of = {c: k for k, v in families.items() for c in v}
    el = df[df["eligible"]].copy()
    el["feature_family"] = el["feature"].map(fam_of)
    fam_rows = []
    for (fam,), g in el.groupby(["family"], sort=True):
        fam_rows.append({"family": fam, "n_tests": int(len(g)), "raw_p_below_0_05": int((g["p"] < ALPHA).sum()),
                         "raw_reject_rate": float((g["p"] < ALPHA).mean()),
                         "holm_survivors_raw": int((g["p_holm"] < ALPHA).sum()),
                         "calibrated_p_below_0_05": int((g["p_calibrated"] < ALPHA).sum()),
                         "holm_survivors_calibrated": int((g["p_holm_calibrated"] < ALPHA).sum())})
    by_ff = []
    pooled = el[el["level"] == "pooled"]
    for (scope, target, h, kind, ff), g in pooled.groupby(["scope", "target", "horizon", "kind", "feature_family"], sort=True):
        best = g.loc[g["t_calibrated"].abs().idxmax()]
        by_ff.append({"scope": scope, "target": target, "horizon": int(h), "kind": kind, "feature_family": ff,
                      "n_features": int(len(g)), "median_abs_rho": float(g["rho"].abs().median()),
                      "max_abs_rho": float(g["rho"].abs().max()),
                      "holm_survivors_raw": int((g["p_holm"] < ALPHA).sum()),
                      "holm_survivors_calibrated": int((g["p_holm_calibrated"] < ALPHA).sum()),
                      "best_feature": best["feature"], "best_rho": float(best["rho"]), "best_t": float(best["t"]),
                      "best_t_calibrated": float(best["t_calibrated"]), "best_p_empirical": float(best["p_empirical"]),
                      "best_p_holm_calibrated": float(best["p_holm_calibrated"])})
    per = el[(el["level"] == "per_stock")]
    consistency = []
    for (scope, target, h, feat), g in per.groupby(["scope", "target", "horizon", "feature"], sort=True):
        consistency.append({"scope": scope, "target": target, "horizon": int(h), "feature": feat,
                            "feature_family": fam_of[feat], "n_stocks": int(len(g)),
                            "share_positive": float((g["rho"] > 0).mean()), "median_rho": float(g["rho"].median()),
                            "n_holm_survivors_raw": int((g["p_holm"] < ALPHA).sum()),
                            "n_holm_survivors_calibrated": int((g["p_holm_calibrated"] < ALPHA).sum())})
    top = (pooled[~pooled["kind"].str.startswith("control")].assign(abs_t=lambda d: d["t_calibrated"].abs())
           .sort_values(["abs_t", "family", "feature"], ascending=[False, True, True]).head(60))
    cols = ["family", "feature", "feature_family", "horizon", "n", "rho", "se", "t", "t_2L", "p", "p_holm",
            "placebo_rms_t", "t_calibrated", "p_empirical", "p_holm_calibrated"]
    return {"families": fam_rows, "pooled_by_feature_family": by_ff, "per_stock_consistency": consistency,
            "top_pooled_candidate_tests": top[cols].to_dict(orient="records"), "placebo": placebo,
            "sample_sizes": sizes}


# ============================================================
# Run
# ============================================================


def diagnose(frames: dict[str, pd.DataFrame], log=print) -> tuple[pd.DataFrame, dict]:
    """Pure computation: results table (with Holm) and summary. No I/O."""
    check_development_only(frames)
    families = feature_families()
    features = [c for v in families.values() for c in v]
    kinds = ["control" if c in families[CONTROL] else "candidate" for c in features]
    samples, dates, sizes = build_samples(frames, features)
    real = run_tests(samples, features, kinds, len(dates))
    log(f"[12.S1] {len(real)} tests, {int(real['eligible'].sum())} eligible")
    acc = PlaceboAccumulator(real)
    for i, frac in enumerate(PLACEBO_FRACTIONS):
        acc.add(run_tests(samples, features, kinds, len(dates), shift=frac))
        if (i + 1) % 10 == 0:
            log(f"[12.S1] placebo shift {i + 1}/{len(PLACEBO_FRACTIONS)} done")
    df = apply_holm(apply_holm(acc.calibrate(real)), "p_calibrated", "p_holm_calibrated")
    summary = summarize(df, families, acc.summary(), sizes)
    summary["feature_families"] = families
    summary["duplicates"] = {"sw_count_1": "identical to event__article_count by construction (Phase 11B T13); "
                                           "both kept, each in its own feature family"}
    summary["dates"] = {"first": str(dates[0]), "last": str(dates[-1]), "n_sessions": len(dates)}
    return df, summary


def _round(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: "float64" for c in ("rho", "se", "t", "p", "se_2L", "t_2L", "p_holm", "placebo_rms_t",
                                             "placebo_max_abs_t", "p_empirical", "t_calibrated", "p_calibrated",
                                             "p_holm_calibrated")})


def run(out_dir: Path = OUT_DIR, log=print) -> dict:
    final, staging = Path(out_dir), Path(str(out_dir) + ".incomplete")
    if final.exists() or staging.exists():
        raise Phase12Error(f"{final.name} (or its .incomplete) already exists; Stage 1 outputs are write-once")
    started = time.time()
    inputs = r.verified_inputs(r.FROZEN_SPEC)                    # hash-verified development frames only
    frames = r.load_frames(inputs, r.FROZEN_SPEC)               # refuses any date outside the development period
    df, summary = diagnose(frames, log)
    staging.mkdir(parents=True)
    csv_path = staging / "tests.csv"
    with open(csv_path, "w", encoding="utf-8", newline="\n") as fh:
        _round(df).to_csv(fh, index=False, float_format="%.10g", lineterminator="\n")
    config = {"diagnostic_version": DIAGNOSTIC_VERSION, "phase": PHASE, "stage": STAGE, "horizons": list(HORIZONS),
              "targets": {"return": "future_excess_return_h (excess_return_v1)",
                          "direction": "1 if future_excess_return_h > 0 else 0 (excess_return_v1)",
                          "magnitude": "|future_excess_return_h| (derived in Phase 12 Stage 1)"},
              "scopes": {"all_days": "all labelled rows", "news_days": f"{NEWS_COLUMN} > 0 at the row date"},
              "min_n": MIN_N, "lag_rule": "L = max(2h, ceil(n^(1/3))); t_2L uses 2L",
              "statistic": "Spearman rho = OLS slope of standardized ranks; Newey-West (per stock) / "
                           "Driscoll-Kraay (pooled), Bartlett kernel, two-sided normal p",
              "placebo": {"type": "circular shift of the target relative to the features, same offset for all stocks",
                          "n_shifts": len(PLACEBO_FRACTIONS),
                          "fractions_of_series_length": list(PLACEBO_FRACTIONS)},
              "calibration": "t_calibrated = t / max(1, placebo_rms_t); p_calibrated two-sided normal; "
                             "p_empirical = (1 + #|t_placebo| >= |t|) / (1 + #placebo); primary Holm on p_calibrated",
              "family_rule": FAMILY_RULE, "alpha": ALPHA, "status": "EXPLORATORY - no qualification"}
    summary_doc = {"document_type": "phase12_stage1_signal_diagnostics", **config, "summary": summary,
                   "inputs_sha256": r.inputs_sha256(inputs), "feature_run_id": inputs["feature_run_id"],
                   "tests_csv_sha256": file_sha256(csv_path), "n_tests": int(len(df)),
                   "n_eligible": int(df["eligible"].sum()), "sealed_data_read": False,
                   "code_sha256": {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES},
                   "environment": environment(), "git": git_info(),
                   "runtime_seconds": round(time.time() - started, 1),
                   "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_new_json(staging / "summary.json", summary_doc)
    _write_new_json(staging / "manifest.json", {
        "files": {n: file_sha256(staging / n) for n in ("tests.csv", "summary.json")},
        "inputs_sha256": summary_doc["inputs_sha256"], "code_sha256": summary_doc["code_sha256"]})
    os.replace(staging, final)
    return summary_doc


def status_lines(out_dir: Path = OUT_DIR) -> list[str]:
    final = Path(out_dir)
    if not final.exists():
        return ["stage 1 diagnostics: not run" + (" (incomplete run present)" if Path(str(final) + ".incomplete").exists() else "")]
    m = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, h in m["files"].items() if file_sha256(final / n) != h]
    s = json.loads((final / "summary.json").read_text(encoding="utf-8"))
    return [f"stage 1 diagnostics: COMPLETE ({s['n_eligible']} eligible tests); manifest {'verified' if not bad else 'MISMATCH ' + str(bad)}",
            "status: EXPLORATORY - no qualification; sealed data read: False"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Stage 1 development-data signal diagnostics")
    parser.add_argument("command", choices=["run", "status"])
    args = parser.parse_args(argv)
    if args.command == "run":
        run()
    print("\n".join(status_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
