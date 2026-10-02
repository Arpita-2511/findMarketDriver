"""
Phase 11 amendment 01 - AAPL reproducibility check with a frozen numerical tolerance.

Why: Phase 11.1 showed that the canonical Phase 11 price batch (yfinance,
retrieved 2026-10-02) differs from the Phase 10 snapshots (retrieved
2026-09-29 / 2026-10-01) by up to ~1e-6 relative in adjusted closes - the
provider recomputes adjustments at retrieval time. AAPL per-stock A/E results
can therefore not be bit-identical to Phase 10A, so the frozen check "checked
for equivalence against Phase 10A where data and design are mathematically
identical" (design sha256 266daadd...) is replaced by the numerical tolerance
below. The frozen design registry itself is NOT edited.

Unchanged: research methodology, target, features, models, data. The canonical
Phase 11 input is the Phase 11.1 batch (data/processed/phase11/market/,
verification report data/results/research/phase11/price_verification.json).
The reproducibility check is a check only: its outcome never changes the
Phase 11 design, data or results.

Frozen BEFORE any Phase 11A result exists.

    python -m training.phase11_reproducibility freeze    # write-once amendment + tier-1 (data) observation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11_design as p11d  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.phase9_research import ResearchError, _write_json_atomic  # noqa: E402

AMENDMENT_PATH = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "amendment_01_reproducibility.json"
FROZEN_DESIGN_SHA = "266daaddba0e8cb18163b4ec9db7eadabe2d5ab574bf410e043a7d28d3247650"
CANONICAL_INPUT = {"directory": "data/processed/phase11/market",
                   "verification_report": "data/results/research/phase11/price_verification.json",
                   "verification_report_sha256": "c6afc03fec249e146c37c0b4f5c512835c1cda9d344229b8ac3f0ce7bd1b15b0"}
PHASE10_SNAPSHOTS = {"AAPL": "data/raw/stocks/AAPL_1d_10y_20260929T205051Z.csv",
                     "SPY": "data/raw/stocks/SPY_1d_10y_20261001T104943Z.csv",
                     "QQQ": "data/raw/stocks/QQQ_1d_10y_20261001T104948Z.csv"}

TOLERANCE = {
    "tier1_data": {
        "symbols": ["AAPL", "SPY", "QQQ"],
        "max_rel_close_diff": 1e-5,          # |Close_p11 / Close_p10 - 1| on common dates
        "max_abs_return_diff": 1e-5,         # |r_p11 - r_p10| of daily simple returns on common dates
        "rule": "every common date within both limits",
    },
    "tier2_results": {
        "scope": "the 66 AAPL per-stock comparisons of feature sets A and E (2 x 3 horizons x 11 models) vs the "
                 "matching Phase 10A development experiments (p10a_A_*, p10a_E_*)",
        "gate_status": "identical gate_v1 status for every comparison",
        "max_rel_primary_metric_diff": 5e-3,  # |m_p11 / m_p10a - 1| for the primary metric (MSE or Log_Loss)
        "max_abs_dm_p_diff": 0.02,            # |p_p11 - p_p10a| raw one-sided DM p-value
        "rule": "every comparison within all three limits",
    },
    "outcome": ("REPRODUCED_WITHIN_TOLERANCE if tier 1 and tier 2 pass, otherwise REPRODUCIBILITY_CHECK_FAILED with "
                "every difference reported; the outcome never alters the Phase 11 design, data or results"),
}


class Phase11AmendmentError(ResearchError):
    """Amendment protocol violation."""


def tier1_compare(new: pd.DataFrame, old: pd.DataFrame, tol: dict = TOLERANCE["tier1_data"]) -> dict:
    """new/old: columns Date (YYYY-MM-DD) and Close. Compares common dates only."""
    j = new[["Date", "Close"]].merge(old[["Date", "Close"]], on="Date", suffixes=("_new", "_old"))
    if j.empty:
        raise Phase11AmendmentError("no common dates")
    rel = np.abs(j["Close_new"].to_numpy() / j["Close_old"].to_numpy() - 1)
    r_new = j["Close_new"].pct_change().to_numpy()[1:]
    r_old = j["Close_old"].pct_change().to_numpy()[1:]
    ret = np.abs(r_new - r_old)
    out = {"common_dates": int(len(j)), "first": j["Date"].iloc[0], "last": j["Date"].iloc[-1],
           "identical_closes": int((j["Close_new"] == j["Close_old"]).sum()),
           "max_rel_close_diff": float(rel.max()), "max_abs_return_diff": float(ret.max()) if len(ret) else 0.0}
    out["pass"] = out["max_rel_close_diff"] <= tol["max_rel_close_diff"] and out["max_abs_return_diff"] <= tol["max_abs_return_diff"]
    return out


def tier2_compare(p11: dict[str, dict], p10a: dict[str, dict], tol: dict = TOLERANCE["tier2_results"]) -> dict:
    """
    p11 / p10a: {key: {"status", "primary_value", "p_value"}} keyed identically (feature set, horizon, model).
    Every key of p10a must be present in p11.
    """
    missing = sorted(set(p10a) - set(p11))
    if missing:
        raise Phase11AmendmentError(f"missing Phase 11 comparisons: {missing[:5]}")
    rows, ok_all = {}, True
    for k, a in p10a.items():
        b = p11[k]
        rel = abs(b["primary_value"] / a["primary_value"] - 1) if a["primary_value"] else float("inf")
        dp = abs(b["p_value"] - a["p_value"])
        ok = (b["status"] == a["status"] and rel <= tol["max_rel_primary_metric_diff"] and dp <= tol["max_abs_dm_p_diff"])
        ok_all &= ok
        rows[k] = {"status_p10a": a["status"], "status_p11": b["status"], "rel_primary_diff": rel,
                   "abs_p_diff": dp, "pass": ok}
    return {"n": len(rows), "n_pass": sum(r["pass"] for r in rows.values()), "pass": ok_all, "comparisons": rows}


def outcome(tier1: dict[str, dict], tier2: dict | None) -> str:
    t1 = all(v["pass"] for v in tier1.values())
    if tier2 is None:
        return "TIER1_PASS_TIER2_PENDING" if t1 else "REPRODUCIBILITY_CHECK_FAILED"
    return "REPRODUCED_WITHIN_TOLERANCE" if t1 and tier2["pass"] else "REPRODUCIBILITY_CHECK_FAILED"


def _canonical_frame(symbol: str) -> pd.DataFrame:
    path = next((PROJECT_ROOT / CANONICAL_INPUT["directory"]).glob(f"{symbol}_1d_*.csv"))
    return pd.read_csv(path, dtype={"Date": str}, float_precision="round_trip")


def _phase10_frame(symbol: str) -> pd.DataFrame:
    bars, _ = load_raw_snapshot(PROJECT_ROOT / PHASE10_SNAPSHOTS[symbol])
    return pd.DataFrame({"Date": bars["Date"].dt.strftime("%Y-%m-%d"), "Close": bars["Close"].to_numpy()})


def freeze(path: Path = AMENDMENT_PATH) -> dict:
    path = Path(path)
    if path.exists():
        raise Phase11AmendmentError(f"{path} already exists; amendment 01 is written once")
    reg = p11d.load_registry()
    if reg["design_sha256"] != FROZEN_DESIGN_SHA:
        raise Phase11AmendmentError("frozen Phase 11 design mismatch")
    report = PROJECT_ROOT / CANONICAL_INPUT["verification_report"]
    if file_sha256(report) != CANONICAL_INPUT["verification_report_sha256"]:
        raise Phase11AmendmentError("Phase 11.1 verification report changed")
    tier1 = {s: tier1_compare(_canonical_frame(s), _phase10_frame(s)) for s in TOLERANCE["tier1_data"]["symbols"]}
    record = {
        "amendment": "phase11_amendment_01_reproducibility_tolerance",
        "amends": {"design_sha256": FROZEN_DESIGN_SHA, "field": "design.aapl_reproducibility",
                   "frozen_text": reg["design"]["aapl_reproducibility"],
                   "frozen_registry_edited": False},
        "reason": ("the canonical Phase 11.1 batch differs from the Phase 10 snapshots by ~1e-6 relative (provider "
                   "re-adjustment at retrieval), so bit-identical equivalence with Phase 10A is impossible"),
        "unchanged": ["research methodology", "target", "features", "models", "data", "gate_v1", "families",
                      "qualification", "intervals", "universe"],
        "canonical_phase11_input": CANONICAL_INPUT,
        "phase10_reference_snapshots": {s: {"path": p, "sha256": file_sha256(PROJECT_ROOT / p)}
                                        for s, p in PHASE10_SNAPSHOTS.items()},
        "tolerance": TOLERANCE,
        "tier1_observed": tier1,
        "status": outcome(tier1, None),
        "frozen_before_any_phase11a_result": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
    }
    _write_json_atomic(path, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11 amendment 01 - reproducibility tolerance")
    parser.add_argument("command", choices=["freeze", "status"])
    args = parser.parse_args(argv)
    rec = freeze() if args.command == "freeze" else json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    print(f"Amendment: {rec['amendment']}  status {rec['status']}")
    for s, v in rec["tier1_observed"].items():
        print(f"  {s}: common {v['common_dates']}, max rel close diff {v['max_rel_close_diff']:.3e}, "
              f"max abs return diff {v['max_abs_return_diff']:.3e}, pass {v['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
