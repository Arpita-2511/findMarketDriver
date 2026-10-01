"""
Incremental evaluation of the feature store through the CENTRAL harness (Phase 7).

    A  technical
    B  technical + sentiment
    C  technical + sentiment + events

Apples-to-apples by construction: all three experiments use the SAME rows
(sessions covered by both news families, so no feature is unknown), the
same targets (rebuilt by the harness from Close: next-day return and
direction_v1), the same candidates (training.evaluation_harness
.default_candidates: Mean/Zero/Linear/Ridge, Always UP/Base Rate/Logistic),
the same TimeSeriesSplit(n_splits, gap = horizon), the same metrics and the
same qualification gate (gate_v1). Nothing is tuned.

Minimum-data guard: with fewer than MIN_EVALUATION_ROWS evaluable rows the
experiments are NOT run and the report says INSUFFICIENT_DATA - a 20-fold
walk-forward on a few weeks of data would produce numbers without meaning.

The harness evaluates a single price series; multi-symbol evaluation is
future work and is rejected here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluation_harness import (  # noqa: E402
    CLASSIFICATION,
    DEFAULT_RESULTS_DIR,
    REGRESSION,
    HarnessConfig,
    _json_safe,
    default_candidates,
    run_evaluation,
)
from training.feature_store import COVERAGE_COLUMNS, FEATURE_FAMILIES, load_feature_store  # noqa: E402

MIN_EVALUATION_ROWS = 252          # about one trading year
EXPERIMENTS = {
    "A_technical": ("technical",),
    "B_technical_sentiment": ("technical", "sentiment"),
    "C_technical_sentiment_events": ("technical", "sentiment", "events"),
}
SMALL_SAMPLE_NOTE = ("This is a pipeline/integration validation dataset, not sufficient evidence of "
                     "generalizable model performance.")


class IncrementalEvaluationError(ValueError):
    """The feature store cannot be evaluated as requested."""


def experiment_columns(families: tuple[str, ...]) -> list[str]:
    return [c for fam in families for c in FEATURE_FAMILIES[fam]]


def evaluation_frame(store: pd.DataFrame) -> pd.DataFrame:
    """Rows covered by BOTH news families, in chronological order, in the harness's input format."""
    symbols = store["symbol"].unique()
    if len(symbols) != 1:
        raise IncrementalEvaluationError(f"the harness evaluates one series; got symbols {sorted(symbols)}")
    covered = store[store[COVERAGE_COLUMNS[0]] & store[COVERAGE_COLUMNS[1]]]
    covered = covered.sort_values("trading_date").reset_index(drop=True)
    frame = pd.DataFrame({"Date": pd.to_datetime([d.isoformat() for d in covered["trading_date"]]),
                          "Close": covered["Close"], "Target": covered["target_return_1d"]})
    for col in experiment_columns(EXPERIMENTS["C_technical_sentiment_events"]):    # superset of A and B
        frame[col] = covered[col].astype("float64")
    return frame


def _experiment_summary(result) -> dict:
    r = result.report
    models = {}
    for m in r["models"]:
        models[m["model"]] = {"task": m["task"], "is_baseline": m["is_baseline"],
                              "status": m["qualification"]["status"], "metrics": m["metrics"]}
    return {"status": "RUN", "n_features": len(r["features"]), "evaluation_period": r["evaluation_period"],
            "qualification_result": r["qualification_result"], "models": models}


def _verdict(experiments: dict) -> dict:
    """Factual comparison of qualification outcomes per track (no subjective ranking)."""
    verdict = {}
    for track in (REGRESSION, CLASSIFICATION):
        outcomes = {name: e["qualification_result"].get(track) for name, e in experiments.items()}
        qualified = {name: o for name, o in outcomes.items() if isinstance(o, list) and o}
        if not qualified:
            verdict[track] = "NO QUALIFIED MODEL in any experiment - no evidence that adding features helps"
        else:
            verdict[track] = f"qualified models: {qualified} (see per-model metrics; compare against A)"
    return verdict


def run_incremental_evaluation(store: pd.DataFrame, *, horizon: int = 1, n_splits: int = 20,
                               min_rows: int = MIN_EVALUATION_ROWS, candidates_factory=default_candidates) -> dict:
    frame = evaluation_frame(store)
    report = {
        "evaluation": "feature_store_incremental",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_in_store": len(store),
        "evaluable_rows": len(frame),
        "evaluable_period": ([frame["Date"].iloc[0].date().isoformat(), frame["Date"].iloc[-1].date().isoformat()]
                             if len(frame) else None),
        "config": {"horizon": horizon, "n_splits": n_splits, "gap": horizon, "min_rows": min_rows,
                   "candidates": [c.name for c in candidates_factory()]},
        "experiments": {},
    }
    if len(frame) < min_rows:
        report["status"] = "INSUFFICIENT_DATA"
        report["reason"] = (f"{len(frame)} evaluable rows (sessions with technical, sentiment and event "
                            f"features) < {min_rows}; a {n_splits}-fold walk-forward would not be meaningful. "
                            "No experiment was run and no metric is reported.")
        report["experiments"] = {name: {"status": "NOT_RUN", "families": list(fams),
                                        "n_features": len(experiment_columns(fams))}
                                 for name, fams in EXPERIMENTS.items()}
        report["note"] = SMALL_SAMPLE_NOTE
        return report

    config = HarnessConfig(horizon=horizon, n_splits=n_splits)
    for name, fams in EXPERIMENTS.items():
        result = run_evaluation(data=frame, feature_columns=experiment_columns(fams),
                                feature_version="+".join(fams), candidates=candidates_factory(),
                                config=config, experiment=name)
        report["experiments"][name] = {"families": list(fams), **_experiment_summary(result)}

    periods = {json.dumps(e["evaluation_period"], sort_keys=True) for e in report["experiments"].values()}
    if len(periods) != 1:
        raise IncrementalEvaluationError("experiments were not evaluated on identical out-of-sample rows")
    report["status"] = "RUN"
    report["comparison"] = _verdict(report["experiments"])
    if len(frame) < 2 * min_rows:
        report["note"] = SMALL_SAMPLE_NOTE
    return report


def save_report(report: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report["timestamp"]).strftime("%Y%m%d")
    path = results_dir / f"feature_store_incremental_{stamp}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incremental A/B/C evaluation of the feature store")
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=MIN_EVALUATION_ROWS)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    store, _ = load_feature_store(args.feature_store)
    report = run_incremental_evaluation(store, n_splits=args.n_splits, min_rows=args.min_rows)
    print(f"Status: {report['status']}  evaluable rows: {report['evaluable_rows']}  period: {report['evaluable_period']}")
    if report["status"] != "RUN":
        print(report["reason"])
    for name, e in report["experiments"].items():
        line = f"{name:32s} features={e['n_features']:3d}  {e['status']}"
        if e["status"] == "RUN":
            line += f"  regression={e['qualification_result'].get(REGRESSION)}  " \
                    f"classification={e['qualification_result'].get(CLASSIFICATION)}"
        print(line)
    if report.get("comparison"):
        print(json.dumps(report["comparison"], indent=2))
    if report.get("note"):
        print(report["note"])
    if not args.no_save:
        print(f"Saved: {save_report(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
