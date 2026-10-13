"""
Phase 11B Stage 11B.3: development-only feature build and Gate C evidence.

    Gate B integrity report, 11B.2 scoring and classification run records
        -> every input hash and provenance verified; nothing selectable on the command line
        -> partition_canonical            same streaming split as 11B.2; sealed records keep
                                          only their key and must reproduce Gate B exactly
        -> development scores + events    joined to development articles only; event labels
                                          re-derived with rules_v1 and compared to the stored ones
        -> UTX->RTX mapping in memory     map_symbols_for_rtx (frozen rule); canonical file untouched
        -> frozen generators, unchanged   sentiment_features_v1 (sentiment__*), event_features_v1
                                          numeric (event__*), sentiment_window_v1 (sw_*)
        -> 11A price frames (unchanged)   Date, Close, SPY_Close, technical A, market context E
        -> checks (all before any write)  1928 rows/stock, no NaN, every sw row covered,
                                          sw_count_1 == event__article_count everywhere,
                                          prefix invariance, horizon independence, panel OOS rows,
                                          Holm family sizes, 11A p-value artefacts by hash
        -> write once, atomically         data/processed/features/phase11b/<run id>/<SYMBOL>.csv
                                          (read back and compared with freshly built 11A frames)
                                          data/results/research/phase11/11B/feature_verification.json (last)

Never called here: load_canonical_dataset, load_sentiment_dataset, load_event_dataset.
No model is trained and no experiment is run.

Usage (project root; operator-run only; no network, no model, no keys):

    python -m training.phase11b_feature_build
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11_design as p11d  # noqa: E402
import training.phase11a_research as a11  # noqa: E402
from services.event_classifier import KeywordEventClassifier  # noqa: E402
from services.event_features import EVENT_FEATURE_COLUMNS, ClassifiedArticle, generate_event_features  # noqa: E402
from services.historical_events import (  # noqa: E402
    HistoricalEventError,
    _record_to_classified,
    classifier_provenance,
    classify_scored_articles,
)
from services.historical_sentiment import PROVENANCE_FIELDS  # noqa: E402
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_alignment import row_prediction_timestamp  # noqa: E402
from services.sentiment_features import FEATURE_COLUMNS as SENTIMENT_FEATURE_COLUMNS  # noqa: E402
from services.sentiment_features import ScoredArticle, generate_sentiment_features  # noqa: E402
from services.sentiment_window import COLUMNS as SW_COLUMNS  # noqa: E402
from services.sentiment_window import generate_sentiment_window_features  # noqa: E402
from training.build_dataset import meta_path_for  # noqa: E402
from training.excess_targets import SPY_CLOSE  # noqa: E402
from training.phase11b_event_classification import (  # noqa: E402
    EVENTS_DIR_11B,
    EVENTS_DIR_PHASE8,
    join_development_scores,
    load_scoring_record,
    verify_rules_v1,
    verify_score_file,
)
from training.phase11b_ingest_closeout import RTX_UTX_BOUNDARY_UTC, map_symbols_for_rtx  # noqa: E402
from training.phase11b_sentiment_scoring import (  # noqa: E402
    BOUNDARY_RULE,
    CUTOFF_TEXT,
    DEVELOPMENT_CUTOFF_UTC,
    PHASE_11B_DIR,
    REPORT_PATH,
    SENTIMENT_DIR,
    SENTIMENT_DIR_11B,
    ScoringError,
    _atomic_write,
    _json_bytes,
    _sha256_file,
    load_integrity_report,
    partition_canonical,
    run_id_for,
    verify_frozen_artifacts,
    verify_partition_against_report,
)

STAGE = "11B.3"
BUILD_VERSION = "phase11b_development_features_v1"
FEATURES_DIR_11B = PROJECT_ROOT / "data" / "processed" / "features" / "phase11b"
FEATURES_DIR_PHASE8 = PROJECT_ROOT / "data" / "processed" / "features"
VERIFICATION_PATH = PHASE_11B_DIR / "feature_verification.json"
CLASSIFICATION_RECORD_TYPE = "phase_11b_rules_v1_development_classification_run"
SW_COVERAGE_START = datetime(2017, 1, 1, tzinfo=timezone.utc)          # design_freeze coverage_interval.start_utc
SW_COVERAGE_END = DEVELOPMENT_CUTOFF_UTC                              # design_freeze coverage_interval.end_utc
DESIGN_FREEZE_PATH = PHASE_11B_DIR / "design_freeze.json"

SENTIMENT_COLUMNS = tuple("sentiment__" + c for c in SENTIMENT_FEATURE_COLUMNS)
EVENT_NUMERIC = tuple(c for c in EVENT_FEATURE_COLUMNS if c != "dominant_event_type")
EVENT_COLUMNS = tuple("event__" + c for c in EVENT_NUMERIC)
NEWS_COLUMNS = (*SENTIMENT_COLUMNS, *EVENT_COLUMNS, *SW_COLUMNS)
FEATURE_SETS_11B = {k: tuple(p11d.FEATURE_SETS[k]["columns"]) for k in ("B", "C", "D", "F")}
PRICE_COLUMNS = tuple(dict.fromkeys([*p11d.FEATURE_SETS["A"]["columns"], *p11d.FEATURE_SETS["E"]["columns"]]))
FRAME_COLUMNS = ("Date", "Close", SPY_CLOSE, *PRICE_COLUMNS, *NEWS_COLUMNS)


class FeatureBuildError(ScoringError):
    """Fail-closed signal for Stage 11B.3."""


@dataclass(frozen=True)
class BuildSpec:
    """Frozen expectations of the real build; the library only accepts other values for synthetic tests."""
    universe: tuple = tuple(p11d.UNIVERSE)
    rows_per_stock: int = 1928
    panel_oos_rows_h1: int = 52780
    n_splits: int = 20
    prefix_cut_fractions: tuple = (0.25, 0.5, 0.75)


FROZEN_SPEC = BuildSpec()


def verify_frozen_spec(spec: BuildSpec) -> dict:
    """The real build's expectations must equal the frozen design; returns the frozen family sizes."""
    design = json.loads(DESIGN_FREEZE_PATH.read_text(encoding="utf-8"))
    period, matrix = design["development_period"], design["experiment_matrix"]
    for name, cols in FEATURE_SETS_11B.items():
        if len(cols) != matrix["feature_sets"][name]["n_features"] or len(set(cols)) != len(cols):
            raise FeatureBuildError(f"feature set {name} differs from the frozen design")
    if not set(NEWS_COLUMNS) >= {c for cols in FEATURE_SETS_11B.values() for c in cols} - set(PRICE_COLUMNS):
        raise FeatureBuildError("a frozen feature column is not produced by this build")
    if spec == FROZEN_SPEC and (tuple(sorted(spec.universe)) != tuple(sorted(matrix["universe"]))
                                or spec.rows_per_stock != period["n_sessions_per_stock"]
                                or spec.panel_oos_rows_h1 != period["panel_oos_rows"]["all_29_stocks_h1"]):
        raise FeatureBuildError("the frozen build specification differs from the design freeze")
    return design["multiple_testing"]["final_families"]


# ============================================================
# Inputs
# ============================================================


def load_classification_record(record_dir: Path, *, run_id: str, frozen: dict, rules_sha: str, report_path: Path,
                               scoring_record_path: Path, scoring_record: dict) -> dict:
    path = record_dir / f"rules_v1_development_classification_{run_id}.json"
    if not path.is_file():
        raise FeatureBuildError(f"11B.2 classification run record not found: {path.name}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "document_type": rec.get("document_type") == CLASSIFICATION_RECORD_TYPE,
        "stage": rec.get("stage") == "11B.2",
        "run_id": rec.get("run_id") == run_id,
        "frozen_artifact_sha256": rec.get("frozen_artifact_sha256") == frozen,
        "rules_v1_spec_sha256": rec.get("rules_v1_spec_sha256") == rules_sha,
        "integrity_report.sha256": (rec.get("integrity_report") or {}).get("sha256") == _sha256_file(report_path),
        "scoring_run_record.sha256": (rec.get("scoring_run_record") or {}).get("sha256") == _sha256_file(scoring_record_path),
        "score_input.sha256": (rec.get("score_input") or {}).get("sha256") == scoring_record["outputs"]["scores"]["sha256"],
        "development_cutoff_utc": rec.get("development_cutoff_utc") == CUTOFF_TEXT,
        "boundary_rule": rec.get("boundary_rule") == BOUNDARY_RULE,
        "development_article_count": rec.get("development_article_count") == scoring_record["development_article_count"],
        "classified_article_count": rec.get("classified_article_count") == scoring_record["development_article_count"],
        "excluded_article_count": rec.get("excluded_article_count") == 0,
        "phase_8_score_reuse": rec.get("phase_8_score_reuse") is False,
        "sealed_content_sent_to_classifier": rec.get("sealed_content_sent_to_classifier") is False,
        "classifier": rec.get("classifier") == classifier_provenance(KeywordEventClassifier()),
        "outputs.events.file": ((rec.get("outputs") or {}).get("events") or {}).get("file") == f"{run_id}.events-rules_v1.jsonl",
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise FeatureBuildError(f"11B.2 classification run record fails verification: {failed}")
    return rec


def verify_events_file(events_dir: Path, cls_record: dict, scoring_record: dict) -> Path:
    out = cls_record["outputs"]["events"]
    path = events_dir / out["file"]
    if not path.is_file() or not meta_path_for(path).is_file():
        raise FeatureBuildError(f"development events file or its metadata not found: {path.name}")
    meta = json.loads(meta_path_for(path).read_text(encoding="utf-8"))
    if _sha256_file(path) != out["sha256"] or meta.get("sha256") != out["sha256"] or meta.get("rows") != out["rows"]:
        raise FeatureBuildError("development events file does not match the 11B.2 classification run record")
    if (meta.get("impact_source") or {}).get("sha256") != scoring_record["outputs"]["scores"]["sha256"] or \
            meta.get("classifier") != cls_record["classifier"]:
        raise FeatureBuildError("development events metadata does not match its scoring input or classifier")
    return path


def join_development_events(events_path: Path, scored: list[ScoredArticle]) -> list[ClassifiedArticle]:
    """Event records joined to scored development articles only; labels must equal a fresh rules_v1 run."""
    by_key = {s.article.key: s for s in scored}
    provenance = classifier_provenance(KeywordEventClassifier())
    classified: dict[tuple[str, str], ClassifiedArticle] = {}
    with open(events_path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = (rec["provider"], str(rec["provider_article_id"]))
            except (ValueError, KeyError, TypeError):
                raise FeatureBuildError(f"event line {n} is invalid") from None
            if key in classified:
                raise FeatureBuildError(f"duplicate event record {key[0]}:{key[1]}")
            if key not in by_key:
                raise FeatureBuildError(f"event record {key[0]}:{key[1]} is not a development article")
            try:
                classified[key] = _record_to_classified(rec, by_key, provenance)
            except HistoricalEventError as e:
                raise FeatureBuildError(f"event record {key[0]}:{key[1]} rejected: {e}") from None
    if set(classified) != set(by_key):
        raise FeatureBuildError(f"event article set differs from the development set "
                                f"({len(set(by_key) - set(classified))} missing)")
    stored = sorted(classified.values(), key=lambda c: c.key)
    fresh = {c.key: c.prediction for c in classify_scored_articles(scored, KeywordEventClassifier())}
    if any(fresh[c.key] != c.prediction for c in stored):
        raise FeatureBuildError("stored event labels are not reproduced by rules_v1")
    return stored


def apply_rtx_mapping(scored: list[ScoredArticle], classified: list[ClassifiedArticle]):
    """Frozen UTX->RTX attribution, in memory; sentiment and event inputs get identical symbol sets."""
    counts = Counter()
    mapped_articles = {}
    for s in scored:
        a = s.article
        mapped = map_symbols_for_rtx(a.symbols, a.created_at)
        before = a.created_at < RTX_UTX_BOUNDARY_UTC
        counts["utx_mapped_to_rtx"] += int("UTX" in a.symbols and before)
        counts["utx_off_period_dropped"] += int("UTX" in a.symbols and not before)
        counts["rtx_off_period_dropped"] += int("RTX" in a.symbols and before)
        counts["rtn_tag_never_substituted"] += int("RTN" in a.symbols)
        mapped_articles[a.key] = a if mapped == a.symbols else dataclasses.replace(a, symbols=mapped)
    scored_m = [ScoredArticle(mapped_articles[s.article.key], s.sentiment) for s in scored]
    by_key = {s.article.key: s for s in scored_m}
    classified_m = [ClassifiedArticle(by_key[c.key], c.prediction) for c in classified]
    return scored_m, classified_m, dict(sorted(counts.items()))


# ============================================================
# Features
# ============================================================


def news_features(scored: list[ScoredArticle], classified: list[ClassifiedArticle], symbol: str,
                  dates: list[date], calendar: TradingCalendar) -> pd.DataFrame:
    """One row per development session for one stock: sentiment__*, event__* (numeric), sw_*."""
    s_rel = [s for s in scored if symbol in s.article.symbols]
    c_rel = [c for c in classified if symbol in c.article.symbols]
    sent = generate_sentiment_features(s_rel, dates, calendar, [symbol])
    events = generate_event_features(c_rel, dates, calendar, [symbol])
    window = generate_sentiment_window_features(s_rel, dates, calendar, symbol,
                                                coverage_start=SW_COVERAGE_START, coverage_end=SW_COVERAGE_END)
    ts = [row_prediction_timestamp(d, calendar) for d in dates]
    if [r.trading_date for r in sent] != dates or [r["trading_date"] for r in events] != dates or \
            list(window["trading_date"]) != dates:
        raise FeatureBuildError(f"{symbol}: news families are not aligned to the development sessions")
    if [r.prediction_timestamp for r in sent] != ts or [r["prediction_timestamp"] for r in events] != ts or \
            list(window["prediction_timestamp"]) != ts:
        raise FeatureBuildError(f"{symbol}: a news family uses a different prediction timestamp")
    if not window["covered"].all():
        raise FeatureBuildError(f"{symbol}: {int((~window['covered']).sum())} session(s) lack a covered 20-session window")
    out = pd.DataFrame({"trading_date": dates})
    for c in SENTIMENT_FEATURE_COLUMNS:
        out["sentiment__" + c] = np.array([float(getattr(r, c)) for r in sent], dtype="float64")
    for c in EVENT_NUMERIC:
        out["event__" + c] = np.array([float(r[c]) for r in events], dtype="float64")
    for c in SW_COLUMNS:
        out[c] = window[c].to_numpy(dtype="float64")
    return out


def build_all_news(scored, classified, universe, dates, calendars) -> dict[str, pd.DataFrame]:
    return {sym: news_features(scored, classified, sym, dates[sym], calendars[sym]) for sym in universe}


def assemble(price_frames: dict[str, pd.DataFrame], news: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frames = {}
    for sym, price in price_frames.items():
        n = news[sym]
        if list(price["Date"].dt.date) != list(n["trading_date"]):
            raise FeatureBuildError(f"{sym}: news rows do not match the 11A price rows")
        frame = price[["Date", "Close", SPY_CLOSE, *PRICE_COLUMNS]].reset_index(drop=True).copy()
        for c in NEWS_COLUMNS:
            frame[c] = n[c].to_numpy(dtype="float64")
        frames[sym] = frame[list(FRAME_COLUMNS)]
    return frames


# ============================================================
# Checks (all run before anything is written)
# ============================================================


def check_rows_and_values(frames: dict[str, pd.DataFrame], spec: BuildSpec) -> int:
    dates = {tuple(f["Date"]) for f in frames.values()}
    if len(dates) != 1:
        raise FeatureBuildError("stocks do not share the same development sessions")
    bad_rows = [s for s, f in frames.items() if len(f) != spec.rows_per_stock]
    if bad_rows:
        raise FeatureBuildError(f"row count differs from {spec.rows_per_stock} for {bad_rows}")
    feature_cols = sorted({c for cols in FEATURE_SETS_11B.values() for c in cols})
    nans = sum(int((~np.isfinite(f[feature_cols].to_numpy(dtype="float64"))).sum()) for f in frames.values())
    if nans:
        raise FeatureBuildError(f"{nans} non-finite value(s) in development features; refusing to continue")
    return nans


def check_f_duplicate(frames: dict[str, pd.DataFrame]) -> None:
    for sym, f in frames.items():
        mismatch = int((f["sw_count_1"].to_numpy() != f["event__article_count"].to_numpy()).sum())
        if mismatch:
            raise FeatureBuildError(f"F duplicate check failed for {sym}: sw_count_1 != event__article_count "
                                    f"on {mismatch} row(s)")


def check_prefix_invariance(scored, classified, frames, dates, calendars, spec: BuildSpec) -> list[str]:
    """Rebuild with only articles available by ts(cut); every row <= cut must be unchanged."""
    common = dates[spec.universe[0]]
    cuts = sorted({common[min(len(common) - 1, int(len(common) * q))] for q in spec.prefix_cut_fractions})
    for cut in cuts:
        cut_ts = row_prediction_timestamp(cut, calendars[spec.universe[0]])
        s_cut = [s for s in scored if s.article.information_available_at <= cut_ts]
        keys = {s.article.key for s in s_cut}
        c_cut = [c for c in classified if c.key in keys]
        for sym in spec.universe:
            part = news_features(s_cut, c_cut, sym, dates[sym], calendars[sym])
            rows = part["trading_date"] <= cut
            full = frames[sym].loc[rows.to_numpy(), list(NEWS_COLUMNS)].to_numpy()
            if not np.array_equal(part.loc[rows, list(NEWS_COLUMNS)].to_numpy(), full):
                raise FeatureBuildError(f"prefix invariance failed for {sym} at cut {cut.isoformat()}")
    return [c.isoformat() for c in cuts]


def check_horizon_independence(frames: dict[str, pd.DataFrame]) -> None:
    cols = list(NEWS_COLUMNS)
    base = a11.panel_rows(frames, cols, 1).set_index(["symbol", "Date"])[cols]
    for h in (3, 5):
        other = a11.panel_rows(frames, cols, h).set_index(["symbol", "Date"])[cols]
        if not other.equals(base.loc[other.index]):
            raise FeatureBuildError(f"news features differ between horizon 1 and horizon {h}")


def panel_oos_rows_h1(frames: dict[str, pd.DataFrame], n_splits: int) -> int:
    data = a11.panel_rows(frames, list(SENTIMENT_COLUMNS), 1)
    test = np.concatenate([te for _, te in a11.date_splits(data["Date"], 1, n_splits)])
    return int(data["Date"].isin(test).sum())


def verify_phase11a_p_values(design_builds_on_matrix: str) -> dict:
    """11A p-values are only ever loaded by hash: every 11A result artefact must match its recorded sha256."""
    reg = a11.load_registry()
    if reg["matrix_sha256"] != design_builds_on_matrix:
        raise FeatureBuildError("11A registry matrix hash differs from the design freeze")
    dev = reg.get("development") or {}
    if dev.get("status") != "COMPLETE":
        raise FeatureBuildError("11A development results are not complete")
    for name, art in dev["artifacts"].items():
        if _sha256_file(PROJECT_ROOT / art["path"]) != art["sha256"]:
            raise FeatureBuildError(f"11A result artefact {name} does not match its recorded hash")
    return {"registry_sha256": _sha256_file(a11.REGISTRY_PATH), "n_result_artifacts": len(dev["artifacts"]),
            "families": dev["families"]}


# ============================================================
# Real price inputs
# ============================================================


def real_price_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, TradingCalendar]]:
    """Phase 11A development price frames and each stock's development calendar (prices only)."""
    inputs = a11.verified_inputs()
    frames = a11.build_frames(inputs)
    calendars = {s: TradingCalendar.from_bars(a11.dev_cut(a11.load_bars(inputs["artifacts"][s]["path"])))
                 for s in a11.UNIVERSE}
    return frames, calendars


# ============================================================
# End to end
# ============================================================


def _frame_csv(frame: pd.DataFrame) -> bytes:
    out = frame.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    buf = io.StringIO()
    out.to_csv(buf, index=False, lineterminator="\n")         # shortest round-trip repr; counts stay floats ("1.0")
    return buf.getvalue().encode("utf-8")


def _rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)


def run_feature_build(*, report_path: Path = REPORT_PATH, sentiment_dir: Path = SENTIMENT_DIR_11B,
                      events_dir: Path = EVENTS_DIR_11B, record_dir: Path = PHASE_11B_DIR,
                      features_dir: Path = FEATURES_DIR_11B, verification_path: Path = VERIFICATION_PATH,
                      price_inputs: Callable = real_price_inputs, spec: BuildSpec = FROZEN_SPEC,
                      clock: Callable[[], datetime] | None = None) -> dict:
    clock = clock or (lambda: datetime.now(timezone.utc))
    report_path, sentiment_dir, events_dir = Path(report_path), Path(sentiment_dir), Path(events_dir)
    record_dir, features_dir, verification_path = Path(record_dir), Path(features_dir), Path(verification_path)
    for given, phase8, what in ((sentiment_dir, SENTIMENT_DIR, "sentiment"), (events_dir, EVENTS_DIR_PHASE8, "events"),
                                (features_dir, FEATURES_DIR_PHASE8, "features")):
        if given.resolve() == Path(phase8).resolve():
            raise FeatureBuildError(f"{what} directory is the Phase 8 directory; only Phase 11B paths are accepted")
    if verification_path.exists():
        raise FeatureBuildError(f"{verification_path.name} already exists; the 11B.3 build is complete and write-once")

    families = verify_frozen_spec(spec)
    frozen = verify_frozen_artifacts()
    rules_sha = verify_rules_v1()
    report = load_integrity_report(report_path)
    canonical_sha = report["canonical_input"]["sha256"]
    run_id = run_id_for(canonical_sha)
    scoring_record_path = record_dir / f"finbert_development_scoring_{run_id}.json"
    scoring_record = load_scoring_record(scoring_record_path, report=report, report_path=report_path,
                                         frozen=frozen, run_id=run_id)
    score_path, _ = verify_score_file(sentiment_dir, scoring_record)
    cls_record = load_classification_record(record_dir, run_id=run_id, frozen=frozen, rules_sha=rules_sha,
                                            report_path=report_path, scoring_record_path=scoring_record_path,
                                            scoring_record=scoring_record)
    events_path = verify_events_file(events_dir, cls_record, scoring_record)
    design = json.loads(DESIGN_FREEZE_PATH.read_text(encoding="utf-8"))
    p11a = verify_phase11a_p_values(design["builds_on"]["phase_11a_matrix_sha256"])
    final = {"panel": p11a["families"]["panel"] + design["experiment_matrix"]["comparison_counts"]["panel_primary"],
             "per_stock": p11a["families"]["per_stock"] + design["experiment_matrix"]["comparison_counts"]["per_stock"]}
    if final != {"panel": families["panel"]["total"], "per_stock": families["per_stock"]["total"]}:
        raise FeatureBuildError(f"final Holm family sizes {final} differ from the design freeze")

    canonical_path = Path(report["canonical_input"]["path"])
    if not canonical_path.is_absolute():
        canonical_path = PROJECT_ROOT / canonical_path
    if not canonical_path.is_file():
        raise FeatureBuildError(f"canonical dataset named by the integrity report not found: {canonical_path.name}")

    started = clock()
    partition = partition_canonical(canonical_path, expected_sha256=canonical_sha)
    verify_partition_against_report(partition, report)
    development, sealed_count = partition.development, len(partition.sealed_keys)
    del partition
    provenance = {k: scoring_record["model"][k] for k in PROVENANCE_FIELDS}
    scored = join_development_scores(score_path, development, provenance)
    del development
    classified = join_development_events(events_path, scored)
    for s in scored:
        if s.article.information_available_at > DEVELOPMENT_CUTOFF_UTC:
            raise FeatureBuildError(f"{s.article.article_key} is not a development article")
    scored_m, classified_m, mapping_counts = apply_rtx_mapping(scored, classified)

    price_frames, calendars = price_inputs()
    if tuple(sorted(price_frames)) != tuple(sorted(spec.universe)):
        raise FeatureBuildError("price frames do not cover exactly the universe")
    dates = {s: list(f["Date"].dt.date) for s, f in price_frames.items()}
    frames = assemble(price_frames, build_all_news(scored_m, classified_m, spec.universe, dates, calendars))
    nans = check_rows_and_values(frames, spec)
    check_f_duplicate(frames)
    cuts = check_prefix_invariance(scored_m, classified_m, frames, dates, calendars, spec)
    check_horizon_independence(frames)
    oos = panel_oos_rows_h1(frames, spec.n_splits)
    if oos != spec.panel_oos_rows_h1:
        raise FeatureBuildError(f"panel out-of-sample rows (h=1) {oos} differ from the frozen {spec.panel_oos_rows_h1}")

    out_dir = features_dir / run_id
    payloads = {sym: _frame_csv(frames[sym]) for sym in sorted(frames)}
    for sym, payload in payloads.items():
        path = out_dir / f"{sym}.csv"
        if path.exists() and path.read_bytes() != payload:
            raise FeatureBuildError(f"{_rel(path)} exists with different content; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym, payload in payloads.items():
        _atomic_write(out_dir / f"{sym}.csv", payload)

    fresh_prices, _ = price_inputs()                                   # independent rebuild of the 11A frames
    shared = ["Close", SPY_CLOSE, *PRICE_COLUMNS]
    for sym in sorted(frames):
        back = pd.read_csv(out_dir / f"{sym}.csv", float_precision="round_trip")
        fresh = fresh_prices[sym].reset_index(drop=True)
        if list(back.columns) != list(FRAME_COLUMNS) or \
                list(back["Date"]) != list(fresh["Date"].dt.strftime("%Y-%m-%d")) or \
                not np.array_equal(back[shared].to_numpy(dtype="float64"), fresh[shared].to_numpy(dtype="float64")) or \
                not np.array_equal(back[list(NEWS_COLUMNS)].to_numpy(dtype="float64"),
                                   frames[sym][list(NEWS_COLUMNS)].to_numpy(dtype="float64")):
            raise FeatureBuildError(f"{sym}: written frame does not reproduce the 11A shared columns or the news features")

    coverage = defaultdict(Counter)
    for s in scored_m:
        for sym in set(s.article.symbols) & set(spec.universe):
            coverage[sym][s.article.information_available_at.strftime("%Y-%m")] += 1
    frame_files = {sym: {"path": _rel(out_dir / f"{sym}.csv"), "sha256": hashlib.sha256(p).hexdigest()}
                   for sym, p in payloads.items()}
    finished = clock()
    verification = {
        "document_type": "phase_11b_feature_verification", "phase": "11B", "stage": STAGE,
        "build_version": BUILD_VERSION, "run_id": run_id,
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "frozen_artifact_sha256": frozen, "rules_v1_spec_sha256": rules_sha,
        "inputs": {
            "integrity_report": {"file": report_path.name, "sha256": _sha256_file(report_path)},
            "canonical": {"file": canonical_path.name, "sha256": canonical_sha},
            "scoring_run_record": {"file": scoring_record_path.name, "sha256": _sha256_file(scoring_record_path)},
            "scores": {"file": score_path.name, "sha256": scoring_record["outputs"]["scores"]["sha256"]},
            "classification_run_record": {"file": f"rules_v1_development_classification_{run_id}.json",
                                          "sha256": _sha256_file(record_dir / f"rules_v1_development_classification_{run_id}.json")},
            "events": {"file": events_path.name, "sha256": cls_record["outputs"]["events"]["sha256"]},
        },
        "development_cutoff_utc": CUTOFF_TEXT, "boundary_rule": BOUNDARY_RULE,
        "partition_reproduces_integrity_report": True,
        "development_article_count": len(scored), "sealed_article_count": sealed_count,
        "sentiment_window_coverage": {"start_utc": "2017-01-01T00:00:00Z", "end_utc": CUTOFF_TEXT,
                                      "all_rows_covered": True},
        "rtx_utx_mapping": {"boundary_utc": "2020-04-03T04:00:00Z", "applied_in_memory": True,
                            "canonical_file_edited": False, "development_counts": mapping_counts},
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS_11B.items()},
        "frame_columns": list(FRAME_COLUMNS),
        "frame_files": frame_files,
        "frame_paths": {fs: {sym: frame_files[sym]["path"] for sym in sorted(frames)} for fs in FEATURE_SETS_11B},
        "row_count_per_stock": spec.rows_per_stock,
        "first_session": dates[spec.universe[0]][0].isoformat(), "last_session": dates[spec.universe[0]][-1].isoformat(),
        "shared_columns_match_11a": True,
        "nans_in_development_features": nans,
        "f_duplicate_check_passed": True,
        "prefix_invariance_check_passed": True, "prefix_invariance_cut_dates": cuts,
        "horizon_independence_check_passed": True,
        "panel_oos_rows_all29_h1": oos,
        "final_panel_family_size": final["panel"], "final_per_stock_family_size": final["per_stock"],
        "phase_11a_p_values_hash_verified": True, "phase_11a_registry": p11a,
        "news_file_hashes_match_records": True, "sentiment_file_hashes_match_records": True,
        "event_file_hashes_match_records": True,
        "development_article_months_by_stock": {s: dict(sorted(coverage[s].items())) for s in sorted(coverage)},
        "phase_8_score_reuse": False, "sealed_content_used": False,
        "models_trained": False,
    }
    if verification_path.exists():
        raise FeatureBuildError(f"{verification_path.name} already exists; refusing to overwrite")
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(verification_path, _json_bytes(verification))
    return verification


# ============================================================
# CLI (operator-run only)
# ============================================================


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Phase 11B Stage 11B.3: development-only feature build and Gate C evidence. All inputs come "
                    "from the Gate B report and the 11B.2 run records; no paths, dates or selections can be passed."
    ).parse_args(argv)
    v = run_feature_build()
    print(f"Run id       : {v['run_id']}")
    print(f"Stocks       : {len(v['frame_files'])} x {v['row_count_per_stock']} rows "
          f"({v['first_session']} .. {v['last_session']})")
    print(f"Development  : {v['development_article_count']} articles; sealed {v['sealed_article_count']} (unused)")
    print(f"Checks       : NaN {v['nans_in_development_features']}, F duplicate passed, prefix invariance passed, "
          f"panel OOS rows {v['panel_oos_rows_all29_h1']}")
    print(f"Verification : {VERIFICATION_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
