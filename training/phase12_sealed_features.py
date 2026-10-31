"""
Phase 12 Stage 3B: the sealed feature builder for the Amendment 07 confirmation.

The 8 finalist features come from the pinned generators only, through one assembly path (`build_frames`) used
for both periods:

    Volatility, Volume_Change_1, Volume_Change_5   features.feature_engineering.engineer_features (stock bars)
    spy_volatility_20                              services.market_context.compute_market_context (SPY bars)
    event__earnings_event_count, event__event_count, event__negative_event_count
                                                   services.event_features.generate_event_features: rules_v1
                                                   labels; "negative" = the article's FinBERT label
    sw_count_surprise_1_20                         services.sentiment_window.generate_sentiment_window_features

Timestamps: row D is predicted at ts(D) = D 16:30 America/New_York. Price features use bars dated <= D
(bars are cut at the period end before anything is computed). An article counts for row D only if
ts(previous session) < information_available_at <= ts(D), where information_available_at =
max(created_at, updated_at) of the stored (latest) revision.

`reproduce` (development only) rebuilds 2017-02-01 .. 2024-09-30 from the frozen 11B inputs and must equal the
Stage 2 input frames exactly. It never parses a sealed record's content. Price snapshots are parsed only up to
the development end; their full bytes are hashed by the 11A verifier. The canonical news file is sorted by
created_at, so parsing stops at the first article created after the cutoff, and the rest is hashed only.
Revised articles created before the cutoff are routed by their timestamps alone.

Erratum 01 (phase12/stage3/erratum_01.json, pinned below) corrects Amendment 07: event__negative_event_count
needs each article's FinBERT label, so sealed articles are scored with the frozen 11B.2 FinBERT, offline. Those
labels are used for that feature only. `build_sealed` needs Amendment 07, the owner's authorization record
(naming this file's hash), Erratum 01, a passing reproduction, the frozen FinBERT configuration and 500/500
development labels reproduced, all in the same process, before any sealed record is read.

    python -m training.phase12_sealed_features reproduce      # development only -> phase12/stage3b/ (write-once)
    python -m training.phase12_sealed_features erratum        # phase12/stage3/erratum_01.json (write-once)
    python -m training.phase12_sealed_features finbert-check  # 500 development labels (HF_HUB_OFFLINE=1; write-once)
    python -m training.phase12_sealed_features status
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11a_research as a11  # noqa: E402
import training.phase11b_feature_build as fb  # noqa: E402
import training.phase11b_research as r  # noqa: E402
import training.phase12_confirmation as c  # noqa: E402
import training.phase12_magnitude as m2  # noqa: E402
from features.feature_engineering import engineer_features  # noqa: E402
from services import market_context as mc  # noqa: E402
from services.event_classifier import KeywordEventClassifier  # noqa: E402
from services.event_features import generate_event_features  # noqa: E402
from services.historical_events import classify_scored_articles  # noqa: E402
from services.historical_sentiment import score_canonical_articles  # noqa: E402
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_alignment import row_prediction_timestamp  # noqa: E402
from services.news_schema import NewsArticle, NewsValidationError, parse_timestamp  # noqa: E402
from services.sentiment_window import generate_sentiment_window_features  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.excess_targets import SPY_CLOSE  # noqa: E402
from training.phase11b_event_classification import (  # noqa: E402
    join_development_scores,
    load_scoring_record,
    verify_rules_v1,
    verify_score_file,
)
from training.phase11b_sentiment_scoring import (  # noqa: E402
    DEVELOPMENT_CUTOFF_UTC,
    PARTITION_FIELDS,
    _development_article_sha256,
    historical_information_available_at,
    load_integrity_report,
    run_id_for,
    verify_frozen_artifacts,
)
from training.phase9_research import ResearchError, _write_new_json, environment  # noqa: E402

STAGE = "12.S3B"
FEATURES = m2.FEATURES
PRICE_FEATURES = ("Volatility", "spy_volatility_20", "Volume_Change_1", "Volume_Change_5")
NEWS_FEATURES = ("event__earnings_event_count", "event__event_count", "event__negative_event_count",
                 "sw_count_surprise_1_20")
OUTPUT_COLUMNS = ("Date", "Close", SPY_CLOSE, *FEATURES)
EVENT_SOURCE = {"event__earnings_event_count": "earnings_event_count", "event__event_count": "event_count",
                "event__negative_event_count": "negative_event_count"}
DEV_START, DEV_END = a11.DEV_START, a11.DEV_END
SW_COVERAGE_START, DEV_COVERAGE_END = fb.SW_COVERAGE_START, fb.SW_COVERAGE_END
SEALED_COVERAGE_END = datetime(2026, 9, 29, tzinfo=timezone.utc)   # canonical news interval end (created_at <)
OUT_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage3b"
REPRODUCTION_FILE = "development_reproduction.json"
SEALED_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features" / "phase12_sealed"
FROZEN_DIRS = (PROJECT_ROOT / "data" / "results" / "research" / "phase11",
               PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage1",
               m2.OUT_DIR, c.OUT_DIR, fb.FEATURES_DIR_11B, PROJECT_ROOT / "data" / "processed" / "phase11")
ERRATUM_ID = "amendment_07_erratum_01_finbert_labels"
ERRATUM_FILE = c.OUT_DIR / "erratum_01.json"
ERRATUM_SHA256 = "0dd61b0b12bca9951387ea8568ac536b6afc13e6a08e322a06d137e8cd071713"   # written by `erratum`
EQUIVALENCE_ARTICLES = 500                 # development articles re-scored; every label must match (500/500)
SELECTION_SALT = "phase12|erratum01|"
EQUIVALENCE_FILE = "finbert_label_equivalence.json"
FINBERT_CONFIG = {                         # the frozen 11B.2 pipeline (scoring run record + services.finbert_sentiment)
    "model_name": "ProsusAI/finbert", "model_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
    "inference_version": "finbert_sentiment_v1", "labels": ["positive", "negative", "neutral"],
    "label_rule": "argmax of softmax(logits); ties broken in the label order above",
    "text_policy": ("text_v1: headline + ' ' + summary (whitespace-collapsed; summary dropped if it repeats the "
                    "headline); content (HTML stripped) only if both are empty; never URLs, ids, symbols, timestamps "
                    "or source"),
    "tokenizer": ("AutoTokenizer of the pinned revision; truncation by tokens at max_length; padding to the longest "
                  "text of a batch (none for a single text); attention mask"),
    "max_length": 512, "batch_size": 16, "device": "cpu", "hf_hub_offline": True,
    "libraries": {"torch": "2.9.0+cpu", "transformers": "4.57.1"},
}


class SealedFeatureError(ResearchError):
    """Phase 12 Stage 3B violation; the build stops and writes nothing."""


def _check_output_dir(path: Path) -> Path:
    path = Path(path).resolve()
    for frozen in FROZEN_DIRS:
        if path == frozen.resolve() or path.is_relative_to(frozen.resolve()):
            raise SealedFeatureError(f"{path} is inside frozen evidence ({frozen.name}); refusing to write there")
    return path


# ============================================================
# Prices
# ============================================================


def read_bars_until(path: Path, end: date) -> pd.DataFrame:
    """a11.load_bars restricted to Date <= end; lines after the first later date are never parsed."""
    keep = []
    with open(path, encoding="utf-8", newline="") as f:
        keep.append(f.readline())
        for line in f:
            if not line.strip():
                continue
            if date.fromisoformat(line.split(",", 1)[0]) > end:
                break
            keep.append(line)
    df = pd.read_csv(io.StringIO("".join(keep)), dtype={"Date": str}, float_precision="round_trip")
    bars = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    bars["Date"] = pd.to_datetime(bars["Date"], format="%Y-%m-%d").astype("datetime64[ns]")
    return bars[a11.OHLCV_COLUMNS]


def price_features(stock: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """a11.stock_frame's composition for [start, end], restricted to the 4 finalist price features."""
    for b in (stock, spy, qqq):
        if b["Date"].dt.date.max() > end:
            raise SealedFeatureError(f"bars must be cut at {end} before features are built")
    tech = engineer_features(stock.copy())
    tech = tech[(tech["Date"].dt.date >= start) & (tech["Date"].dt.date <= end)].reset_index(drop=True)
    dates = list(tech["Date"].dt.date)
    ctx = mc.compute_market_context(stock, spy, qqq, dates)
    spy_close = dict(zip(spy["Date"].dt.date, spy["Close"].astype("float64")))
    out = pd.DataFrame({"Date": tech["Date"], "Close": tech["Close"].astype("float64"),
                        SPY_CLOSE: [spy_close[d] for d in dates]})
    for col in PRICE_FEATURES:
        src = ctx if col == "spy_volatility_20" else tech
        out[col] = src[col].to_numpy(dtype="float64")
    return out


# ============================================================
# News
# ============================================================


def news_features(scored, classified, symbol: str, dates: list[date], calendar: TradingCalendar,
                  coverage_end: datetime) -> pd.DataFrame:
    """fb.news_features' generator calls, restricted to the 4 finalist news features; uncovered rows stay NaN."""
    s_rel = [s for s in scored if symbol in s.article.symbols]
    c_rel = [x for x in classified if symbol in x.article.symbols]
    events = generate_event_features(c_rel, dates, calendar, [symbol])
    window = generate_sentiment_window_features(s_rel, dates, calendar, symbol,
                                                coverage_start=SW_COVERAGE_START, coverage_end=coverage_end)
    ts = [row_prediction_timestamp(d, calendar) for d in dates]
    if [e["trading_date"] for e in events] != dates or list(window["trading_date"]) != dates:
        raise SealedFeatureError(f"{symbol}: news families are not aligned to the sessions")
    if [e["prediction_timestamp"] for e in events] != ts or list(window["prediction_timestamp"]) != ts:
        raise SealedFeatureError(f"{symbol}: a news family uses a different prediction timestamp")
    out = pd.DataFrame({"trading_date": dates})
    for col, src in EVENT_SOURCE.items():
        out[col] = np.array([float(e[src]) for e in events], dtype="float64")
    out["sw_count_surprise_1_20"] = window["sw_count_surprise_1_20"].to_numpy(dtype="float64")
    return out


def build_frames(bars: dict[str, pd.DataFrame], scored, classified, universe, start: date, end: date,
                 coverage_end: datetime) -> dict[str, pd.DataFrame]:
    """Date, Close, SPY_Close and the 8 features per stock for sessions in [start, end]. One path, both periods."""
    spy, qqq = bars["SPY"], bars["QQQ"]
    frames = {}
    for sym in sorted(universe):
        price = price_features(bars[sym], spy, qqq, start, end)
        dates = list(price["Date"].dt.date)
        news = news_features(scored, classified, sym, dates, TradingCalendar.from_bars(bars[sym]), coverage_end)
        frame = price.reset_index(drop=True)
        for col in NEWS_FEATURES:
            frame[col] = news[col].to_numpy(dtype="float64")
        frames[sym] = frame[list(OUTPUT_COLUMNS)]
    if len({tuple(f["Date"]) for f in frames.values()}) != 1:
        raise SealedFeatureError("stocks do not share the same sessions")
    return frames


def development_prefix(canonical_path: Path, expected_sha256: str) -> tuple[list[NewsArticle], dict]:
    """
    Development articles of the canonical file (sorted by created_at). Partition fields are parsed until the first
    article created after the cutoff; every later byte is only hashed. Records created before the cutoff but
    revised after it are routed by their timestamps and dropped (nothing of them is kept).
    """
    h, development, seen = hashlib.sha256(), [], set()
    stats = {"lines_parsed": 0, "revised_after_cutoff_dropped": 0, "stopped_at_line": None}
    last_created = None
    with open(canonical_path, "rb") as f:
        for n, raw in enumerate(f, start=1):
            h.update(raw)
            if stats["stopped_at_line"] is not None or not raw.strip():
                continue
            view = json.loads(raw, object_pairs_hook=lambda pairs: {k: v for k, v in pairs if k in PARTITION_FIELDS})
            try:
                created = parse_timestamp(view.get("created_at"), "created_at")
            except NewsValidationError as e:
                raise SealedFeatureError(f"canonical line {n}: {e}") from None
            if created > DEVELOPMENT_CUTOFF_UTC:
                stats["stopped_at_line"] = n                         # every later record is sealed
                continue
            if last_created is not None and created < last_created:
                raise SealedFeatureError(f"canonical line {n} breaks the created_at order; prefix reading is unsafe")
            last_created = created
            stats["lines_parsed"] += 1
            key = f"{view.get('provider')}:{view.get('provider_article_id')}"
            if key in seen:
                raise SealedFeatureError(f"duplicate article identifier {key}")
            seen.add(key)
            updated = parse_timestamp(view["updated_at"], "updated_at") if view.get("updated_at") else None
            available = historical_information_available_at(created, updated)
            if available != parse_timestamp(view.get("information_available_at"), "information_available_at"):
                raise SealedFeatureError(f"{key}: information_available_at is not max(created_at, updated_at)")
            if available > DEVELOPMENT_CUTOFF_UTC:
                stats["revised_after_cutoff_dropped"] += 1
                continue
            article = NewsArticle.from_dict(json.loads(raw))
            if article.article_key != key or article.information_available_at != available:
                raise SealedFeatureError(f"{key}: development record does not match its partition fields")
            development.append(article)
    if h.hexdigest() != expected_sha256:
        raise SealedFeatureError("canonical dataset SHA256 differs from the integrity report")
    return development, stats


def development_news(report_path: Path = fb.REPORT_PATH, sentiment_dir: Path = fb.SENTIMENT_DIR_11B,
                     events_dir: Path = fb.EVENTS_DIR_11B, record_dir: Path = fb.PHASE_11B_DIR) -> tuple[list, list, dict]:
    """The 11B.3 news inputs, verified by the 11B.3 functions; development articles from `development_prefix`."""
    frozen = verify_frozen_artifacts()
    rules_sha = verify_rules_v1()
    report = load_integrity_report(Path(report_path))
    canonical_sha = report["canonical_input"]["sha256"]
    run_id = run_id_for(canonical_sha)
    scoring_record_path = Path(record_dir) / f"finbert_development_scoring_{run_id}.json"
    scoring_record = load_scoring_record(scoring_record_path, report=report, report_path=Path(report_path),
                                         frozen=frozen, run_id=run_id)
    score_path, _ = verify_score_file(Path(sentiment_dir), scoring_record)
    cls_record = fb.load_classification_record(Path(record_dir), run_id=run_id, frozen=frozen, rules_sha=rules_sha,
                                               report_path=Path(report_path), scoring_record_path=scoring_record_path,
                                               scoring_record=scoring_record)
    events_path = fb.verify_events_file(Path(events_dir), cls_record, scoring_record)
    canonical = Path(report["canonical_input"]["path"])
    canonical = canonical if canonical.is_absolute() else PROJECT_ROOT / canonical
    development, stats = development_prefix(canonical, canonical_sha)
    if len(development) != report["development_article_count"] or \
            _development_article_sha256(development) != report.get("development_article_sha256"):
        raise SealedFeatureError("development articles do not reproduce the Gate B count and hash")
    provenance = {k: scoring_record["model"][k] for k in fb.PROVENANCE_FIELDS}
    scored = join_development_scores(score_path, development, provenance)
    classified = fb.join_development_events(events_path, scored)            # also re-derives every rules_v1 label
    if any(s.article.information_available_at > DEVELOPMENT_CUTOFF_UTC for s in scored):
        raise SealedFeatureError("a scored article is not a development article")
    scored_m, classified_m, mapping = fb.apply_rtx_mapping(scored, classified)
    inputs = {"integrity_report_sha256": file_sha256(Path(report_path)), "canonical_sha256": canonical_sha,
              "scores_sha256": scoring_record["outputs"]["scores"]["sha256"],
              "events_sha256": cls_record["outputs"]["events"]["sha256"], "rules_v1_spec_sha256": rules_sha,
              "finbert": provenance, "development_articles": len(development), "canonical_read": stats,
              "rtx_mapping": mapping}
    return scored_m, classified_m, inputs


def development_bars() -> tuple[dict[str, pd.DataFrame], dict]:
    """11A-verified price snapshots (full-file hashes), parsed only up to the development end."""
    inputs = a11.verified_inputs()
    bars = {s: read_bars_until(PROJECT_ROOT / a["path"], DEV_END) for s, a in inputs["artifacts"].items()}
    return bars, {s: a["sha256"] for s, a in sorted(inputs["artifacts"].items())}


# ============================================================
# Development reproduction gate
# ============================================================


def frame_sha256(frame: pd.DataFrame) -> str:
    out = frame.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    buf = io.StringIO()
    out.to_csv(buf, index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def compare_frames(rebuilt: dict[str, pd.DataFrame], frozen: dict[str, pd.DataFrame]) -> list[str]:
    """Exact agreement: symbols, row counts, dates, column order, float64 dtype, NaN locations and values."""
    if sorted(rebuilt) != sorted(frozen):
        return [f"symbols differ: {sorted(set(rebuilt) ^ set(frozen))}"]
    problems = []
    for sym in sorted(frozen):
        a, b = rebuilt[sym].reset_index(drop=True), frozen[sym][list(OUTPUT_COLUMNS)].reset_index(drop=True)
        if list(a.columns) != list(OUTPUT_COLUMNS):
            problems.append(f"{sym}: column order {list(a.columns)}")
            continue
        if len(a) != len(b):
            problems.append(f"{sym}: {len(a)} rows, frozen {len(b)}")
            continue
        if not a["Date"].equals(b["Date"]):
            problems.append(f"{sym}: trading dates differ")
            continue
        for col in OUTPUT_COLUMNS[1:]:
            if a[col].dtype != np.float64 or b[col].dtype != np.float64:
                problems.append(f"{sym}.{col}: dtype {a[col].dtype} vs {b[col].dtype}")
                continue
            x, y = a[col].to_numpy(), b[col].to_numpy()
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                problems.append(f"{sym}.{col}: missing-value locations differ")
            elif not np.array_equal(x, y, equal_nan=True):
                i = int(np.flatnonzero(~((x == y) | (np.isnan(x) & np.isnan(y))))[0])
                problems.append(f"{sym}.{col}: {int((x != y).sum())} value(s) differ, first {a['Date'][i].date()} "
                                f"rebuilt {x[i]!r} frozen {y[i]!r}")
    return problems


def generator_pins() -> dict:
    """The generator files and rules_v1 must equal the Amendment 07 pins."""
    pins = c.load_amendment()["pins"]
    live = {f: file_sha256(PROJECT_ROOT / f) for f in c.GENERATOR_FILES}
    bad = [f for f, h in live.items() if pins["generator_code_sha256"].get(f) != h]
    if bad or pins["rules_v1_spec_sha256"] != verify_rules_v1():
        raise SealedFeatureError(f"generator code differs from the Amendment 07 pins: {bad or ['rules_v1']}")
    return live


def reproduce(out_dir: Path | None = OUT_DIR, *, bars_loader: Callable = development_bars,
              news_loader: Callable = development_news, frozen_loader: Callable | None = None,
              universe=None, pins: Callable = generator_pins) -> dict:
    """Rebuild the development period; any difference stops everything. Writes one record (write-once)."""
    if out_dir is not None:
        out_dir = _check_output_dir(out_dir)
        if (out_dir / REPRODUCTION_FILE).exists():
            raise SealedFeatureError(f"{REPRODUCTION_FILE} already exists; the record is written once")
    generators = pins()
    if frozen_loader is None:
        inputs = r.verified_inputs(r.FROZEN_SPEC)
        if r.inputs_sha256(inputs) != c.INPUTS_SHA256:
            raise SealedFeatureError("frozen development frames differ from the Stage 2 inputs")
        frozen = r.load_frames(inputs, r.FROZEN_SPEC)
    else:
        frozen = frozen_loader()
    universe = sorted(frozen) if universe is None else sorted(universe)
    bars, bar_sha = bars_loader()
    scored, classified, news_inputs = news_loader()
    first = min(pd.Timestamp(f["Date"].min()).date() for f in frozen.values())
    last = max(pd.Timestamp(f["Date"].max()).date() for f in frozen.values())
    rebuilt = build_frames(bars, scored, classified, universe, first, last, DEV_COVERAGE_END)
    problems = compare_frames(rebuilt, frozen)
    if problems:
        raise SealedFeatureError(f"development reproduction FAILED ({len(problems)}): {problems[:10]}")
    rec = {"document_type": "phase12_development_reproduction", "stage": STAGE, "result": "EXACT_MATCH",
           "period": [first.isoformat(), last.isoformat()], "symbols": universe,
           "rows_per_stock": {s: int(len(f)) for s, f in sorted(rebuilt.items())},
           "columns": list(OUTPUT_COLUMNS), "comparison": "exact (np.array_equal, identical NaN locations, float64)",
           "frame_sha256": {s: frame_sha256(f) for s, f in sorted(rebuilt.items())},
           "inputs": {"stage2_inputs_sha256": c.INPUTS_SHA256, "price_snapshots_sha256": bar_sha, "news": news_inputs},
           "generator_code_sha256": generators, "builder_code_sha256": file_sha256(Path(__file__)),
           "sealed_rows_generated": 0, "sealed_record_content_parsed": False,
           "environment": environment(), "git": git_info(),
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_new_json(out_dir / REPRODUCTION_FILE, rec)
    return rec


# ============================================================
# Erratum 01: FinBERT labels for event__negative_event_count
# ============================================================


def erratum_rules() -> dict:
    """Every fixed statement of Erratum 01 (the model-file fingerprint is added when the file is written)."""
    return {
        "erratum_id": ERRATUM_ID, "phase": "12", "stage": "12.S3C",
        "amends": {"amendment_id": c.AMENDMENT_ID, "file": f"phase12/stage3/{c.AMENDMENT_FILE}",
                   "amendment_sha256": c.AMENDMENT_SHA256,
                   "relation": "supplements Amendment 07; Amendment 07 itself is not modified"},
        "status": "APPROVED by the project owner (Stage 3C); written before any sealed record is read",
        "reason": ("Amendment 07 sealed_feature_build says no FinBERT run is needed and sentiment is not an input of "
                   "the 8 features. That is wrong for event__negative_event_count: event_features_v1 counts events "
                   "whose article FinBERT label is negative (services.event_taxonomy.impact_from_sentiment_label). "
                   "A placeholder label would make the feature constant on sealed data."),
        "correction": {
            "replaces": ["no FinBERT or any other model is run",
                         ("sentiment scores are not an input of any of the 8 features; where a generator needs one "
                          "a constant placeholder is passed")],
            "with": ("sealed articles are scored with the frozen 11B.2 FinBERT pipeline below, offline, and their "
                     "labels are used only for event__negative_event_count")},
        "finbert": dict(FINBERT_CONFIG),
        "permitted_use": "the label of each sealed article, as an input of event__negative_event_count only",
        "restriction": ("no FinBERT output feeds any other feature: the other 7 features and their generation rules "
                        "are unchanged and must be identical whatever the sentiment values (tested); sentiment "
                        "scores, ratios and means are never used"),
        "no_substitution": ("no other model, revision, tokenizer, text policy, truncation, padding, batch size, device "
                            "or label mapping; any difference stops the build"),
        "equivalence_gate": {
            "articles": EQUIVALENCE_ARTICLES, "population": "11B.2 development articles only",
            "selection": f"sort by sha256('{SELECTION_SALT}' + article_key), take the first {EQUIVALENCE_ARTICLES}",
            "requirement": f"{EQUIVALENCE_ARTICLES}/{EQUIVALENCE_ARTICLES} fresh labels equal the stored 11B.2 labels",
            "when": "in the same process, after the development feature reproduction and before any sealed inference",
            "on_failure": "stop; no sealed record is read; the requirement is never weakened"},
        "unchanged": ["the other 7 features and every generation rule", "finalist artifact", "baseline artifact",
                      "preprocessing", "target", "horizon", "sealed window", "eligibility and missing-data rules",
                      "metrics", "statistics", "outcomes", "prohibitions"],
        "qualification_unchanged": c.rules()["qualification"],
        "qualification_statement": "Q1-Q6 and the CONFIRMED rule are exactly those of Amendment 07",
        "sealed_access": ("this erratum authorizes no sealed access, sealed inference, sealed feature generation or "
                          "confirmation; those need the owner's separate authorization record"),
    }


def model_files_sha256(snapshot: Path | None = None) -> dict:
    """SHA-256 of every file in the cached snapshot of the pinned revision (no model is loaded)."""
    if snapshot is None:                    # the hub cache location rule (HF_HUB_CACHE, else HF_HOME/hub, else default)
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cache = Path(os.environ.get("HF_HUB_CACHE") or Path(hf_home) / "hub")
        snapshot = cache / "models--ProsusAI--finbert" / "snapshots" / FINBERT_CONFIG["model_revision"]
    snapshot = Path(snapshot)
    files = {p.name: file_sha256(p) for p in sorted(snapshot.iterdir()) if p.is_file()} if snapshot.is_dir() else {}
    if not files:
        raise SealedFeatureError(f"pinned FinBERT revision is not in the local cache ({snapshot})")
    return files


def write_erratum(path: Path | None = None, snapshot: Path | None = None) -> dict:
    """Write-once; development-side only (hashes the cached model files, loads no model)."""
    path = Path(path or ERRATUM_FILE)
    if path.exists():
        raise SealedFeatureError(f"{path.name} already exists; the erratum is written once")
    c.load_amendment()
    doc = {**erratum_rules(), "model_files_sha256": model_files_sha256(snapshot), "sealed_data_read": False,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_new_json(path, doc)
    return doc


def load_erratum(path: Path | None = None, pinned_sha: str | None = None) -> dict:
    """Erratum 01 equals its pin and the rules in code, and amends the pinned Amendment 07."""
    path, pinned_sha = Path(path or ERRATUM_FILE), pinned_sha or ERRATUM_SHA256
    if pinned_sha is None:
        raise SealedFeatureError("Erratum 01 is not pinned in code; refusing sealed generation")
    if not path.is_file() or file_sha256(path) != pinned_sha:
        raise SealedFeatureError("Erratum 01 is missing or differs from its pinned SHA256")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if {k: doc.get(k) for k in erratum_rules()} != json.loads(json.dumps(erratum_rules())):
        raise SealedFeatureError("Erratum 01 rules differ from the rules in code")
    if file_sha256(c.OUT_DIR / c.AMENDMENT_FILE) != doc["amends"]["amendment_sha256"]:
        raise SealedFeatureError("Amendment 07 differs from the hash Erratum 01 amends")
    return doc


def check_finbert(service, expected_files: dict, snapshot: Path | None = None) -> dict:
    """The loaded service is exactly the frozen pipeline: revision, labels, tokenizer limit, batching, files."""
    from training.phase11b_sentiment_scoring import check_model
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SealedFeatureError("HF_HUB_OFFLINE=1 is required (offline inference from the local cache)")
    check_model(service)
    b = service.backend
    live = {"model_name": b.model_name, "model_revision": b.model_revision, "labels": [str(x) for x in b.labels],
            "max_length": b.max_length, "device": str(b.device), "batch_size": service.batch_size}
    want = {k: FINBERT_CONFIG[k] for k in live}
    if live != want:
        raise SealedFeatureError(f"FinBERT configuration differs from Erratum 01: {live} != {want}")
    libs = {k: _version(k) for k in FINBERT_CONFIG["libraries"]}
    import torch
    libs["torch"] = torch.__version__
    if libs != FINBERT_CONFIG["libraries"]:
        raise SealedFeatureError(f"FinBERT libraries differ from Erratum 01: {libs}")
    if model_files_sha256(snapshot) != expected_files:
        raise SealedFeatureError("cached FinBERT files differ from the Erratum 01 fingerprint")
    return {**live, "libraries": libs}


def select_equivalence_articles(scored_dev: list) -> list:
    key = lambda x: hashlib.sha256((SELECTION_SALT + x.article.article_key).encode()).hexdigest()   # noqa: E731
    return sorted(scored_dev, key=key)[:EQUIVALENCE_ARTICLES]


def label_equivalence(scored_dev: list, service) -> dict:
    """Re-score the selected development articles; PASS only if all 500 labels equal the stored labels."""
    pick = select_equivalence_articles(scored_dev)
    if any(s.article.information_available_at > DEVELOPMENT_CUTOFF_UTC for s in pick):
        raise SealedFeatureError("the equivalence check may use development articles only")
    fresh, _ = score_canonical_articles([s.article for s in pick], service)
    got = {x.article.key: x.sentiment.label for x in fresh}
    rows = [{"article_key": s.article.article_key, "stored": s.sentiment.label, "fresh": got.get(s.article.key)}
            for s in pick]
    bad = [x for x in rows if x["stored"] != x["fresh"]]
    return {"result": "PASS" if not bad and len(rows) == EQUIVALENCE_ARTICLES else "FAIL",
            "checked": len(rows), "matches": len(rows) - len(bad), "mismatches": bad,
            "selection": erratum_rules()["equivalence_gate"]["selection"],
            "article_keys_sha256": hashlib.sha256("\n".join(x["article_key"] for x in rows).encode()).hexdigest(),
            "articles": rows}


def require_equivalence(scored_dev: list, service) -> dict:
    rec = label_equivalence(scored_dev, service)
    if rec["result"] != "PASS":
        raise SealedFeatureError(f"FinBERT label equivalence FAILED: {rec['matches']}/{rec['checked']} match "
                                 f"(required {EQUIVALENCE_ARTICLES}/{EQUIVALENCE_ARTICLES}); "
                                 f"first mismatches {rec['mismatches'][:3]}")
    return rec


def run_label_equivalence(out_dir: Path = OUT_DIR, service_factory: Callable | None = None) -> dict:
    """Development only: the 500-article check with the frozen FinBERT; the record is written once, PASS or FAIL."""
    out_dir = _check_output_dir(out_dir)
    if (out_dir / EQUIVALENCE_FILE).exists():
        raise SealedFeatureError(f"{EQUIVALENCE_FILE} already exists; the record is written once")
    erratum = load_erratum()
    scored, _, news_inputs = development_news()
    service = (service_factory or frozen_finbert)()
    config = check_finbert(service, erratum["model_files_sha256"])
    rec = label_equivalence(scored, service)
    doc = {"document_type": "phase12_finbert_label_equivalence", "stage": "12.S3C",
           "erratum_sha256": ERRATUM_SHA256, "amendment_sha256": c.AMENDMENT_SHA256, "finbert_config": config,
           "model_files_sha256": erratum["model_files_sha256"], "stored_scores_sha256": news_inputs["scores_sha256"],
           **rec, "builder_code_sha256": file_sha256(Path(__file__)), "sealed_data_read": False,
           "environment": environment(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_new_json(out_dir / EQUIVALENCE_FILE, doc)
    if rec["result"] != "PASS":
        raise SealedFeatureError(f"FinBERT label equivalence FAILED ({rec['matches']}/{rec['checked']}); see the record")
    return doc


def _version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def frozen_finbert():
    """The 11B.2 service: pinned revision, CPU, batch size 16, offline from the local cache."""
    from services.finbert_sentiment import FinBertSentimentService
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SealedFeatureError("HF_HUB_OFFLINE=1 is required (offline inference from the local cache)")
    return FinBertSentimentService.load(revision=FINBERT_CONFIG["model_revision"], device=FINBERT_CONFIG["device"],
                                        batch_size=FINBERT_CONFIG["batch_size"])


# ============================================================
# Sealed build (authorized path; never run before owner authorization)
# ============================================================


def check_sealed_authorization() -> dict:
    """Amendment 07, the owner's record naming this builder, and Erratum 01, before anything else."""
    c.load_amendment()
    auth = c.load_authorization(c.OUT_DIR, c.AMENDMENT_SHA256)
    return {"authorization": auth, "erratum": load_erratum()}


def sealed_articles(canonical_path: Path, expected_sha256: str) -> list[NewsArticle]:
    """Authorized path only: articles with information_available_at after the cutoff."""
    h, out = hashlib.sha256(), []
    with open(canonical_path, "rb") as f:
        for raw in f:
            h.update(raw)
            if raw.strip():
                a = NewsArticle.from_dict(json.loads(raw))
                if a.information_available_at > DEVELOPMENT_CUTOFF_UTC:
                    out.append(a)
    if h.hexdigest() != expected_sha256:
        raise SealedFeatureError("canonical dataset SHA256 differs from the integrity report")
    return out


def build_sealed(service_factory: Callable, out_dir: Path = SEALED_FEATURES_DIR) -> dict[str, pd.DataFrame]:
    """The one authorized sealed build: gates, reproduction, FinBERT checks, then sealed rows (written once)."""
    gates = check_sealed_authorization()
    out_dir = _check_output_dir(out_dir)
    if out_dir.exists():
        raise SealedFeatureError(f"{out_dir.name} already exists; sealed features are built once")
    reproduce(None)                                                    # same process, before any sealed row
    dev_scored, dev_classified, news_inputs = development_news()
    service = service_factory()
    config = check_finbert(service, gates["erratum"]["model_files_sha256"])
    equivalence = require_equivalence(dev_scored, service)            # 500/500 before any sealed inference
    report = load_integrity_report(fb.REPORT_PATH)
    canonical = PROJECT_ROOT / report["canonical_input"]["path"]
    sealed = sealed_articles(canonical, report["canonical_input"]["sha256"])
    if len(sealed) != report["sealed_article_count"]:
        raise SealedFeatureError("sealed article count differs from Gate B")
    sealed_scored, _ = score_canonical_articles(sealed, service)
    sealed_classified = classify_scored_articles(sealed_scored, KeywordEventClassifier())
    scored_m, classified_m, _ = fb.apply_rtx_mapping(sealed_scored, sealed_classified)
    inputs = a11.verified_inputs()
    bars = {s: read_bars_until(PROJECT_ROOT / a["path"], c.CONFIRMATION_END) for s, a in inputs["artifacts"].items()}
    frames = build_frames(bars, [*dev_scored, *scored_m], [*dev_classified, *classified_m],
                          r.FROZEN_SPEC.universe, c.CONFIRMATION_START, c.CONFIRMATION_END, SEALED_COVERAGE_END)
    out_dir.mkdir(parents=True)
    files = {}
    for sym, f in sorted(frames.items()):
        p = out_dir / f"{sym}.csv"
        out = f.copy()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        out.to_csv(p, index=False, lineterminator="\n", float_format="%.17g")
        files[sym] = file_sha256(p)
    _write_new_json(out_dir / "manifest.json", {
        "document_type": "phase12_sealed_features", "stage": STAGE, "authorization": gates["authorization"],
        "erratum_sha256": ERRATUM_SHA256, "builder_code_sha256": file_sha256(Path(__file__)), "files_sha256": files,
        "price_snapshots_sha256": {s: a["sha256"] for s, a in inputs["artifacts"].items()},
        "news": news_inputs, "sealed_articles": len(sealed), "finbert_config": config,
        "finbert_equivalence": {k: equivalence[k] for k in ("result", "checked", "matches", "article_keys_sha256")},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return frames


def sealed_loader() -> dict[str, pd.DataFrame]:
    """The confirmation runner's sealed_loader."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SealedFeatureError("HF_HUB_OFFLINE=1 is required (the pinned FinBERT revision loads from the local cache)")
    return build_sealed(frozen_finbert)


def status_lines(out_dir: Path = OUT_DIR) -> list[str]:
    lines = [f"builder sha256 : {file_sha256(Path(__file__))}"]
    p = Path(out_dir) / REPRODUCTION_FILE
    if p.is_file():
        rec = json.loads(p.read_text(encoding="utf-8"))
        lines.append(f"reproduction   : {rec['result']} ({len(rec['symbols'])} stocks, {rec['period'][0]} .. {rec['period'][1]})")
    else:
        lines.append("reproduction   : not run")
    if ERRATUM_FILE.is_file():
        lines.append(f"erratum 01     : sha256 {file_sha256(ERRATUM_FILE)}; pinned {ERRATUM_SHA256 or 'no'}")
    else:
        lines.append("erratum 01     : not written")
    q = Path(out_dir) / EQUIVALENCE_FILE
    if q.is_file():
        rec = json.loads(q.read_text(encoding="utf-8"))
        lines.append(f"finbert labels : {rec['result']} ({rec['matches']}/{rec['checked']})")
    else:
        lines.append("finbert labels : not checked")
    lines.append(f"sealed features: {'BUILT' if SEALED_FEATURES_DIR.exists() else 'not generated'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Stage 3B/3C sealed feature builder")
    parser.add_argument("command", choices=["reproduce", "erratum", "finbert-check", "status"])
    args = parser.parse_args(argv)
    if args.command == "reproduce":
        print(f"development reproduction: {reproduce()['result']}")
    elif args.command == "erratum":
        write_erratum()
        print(f"erratum written; sha256 {file_sha256(ERRATUM_FILE)}")
    elif args.command == "finbert-check":
        rec = run_label_equivalence()
        print(f"finbert label equivalence: {rec['result']} ({rec['matches']}/{rec['checked']})")
    print("\n".join(status_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
