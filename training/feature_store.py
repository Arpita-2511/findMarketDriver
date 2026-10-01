"""
Unified ML feature store (Phase 7, feature_store_v1).

Combines EXISTING, hash-verified datasets - no feature is recomputed:

    technical  data/final_stock_dataset.csv            (Phase 2, technical_v2)
    sentiment  data/processed/sentiment/*.daily.jsonl  (Phase 5C, sentiment_features_v1)
    events     data/processed/events/*.daily.jsonl     (Phase 6, event_features_v1)
    sessions   data/raw/stocks/<snapshot>.csv          (Phase 2 raw snapshot)

Grain: one row per (symbol, trading_date) = one prediction observation, made
at prediction_timestamp = completion of bar D = D 16:30 America/New_York (the
existing convention, row_prediction_timestamp). The technical dataset is the
spine: a row exists iff technical features exist for that session.

Join contract:
    - keys normalised (upper-case symbol, ISO session date), validated as
      calendar sessions, duplicates / nulls detected BEFORE joining -> error
    - every sentiment/event row's prediction_timestamp must equal
      row_prediction_timestamp(D) (no second timestamp convention)
    - 1:1 left joins on (symbol, trading_date)

Missing-data semantics (never "fill everything with 0"):
    covered  = the session lies inside the family's daily dataset range.
               The family's own zero convention applies (no news -> 0) and
               EVERY covered session must have a row - a gap is an error.
    not covered -> feature values NaN (unknown), <family>__covered = False.

Namespaces: technical names unchanged (technical_v2 / harness); sentiment and
event features prefixed "sentiment__" / "event__". Version columns move to
metadata; event__dominant_event_type stays categorical and is not part of the
numeric feature families.

Targets (direction_v1 contract, never in feature lists):
    target_return_1d     = stored Target = Close[t+1] / Close[t] - 1 (re-verified)
    target_direction_1d  = 1 if target_return_1d > 0 else 0

Output: data/processed/features/feature_store_v1_<SYMBOL>_<first>_<last>.csv
        + .meta.json. The CSV is deterministic (sorted rows/cols, ISO dates,
        full float precision); its sha256 is the CONTENT hash. generated_at and
        git state live only in the metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_engineering import TECHNICAL_FEATURE_VERSION, TECHNICAL_FEATURES  # noqa: E402
from services.event_features import EVENT_FEATURE_COLUMNS, EVENT_FEATURE_VERSION  # noqa: E402
from services.historical_sentiment import _write_meta, _write_once  # noqa: E402
from services.market_calendar_service import TradingCalendar  # noqa: E402
from services.news_alignment import row_prediction_timestamp  # noqa: E402
from services.news_schema import parse_timestamp  # noqa: E402
from services.sentiment_features import FEATURE_COLUMNS as SENTIMENT_FEATURE_COLUMNS  # noqa: E402
from services.sentiment_features import SENTIMENT_FEATURE_VERSION  # noqa: E402
from training.build_dataset import file_sha256, load_raw_snapshot, meta_path_for  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.targets import TARGET_VERSION, direction_label  # noqa: E402

FEATURE_STORE_VERSION = "feature_store_v1"
SCHEMA_VERSION = "feature_store_schema_v1"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

KEY_COLUMNS = ("symbol", "trading_date")
REFERENCE_COLUMNS = ("prediction_timestamp", "Close")     # not features: Close is needed to rebuild targets
SENTIMENT_PREFIX = "sentiment__"
EVENT_PREFIX = "event__"
CATEGORICAL_EVENT = "dominant_event_type"

TECHNICAL_COLUMNS = tuple(TECHNICAL_FEATURES)
SENTIMENT_COLUMNS = tuple(SENTIMENT_PREFIX + c for c in SENTIMENT_FEATURE_COLUMNS)
EVENT_NUMERIC_COLUMNS = tuple(EVENT_PREFIX + c for c in EVENT_FEATURE_COLUMNS if c != CATEGORICAL_EVENT)
EVENT_CATEGORICAL_COLUMNS = (EVENT_PREFIX + CATEGORICAL_EVENT,)
COVERAGE_COLUMNS = (SENTIMENT_PREFIX + "covered", EVENT_PREFIX + "covered")
TARGET_COLUMNS = ("target_return_1d", "target_direction_1d")

FEATURE_FAMILIES = {
    "technical": TECHNICAL_COLUMNS,
    "sentiment": SENTIMENT_COLUMNS,
    "events": EVENT_NUMERIC_COLUMNS,
}

OUTPUT_COLUMNS = (*KEY_COLUMNS, *REFERENCE_COLUMNS, *TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS,
                  *EVENT_NUMERIC_COLUMNS, *EVENT_CATEGORICAL_COLUMNS, *COVERAGE_COLUMNS, *TARGET_COLUMNS)


class FeatureStoreError(ValueError):
    """Invalid, inconsistent or unverifiable feature-store input."""


# ==========================================
# Loading (hash-verified)
# ==========================================


def _read_meta(path: Path) -> dict:
    meta_file = meta_path_for(path)
    if not Path(path).is_file() or not meta_file.is_file():
        raise FileNotFoundError(f"dataset or metadata missing: {path}")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    if file_sha256(path) != meta.get("sha256"):
        raise FeatureStoreError(f"hash mismatch for {Path(path).name}")
    return meta


@dataclass
class Family:
    name: str
    frame: pd.DataFrame          # normalised keys + family columns
    meta: dict
    path: Path


def load_technical(path: Path) -> Family:
    meta = _read_meta(path)
    if meta.get("feature_version") != TECHNICAL_FEATURE_VERSION:
        raise FeatureStoreError(f"technical dataset is {meta.get('feature_version')}, expected {TECHNICAL_FEATURE_VERSION}")
    ticker = (meta.get("raw_snapshot") or {}).get("ticker")
    if not ticker:
        raise FeatureStoreError("technical metadata does not record the ticker (raw_snapshot.ticker)")
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in ("Date", "Close", "Target", *TECHNICAL_COLUMNS) if c not in df.columns]
    if missing:
        raise FeatureStoreError(f"technical dataset is missing columns {missing}")
    out = pd.DataFrame({"symbol": ticker, "trading_date": df["Date"]})
    for c in ("Close", "Target", *TECHNICAL_COLUMNS):
        out[c] = df[c].astype("float64")
    return Family("technical", _normalise_keys(out, "technical"), meta, Path(path))


def _load_daily(path: Path, name: str, version_field: str, version: str, columns: tuple[str, ...],
                prefix: str) -> Family:
    meta = _read_meta(path)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != meta.get("rows"):
        raise FeatureStoreError(f"{name} dataset has {len(rows)} rows, metadata says {meta.get('rows')}")
    df = pd.DataFrame(rows, columns=["symbol", "trading_date", "prediction_timestamp", version_field, *columns])
    if df[version_field].ne(version).any():
        raise FeatureStoreError(f"{name} dataset contains rows that are not {version}")
    if df[list(columns)].isna().any().any():
        raise FeatureStoreError(f"{name} dataset contains null feature values")
    df = df.drop(columns=[version_field]).rename(columns={c: prefix + c for c in columns})
    df = df.rename(columns={"prediction_timestamp": f"{prefix}prediction_timestamp"})
    return Family(name, _normalise_keys(df, name), meta, Path(path))


def load_sentiment(path: Path) -> Family:
    return _load_daily(path, "sentiment", "sentiment_feature_version", SENTIMENT_FEATURE_VERSION,
                       SENTIMENT_FEATURE_COLUMNS, SENTIMENT_PREFIX)


def load_events(path: Path) -> Family:
    return _load_daily(path, "events", "event_feature_version", EVENT_FEATURE_VERSION,
                       EVENT_FEATURE_COLUMNS, EVENT_PREFIX)


# ==========================================
# Keys, calendar, timestamps
# ==========================================


def _normalise_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df["symbol"].isna().any() or df["trading_date"].isna().any():
        raise FeatureStoreError(f"{name}: null symbol or trading_date")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    if (df["symbol"] == "").any():
        raise FeatureStoreError(f"{name}: empty symbol")
    try:
        df["trading_date"] = [date.fromisoformat(str(d)[:10]) for d in df["trading_date"]]
    except ValueError as e:
        raise FeatureStoreError(f"{name}: invalid trading_date ({e})") from None
    dup = df.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if dup.any():
        sample = df.loc[dup, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise FeatureStoreError(f"{name}: {int(dup.sum())} rows share a (symbol, trading_date) key, e.g. {sample}")
    return df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def validate_sessions(family: Family, calendar: TradingCalendar) -> None:
    """Every key must be a calendar session; daily rows must carry the canonical prediction timestamp."""
    ts_col = None if family.name == "technical" else (
        SENTIMENT_PREFIX if family.name == "sentiment" else EVENT_PREFIX) + "prediction_timestamp"
    for i, d in enumerate(family.frame["trading_date"]):
        expected = row_prediction_timestamp(d, calendar)           # raises for non-sessions
        if ts_col is not None:
            actual = parse_timestamp(family.frame[ts_col].iloc[i], "prediction_timestamp")
            if actual != expected:
                raise FeatureStoreError(
                    f"{family.name}: {d} has prediction_timestamp {actual.isoformat()}, expected {expected.isoformat()}")


def _coverage(family: Family, symbol: str) -> tuple[date, date] | None:
    rows = family.frame[family.frame["symbol"] == symbol]
    return (rows["trading_date"].min(), rows["trading_date"].max()) if len(rows) else None


# ==========================================
# Join + targets
# ==========================================


def _join_family(base: pd.DataFrame, family: Family, prefix: str, value_columns: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    """1:1 left join; inside coverage every base session must have a row; outside -> NaN, covered False."""
    keep = [*KEY_COLUMNS, *value_columns]
    joined = base.merge(family.frame[keep], on=list(KEY_COLUMNS), how="left", validate="one_to_one", indicator=True)
    covered = pd.Series(False, index=joined.index)
    for symbol in joined["symbol"].unique():
        span = _coverage(family, symbol)
        if span is None:
            continue
        in_span = (joined["symbol"] == symbol) & joined["trading_date"].between(*span)
        gaps = in_span & (joined["_merge"] == "left_only")
        if gaps.any():
            raise FeatureStoreError(
                f"{family.name}: {int(gaps.sum())} session(s) inside its coverage have no row "
                f"(missing data, not 'no news'), e.g. {joined.loc[gaps, 'trading_date'].iloc[0]}")
        covered |= in_span
    joined[prefix + "covered"] = covered
    # family rows that have no technical row (e.g. before the technical warm-up) are reported, not used
    orphan = len(family.frame) - int((joined["_merge"] == "both").sum())
    return joined.drop(columns="_merge"), orphan


def attach_targets(df: pd.DataFrame) -> pd.DataFrame:
    """target_return_1d = stored Target, re-verified against Close where the next close is in the table."""
    df = df.copy()
    for symbol, idx in df.groupby("symbol").groups.items():
        part = df.loc[idx]
        recomputed = part["Close"].shift(-1) / part["Close"] - 1
        both = recomputed.notna()
        if not np.allclose(part.loc[both, "Target"], recomputed[both]):
            raise FeatureStoreError(f"{symbol}: stored Target disagrees with Close[t+1]/Close[t]-1")
    df["target_return_1d"] = df.pop("Target")
    if df["target_return_1d"].isna().any() or not np.isfinite(df["target_return_1d"]).all():
        raise FeatureStoreError("target_return_1d has missing or non-finite values")
    df["target_direction_1d"] = direction_label(df["target_return_1d"])
    return df


def build_feature_frame(technical: Family, sentiment: Family, events: Family, calendar: TradingCalendar
                        ) -> tuple[pd.DataFrame, dict]:
    for fam in (technical, sentiment, events):
        validate_sessions(fam, calendar)
    base = technical.frame.copy()
    base["prediction_timestamp"] = [row_prediction_timestamp(d, calendar) for d in base["trading_date"]]
    df, orphan_s = _join_family(base, sentiment, SENTIMENT_PREFIX, SENTIMENT_COLUMNS)
    df, orphan_e = _join_family(df, events, EVENT_PREFIX, (*EVENT_NUMERIC_COLUMNS, *EVENT_CATEGORICAL_COLUMNS))
    df = attach_targets(df)
    df = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)[list(OUTPUT_COLUMNS)]
    return df, {"sentiment_rows_without_technical_row": orphan_s, "event_rows_without_technical_row": orphan_e}


# ==========================================
# Serialisation, quality report, metadata
# ==========================================


def serialize_frame(df: pd.DataFrame) -> bytes:
    """Deterministic CSV: fixed column order, ISO dates/timestamps, repr floats, '\\n', NaN as empty."""
    out = df.copy()
    out["trading_date"] = [d.isoformat() for d in out["trading_date"]]
    out["prediction_timestamp"] = [t.isoformat() for t in out["prediction_timestamp"]]
    buf = io.StringIO()
    out.to_csv(buf, index=False, lineterminator="\n", na_rep="")
    return buf.getvalue().encode("utf-8")


def data_quality_report(df: pd.DataFrame) -> dict:
    numeric = [*TECHNICAL_COLUMNS, *SENTIMENT_COLUMNS, *EVENT_NUMERIC_COLUMNS, *TARGET_COLUMNS]
    values = df[numeric].astype("float64")
    covered = {"sentiment": df[SENTIMENT_PREFIX + "covered"], "events": df[EVENT_PREFIX + "covered"]}
    cat = df[EVENT_CATEGORICAL_COLUMNS[0]]
    from services.event_taxonomy import EVENT_TYPES
    allowed = set(EVENT_TYPES) | {"NONE"}
    constant = [c for c in numeric if values[c].dropna().nunique() <= 1 and values[c].notna().any()]
    zs = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    return {
        "rows": len(df),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "date_range": [df["trading_date"].min().isoformat(), df["trading_date"].max().isoformat()] if len(df) else None,
        "duplicate_keys": int(df.duplicated(subset=list(KEY_COLUMNS)).sum()),
        "null_keys": int(df[list(KEY_COLUMNS)].isna().any(axis=1).sum()),
        "infinite_values": int(np.isinf(values.to_numpy()).sum()),
        "missing_values": {c: int(values[c].isna().sum()) for c in numeric if values[c].isna().any()},
        "missing_by_family": {
            "technical": int(df[list(TECHNICAL_COLUMNS)].isna().any(axis=1).sum()),
            "sentiment_not_covered": int((~covered["sentiment"]).sum()),
            "events_not_covered": int((~covered["events"]).sum()),
        },
        "coverage": {
            "technical": len(df),
            "sentiment": int(covered["sentiment"].sum()),
            "events": int(covered["events"].sum()),
            "all_three": int((covered["sentiment"] & covered["events"]).sum()),
            "sentiment_rows_with_news": int((df[SENTIMENT_PREFIX + "news_count"] > 0).sum()),
            "event_rows_with_events": int((df[EVENT_PREFIX + "event_count"] > 0).sum()),
        },
        "unexpected_categorical_values": sorted(set(cat.dropna()) - allowed),
        "constant_features": constant,
        "extreme_values_abs_z_gt_6": {c: int((zs[c].abs() > 6).sum()) for c in numeric if (zs[c].abs() > 6).any()},
        "targets": {
            "target_direction_1d_up_share": float(df["target_direction_1d"].mean()) if len(df) else None,
            "target_return_1d": {k: float(v) for k, v in df["target_return_1d"].describe().items()} if len(df) else None,
        },
    }


def output_path(df: pd.DataFrame, output_dir: Path = FEATURES_DIR) -> Path:
    symbols = "-".join(sorted(df["symbol"].unique()))
    return Path(output_dir) / (f"{FEATURE_STORE_VERSION}_{symbols}_{df['trading_date'].min():%Y%m%d}_"
                               f"{df['trading_date'].max():%Y%m%d}.csv")


def build_feature_store(technical_path: Path, sentiment_path: Path, events_path: Path, market_snapshot: Path,
                        output_dir: Path = FEATURES_DIR, generated_at: datetime | None = None) -> tuple[Path, dict]:
    technical, sentiment, events = load_technical(technical_path), load_sentiment(sentiment_path), load_events(events_path)
    bars, snapshot_meta = load_raw_snapshot(market_snapshot)           # hash-verified
    calendar = TradingCalendar.from_bars(bars)

    df, orphans = build_feature_frame(technical, sentiment, events, calendar)
    if df.empty:
        raise FeatureStoreError("feature store would be empty")
    payload = serialize_frame(df)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = output_path(df, output_dir)
    _write_once(path, payload)

    def source(fam: Family, extra: dict) -> dict:
        return {"file": fam.path.name, "sha256": fam.meta["sha256"], **extra}

    meta = {
        "dataset_version": FEATURE_STORE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "grain": list(KEY_COLUMNS),
        "prediction_timestamp_rule": "row_prediction_timestamp(D) = D 16:30 America/New_York (completed bar)",
        "eligibility_rule": "information_available_at <= prediction_timestamp (inherited from Phase 5B/6 inputs)",
        "sources": {
            "technical": source(technical, {"feature_version": TECHNICAL_FEATURE_VERSION,
                                            "raw_snapshot": technical.meta.get("raw_snapshot")}),
            "sentiment": source(sentiment, {"feature_version": SENTIMENT_FEATURE_VERSION,
                                            "provenance": sentiment.meta.get("provenance")}),
            "events": source(events, {"feature_version": EVENT_FEATURE_VERSION,
                                      "classifier": events.meta.get("classifier")}),
            "calendar": {"file": Path(market_snapshot).name, "sha256": snapshot_meta["sha256"]},
        },
        "target_version": TARGET_VERSION,
        "targets": {"target_return_1d": "Close[t+1] / Close[t] - 1 (stored Target, re-verified)",
                    "target_direction_1d": "1 if target_return_1d > 0 else 0 (direction_v1)"},
        "columns": list(OUTPUT_COLUMNS),
        "feature_families": {k: list(v) for k, v in FEATURE_FAMILIES.items()},
        "family_counts": {**{k: len(v) for k, v in FEATURE_FAMILIES.items()},
                          "categorical": len(EVENT_CATEGORICAL_COLUMNS), "targets": len(TARGET_COLUMNS)},
        "missing_data_semantics": ("inside a family's coverage its own zero convention applies (no news -> 0); "
                                   "outside coverage values are NaN and <family>__covered is False"),
        "row_count": len(df),
        "column_count": len(OUTPUT_COLUMNS),
        "join_report": orphans,
        "data_quality": data_quality_report(df),
        "ordering": "symbol, trading_date",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_hash_note": "sha256 covers the CSV only; generated_at and git are build metadata",
        "git": git_info(),
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    _write_meta(path, meta)
    return path, meta


def load_feature_store(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a feature store after verifying its content hash; restores dtypes and key types."""
    meta = _read_meta(path)
    df = pd.read_csv(path, float_precision="round_trip", keep_default_na=True)
    if list(df.columns) != list(OUTPUT_COLUMNS):
        raise FeatureStoreError("feature store columns do not match the schema")
    df["trading_date"] = [date.fromisoformat(d) for d in df["trading_date"]]
    df["prediction_timestamp"] = [parse_timestamp(t, "prediction_timestamp") for t in df["prediction_timestamp"]]
    for c in COVERAGE_COLUMNS:
        df[c] = df[c].astype(bool)
    return df, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the unified ML feature store (Phase 7)")
    parser.add_argument("--technical", type=Path, default=PROJECT_ROOT / "data" / "final_stock_dataset.csv")
    parser.add_argument("--sentiment", type=Path, required=True, help="Phase 5C *.daily.jsonl")
    parser.add_argument("--events", type=Path, required=True, help="Phase 6 *.daily.jsonl")
    parser.add_argument("--market-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=FEATURES_DIR)
    args = parser.parse_args(argv)

    path, meta = build_feature_store(args.technical, args.sentiment, args.events, args.market_snapshot, args.output_dir)
    q = meta["data_quality"]
    print(f"Feature store : {path}")
    print(f"  rows {q['rows']}, columns {meta['column_count']}, symbols {q['symbols']}, dates {q['date_range']}")
    print(f"  families    : {meta['family_counts']}")
    print(f"  coverage    : {q['coverage']}")
    print(f"  duplicates {q['duplicate_keys']}, infinite {q['infinite_values']}, "
          f"constant features {len(q['constant_features'])}, unexpected categories {q['unexpected_categorical_values']}")
    print(f"  join report : {meta['join_report']}")
    print(f"  sha256      : {meta['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
