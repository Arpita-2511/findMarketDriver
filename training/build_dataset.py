"""
Build the training dataset from a stored raw market-data snapshot.

Reproducibility chain (Phase 2, REQ-DATA-005):

    yfinance (completed bars only)
            |
            v
    raw snapshot    data/raw/stocks/<TICKER>_1d_<period>_<UTC stamp>.csv
    + metadata      ...same name .meta.json  (ticker, period, source,
            |        retrieval time, parameters, sha256, date range)
            v
    features        features/feature_engineering.py (technical_v2)
            |
            v
    final dataset   data/final_stock_dataset.csv
    + metadata      data/final_stock_dataset.meta.json (snapshot hash,
                     feature version, target, rows, dataset sha256)

A fresh download is NOT byte-reproducible (Yahoo revises and
back-adjusts history). The stored snapshot is: rebuilding from the same
snapshot with the same library versions yields an identical dataset.
The builder always reloads the snapshot from disk before building, so
"fresh" and "rebuilt" datasets go through exactly the same code path.

Usage (from the project root):

    python -m training.build_dataset                       # new snapshot, then build
    python -m training.build_dataset --from-snapshot data/raw/stocks/<file>.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_engineering import (  # noqa: E402
    MIN_HISTORY_ROWS,
    TECHNICAL_FEATURE_VERSION,
    TECHNICAL_FEATURES,
    engineer_features,
)
from features.finalize_dataset import finalize_dataset  # noqa: E402
from services.market_calendar_service import (  # noqa: E402
    REGULAR_SESSION_CLOSE,
    SETTLEMENT_DELAY,
)
from services.market_data import (  # noqa: E402
    EXCHANGE_TIMEZONE,
    OHLCV_COLUMNS,
    MarketDataError,
    validate_ohlcv,
)

# ==========================================
# Configuration
# ==========================================

SYMBOL = "AAPL"
PERIOD = "10y"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_FILE = "feature_engineered_stock_data.csv"
DATASET_FILE = "final_stock_dataset.csv"

DATE_FORMAT = "%Y-%m-%d"
TARGET_DEFINITION = "Target = Close[t+1] / Close[t] - 1 (next-day return, adjusted prices)"


# ==========================================
# Helpers
# ==========================================


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def meta_path_for(csv_path: Path) -> Path:
    return Path(csv_path).with_suffix(".meta.json")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV: ISO dates, '\\n' line endings, full float precision."""
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime(DATE_FORMAT)
    out.to_csv(path, index=False, lineterminator="\n")


def _write_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ==========================================
# Raw snapshot
# ==========================================


def save_raw_snapshot(raw: pd.DataFrame, symbol: str, period: str,
                      retrieved_at: datetime, raw_dir: Path = RAW_DIR) -> Path:
    """
    Store canonical, completed-bar OHLCV data plus metadata.
    Returns the snapshot CSV path.
    """
    validate_ohlcv(raw)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    stamp = retrieved_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = raw_dir / f"{symbol}_1d_{period}_{stamp}.csv"
    if csv_path.exists():
        raise FileExistsError(f"Snapshot already exists: {csv_path}")

    _write_csv(raw[OHLCV_COLUMNS], csv_path)

    meta = {
        "ticker": symbol,
        "source": "yfinance",
        "source_method": raw.attrs.get("source_method"),
        "yfinance_version": _package_version("yfinance"),
        "requested_period": period,
        "interval": "1d",
        "auto_adjust": True,
        "price_semantics": "split- and dividend-adjusted as of retrieval time",
        "dropped_columns": ["Dividends", "Stock Splits", "Capital Gains"],
        "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "completed_bar_policy": (
            f"bar kept only if retrieval time >= bar date {REGULAR_SESSION_CLOSE:%H:%M} "
            f"{EXCHANGE_TIMEZONE} + {int(SETTLEMENT_DELAY.total_seconds() // 60)} min"
        ),
        "incomplete_bars_dropped": raw.attrs.get("incomplete_bars_dropped"),
        "columns": OHLCV_COLUMNS,
        "rows": int(len(raw)),
        "first_date": raw["Date"].iloc[0].strftime(DATE_FORMAT),
        "last_date": raw["Date"].iloc[-1].strftime(DATE_FORMAT),
        "file": csv_path.name,
        "sha256": file_sha256(csv_path),
    }
    _write_json(meta, meta_path_for(csv_path))
    return csv_path


def load_raw_snapshot(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Load a snapshot and verify it against its metadata (hash, rows,
    columns). Raises MarketDataError if the file was modified.
    """
    csv_path = Path(csv_path)
    meta_path = meta_path_for(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {csv_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Snapshot metadata not found: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    actual = file_sha256(csv_path)
    if actual != meta.get("sha256"):
        raise MarketDataError(
            f"Snapshot hash mismatch for {csv_path.name}: metadata {meta.get('sha256')}, file {actual}"
        )

    df = pd.read_csv(csv_path, float_precision="round_trip")
    if list(df.columns) != OHLCV_COLUMNS:
        raise MarketDataError(f"Snapshot columns {list(df.columns)} != {OHLCV_COLUMNS}")
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT).astype("datetime64[ns]")
    for col in OHLCV_COLUMNS[1:]:
        df[col] = df[col].astype("float64")
    if len(df) != meta.get("rows"):
        raise MarketDataError(f"Snapshot has {len(df)} rows, metadata says {meta.get('rows')}")

    return validate_ohlcv(df), meta


# ==========================================
# Dataset build
# ==========================================


def build_from_snapshot(snapshot_path: Path, output_dir: Path = DATA_DIR) -> dict:
    """Build features + final dataset from a verified snapshot. Returns dataset metadata."""
    raw, snapshot_meta = load_raw_snapshot(snapshot_path)

    features_df = engineer_features(raw.copy())
    final_df = finalize_dataset(features_df.copy())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / FEATURE_FILE
    dataset_path = output_dir / DATASET_FILE
    _write_csv(features_df, feature_path)
    _write_csv(final_df, dataset_path)

    meta = {
        "dataset": DATASET_FILE,
        "sha256": file_sha256(dataset_path),
        "rows": int(len(final_df)),
        "columns": list(final_df.columns),
        "first_date": final_df["Date"].iloc[0].strftime(DATE_FORMAT),
        "last_date": final_df["Date"].iloc[-1].strftime(DATE_FORMAT),
        "target": TARGET_DEFINITION,
        "feature_version": TECHNICAL_FEATURE_VERSION,
        "features": TECHNICAL_FEATURES,
        "min_history_rows": MIN_HISTORY_ROWS,
        "raw_snapshot": {
            "path": _rel(snapshot_path),
            "sha256": snapshot_meta["sha256"],
            "ticker": snapshot_meta["ticker"],
            "retrieved_at_utc": snapshot_meta["retrieved_at_utc"],
            "rows": snapshot_meta["rows"],
        },
        "feature_file": {"path": FEATURE_FILE, "sha256": file_sha256(feature_path)},
        "environment": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
        },
    }
    _write_json(meta, meta_path_for(dataset_path))
    return meta


def fetch_and_snapshot(symbol: str = SYMBOL, period: str = PERIOD, raw_dir: Path = RAW_DIR,
                       now: datetime | None = None) -> Path:
    """Download completed bars and store them as a new raw snapshot."""
    from services.live_stock_service import fetch_latest_stock_data  # network dependency

    retrieved_at = now or datetime.now(timezone.utc)
    raw = fetch_latest_stock_data(symbol=symbol, period=period, now=retrieved_at)
    return save_raw_snapshot(raw, symbol, period, retrieved_at, raw_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the FindMarketDriver training dataset")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--period", default=PERIOD)
    parser.add_argument("--from-snapshot", type=Path, default=None,
                        help="rebuild from an existing raw snapshot instead of downloading")
    args = parser.parse_args(argv)

    print("=" * 50)
    print("Building Training Dataset")
    print("=" * 50)

    if args.from_snapshot is None:
        print(f"\nDownloading {args.period} of completed daily bars for {args.symbol}...")
        snapshot = fetch_and_snapshot(args.symbol, args.period)
        print(f"Raw snapshot saved: {_rel(snapshot)}")
    else:
        snapshot = args.from_snapshot
        print(f"\nUsing existing raw snapshot: {_rel(snapshot)}")

    meta = build_from_snapshot(snapshot)

    print("\nDataset Saved Successfully!")
    print(f"Rows       : {meta['rows']}  ({meta['first_date']} -> {meta['last_date']})")
    print(f"Snapshot   : {meta['raw_snapshot']['path']}  sha256={meta['raw_snapshot']['sha256'][:12]}...")
    print(f"Dataset    : data/{DATASET_FILE}  sha256={meta['sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
