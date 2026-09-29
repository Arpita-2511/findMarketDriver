"""Raw snapshot + metadata + deterministic dataset build (no network)."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import services.live_stock_service as live
import training.build_dataset as bd
from features.feature_engineering import MIN_HISTORY_ROWS, TECHNICAL_FEATURE_VERSION, TECHNICAL_FEATURES
from services.market_data import MarketDataError
from tests.conftest import as_ticker_history, make_bars
from training.evaluation_harness import build_targets, load_dataset, load_dataset_metadata, validate_dataset

RETRIEVED = datetime(2025, 3, 3, 22, 0, tzinfo=timezone.utc)


@pytest.fixture
def snapshot(tmp_path):
    bars = make_bars(220)
    bars.attrs = {"source_method": "Ticker.history", "incomplete_bars_dropped": 0}
    path = bd.save_raw_snapshot(bars, "TEST", "1y", RETRIEVED, raw_dir=tmp_path / "raw")
    return path, bars


# ==========================================
# Snapshot + metadata
# ==========================================


def test_snapshot_files_and_metadata(snapshot):
    path, bars = snapshot
    meta = json.loads(bd.meta_path_for(path).read_text(encoding="utf-8"))

    assert path.name == "TEST_1d_1y_20250303T220000Z.csv"
    assert meta["sha256"] == bd.file_sha256(path)
    for key in ["ticker", "source", "source_method", "requested_period", "interval", "auto_adjust",
                "retrieved_at_utc", "completed_bar_policy", "rows", "first_date", "last_date", "columns"]:
        assert key in meta
    assert meta["ticker"] == "TEST"
    assert meta["auto_adjust"] is True
    assert meta["rows"] == len(bars)
    assert meta["retrieved_at_utc"] == "2025-03-03T22:00:00+00:00"
    assert meta["first_date"] == bars["Date"].iloc[0].strftime("%Y-%m-%d")


def test_snapshot_round_trip_is_exact(snapshot):
    path, bars = snapshot
    loaded, _ = bd.load_raw_snapshot(path)
    pd.testing.assert_frame_equal(loaded, bars, check_exact=True)


def test_tampered_snapshot_is_rejected(snapshot):
    path, _ = snapshot
    text = path.read_text(encoding="utf-8").splitlines()
    text[5] = text[5].replace(",", ",9", 1)          # alter one price
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(MarketDataError, match="hash mismatch"):
        bd.load_raw_snapshot(path)


def test_snapshot_without_metadata_is_rejected(snapshot):
    path, _ = snapshot
    bd.meta_path_for(path).unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        bd.load_raw_snapshot(path)


def test_snapshot_is_never_overwritten(snapshot):
    path, bars = snapshot
    with pytest.raises(FileExistsError):
        bd.save_raw_snapshot(bars, "TEST", "1y", RETRIEVED, raw_dir=path.parent)


def test_invalid_data_cannot_be_snapshotted(tmp_path):
    bars = make_bars(100).iloc[::-1].reset_index(drop=True)
    with pytest.raises(MarketDataError):
        bd.save_raw_snapshot(bars, "TEST", "1y", RETRIEVED, raw_dir=tmp_path)


# ==========================================
# Deterministic build from a snapshot
# ==========================================


def test_rebuild_from_same_snapshot_is_byte_identical(snapshot, tmp_path):
    path, _ = snapshot
    meta_a = bd.build_from_snapshot(path, tmp_path / "a")
    meta_b = bd.build_from_snapshot(path, tmp_path / "b")

    assert meta_a == meta_b
    for name in [bd.DATASET_FILE, bd.FEATURE_FILE, "final_stock_dataset.meta.json"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_dataset_metadata_links_to_snapshot(snapshot, tmp_path):
    path, bars = snapshot
    meta = bd.build_from_snapshot(path, tmp_path / "out")

    assert meta["raw_snapshot"]["sha256"] == bd.file_sha256(path)
    assert meta["feature_version"] == TECHNICAL_FEATURE_VERSION
    assert meta["features"] == TECHNICAL_FEATURES
    assert meta["min_history_rows"] == MIN_HISTORY_ROWS
    # MIN_HISTORY_ROWS - 1 warm-up rows removed, and the last row has no next-day target
    assert meta["rows"] == len(bars) - (MIN_HISTORY_ROWS - 1) - 1
    assert meta["sha256"] == bd.file_sha256(tmp_path / "out" / bd.DATASET_FILE)


def test_built_dataset_satisfies_phase1_contract(snapshot, tmp_path):
    path, _ = snapshot
    bd.build_from_snapshot(path, tmp_path / "out")
    dataset_path = tmp_path / "out" / bd.DATASET_FILE

    df = load_dataset(dataset_path)
    validate_dataset(df, TECHNICAL_FEATURES)        # schema, order, duplicates, NaN/inf
    build_targets(df, horizon=1)                    # stored Target aligned with Close[t+1]/Close[t]-1

    meta = load_dataset_metadata(dataset_path, bd.file_sha256(dataset_path))
    assert meta["matches_dataset_file"] is True

    dataset_path.write_text(dataset_path.read_text() + "\n", encoding="utf-8")
    edited = load_dataset_metadata(dataset_path, bd.file_sha256(dataset_path))
    assert edited["matches_dataset_file"] is False


def test_fresh_fetch_goes_through_snapshot(monkeypatch, tmp_path):
    bars = make_bars(150)
    monkeypatch.setattr(live, "_fetch_history", lambda s, p: as_ticker_history(bars))
    path = bd.fetch_and_snapshot("TEST", "1y", raw_dir=tmp_path, now=RETRIEVED)

    loaded, meta = bd.load_raw_snapshot(path)
    assert meta["source_method"] == "Ticker.history"
    assert meta["incomplete_bars_dropped"] == 0
    pd.testing.assert_frame_equal(loaded, bars, check_exact=True)
