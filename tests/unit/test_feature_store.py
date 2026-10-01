"""Phase 7 feature store: keys, joins, missing-data semantics, namespaces, leakage, determinism, metadata."""

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from services.market_data import MarketDataError
from tests.event_fakes import classified
from tests.feature_store_fakes import ANALYST, NEWS_DATES, TECH_DATES, default_items, ny, rewrite_jsonl, write_inputs
from training.build_dataset import file_sha256
from training.feature_store import (
    COVERAGE_COLUMNS,
    EVENT_CATEGORICAL_COLUMNS,
    FEATURE_FAMILIES,
    OUTPUT_COLUMNS,
    TARGET_COLUMNS,
    FeatureStoreError,
    build_feature_store,
    load_feature_store,
)

UTC = timezone.utc


def build(tmp_path, inputs=None, out="out"):
    inputs = inputs or write_inputs(tmp_path / "in")
    path, meta = build_feature_store(inputs["technical"], inputs["sentiment"], inputs["events"],
                                     inputs["snapshot"], tmp_path / out)
    store, _ = load_feature_store(path)
    return path, meta, store, inputs


def at(store, d):
    return store[store["trading_date"] == d].iloc[0]


# ==========================================
# Spine, alignment, coverage
# ==========================================


def test_one_row_per_technical_session(tmp_path):
    _, meta, store, _ = build(tmp_path)
    assert list(store.columns) == list(OUTPUT_COLUMNS)
    assert list(store["trading_date"]) == TECH_DATES                             # sorted, technical spine
    assert not store.duplicated(subset=["symbol", "trading_date"]).any()
    assert int(store[COVERAGE_COLUMNS[0]].sum()) == int(store[COVERAGE_COLUMNS[1]].sum()) == len(NEWS_DATES)
    assert meta["row_count"] == len(TECH_DATES) and meta["column_count"] == len(OUTPUT_COLUMNS)


def test_values_come_from_the_source_rows(tmp_path):
    _, _, store, inputs = build(tmp_path)
    import json
    src_s = {r["trading_date"]: r for r in map(json.loads, inputs["sentiment"].read_text(encoding="utf-8").splitlines())}
    src_e = {r["trading_date"]: r for r in map(json.loads, inputs["events"].read_text(encoding="utf-8").splitlines())}
    row = at(store, date(2024, 7, 10))
    assert row["sentiment__mean_sentiment"] == src_s["2024-07-10"]["mean_sentiment"]
    assert row["sentiment__news_count"] == src_s["2024-07-10"]["news_count"]
    assert row["event__earnings_event_count"] == src_e["2024-07-10"]["earnings_event_count"] == 1
    assert row["event__dominant_event_type"] == src_e["2024-07-10"]["dominant_event_type"]
    assert row["prediction_timestamp"] == datetime(2024, 7, 10, 20, 30, tzinfo=UTC)


def test_no_news_and_no_event_days_are_zero_not_missing(tmp_path):
    _, _, store, _ = build(tmp_path)
    first = at(store, date(2024, 7, 1))                                        # covered, nothing yet
    assert first["sentiment__covered"] and first["sentiment__news_count"] == 0
    quiet = at(store, date(2024, 7, 9))                                        # covered, no new events
    assert quiet["event__covered"] and quiet["event__event_count"] == 0
    assert quiet["event__dominant_event_type"] == "NONE"


def test_outside_coverage_is_missing_not_zero(tmp_path):
    _, meta, store, _ = build(tmp_path)
    row = at(store, date(2024, 6, 20))
    assert not row["sentiment__covered"] and not row["event__covered"]
    assert math.isnan(row["sentiment__news_count"]) and math.isnan(row["event__event_count"])
    assert not isinstance(row["event__dominant_event_type"], str)                  # NaN, not "NONE"
    assert not store[list(FEATURE_FAMILIES["technical"])].isna().any().any()
    assert meta["data_quality"]["missing_by_family"]["sentiment_not_covered"] == len(TECH_DATES) - len(NEWS_DATES)


def test_gap_inside_coverage_is_an_error(tmp_path):
    inputs = write_inputs(tmp_path / "in")
    rewrite_jsonl(inputs["sentiment"], lambda rows: rows.pop(7))
    with pytest.raises(FeatureStoreError, match="missing data"):
        build(tmp_path, inputs)


# ==========================================
# Key integrity & input validation
# ==========================================


@pytest.mark.parametrize("mutate, error, message", [
    (lambda rows: rows.append(dict(rows[3])), FeatureStoreError, "share a"),
    (lambda rows: rows[2].update(symbol=None), FeatureStoreError, "null"),
    (lambda rows: rows[2].update(trading_date="2024-13-40"), FeatureStoreError, "invalid trading_date"),
    (lambda rows: rows[2].update(sentiment_feature_version="sentiment_features_v0"), FeatureStoreError, "not sentiment"),
    (lambda rows: rows[2].update(prediction_timestamp=(datetime.fromisoformat(rows[2]["prediction_timestamp"])
                                                       + timedelta(hours=1)).isoformat()), FeatureStoreError, "expected"),
    (lambda rows: rows[2].update(trading_date="2024-07-06"), MarketDataError, "not a trading session"),
])
def test_invalid_sentiment_inputs_rejected(tmp_path, mutate, error, message):
    inputs = write_inputs(tmp_path / "in")
    rewrite_jsonl(inputs["sentiment"], mutate)
    with pytest.raises(error, match=message):
        build(tmp_path, inputs)


def test_duplicate_technical_session_rejected(tmp_path):
    inputs = write_inputs(tmp_path / "in")
    lines = inputs["technical"].read_text(encoding="utf-8").splitlines()
    from tests.feature_store_fakes import _write_with_meta
    _write_with_meta(inputs["technical"], ("\n".join(lines + [lines[5]]) + "\n").encode("utf-8"),
                     feature_version="technical_v2", raw_snapshot={"ticker": "AAPL"})
    with pytest.raises(FeatureStoreError, match="share a"):
        build(tmp_path, inputs)


def test_tampered_input_rejected(tmp_path):
    inputs = write_inputs(tmp_path / "in")
    inputs["events"].write_bytes(inputs["events"].read_bytes().replace(b"AAPL", b"MSFT", 1))
    with pytest.raises(FeatureStoreError, match="hash mismatch"):
        build(tmp_path, inputs)


# ==========================================
# Namespaces & targets
# ==========================================


def test_namespaces_and_families():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS))
    fams = FEATURE_FAMILIES
    assert (len(fams["technical"]), len(fams["sentiment"]), len(fams["events"])) == (20, 15, 23)   # 24 Phase 6 cols - dominant_event_type
    all_features = [c for cols in fams.values() for c in cols]
    assert len(all_features) == len(set(all_features))                         # disjoint
    assert all(c.startswith("sentiment__") for c in fams["sentiment"])
    assert all(c.startswith("event__") for c in fams["events"])
    for excluded in (*TARGET_COLUMNS, *COVERAGE_COLUMNS, *EVENT_CATEGORICAL_COLUMNS, "Close", "prediction_timestamp"):
        assert excluded not in all_features                                    # targets never features


def test_targets_preserved(tmp_path):
    _, meta, store, _ = build(tmp_path)
    recomputed = store["Close"].shift(-1) / store["Close"] - 1
    assert np.allclose(store["target_return_1d"][:-1], recomputed[:-1])
    assert (store["target_direction_1d"] == (store["target_return_1d"] > 0).astype(int)).all()
    assert meta["target_version"] == "direction_v1"


def test_inconsistent_stored_target_rejected(tmp_path):
    inputs = write_inputs(tmp_path / "in")
    import pandas as pd
    from tests.feature_store_fakes import _write_with_meta
    df = pd.read_csv(inputs["technical"])
    df.loc[10, "Target"] = 0.5
    _write_with_meta(inputs["technical"], df.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                     feature_version="technical_v2", raw_snapshot={"ticker": "AAPL"})
    with pytest.raises(FeatureStoreError, match="Target disagrees"):
        build(tmp_path, inputs)


# ==========================================
# Leakage through the join
# ==========================================


def test_future_article_cannot_change_earlier_feature_vectors(tmp_path):
    _, _, before, _ = build(tmp_path / "a")
    late = classified(9, ny(2024, 7, 12, 17, 0), ANALYST + " late")           # after the 07-12 16:30 boundary
    _, _, after, _ = build(tmp_path / "b", write_inputs(tmp_path / "b" / "in", default_items() + [late]))
    cut = date(2024, 7, 12)
    early_b = before[before["trading_date"] <= cut].reset_index(drop=True)
    early_a = after[after["trading_date"] <= cut].reset_index(drop=True)
    assert early_b.equals(early_a)                                             # sentiment AND event features
    nxt = date(2024, 7, 15)
    assert at(after, nxt)["sentiment__news_count"] == at(before, nxt)["sentiment__news_count"] + 1
    assert at(after, nxt)["event__analyst_rating_event_count"] == at(before, nxt)["event__analyst_rating_event_count"] + 1


# ==========================================
# Determinism & metadata
# ==========================================


def test_rebuild_is_byte_identical_and_order_independent(tmp_path):
    p1, m1, _, inputs = build(tmp_path)
    p2, m2, _, _ = build(tmp_path, inputs)                                    # same output dir: identical accepted
    rewrite_jsonl(inputs["sentiment"], lambda rows: rows.reverse())           # different input file order
    rewrite_jsonl(inputs["events"], lambda rows: rows.reverse())
    p3, m3, _, _ = build(tmp_path, inputs, out="other")
    assert p1.read_bytes() == p2.read_bytes() == p3.read_bytes()
    assert m1["sha256"] == m2["sha256"] == m3["sha256"] == file_sha256(p1)


def test_different_content_never_overwrites(tmp_path):
    build(tmp_path)
    changed = write_inputs(tmp_path / "in2", default_items()[:3])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(tmp_path, changed)


def test_metadata(tmp_path):
    _, meta, store, inputs = build(tmp_path)
    assert (meta["dataset_version"], meta["schema_version"], meta["grain"]) == (
        "feature_store_v1", "feature_store_schema_v1", ["symbol", "trading_date"])
    assert meta["family_counts"] == {"technical": 20, "sentiment": 15, "events": 23, "categorical": 1, "targets": 2}
    for fam in ("technical", "sentiment", "events"):
        assert meta["sources"][fam]["sha256"] == file_sha256(inputs[fam])
    assert meta["sources"]["calendar"]["file"] == inputs["snapshot"].name
    q = meta["data_quality"]
    assert (q["duplicate_keys"], q["null_keys"], q["infinite_values"], q["unexpected_categorical_values"]) == (0, 0, 0, [])
    assert q["coverage"]["all_three"] == len(NEWS_DATES) and q["symbols"] == ["AAPL"]
    assert meta["join_report"] == {"sentiment_rows_without_technical_row": 0, "event_rows_without_technical_row": 0}
    assert "generated_at" in meta and "content_hash_note" in meta


def test_news_rows_without_technical_rows_are_reported(tmp_path):
    inputs = write_inputs(tmp_path / "in")
    from tests.feature_store_fakes import write_technical
    inputs["technical"] = write_technical(tmp_path / "in" / "tech2",
                                          dates=[d for d in TECH_DATES if d >= date(2024, 7, 8)])
    _, meta, store, _ = build(tmp_path, inputs)
    assert meta["join_report"]["sentiment_rows_without_technical_row"] == 4  # 07-01, 02, 03, 05
    assert store["trading_date"].iloc[0] == date(2024, 7, 8)
