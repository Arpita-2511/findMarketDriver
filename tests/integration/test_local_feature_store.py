"""
Phase 7 on the REAL local datasets (opt-in; no network, no model, no keys).
Builds the feature store twice into a temp directory, checks the content hash
is reproducible, and runs the incremental A/B/C evaluation. Skipped if inputs
are missing.

    pytest -m integration tests/integration/test_local_feature_store.py -s
"""

import json
from pathlib import Path

import pytest

from training.feature_store import build_feature_store, load_feature_store
from training.incremental_evaluation import run_incremental_evaluation

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _inputs():
    technical = ROOT / "data" / "final_stock_dataset.csv"
    sentiment = sorted((ROOT / "data" / "processed" / "sentiment").glob("*.daily.jsonl"))
    events = sorted((ROOT / "data" / "processed" / "events").glob("*.daily.jsonl"))
    if not technical.is_file() or not sentiment or not events:
        pytest.skip("technical / sentiment daily / event daily datasets not found")
    calendar = json.loads(events[-1].with_suffix(".meta.json").read_text(encoding="utf-8"))["calendar_source"]["file"]
    snapshot = ROOT / "data" / "raw" / "stocks" / calendar
    if not snapshot.is_file():
        pytest.skip(f"market snapshot {calendar} not found")
    return technical, sentiment[-1], events[-1], snapshot


def test_feature_store_on_local_data(tmp_path):
    technical, sentiment, events, snapshot = _inputs()
    p1, m1 = build_feature_store(technical, sentiment, events, snapshot, tmp_path / "a")
    p2, m2 = build_feature_store(technical, sentiment, events, snapshot, tmp_path / "b")
    assert p1.read_bytes() == p2.read_bytes() and m1["sha256"] == m2["sha256"]

    q = m1["data_quality"]
    assert q["duplicate_keys"] == 0 and q["infinite_values"] == 0 and q["unexpected_categorical_values"] == []
    store, _ = load_feature_store(p1)
    report = run_incremental_evaluation(store)

    print(f"\nrows={q['rows']} columns={m1['column_count']} dates={q['date_range']} families={m1['family_counts']}")
    print(f"coverage={q['coverage']} join={m1['join_report']}")
    print(f"constant features={len(q['constant_features'])}  sha256={m1['sha256']}")
    print(f"evaluation: {report['status']} evaluable_rows={report['evaluable_rows']} period={report['evaluable_period']}")
    print(report.get("reason") or json.dumps(report.get("comparison"), indent=2))
