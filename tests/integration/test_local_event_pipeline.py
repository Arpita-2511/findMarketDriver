"""
Phase 6 on the REAL local datasets (opt-in; no network, no model, no keys):
the newest 4C canonical news dataset, its 5C sentiment dataset and the
newest Phase 2 market snapshot. Outputs go to a pytest temp directory; the
real datasets are only read. Skipped if the files are not present.

    pytest -m integration tests/integration/test_local_event_pipeline.py -s
"""

import json
import time
from pathlib import Path

import pytest

from services.event_taxonomy import EVENT_TYPES
from services.historical_events import run_historical_events
from training.build_dataset import load_raw_snapshot

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _local_inputs():
    news = sorted((ROOT / "data" / "processed" / "news").glob("*.jsonl"))
    if not news:
        pytest.skip("no canonical news dataset in data/processed/news")
    canonical = news[-1]
    sentiment = sorted((ROOT / "data" / "processed" / "sentiment").glob(canonical.stem + ".finbert-*.sentiment.jsonl"))
    snapshots = sorted((ROOT / "data" / "raw" / "stocks").glob("*.csv"))
    if not sentiment or not snapshots:
        pytest.skip("matching 5C sentiment dataset or market snapshot not found")
    return canonical, sentiment[-1], snapshots[-1]


def test_event_pipeline_on_local_data_is_valid_and_reproducible(tmp_path):
    canonical, sentiment, snapshot = _local_inputs()
    bars, snap_meta = load_raw_snapshot(snapshot)
    source = {"file": snapshot.name, "sha256": snap_meta["sha256"]}

    started = time.perf_counter()
    a = run_historical_events(canonical, sentiment, bars, calendar_source=source, output_dir=tmp_path / "a")
    elapsed = time.perf_counter() - started
    b = run_historical_events(canonical, sentiment, bars, calendar_source=source, output_dir=tmp_path / "b")

    q = a["event_meta"]["data_quality"]
    sentiment_rows = json.loads(sentiment.with_suffix(".meta.json").read_text(encoding="utf-8"))["rows"]
    assert q["n_articles"] == sentiment_rows and q["n_duplicates_collapsed"] == 0
    assert set(q["event_type_distribution"]) == set(EVENT_TYPES)
    assert sum(q["event_type_distribution"].values()) == q["n_articles"]
    assert a["event_path"].read_bytes() == b["event_path"].read_bytes()             # deterministic
    assert a["daily_path"].read_bytes() == b["daily_path"].read_bytes()

    print(f"\narticles={q['n_articles']} events={q['n_events']} other={q['n_other']} "
          f"daily_rows={a['daily_meta']['rows']} classify={a['classify_seconds']:.3f}s total={elapsed:.2f}s")
    print("types: " + ", ".join(f"{t}={n}" for t, n in q["event_type_distribution"].items() if n))
    print(f"impacts: {q['event_impact_distribution']}  confidence: {q['event_confidence']}")
