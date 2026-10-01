"""
Phase 6 end to end (offline): canonical news -> 5C sentiment (fake FinBERT) ->
event records -> reload + re-classification -> daily event dataset.
"""

import hashlib
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.event_classifier import KeywordEventClassifier
from services.event_features import EVENT_OUTPUT_COLUMNS
from services.historical_events import (
    RECORD_FIELDS,
    HistoricalEventError,
    load_event_dataset,
    run_historical_events,
)
from services.historical_sentiment import run_historical_sentiment
from tests.conftest import calendar_2024
from tests.sentiment_fakes import bars_for, fake_service, news, write_canonical
from training.build_dataset import meta_path_for

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
START, END = datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 13, tzinfo=UTC)
SESSIONS = [d for d in pd.bdate_range("2024-06-24", "2024-07-31").date if calendar_2024().is_session(d)]
BARS = bars_for(SESSIONS)
SOURCE = {"file": "fixture-bars.csv", "sha256": "0" * 64}

ARTICLES = [
    news(1, datetime(2024, 7, 8, 10, 0, tzinfo=NY), "Apple Reports Q1 EPS that beats estimates"),
    news(2, datetime(2024, 7, 8, 12, 0, tzinfo=NY), "Apple shares plunge as Qualcomm lawsuit widens"),
    news(3, datetime(2024, 7, 9, 9, 0, tzinfo=NY), "Inside The Hidden Economy Of Pawn Shops"),
    news(4, datetime(2024, 7, 10, 17, 0, tzinfo=NY), "Barclays downgrade hits Apple", symbols=("AAPL", "MSFT")),
    news(5, datetime(2024, 7, 13, 11, 0, tzinfo=NY), "Jabil to be supplier for next iPhone"),        # Saturday
]


def pipeline(tmp_path, articles=ARTICLES, classifier=None, out="events"):
    canonical = write_canonical(tmp_path / "news", articles, start=START, end=END)
    sentiment = run_historical_sentiment(canonical, BARS, fake_service(), calendar_source=SOURCE,
                                         output_dir=tmp_path / "sentiment")
    result = run_historical_events(canonical, sentiment["sentiment_path"], BARS, classifier or KeywordEventClassifier(),
                                   calendar_source=SOURCE, symbols=["AAPL", "MSFT"], output_dir=tmp_path / out)
    return canonical, sentiment["sentiment_path"], result


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def rewrite_consistently(path, mutate):
    rows = lines(path)
    mutate(rows)
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    path.write_bytes(payload.encode("utf-8"))
    meta_file = meta_path_for(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    meta_file.write_text(json.dumps(meta), encoding="utf-8")


# ==========================================
# Article-level dataset
# ==========================================


def test_article_records_schema_and_values(tmp_path):
    _, _, out = pipeline(tmp_path)
    recs = lines(out["event_path"])
    assert len(recs) == 5
    assert all(tuple(sorted(r)) == tuple(sorted(RECORD_FIELDS)) for r in recs)
    by_id = {r["provider_article_id"]: r for r in recs}
    assert (by_id["1"]["event_type"], by_id["1"]["event_impact"]) == ("EARNINGS", "POSITIVE")
    assert (by_id["2"]["event_type"], by_id["2"]["event_impact"]) == ("LEGAL", "NEGATIVE")   # type != impact
    assert (by_id["3"]["event_type"], by_id["3"]["event_confidence"]) == ("OTHER", 0.0)
    assert by_id["4"]["symbols"] == ["AAPL", "MSFT"] and by_id["4"]["event_type"] == "ANALYST_RATING"
    assert by_id["5"]["event_type"] == "OPERATIONS"
    assert by_id["4"]["information_available_at"] == "2024-07-10T21:00:00+00:00"
    assert all(r["classifier_version"] == "rules_v1" and r["taxonomy_version"] == "taxonomy_v1" for r in recs)
    assert "headline" not in recs[0] and "content" not in recs[0]                    # no article text copied


def test_metadata_provenance_and_data_quality(tmp_path):
    canonical, sentiment_path, out = pipeline(tmp_path)
    m = out["event_meta"]
    assert m["classifier"] == {"classifier_name": "keyword_event_classifier", "classifier_version": "rules_v1",
                               "taxonomy_version": "taxonomy_v1"}
    assert m["impact_source"]["file"] == sentiment_path.name and m["impact_source"]["sha256"]
    assert m["impact_source"]["sentiment_provenance"]["model_name"] == "ProsusAI/finbert"
    assert m["labels_are_ground_truth"] is False
    q = m["data_quality"]
    assert (q["n_input_articles"], q["n_articles"], q["n_duplicates_collapsed"], q["n_events"], q["n_other"]) == (5, 5, 0, 4, 1)
    assert sum(q["event_type_distribution"].values()) == 5
    assert q["symbol_distribution"] == {"AAPL": 5, "MSFT": 1}
    assert "not a probability" in q["event_confidence"]["meaning"]


# ==========================================
# Daily dataset
# ==========================================


def test_daily_rows_follow_availability_and_calendar(tmp_path):
    _, _, out = pipeline(tmp_path)
    rows = {(r["symbol"], r["trading_date"]): r for r in lines(out["daily_path"])}
    assert tuple(lines(out["daily_path"])[0]) == tuple(sorted(EVENT_OUTPUT_COLUMNS))     # sorted JSON keys
    a8, a10, a11 = rows[("AAPL", "2024-07-08")], rows[("AAPL", "2024-07-10")], rows[("AAPL", "2024-07-11")]
    assert (a8["earnings_event_count"], a8["legal_event_count"], a8["event_count"]) == (1, 1, 2)
    assert a10["event_count"] == 0                                                  # 17:00 article not yet eligible
    assert a11["analyst_rating_event_count"] == 1 and rows[("MSFT", "2024-07-11")]["event_count"] == 1
    assert ("AAPL", "2024-07-13") not in rows                                       # Saturday: no row
    assert rows[("AAPL", "2024-07-12")]["operations_event_count"] == 0             # Saturday news not yet
    assert out["daily_meta"]["source_events"]["sha256"] == out["event_meta"]["sha256"]


# ==========================================
# Determinism & validation
# ==========================================


def test_rerun_is_identical_and_separate_runs_match(tmp_path):
    _, _, first = pipeline(tmp_path)
    _, _, again = pipeline(tmp_path)                                              # same dir: identical bytes accepted
    _, _, other = pipeline(tmp_path / "second")
    for key in ("event_path", "daily_path"):
        assert first[key].read_bytes() == again[key].read_bytes() == other[key].read_bytes()


def test_tampered_event_file_rejected(tmp_path):
    canonical, sentiment_path, out = pipeline(tmp_path)
    path = out["event_path"]
    path.write_bytes(path.read_bytes().replace(b"EARNINGS", b"GUIDANCE", 1))
    with pytest.raises(HistoricalEventError, match="hash mismatch"):
        load_event_dataset(path, sentiment_path, canonical)


def test_relabelled_record_fails_reclassification(tmp_path):
    canonical, sentiment_path, out = pipeline(tmp_path)
    rewrite_consistently(out["event_path"], lambda rows: rows[0].update(event_type="GUIDANCE"))
    load_event_dataset(out["event_path"], sentiment_path, canonical)                # structurally valid...
    with pytest.raises(HistoricalEventError, match="does not reproduce"):
        load_event_dataset(out["event_path"], sentiment_path, canonical, verify_with=KeywordEventClassifier())


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(event_type="NEWS"), "invalid event record"),
    (lambda r: r.update(event_impact="MIXED"), "invalid event record"),
    (lambda r: r.update(event_confidence=1.7), "invalid event record"),
    (lambda r: r.update(event_impact="NEGATIVE" if r["event_impact"] != "NEGATIVE" else "POSITIVE"), "impact"),
    (lambda r: r.update(information_available_at="2024-07-01T00:00:00+00:00"), "does not match"),
    (lambda r: r.update(provider_article_id="999"), "no scored canonical article"),
    (lambda r: r.update(classifier_version="rules_v9"), "classifier provenance"),
    (lambda r: r.pop("input_text_hash"), "missing"),
])
def test_invalid_records_rejected(tmp_path, mutate, message):
    canonical, sentiment_path, out = pipeline(tmp_path)
    rewrite_consistently(out["event_path"], lambda rows: mutate(rows[0]))
    with pytest.raises(HistoricalEventError, match=message):
        load_event_dataset(out["event_path"], sentiment_path, canonical)


def test_wrong_sentiment_lineage_rejected(tmp_path):
    canonical, _, out = pipeline(tmp_path)
    _, other_sentiment, _ = pipeline(tmp_path / "other", articles=ARTICLES[:3])
    with pytest.raises(ValueError, match="does not match"):           # 5C lineage check fires first
        load_event_dataset(out["event_path"], other_sentiment, canonical)


def test_changed_rules_with_same_version_never_overwrite(tmp_path):
    pipeline(tmp_path)

    class Tweaked(KeywordEventClassifier):
        def classify_text(self, text):
            p = super().classify_text(text)
            return p if p.event_type != "LEGAL" else super().classify_text("Option Alert: x")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pipeline(tmp_path, classifier=Tweaked())


def test_empty_dataset(tmp_path):
    _, _, out = pipeline(tmp_path, articles=[])
    assert out["event_path"].read_bytes() == b"" and out["event_meta"]["data_quality"]["n_articles"] == 0
    assert all(r["event_count"] == 0 and r["dominant_event_type"] == "NONE" for r in lines(out["daily_path"]))
