"""
Phase 5C: historical FinBERT scoring -> article sentiment dataset -> daily sentiment dataset.
Offline: fake FinBERT backend, local fixtures, no network, no model download.
"""

import json
import math
import random
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from services.finbert_sentiment import INFERENCE_VERSION, TEXT_POLICY, SentimentResult, text_hash
from services.historical_sentiment import (
    HistoricalSentimentError,
    SentimentRecord,
    backend_provenance,
    check_consistent_provenance,
    load_sentiment_dataset,
    run_historical_sentiment,
    save_sentiment_dataset,
    score_canonical_articles,
    serialize_records,
    trading_dates_from_bars,
)
from services.news_alignment import FutureInformationError
from services.sentiment_features import FEATURE_COLUMNS, OUTPUT_COLUMNS
from tests.conftest import calendar_2024
from tests.sentiment_fakes import bars_for, fake_service, news, write_canonical
from training.build_dataset import meta_path_for

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
START, END = datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 20, tzinfo=UTC)
SESSIONS = [d for d in pd.bdate_range("2024-06-24", "2024-07-31").date if calendar_2024().is_session(d)]
BARS = bars_for(SESSIONS)
SOURCE = {"file": "fixture-bars.csv", "sha256": "0" * 64}


def ny(*args):
    return datetime(*args, tzinfo=NY)


def run(tmp_path, articles, service=None, **kwargs):
    canonical = write_canonical(tmp_path / "news", articles, start=START, end=END)
    out = run_historical_sentiment(canonical, BARS, service or fake_service(), calendar_source=SOURCE,
                                   output_dir=tmp_path / "out", **kwargs)
    return canonical, out


def daily(out):
    lines = out["daily_path"].read_text(encoding="utf-8").splitlines()
    return {row["trading_date"]: row for row in map(json.loads, lines)}


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rewrite(path, mutate):
    """Rewrite the sentiment file with `mutate` applied AND a matching hash - simulates a consistent-looking bad file."""
    rows = records(path)
    mutate(rows)
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    path.write_bytes(payload.encode("utf-8"))
    meta_file = meta_path_for(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    import hashlib
    meta["sha256"], meta["rows"] = hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)
    meta_file.write_text(json.dumps(meta), encoding="utf-8")


# ==========================================
# A-C. Empty, one, many
# ==========================================


def test_A_empty_dataset(tmp_path):
    _, out = run(tmp_path, [])
    assert out["sentiment_path"].read_bytes() == b"" and out["sentiment_meta"]["rows"] == 0
    rows = daily(out)
    assert len(rows) == len([d for d in SESSIONS if date(2024, 7, 1) <= d < date(2024, 7, 20)])
    assert all(r["news_count"] == 0 for r in rows.values())


def test_B_one_article(tmp_path):
    _, out = run(tmp_path, [news(1, ny(2024, 7, 10, 10, 0), "Apple beats estimates")])
    [rec] = records(out["sentiment_path"])
    assert (rec["provider"], rec["provider_article_id"], rec["label"]) == ("alpaca", "1", "positive")
    assert rec["symbols"] == ["AAPL"] and rec["information_available_at"] == "2024-07-10T14:00:00+00:00"
    assert daily(out)["2024-07-10"]["news_count"] == 1 and daily(out)["2024-07-09"]["news_count"] == 0


def test_C_multiple_articles(tmp_path):
    arts = [news(1, ny(2024, 7, 8, 9, 0), "Apple beats"), news(2, ny(2024, 7, 9, 9, 0), "Apple shares plunge"),
            news(3, ny(2024, 7, 10, 9, 0), "Apple meeting")]
    _, out = run(tmp_path, arts)
    assert [r["label"] for r in records(out["sentiment_path"])] == ["positive", "negative", "neutral"]
    row = daily(out)["2024-07-10"]
    assert (row["news_count"], row["positive_count"], row["negative_count"], row["neutral_count"]) == (3, 1, 1, 1)


# ==========================================
# D-G. Determinism, batching, duplicates
# ==========================================

ARTS = [news(i, ny(2024, 7, 8 + i % 5, 9 + i % 7, 0), ["Apple beats", "Apple shares plunge", "Apple meeting"][i % 3])
        for i in range(12)]


def test_D_deterministic_ordering_regardless_of_input_order():
    reference = serialize_records(score_canonical_articles(ARTS, fake_service())[0])
    rng = random.Random(0)
    for _ in range(5):
        shuffled = ARTS[:]
        rng.shuffle(shuffled)
        assert serialize_records(score_canonical_articles(shuffled, fake_service())[0]) == reference


def test_E_batch_size_does_not_change_output():
    outputs = {bs: serialize_records(score_canonical_articles(ARTS, fake_service(batch_size=bs))[0]) for bs in (1, 3, 16)}
    assert outputs[1] == outputs[3] == outputs[16]


def test_F_identical_duplicate_scored_once():
    svc = fake_service()
    scored, stats = score_canonical_articles([ARTS[0], ARTS[0], ARTS[1]], svc)
    assert len(scored) == 2 and stats["n_duplicates_collapsed"] == 1
    assert sum(len(c) for c in svc.backend.calls) == 2                       # no double inference


def test_G_conflicting_duplicate_raises():
    other = news(0, ARTS[0].created_at, "Different headline, same id")
    with pytest.raises(HistoricalSentimentError, match="conflicting"):
        score_canonical_articles([ARTS[0], other], fake_service())


# ==========================================
# H-M. Record validation & provenance
# ==========================================


def record_dict(**overrides):
    a = news(1, ny(2024, 7, 10, 10, 0), "Apple beats")
    d = {"record_version": "sentiment_records_v1", "provider": "alpaca", "provider_article_id": "1",
         "symbols": ["AAPL"], "information_available_at": a.information_available_at.isoformat(),
         "inference_version": INFERENCE_VERSION, "model_name": "ProsusAI/finbert", "model_revision": "rev1",
         "text_policy": TEXT_POLICY, "label": "positive", "positive_probability": 0.7,
         "negative_probability": 0.1, "neutral_probability": 0.2, "sentiment_score": 0.6,
         "input_text_hash": text_hash("Apple beats")}
    d.update(overrides)
    return d


def test_valid_record_round_trip():
    rec = SentimentRecord.from_dict(record_dict())
    assert SentimentRecord.from_dict(rec.to_dict()) == rec


@pytest.mark.parametrize("overrides, message", [
    ({"positive_probability": 0.9}, "sum"),                                              # H
    ({"positive_probability": 1.5, "negative_probability": -0.7}, r"\[0, 1\]"),
    ({"label": "negative"}, "most probable"),                                            # I
    ({"label": "bullish"}, "unknown label"),
    ({"sentiment_score": 0.1}, "sentiment_score"),
    ({"input_text_hash": "not-a-hash"}, "sha256"),
])
def test_H_I_invalid_sentiment_values_rejected(overrides, message):
    with pytest.raises(HistoricalSentimentError, match=message):
        SentimentRecord.from_dict(record_dict(**overrides))


@pytest.mark.parametrize("field", ["model_revision", "model_name", "inference_version", "text_policy",
                                   "input_text_hash", "information_available_at"])
def test_K_missing_provenance_rejected(field):
    d = record_dict()
    del d[field]
    with pytest.raises(HistoricalSentimentError, match="missing"):
        SentimentRecord.from_dict(d)


def test_K_backend_without_revision_rejected():
    with pytest.raises(HistoricalSentimentError, match="missing sentiment provenance"):
        backend_provenance(fake_service(revision=None))


def _result(**overrides):
    fields = dict(inference_version=INFERENCE_VERSION, model_name="ProsusAI/finbert", model_revision="rev1",
                  label="positive", positive_probability=0.7, negative_probability=0.1, neutral_probability=0.2,
                  sentiment_score=0.6, input_text_hash="0" * 64, text_policy=TEXT_POLICY)
    fields.update(overrides)
    return SentimentResult(**fields)


EXPECTED = {"model_name": "ProsusAI/finbert", "model_revision": "rev1",
            "inference_version": INFERENCE_VERSION, "text_policy": TEXT_POLICY}


def test_L_mixed_model_revisions_rejected():
    with pytest.raises(HistoricalSentimentError, match="mixed sentiment provenance"):
        check_consistent_provenance([_result(), _result(model_revision="rev2")], EXPECTED)


def test_M_mixed_text_policies_rejected():
    with pytest.raises(HistoricalSentimentError, match="mixed sentiment provenance"):
        check_consistent_provenance([_result(), _result(text_policy="text_v2: something else")], EXPECTED)


def test_L_save_refuses_results_from_another_revision(tmp_path):
    scored, stats = score_canonical_articles(ARTS[:2], fake_service(revision="revAAA"))
    canonical = write_canonical(tmp_path, ARTS[:2], start=START, end=END)
    with pytest.raises(HistoricalSentimentError, match="mixed"):
        save_sentiment_dataset(scored, stats, dict(EXPECTED, model_revision="revBBB"), canonical, {"sha256": "x"},
                               tmp_path / "out")


# ==========================================
# J, T. Stored-file validation
# ==========================================


def test_J_wrong_input_text_hash_rejected_on_reload(tmp_path):
    canonical, out = run(tmp_path, ARTS[:3])
    rewrite(out["sentiment_path"], lambda rows: rows[0].update(input_text_hash=text_hash("some other text")))
    with pytest.raises(HistoricalSentimentError, match="not computed from this article"):
        load_sentiment_dataset(out["sentiment_path"], canonical)


def test_wrong_article_identity_rejected_on_reload(tmp_path):
    canonical, out = run(tmp_path, ARTS[:3])
    rewrite(out["sentiment_path"], lambda rows: rows[0].update(provider_article_id="999"))
    with pytest.raises(HistoricalSentimentError, match="no canonical article"):
        load_sentiment_dataset(out["sentiment_path"], canonical)

    canonical2, out2 = run(tmp_path / "b", ARTS[:3])
    rewrite(out2["sentiment_path"], lambda rows: rows[0].update(information_available_at="2024-07-01T00:00:00+00:00"))
    with pytest.raises(HistoricalSentimentError, match="does not match its canonical article"):
        load_sentiment_dataset(out2["sentiment_path"], canonical2)


def test_T_file_hash_and_source_dataset_verified(tmp_path):
    canonical, out = run(tmp_path, ARTS[:3])
    path = out["sentiment_path"]
    original = path.read_bytes()

    path.write_bytes(original.replace(b'"alpaca"', b'"alpacA"', 1))              # tampered, hash not updated
    with pytest.raises(HistoricalSentimentError, match="hash mismatch"):
        load_sentiment_dataset(path, canonical)
    path.write_bytes(original)

    other = write_canonical(tmp_path / "other", ARTS[:4], start=START, end=END)  # a different canonical dataset
    with pytest.raises(HistoricalSentimentError, match="does not match the one"):
        load_sentiment_dataset(path, other)

    meta_file = meta_path_for(path)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["provenance"].pop("model_revision")
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(HistoricalSentimentError, match="missing sentiment provenance"):
        load_sentiment_dataset(path, canonical)


def test_T_metadata_contents_and_content_identity(tmp_path):
    canonical, out = run(tmp_path, ARTS)
    m = out["sentiment_meta"]
    assert m["provenance"] == {"model_name": "ProsusAI/finbert", "model_revision": "fakerev000001",
                               "inference_version": INFERENCE_VERSION, "text_policy": TEXT_POLICY}
    assert (m["n_input_articles"], m["n_scored_articles"], m["rows"]) == (12, 12, 12)
    assert m["source_dataset"]["file"] == canonical.name and m["ordering"]
    assert "generated_at" in m and "generated_at" not in out["sentiment_path"].read_text(encoding="utf-8")
    assert "finbert-fakerev00000" in out["sentiment_path"].name                      # revision in the name

    _, again = run(tmp_path, ARTS)                                                     # identical rerun: allowed
    assert again["sentiment_meta"]["sha256"] == m["sha256"]
    assert again["daily_meta"]["sha256"] == out["daily_meta"]["sha256"]


def test_different_content_is_never_silently_overwritten(tmp_path):
    run(tmp_path, ARTS[:3])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(tmp_path, ARTS[:3], service=fake_service(bias=5.0))                       # same revision, other values


def test_S_serialization_and_reload(tmp_path):
    canonical, out = run(tmp_path, ARTS)
    reloaded, meta = load_sentiment_dataset(out["sentiment_path"], canonical)
    scored, _ = score_canonical_articles(ARTS, fake_service())
    assert reloaded == scored and meta["sha256"] == out["sentiment_meta"]["sha256"]

    lines = out["daily_path"].read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert tuple(sorted(first)) == tuple(sorted(OUTPUT_COLUMNS))
    assert first["prediction_timestamp"].endswith("+00:00")                         # timezone-aware, UTC
    assert first["sentiment_feature_version"] == "sentiment_features_v1"
    assert out["daily_meta"]["source_sentiment"]["sha256"] == out["sentiment_meta"]["sha256"]


# ==========================================
# N-R. Future news, leakage, calendar, zero news
# ==========================================


def test_N_impossible_future_article_rejected_before_inference():
    bad = news(1, ny(2024, 7, 10, 10, 0), fetched=ny(2024, 7, 10, 9, 0))
    svc = fake_service()
    with pytest.raises(FutureInformationError):
        score_canonical_articles([bad], svc)
    assert svc.backend.calls == []


def test_O_aggregation_uses_information_available_at(tmp_path):
    revised = news(1, ny(2024, 7, 10, 10, 0), "Apple beats", updated=ny(2024, 7, 10, 17, 0))
    rows = daily(run(tmp_path, [revised])[1])
    assert rows["2024-07-10"]["news_count"] == 0                                    # created 10:00, available 17:00
    assert rows["2024-07-11"]["news_count"] == 1


def test_P_post_prediction_article_excluded(tmp_path):
    arts = [news(1, ny(2024, 7, 10, 16, 30), "Apple beats"), news(2, ny(2024, 7, 10, 16, 31), "Apple shares plunge")]
    rows = daily(run(tmp_path, arts)[1])
    assert (rows["2024-07-10"]["news_count"], rows["2024-07-10"]["negative_count"]) == (1, 0)
    assert rows["2024-07-11"]["news_count"] == 2


def test_Q_weekend_and_holiday_alignment(tmp_path):
    arts = [news(1, ny(2024, 7, 13, 11, 0), "Apple beats"),          # Saturday
            news(2, ny(2024, 7, 4, 12, 0), "Apple shares plunge")]   # Independence Day
    rows = daily(run(tmp_path, arts)[1])
    assert "2024-07-04" not in rows and "2024-07-13" not in rows    # no artificial rows
    assert rows["2024-07-03"]["news_count"] == 0 and rows["2024-07-05"]["news_count"] == 1
    assert rows["2024-07-12"]["news_count"] == 1 and rows["2024-07-15"]["news_count"] == 2


def test_R_zero_news_rows_are_zero_not_nan(tmp_path):
    rows = daily(run(tmp_path, [news(1, ny(2024, 7, 15, 10, 0), "Apple beats")])[1])
    early = rows["2024-07-01"]
    for name in FEATURE_COLUMNS:
        assert early[name] == 0 and not (isinstance(early[name], float) and math.isnan(early[name]))


def test_trading_dates_come_from_bars_half_open():
    dates = trading_dates_from_bars(BARS, date(2024, 7, 3), date(2024, 7, 8))
    assert dates == [date(2024, 7, 3), date(2024, 7, 5)]            # 07-04 holiday, weekend absent, 07-08 excluded
