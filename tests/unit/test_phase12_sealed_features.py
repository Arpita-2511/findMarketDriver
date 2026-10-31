"""
Phase 12 Stage 3B sealed feature builder - synthetic tests.

News comes from the synthetic 11B.3 environment (fake FinBERT, rules_v1, synthetic canonical file); prices are
fabricated OHLCV bars. No real sealed record is read; the only real-data tests check hashes and pins.
"""

from __future__ import annotations

import hashlib
import json
import types
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import training.phase11a_research as a11
import training.phase11b_feature_build as fb
import training.phase12_confirmation as c
import training.phase12_magnitude as m2
import training.phase12_sealed_features as s
from services.event_classifier import KeywordEventClassifier
from services.historical_events import classify_scored_articles
from services.historical_sentiment import score_canonical_articles
from services.market_calendar_service import TradingCalendar
from services.news_alignment import row_prediction_timestamp
from tests.sentiment_fakes import news
from services.news_dataset import sort_key
from tests.unit.test_phase11b_feature_build import DEV_DATES, articles, build_env
from tests.unit.test_phase11b_sentiment_scoring import CUT, H, US, fake, make_env
from training.build_dataset import file_sha256
from training.phase11b_sentiment_scoring import partition_canonical

SYMS = ("AAPL", "RTX")
DATES = pd.bdate_range("2023-06-01", "2024-09-30")


def ohlcv(seed=0, dates=DATES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame({"Date": pd.to_datetime(dates).astype("datetime64[ns]"), "Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1_000_000, 2_000_000, len(dates)).astype("float64")})


def all_bars(dates=DATES) -> dict:
    return {sym: ohlcv(i, dates) for i, sym in enumerate((*SYMS, "SPY", "QQQ"))}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = build_env(tmp_path_factory.mktemp("s3b"), dev=sorted(articles(), key=sort_key))   # canonical order
    e.news = s.development_news(e.report, e.out, e.events, e.records)
    return e


def frames_for(env, bars=None, start=None, end=None, coverage_end=s.DEV_COVERAGE_END):
    scored, classified, _ = env.news
    return s.build_frames(bars or all_bars(), scored, classified, SYMS, start or DATES[200].date(),
                          end or DATES[-1].date(), coverage_end)


# ============================================================
# Schema and generator reuse
# ============================================================


def test_schema_and_order_match_the_finalist(env):
    assert s.OUTPUT_COLUMNS == ("Date", "Close", "SPY_Close", *m2.FEATURES)
    assert c.rules()["finalist"]["features_in_order"] == list(m2.FEATURES)
    assert set(s.PRICE_FEATURES) | set(s.NEWS_FEATURES) == set(m2.FEATURES)
    f = frames_for(env)["AAPL"]
    assert list(f.columns) == list(s.OUTPUT_COLUMNS) and f["Date"].dtype == "datetime64[ns]"
    assert all(f[col].dtype == np.float64 for col in s.OUTPUT_COLUMNS[1:])


def test_price_features_equal_the_11a_stock_frame():
    b = all_bars()
    ours = s.price_features(b["AAPL"], b["SPY"], b["QQQ"], s.DEV_START, s.DEV_END)
    theirs = a11.stock_frame(b["AAPL"], b["SPY"], b["QQQ"])
    assert ours["Date"].equals(theirs["Date"])
    for col in ("Close", "SPY_Close", *s.PRICE_FEATURES):
        assert np.array_equal(ours[col].to_numpy(), theirs[col].to_numpy()), col


def test_news_features_equal_the_11b3_news_features(env):
    scored, classified, _ = env.news
    _, calendars = env.prices()
    for sym in SYMS:
        ours = s.news_features(scored, classified, sym, DEV_DATES, calendars[sym], s.DEV_COVERAGE_END)
        theirs = fb.news_features(scored, classified, sym, DEV_DATES, calendars[sym])
        for col in s.NEWS_FEATURES:
            assert np.array_equal(ours[col].to_numpy(), theirs[col].to_numpy()), (sym, col)


def test_development_prefix_equals_the_frozen_partition_and_never_parses_the_sealed_tail(tmp_path):
    dev = [news(1, CUT - 5 * H), news(2, CUT - 2 * H, updated=CUT + H), news(3, CUT)]   # 2: revised after the cutoff
    e = make_env(tmp_path, dev=[dev[0], dev[2]], sealed=[news(4, CUT + US)])
    lines = [json.dumps(a.to_dict(), sort_keys=True) for a in (dev[0], dev[1], dev[2], news(4, CUT + US))]
    path = tmp_path / "sorted.jsonl"
    payload = ("\n".join(lines) + "\n" + "{not json: sealed tail is only hashed}\n").encode()
    path.write_bytes(payload)
    got, stats = s.development_prefix(path, hashlib.sha256(payload).hexdigest())
    assert [a.provider_article_id for a in got] == ["1", "3"]
    assert stats == {"lines_parsed": 3, "revised_after_cutoff_dropped": 1, "stopped_at_line": 4}
    frozen = partition_canonical(e.canonical, expected_sha256=file_sha256(e.canonical)).development
    assert [a.article_key for a in frozen] == [a.article_key for a in got]
    with pytest.raises(s.SealedFeatureError, match="SHA256"):
        s.development_prefix(path, "0" * 64)


def test_development_prefix_refuses_an_unsorted_file(tmp_path):
    path = tmp_path / "unsorted.jsonl"
    path.write_text("".join(json.dumps(a.to_dict()) + "\n" for a in (news(1, CUT - H), news(2, CUT - 3 * H))))
    with pytest.raises(s.SealedFeatureError, match="created_at order"):
        s.development_prefix(path, file_sha256(path))


def test_read_bars_until_equals_load_bars_and_stops_at_the_end(tmp_path):
    b = ohlcv(3)
    raw = b.assign(Date=b["Date"].dt.strftime("%Y-%m-%d"), Dividends=0.0, Stock_Splits=0.0)
    path = tmp_path / "bars.csv"
    raw.to_csv(path, index=False, float_format="%.17g")
    end = DATES[100].date()
    want = a11.load_bars(str(path))
    want = want[want["Date"].dt.date <= end].reset_index(drop=True)
    assert s.read_bars_until(path, end).equals(want)
    with open(path, "a", encoding="utf-8") as f:
        f.write("2099-01-01,garbage,never,parsed\nnot,a,csv,line,at,all\n")
    assert s.read_bars_until(path, end).equals(want)


# ============================================================
# Point in time, alignment, missing values, determinism
# ============================================================


def test_future_prices_never_change_earlier_rows():
    b = all_bars()
    cut = DATES[250].date()
    early = {k: v[v["Date"].dt.date <= cut].reset_index(drop=True) for k, v in b.items()}
    late = {k: v.copy() for k, v in b.items()}
    for v in late.values():
        v.loc[v["Date"].dt.date > cut, ["Open", "High", "Low", "Close"]] *= 3.0
        v.loc[v["Date"].dt.date > cut, "Volume"] *= 7.0
    a = s.price_features(early["AAPL"], early["SPY"], early["QQQ"], DATES[60].date(), cut)
    z = s.price_features(late["AAPL"], late["SPY"], late["QQQ"], DATES[60].date(), DATES[-1].date())
    z = z[z["Date"].dt.date <= cut].reset_index(drop=True)
    assert a.equals(z)
    with pytest.raises(s.SealedFeatureError, match="cut at"):
        s.price_features(late["AAPL"], late["SPY"], late["QQQ"], DATES[60].date(), cut)


def test_later_articles_never_change_earlier_rows_and_the_boundary_is_inclusive(env):
    scored, classified, _ = env.news
    cal = TradingCalendar.from_bars(ohlcv())
    dates = [d.date() for d in DATES[300:]]
    d = dates[10]
    ts = row_prediction_timestamp(d, cal)
    extra = [news(101, ts, headline="Acme beats EPS estimates", symbols=("AAPL",)),
             news(102, ts + US, headline="Acme beats EPS estimates", symbols=("AAPL",))]
    sc, _ = score_canonical_articles(extra, fake())
    cl = classify_scored_articles(sc, KeywordEventClassifier())
    base = s.news_features(scored, classified, "AAPL", dates, cal, s.DEV_COVERAGE_END)
    more = s.news_features([*scored, *sc], [*classified, *cl], "AAPL", dates, cal, s.DEV_COVERAGE_END)
    before = base["trading_date"] < d
    assert base[before].equals(more[before])
    i = dates.index(d)
    assert more["event__event_count"][i] - base["event__event_count"][i] == 1            # available exactly at ts(D)
    assert more["event__event_count"][i + 1] - base["event__event_count"][i + 1] == 1    # 1 microsecond later
    # adding only the later article changes nothing at D
    only_late = s.news_features([*scored, sc[1]], [*classified, cl[1]], "AAPL", dates, cal, s.DEV_COVERAGE_END)
    assert only_late.loc[: i, s.NEWS_FEATURES].equals(base.loc[: i, s.NEWS_FEATURES])


def test_negative_event_count_follows_the_finbert_label(env):
    scored, classified, _ = env.news
    cal = TradingCalendar.from_bars(ohlcv())
    dates = [d.date() for d in DATES[300:]]
    ts = row_prediction_timestamp(dates[5], cal)
    sc, _ = score_canonical_articles([news(201, ts - H, headline="Acme misses EPS estimates; shares plunge",
                                           symbols=("AAPL",))], fake())
    cl = classify_scored_articles(sc, KeywordEventClassifier())
    out = s.news_features(sc, cl, "AAPL", dates, cal, s.DEV_COVERAGE_END)
    assert cl[0].is_event and cl[0].event_impact == "NEGATIVE"
    assert out["event__negative_event_count"][5] == 1.0 and out["event__event_count"][5] == 1.0


def test_uncovered_news_windows_stay_missing_and_counts_stay_zero(env):
    end = DATES[-1].date()
    cov_end = row_prediction_timestamp(DATES[-6].date(), TradingCalendar.from_bars(ohlcv())) + timedelta(minutes=1)
    f = frames_for(env, coverage_end=cov_end)["AAPL"]
    late = f["Date"].dt.date > DATES[-6].date()
    assert late.sum() == 5 and f.loc[late, "sw_count_surprise_1_20"].isna().all()
    assert f.loc[~late, "sw_count_surprise_1_20"].notna().all()
    assert f[list(s.NEWS_FEATURES[:3])].notna().all().all()                 # zero counts are values, never NaN
    assert f["Date"].dt.date.max() == end


def test_stock_and_spy_sessions_must_align():
    b = all_bars()
    b["SPY"] = b["SPY"].drop(index=280).reset_index(drop=True)
    with pytest.raises(Exception, match="missing"):
        s.build_frames(b, [], [], SYMS, DATES[200].date(), DATES[-1].date(), s.DEV_COVERAGE_END)


def test_build_is_deterministic_and_input_order_free(env):
    scored, classified, _ = env.news
    a = frames_for(env)
    b = s.build_frames(all_bars(), list(reversed(scored)), list(reversed(classified)), list(reversed(SYMS)),
                       DATES[200].date(), DATES[-1].date(), s.DEV_COVERAGE_END)
    assert {k: s.frame_sha256(v) for k, v in a.items()} == {k: s.frame_sha256(v) for k, v in b.items()}


# ============================================================
# Reproduction gate
# ============================================================


def gate(env, tmp_path, frozen, **kw):
    return s.reproduce(tmp_path / "rec", bars_loader=lambda: (all_bars(), {"synthetic": "x"}),
                       news_loader=lambda: env.news, frozen_loader=lambda: frozen, pins=lambda: {}, **kw)


def test_reproduction_matches_exactly_and_is_written_once(env, tmp_path):
    frozen = frames_for(env, start=s.DEV_START)
    rec = gate(env, tmp_path, frozen)
    assert rec["result"] == "EXACT_MATCH" and rec["sealed_rows_generated"] == 0
    assert json.loads((tmp_path / "rec" / s.REPRODUCTION_FILE).read_text())["frame_sha256"] == rec["frame_sha256"]
    with pytest.raises(s.SealedFeatureError, match="written once"):
        gate(env, tmp_path, frozen)


@pytest.mark.parametrize("break_it,match", [
    (lambda f: f["AAPL"].__setitem__("Volatility", np.nextafter(f["AAPL"]["Volatility"], 1)), "AAPL.Volatility"),
    (lambda f: f["RTX"].loc.__setitem__((3, "sw_count_surprise_1_20"), np.nan), "missing-value locations"),
    (lambda f: f.__setitem__("AAPL", f["AAPL"].iloc[:-1]), "rows"),
    (lambda f: f.__setitem__("AAPL", f["AAPL"].assign(Date=f["AAPL"]["Date"] + pd.Timedelta(days=1))), "dates"),
    (lambda f: f.__setitem__("RTX", f["RTX"].assign(event__event_count=f["RTX"]["event__event_count"].astype(int))),
     "dtype"),
    (lambda f: f.__setitem__("XOM", f["AAPL"]), "symbols differ"),
])
def test_any_reproduction_mismatch_stops_and_writes_nothing(env, tmp_path, break_it, match):
    frozen = {k: v.copy() for k, v in frames_for(env, start=s.DEV_START).items()}
    break_it(frozen)
    with pytest.raises(s.SealedFeatureError, match=match):
        gate(env, tmp_path, frozen, universe=SYMS)
    assert not (tmp_path / "rec").exists()


def test_reproduction_refuses_frozen_output_directories(env):
    for d in (m2.OUT_DIR, m2.OUT_DIR / "results", c.OUT_DIR, fb.FEATURES_DIR_11B,
              s.PROJECT_ROOT / "data" / "results" / "research" / "phase12" / "stage1" / "x"):
        with pytest.raises(s.SealedFeatureError, match="frozen evidence"):
            s.reproduce(d, pins=lambda: {})


def test_real_generator_files_equal_the_amendment_pins():
    if not (c.OUT_DIR / c.AMENDMENT_FILE).is_file():
        pytest.skip("Amendment 07 not frozen")
    assert set(s.generator_pins()) == set(c.GENERATOR_FILES)


# ============================================================
# Sealed path: refused before every gate passes
# ============================================================


@pytest.fixture
def no_data_access(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("data accessed before authorization")
    for name in ("development_news", "sealed_articles", "read_bars_until", "development_prefix", "reproduce"):
        monkeypatch.setattr(s, name, boom)
    return boom


def test_sealed_build_refuses_without_owner_authorization(no_data_access, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUT_DIR", tmp_path)                              # no authorization.json here
    monkeypatch.setattr(c, "load_amendment", lambda *a, **k: {})
    with pytest.raises(c.Phase12S3Error, match="not authorized"):
        s.build_sealed(no_data_access, tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()


def test_sealed_build_refuses_without_a_verified_erratum(no_data_access, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "load_amendment", lambda *a, **k: {})
    monkeypatch.setattr(c, "load_authorization", lambda *a, **k: {"authorized_by": "owner"})
    monkeypatch.setattr(s, "ERRATUM_SHA256", None)
    with pytest.raises(s.SealedFeatureError, match="not pinned"):
        s.build_sealed(no_data_access, tmp_path / "sealed")
    monkeypatch.setattr(s, "ERRATUM_SHA256", "0" * 64)
    with pytest.raises(s.SealedFeatureError, match="differs from its pinned SHA256"):
        s.build_sealed(no_data_access, tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()


def test_sealed_build_stops_when_reproduction_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "check_sealed_authorization", lambda: {})
    calls = []
    monkeypatch.setattr(s, "reproduce", lambda *a, **k: (_ for _ in ()).throw(s.SealedFeatureError("mismatch")))
    monkeypatch.setattr(s, "sealed_articles", lambda *a, **k: calls.append("sealed"))
    with pytest.raises(s.SealedFeatureError, match="mismatch"):
        s.build_sealed(lambda: calls.append("finbert"), tmp_path / "sealed")
    assert calls == [] and not (tmp_path / "sealed").exists()
    with pytest.raises(s.SealedFeatureError, match="frozen evidence"):
        s.build_sealed(lambda: None, m2.OUT_DIR / "sealed")


def test_sealed_loader_requires_offline_model_loading(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(s, "build_sealed", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(s.SealedFeatureError, match="HF_HUB_OFFLINE"):
        s.sealed_loader()


def test_sealed_article_routing_keeps_only_post_cutoff_articles(tmp_path):
    path = tmp_path / "c.jsonl"
    arts = [news(1, CUT), news(2, CUT - H, updated=CUT + US), news(3, CUT + H)]
    path.write_text("".join(json.dumps(a.to_dict()) + "\n" for a in arts))
    assert [a.provider_article_id for a in s.sealed_articles(path, file_sha256(path))] == ["2", "3"]


def test_confirmation_runner_binds_this_builder_by_hash(tmp_path):
    assert (s.PROJECT_ROOT / c.SEALED_BUILDER_FILE).resolve() == s.Path(s.__file__).resolve()
    rec = {"amendment_sha256": "a" * 64, "runner_code_sha256": file_sha256(s.Path(c.__file__)),
           "sealed_builder_code_sha256": file_sha256(s.Path(s.__file__)), "authorized_by": "owner",
           "authorized_at": "2026-11-01T00:00:00Z", "statement": c.AUTHORIZATION_STATEMENT}
    (tmp_path / c.AUTHORIZATION_FILE).write_text(json.dumps(rec))
    assert c.load_authorization(tmp_path, "a" * 64)["authorized_by"] == "owner"
    (tmp_path / c.AUTHORIZATION_FILE).write_text(json.dumps({**rec, "sealed_builder_code_sha256": "b" * 64}))
    with pytest.raises(c.Phase12S3Error, match="sealed_builder_code_sha256"):
        c.load_authorization(tmp_path, "a" * 64)


# ============================================================
# Erratum 01 and the FinBERT label-equivalence gate
# ============================================================


def dev_scored(n=600):
    """n synthetic development articles scored by the fake FinBERT (some negative, some positive)."""
    heads = ("Acme shares plunge", "Acme beats EPS estimates", "Acme holds annual meeting")
    arts = [news(1000 + i, CUT - (i + 1) * H, headline=f"{heads[i % 3]} {i}") for i in range(n)]
    return score_canonical_articles(arts, fake())[0]


def frozen_like(**over):
    svc = fake()
    svc.backend.max_length, svc.backend.device = 512, "cpu"
    for k, v in over.items():
        setattr(svc.backend if k in ("max_length", "device", "labels", "model_revision") else svc, k, v)
    return svc


def snapshot(tmp_path):
    d = tmp_path / "snap"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text("{}")
    (d / "pytorch_model.bin").write_bytes(b"weights")
    return d


def test_real_erratum_verifies_and_supplements_the_unchanged_amendment():
    doc = s.load_erratum()
    assert file_sha256(c.OUT_DIR / c.AMENDMENT_FILE) == c.AMENDMENT_SHA256 == doc["amends"]["amendment_sha256"] \
        == "54aa14a84de9f76aa79f579171e6908471fbf63ddd9056cabc5ac2c16894e0d7"
    c.load_amendment()                                                  # Amendment 07 rules unchanged
    assert doc["qualification_unchanged"] == c.rules()["qualification"] == c.load_amendment()["qualification"]
    assert doc["finbert"] == s.FINBERT_CONFIG and doc["finbert"]["model_revision"].startswith("4556d130")
    assert doc["equivalence_gate"]["articles"] == 500 and "500/500" in doc["equivalence_gate"]["requirement"]
    assert "event__negative_event_count only" in doc["permitted_use"]
    assert "authorizes no sealed access" in doc["sealed_access"]
    assert set(doc["model_files_sha256"]) >= {"config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json"}


def test_erratum_refuses_any_change(tmp_path, monkeypatch):
    p = tmp_path / "erratum_01.json"
    p.write_bytes(s.ERRATUM_FILE.read_bytes())
    assert s.load_erratum(p, file_sha256(p))["erratum_id"] == s.ERRATUM_ID
    doc = json.loads(p.read_text())
    doc["equivalence_gate"]["requirement"] = "495/500 fresh labels equal the stored labels"
    p.write_text(json.dumps(doc))
    with pytest.raises(s.SealedFeatureError, match="differs from its pinned SHA256"):
        s.load_erratum(p, s.ERRATUM_SHA256)
    with pytest.raises(s.SealedFeatureError, match="rules differ"):
        s.load_erratum(p, file_sha256(p))
    p.write_bytes(s.ERRATUM_FILE.read_bytes())
    monkeypatch.setattr(c, "OUT_DIR", tmp_path)                          # a different "Amendment 07"
    (tmp_path / c.AMENDMENT_FILE).write_text("{}")
    with pytest.raises(s.SealedFeatureError, match="Amendment 07 differs"):
        s.load_erratum(p, file_sha256(p))


def test_write_erratum_is_write_once(tmp_path):
    p = tmp_path / "e.json"
    doc = s.write_erratum(p, snapshot(tmp_path))
    assert doc["amends"]["amendment_sha256"] == c.AMENDMENT_SHA256 and doc["sealed_data_read"] is False
    with pytest.raises(s.SealedFeatureError, match="written once"):
        s.write_erratum(p, snapshot(tmp_path))
    with pytest.raises(s.SealedFeatureError, match="not in the local cache"):
        s.model_files_sha256(tmp_path / "missing")


def test_label_equivalence_passes_only_at_500_of_500():
    scored = dev_scored()
    rec = s.label_equivalence(scored, fake())
    assert (rec["result"], rec["checked"], rec["matches"], rec["mismatches"]) == ("PASS", 500, 500, [])
    assert s.require_equivalence(scored, fake())["result"] == "PASS"
    assert rec["articles"] == s.label_equivalence(list(reversed(scored)), fake())["articles"]   # deterministic pick
    assert len(s.label_equivalence(scored[:499], fake())["articles"]) == 499
    assert s.label_equivalence(scored[:499], fake())["result"] == "FAIL"   # fewer than 500 is never enough


def test_a_single_label_mismatch_stops(monkeypatch):
    scored = dev_scored()
    real = s.score_canonical_articles

    def one_flipped(arts, svc):
        out, st = real(arts, svc)
        x = out[17]
        out[17] = types.SimpleNamespace(article=x.article,
                                        sentiment=types.SimpleNamespace(label="neutral" if x.sentiment.label != "neutral"
                                                                        else "negative"))
        return out, st
    monkeypatch.setattr(s, "score_canonical_articles", one_flipped)
    rec = s.label_equivalence(scored, fake())
    assert (rec["result"], rec["matches"], len(rec["mismatches"])) == ("FAIL", 499, 1)
    with pytest.raises(s.SealedFeatureError, match="499/500"):
        s.require_equivalence(scored, fake())


def test_label_equivalence_uses_development_articles_only():
    sealed = score_canonical_articles([news(9001, CUT + US, headline="Acme shares plunge")], fake())[0]
    with pytest.raises(s.SealedFeatureError, match="development articles only"):
        s.label_equivalence(sealed, fake())


def test_frozen_finbert_configuration_and_offline_requirement(tmp_path, monkeypatch):
    snap = snapshot(tmp_path)
    files = s.model_files_sha256(snap)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(s.SealedFeatureError, match="HF_HUB_OFFLINE"):
        s.check_finbert(frozen_like(), files, snap)
    with pytest.raises(s.SealedFeatureError, match="HF_HUB_OFFLINE"):
        s.frozen_finbert()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    live = s.check_finbert(frozen_like(), files, snap)
    assert live["model_revision"] == s.FINBERT_CONFIG["model_revision"] and live["batch_size"] == 16
    for over in ({"batch_size": 32}, {"max_length": 256}, {"device": "cuda"}):
        with pytest.raises(s.SealedFeatureError, match="configuration differs"):
            s.check_finbert(frozen_like(**over), files, snap)
    with pytest.raises(Exception, match="revision|not the pinned"):
        s.check_finbert(frozen_like(model_revision="7db323f79b751944bcfa66298ec06977e4518306"), files, snap)
    with pytest.raises(Exception, match="classes|labels"):
        s.check_finbert(frozen_like(labels=("positive", "negative", "mixed")), files, snap)
    (snap / "pytorch_model.bin").write_bytes(b"other weights")
    with pytest.raises(s.SealedFeatureError, match="fingerprint"):
        s.check_finbert(frozen_like(), files, snap)


def test_finbert_labels_feed_only_event_negative_event_count(env):
    cal = TradingCalendar.from_bars(ohlcv())
    ts = row_prediction_timestamp(DATES[-3].date(), cal)
    arts = [news(3000 + i, ts - (i + 1) * H, headline=h, symbols=("AAPL",))
            for i, h in enumerate(("Acme misses EPS estimates, shares plunge", "Acme beats EPS estimates",
                                    "Acme cuts revenue guidance; shares plunge"))]
    neg = score_canonical_articles(arts, fake())[0]
    calm = score_canonical_articles(arts, fake(bias=10.0))[0]           # same articles, every label neutral
    assert [x.sentiment.label for x in neg] != [x.sentiment.label for x in calm]
    a = s.build_frames(all_bars(), neg, classify_scored_articles(neg, KeywordEventClassifier()), SYMS,
                       DATES[200].date(), DATES[-1].date(), s.DEV_COVERAGE_END)
    b = s.build_frames(all_bars(), calm, classify_scored_articles(calm, KeywordEventClassifier()), SYMS,
                       DATES[200].date(), DATES[-1].date(), s.DEV_COVERAGE_END)
    others = [col for col in s.OUTPUT_COLUMNS if col != "event__negative_event_count"]
    for sym in SYMS:
        assert a[sym][others].equals(b[sym][others])
    assert a["AAPL"]["event__negative_event_count"].sum() > 0 == b["AAPL"]["event__negative_event_count"].sum()


def test_sealed_build_runs_the_label_gate_before_any_sealed_record(env, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(s, "check_sealed_authorization", lambda: {"authorization": {}, "erratum": {"model_files_sha256": {}}})
    monkeypatch.setattr(s, "reproduce", lambda *a, **k: calls.append("reproduce"))
    monkeypatch.setattr(s, "development_news", lambda *a, **k: env.news)
    monkeypatch.setattr(s, "check_finbert", lambda *a, **k: calls.append("config") or {})
    monkeypatch.setattr(s, "require_equivalence",
                        lambda *a, **k: (_ for _ in ()).throw(s.SealedFeatureError("FinBERT label equivalence FAILED")))
    monkeypatch.setattr(s, "sealed_articles", lambda *a, **k: calls.append("SEALED READ"))
    with pytest.raises(s.SealedFeatureError, match="equivalence FAILED"):
        s.build_sealed(lambda: fake(), tmp_path / "sealed")
    assert calls == ["reproduce", "config"] and not (tmp_path / "sealed").exists()
