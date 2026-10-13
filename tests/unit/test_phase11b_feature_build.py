"""
Phase 11B Stage 11B.3: development-only feature build - synthetic tests.

Every news input is synthetic (fabricated headlines, fake FinBERT backend, real
rules_v1); price frames are synthetic too. The real canonical dataset, scores,
events and price files are never read by these tests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

import services.historical_events as historical_events
import services.historical_sentiment as historical_sentiment
import services.news_dataset as news_dataset
import training.phase11a_research as a11
import training.phase11b_feature_build as fb
from services.market_calendar_service import TradingCalendar
from services.news_schema import NewsArticle
from tests.sentiment_fakes import news
from tests.unit.test_phase11b_event_classification import classify, scored_env
from tests.unit.test_phase11b_sentiment_scoring import CUT, H, US, fake, make_env, run as run_scoring

SENTINEL = "SEALED_FEATURE_SENTINEL_must_never_reach_features"
UNIVERSE = ("AAPL", "RTX")
SESSIONS = [d.date() for d in pd.bdate_range("2024-08-01", "2024-09-30")]       # synthetic calendar
DEV_DATES = SESSIONS[21:]                                                       # >= 20 prior sessions
D = DEV_DATES[5]
TS_D = (pd.Timestamp(D).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=30)).tz_convert("UTC") \
    .to_pydatetime()


# ============================================================
# Builders
# ============================================================


def synthetic_prices(seed=7):
    rng = np.random.default_rng(seed)
    frames, calendars = {}, {}
    for i, sym in enumerate(UNIVERSE):
        n = len(DEV_DATES)
        f = pd.DataFrame({"Date": pd.to_datetime(DEV_DATES).astype("datetime64[ns]"),
                          "Close": 100 + np.cumsum(rng.normal(0, 1, n)) + 10 * i,
                          fb.SPY_CLOSE: 400 + np.cumsum(rng.normal(0, 1, n))})
        for c in fb.PRICE_COLUMNS:
            f[c] = rng.normal(0, 1, n)
        frames[sym] = f
        calendars[sym] = TradingCalendar(SESSIONS)
    return lambda: ({k: v.copy() for k, v in frames.items()}, calendars)


def spec(prices):
    frames, _ = prices()
    data = a11.panel_rows(frames, ["RSI"], 1)
    uniq = np.array(sorted(data["Date"].unique()))
    test = np.concatenate([uniq[te] for _, te in TimeSeriesSplit(n_splits=3, gap=1).split(np.zeros((len(uniq), 1)))])
    return fb.BuildSpec(universe=UNIVERSE, rows_per_stock=len(DEV_DATES),
                        panel_oos_rows_h1=int(data["Date"].isin(test).sum()), n_splits=3)


def articles():
    return [
        news(1, TS_D - 6 * H, headline="Acme beats EPS estimates", symbols=("AAPL", "RTX")),   # both stocks, row D
        news(2, TS_D + 30 * timedelta(minutes=1), headline="Analyst upgrades Acme", symbols=("AAPL",)),  # next row
        news(3, TS_D - 2 * H, headline="UTX tag after the merger", symbols=("UTX",)),          # off-period: nobody
        news(4, CUT, headline="Acme announces new board members", symbols=("RTX",)),          # exactly at cutoff
        news(5, TS_D - 30 * H, headline="A quiet day", symbols=("AAPL",)),
    ]


def build_env(tmp_path, dev=None, sealed=None):
    env = scored_env(tmp_path, dev=dev if dev is not None else articles(),
                     sealed=sealed if sealed is not None else [news(9, CUT + US, headline="Acme beats EPS estimates",
                                                                    symbols=("AAPL", "RTX"))])
    classify(env)
    env.features = tmp_path / "features" / "phase11b"
    env.verification = tmp_path / "feature_verification.json"
    env.prices = synthetic_prices()
    env.spec = spec(env.prices)
    return env


def build(env, **kw):
    return fb.run_feature_build(report_path=env.report, sentiment_dir=env.out, events_dir=env.events,
                                record_dir=env.records, features_dir=env.features,
                                verification_path=env.verification, price_inputs=env.prices, spec=env.spec, **kw)


def frame(env, sym):
    return pd.read_csv(env.features / env.run_id / f"{sym}.csv", float_precision="round_trip")


def row(env, sym, d):
    f = frame(env, sym)
    return f[f["Date"] == d.isoformat()].iloc[0]


def nothing_written(env):
    return not env.verification.exists() and not (env.features.exists() and any(env.features.rglob("*.csv")))


def classification_record_path(env):
    return env.records / f"rules_v1_development_classification_{env.run_id}.json"


def events_path(env):
    return env.events / f"{env.run_id}.events-rules_v1.jsonl"


def edit_classification_record(env, mutate):
    p = classification_record_path(env)
    rec = json.loads(p.read_text(encoding="utf-8"))
    mutate(rec)
    p.write_text(json.dumps(rec), encoding="utf-8")


def reseal_events(env, mutate_lines):
    """Rewrite the events file and update its metadata and the classification record so the hashes agree."""
    lines = events_path(env).read_text(encoding="utf-8").splitlines()
    payload = "".join(line + "\n" for line in mutate_lines(lines)).encode("utf-8")
    events_path(env).write_bytes(payload)
    sha, rows = hashlib.sha256(payload).hexdigest(), payload.count(b"\n")
    meta_file = events_path(env).with_suffix(".meta.json")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta.update(sha256=sha, rows=rows)
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    edit_classification_record(env, lambda r: r["outputs"]["events"].update(sha256=sha, rows=rows))


@pytest.fixture
def banned_loaders(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("a full-canonical loader was called by the 11B.3 build")
    for module, name in [(news_dataset, "load_canonical_dataset"), (historical_sentiment, "load_canonical_dataset"),
                         (historical_sentiment, "load_sentiment_dataset"), (historical_events, "load_canonical_dataset"),
                         (historical_events, "load_sentiment_dataset"), (historical_events, "load_event_dataset")]:
        monkeypatch.setattr(module, name, boom)


# ============================================================
# Columns, attribution, sessions, cutoff
# ============================================================


def test_frames_have_exactly_the_frozen_columns(tmp_path, banned_loaders):
    env = build_env(tmp_path)
    v = build(env)
    for sym in UNIVERSE:
        f = frame(env, sym)
        assert list(f.columns) == list(fb.FRAME_COLUMNS) and len(f) == len(DEV_DATES)
        for name, n in (("B", 35), ("C", 43), ("D", 58), ("F", 64)):
            cols = v["feature_sets"][name]
            assert len(cols) == n and set(cols) <= set(f.columns)
            assert all(pd.api.types.is_float_dtype(f[c]) for c in cols)
        assert np.isfinite(f[list(fb.NEWS_COLUMNS)].to_numpy()).all()
    assert "sw_count_1" not in v["feature_sets"]["F"] and "event__dominant_event_type" not in fb.FRAME_COLUMNS
    assert v["feature_sets"] == {k: list(fb.FEATURE_SETS_11B[k]) for k in "BCDF"}


def test_multi_symbol_attribution_and_session_assignment(tmp_path):
    env = build_env(tmp_path)
    build(env)
    nxt = DEV_DATES[DEV_DATES.index(D) + 1]
    a_d, r_d, a_next = row(env, "AAPL", D), row(env, "RTX", D), row(env, "AAPL", nxt)
    assert a_d["event__article_count"] == 1 and r_d["event__article_count"] == 1      # shared article, once each
    assert a_d["event__earnings_event_count"] == 1 and r_d["event__earnings_event_count"] == 1
    assert a_next["event__article_count"] == 1 and a_next["event__analyst_rating_event_count"] == 1   # after 16:30
    assert row(env, "RTX", nxt)["event__article_count"] == 0
    assert a_d["sentiment__news_count"] == 2 and r_d["sentiment__news_count"] == 1     # cumulative
    v = json.loads(env.verification.read_text(encoding="utf-8"))
    assert v["rtx_utx_mapping"]["development_counts"]["utx_off_period_dropped"] == 1
    for sym in UNIVERSE:                                                                # UTX article reaches nobody
        assert frame(env, sym)["sentiment__news_count"].max() <= 3


def test_cutoff_equality_counts_and_one_microsecond_later_never_does(tmp_path):
    env = build_env(tmp_path)
    build(env)
    last = row(env, "RTX", DEV_DATES[-1])
    assert DEV_DATES[-1].isoformat() == "2024-09-30"
    assert last["event__article_count"] == 1 and last["event__management_governance_event_count"] == 1
    assert row(env, "AAPL", DEV_DATES[-1])["event__article_count"] == 0                 # sealed 1us later: absent


# ============================================================
# Sealed records and future-data invariance
# ============================================================


def test_sealed_text_never_reaches_features_or_outputs(tmp_path, monkeypatch, capsys):
    sealed = news(9, CUT + H, symbols=("AAPL", "RTX")).to_dict()
    sealed.update(headline=123, summary=SENTINEL, content=SENTINEL)                    # a full parse would raise
    env = make_env(tmp_path, dev=articles(), sealed=[sealed])
    run_scoring(env, fake())
    env.events = tmp_path / "events" / "phase11b"
    classify(env)
    env.features, env.verification = tmp_path / "features" / "phase11b", tmp_path / "feature_verification.json"
    env.prices = synthetic_prices()
    env.spec = spec(env.prices)
    parsed = []
    original = NewsArticle.from_dict
    monkeypatch.setattr(NewsArticle, "from_dict",
                        classmethod(lambda cls, d: (parsed.append(d.get("provider_article_id")), original(d))[1]))
    v = build(env)
    assert "9" not in parsed and v["sealed_article_count"] == 1 and v["sealed_content_used"] is False
    for path in [*env.features.rglob("*.csv"), env.verification]:
        assert SENTINEL.encode() not in path.read_bytes(), path.name
    assert SENTINEL not in capsys.readouterr().out


def test_adding_a_sealed_article_changes_no_development_row(tmp_path):
    a = build_env(tmp_path / "a", sealed=[news(9, CUT + US, symbols=("AAPL",))])
    b = build_env(tmp_path / "b", sealed=[news(9, CUT + US, symbols=("AAPL",)),
                                          news(10, CUT + 2 * H, headline="Acme beats EPS estimates",
                                               symbols=("AAPL", "RTX"))])
    build(a)
    build(b)
    for sym in UNIVERSE:
        assert (a.features / a.run_id / f"{sym}.csv").read_bytes() == (b.features / b.run_id / f"{sym}.csv").read_bytes()


def test_module_never_references_full_canonical_loaders():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fb))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"load_canonical_dataset", "load_sentiment_dataset", "load_event_dataset"}


# ============================================================
# Integrity and wrong inputs
# ============================================================


def test_changed_integrity_report_is_refused_by_the_run_record_hash_check(tmp_path):
    env = build_env(tmp_path)
    report = json.loads(env.report.read_text(encoding="utf-8"))
    report["sealed_article_count"] = 2
    env.report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(fb.ScoringError, match=r"integrity_report\.sha256"):
        build(env)
    assert nothing_written(env)


def test_partition_must_reproduce_the_integrity_report(tmp_path, monkeypatch):
    """Every record hash is kept consistent, so the only defect left is the partition mismatch itself."""
    import training.phase11b_sentiment_scoring as scoring
    env = build_env(tmp_path)
    report = json.loads(env.report.read_text(encoding="utf-8"))
    report["sealed_article_id_sha256"] = "0" * 64                   # not cross-checked by any run record
    env.report.write_text(json.dumps(report), encoding="utf-8")
    new_sha = hashlib.sha256(env.report.read_bytes()).hexdigest()
    scoring_path = env.records / f"finbert_development_scoring_{env.run_id}.json"
    rec = json.loads(scoring_path.read_text(encoding="utf-8"))
    rec["integrity_report"]["sha256"] = new_sha
    scoring_path.write_text(json.dumps(rec), encoding="utf-8")
    scoring_sha = hashlib.sha256(scoring_path.read_bytes()).hexdigest()
    edit_classification_record(env, lambda r: (r["integrity_report"].update(sha256=new_sha),
                                               r["scoring_run_record"].update(sha256=scoring_sha)))
    reached = []
    original = scoring.verify_partition_against_report
    monkeypatch.setattr(fb, "verify_partition_against_report",
                        lambda p, r: (reached.append(1), original(p, r))[1])
    with pytest.raises(fb.ScoringError, match=r"partition does not reproduce the integrity report: \['sealed_article_id_sha256'\]"):
        build(env)
    assert reached == [1] and nothing_written(env)


@pytest.mark.parametrize("mutate, field", [
    (lambda r: r.update(stage="11B.3"), "stage"),
    (lambda r: r.update(run_id="phase11b_dev_000000000000_x"), "run_id"),
    (lambda r: r.update(rules_v1_spec_sha256="0" * 64), "rules_v1_spec_sha256"),
    (lambda r: r.update(development_cutoff_utc="2024-09-30T20:30:01Z"), "development_cutoff_utc"),
    (lambda r: r["scoring_run_record"].update(sha256="0" * 64), "scoring_run_record.sha256"),
    (lambda r: r["score_input"].update(sha256="0" * 64), "score_input.sha256"),
    (lambda r: r.update(phase_8_score_reuse=True), "phase_8_score_reuse"),
    (lambda r: r.update(excluded_article_count=1), "excluded_article_count"),
])
def test_classification_record_that_fails_verification_is_refused(tmp_path, mutate, field):
    env = build_env(tmp_path)
    edit_classification_record(env, mutate)
    with pytest.raises(fb.FeatureBuildError, match=field.replace(".", r"\.")):
        build(env)
    assert nothing_written(env)


def test_scoring_record_with_another_model_revision_is_refused(tmp_path):
    env = build_env(tmp_path)
    p = env.records / f"finbert_development_scoring_{env.run_id}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["model"]["model_revision"] = "7db323f79b751944bcfa66298ec06977e4518306"
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(fb.ScoringError, match="model_revision"):
        build(env)
    assert nothing_written(env)


def test_tampered_events_file_is_refused(tmp_path):
    env = build_env(tmp_path)
    events_path(env).write_bytes(events_path(env).read_bytes() + b"\n")
    with pytest.raises(fb.FeatureBuildError, match="does not match the 11B.2 classification run record"):
        build(env)


def test_missing_event_record_is_refused(tmp_path):
    env = build_env(tmp_path)
    reseal_events(env, lambda lines: lines[1:])
    with pytest.raises(fb.ScoringError, match="article set differs|fails verification"):
        build(env)
    assert nothing_written(env)


def test_duplicate_or_extra_event_record_is_refused(tmp_path):
    env = build_env(tmp_path)
    reseal_events(env, lambda lines: lines + [lines[0]])
    with pytest.raises(fb.FeatureBuildError, match="duplicate event record"):
        build(env)
    env2 = build_env(tmp_path / "x")

    def extra(lines):
        rec = json.loads(lines[0])
        rec["provider_article_id"] = "999999"
        return lines + [json.dumps(rec)]
    reseal_events(env2, extra)
    with pytest.raises(fb.FeatureBuildError, match="not a development article"):
        build(env2)


def test_mismatched_event_record_is_refused(tmp_path):
    env = build_env(tmp_path)

    def wrong_hash(lines):
        rec = json.loads(lines[0])
        rec["input_text_hash"] = "0" * 64
        return [json.dumps(rec)] + lines[1:]
    reseal_events(env, wrong_hash)
    with pytest.raises(fb.FeatureBuildError, match="rejected"):
        build(env)


def test_stored_event_label_not_reproduced_by_rules_v1_is_refused(tmp_path):
    env = build_env(tmp_path)

    def relabel(lines):
        out = []
        for line in lines:
            rec = json.loads(line)
            if rec["event_type"] == "EARNINGS":
                rec["event_type"] = "GUIDANCE"
            out.append(json.dumps(rec))
        return out
    reseal_events(env, relabel)
    with pytest.raises(fb.FeatureBuildError, match="not reproduced by rules_v1"):
        build(env)
    assert nothing_written(env)


def test_phase8_directories_are_refused(tmp_path):
    env = build_env(tmp_path)
    for kw, match in ((dict(sentiment_dir=historical_sentiment.SENTIMENT_DIR), "sentiment directory"),
                      (dict(events_dir=fb.EVENTS_DIR_PHASE8), "events directory"),
                      (dict(features_dir=fb.FEATURES_DIR_PHASE8), "features directory")):
        args = dict(report_path=env.report, sentiment_dir=env.out, events_dir=env.events, record_dir=env.records,
                    features_dir=env.features, verification_path=env.verification, price_inputs=env.prices,
                    spec=env.spec)
        args.update(kw)
        with pytest.raises(fb.FeatureBuildError, match=match):
            fb.run_feature_build(**args)
    assert nothing_written(env)


def test_price_frames_must_cover_exactly_the_universe(tmp_path):
    env = build_env(tmp_path)
    frames, cals = env.prices()
    frames.pop("RTX")
    env.prices = lambda: (frames, cals)
    with pytest.raises(fb.FeatureBuildError, match="universe"):
        build(env)


def test_frozen_specification_matches_the_design_freeze():
    families = fb.verify_frozen_spec(fb.FROZEN_SPEC)
    assert (fb.FROZEN_SPEC.rows_per_stock, fb.FROZEN_SPEC.panel_oos_rows_h1) == (1928, 52780)
    assert (families["panel"]["total"], families["per_stock"]["total"]) == (198, 5742)
    assert len(fb.FROZEN_SPEC.universe) == 29


# ============================================================
# Invariants, determinism, immutability, Gate C evidence
# ============================================================


def test_f_duplicate_mismatch_fails_before_anything_is_written(tmp_path, monkeypatch):
    env = build_env(tmp_path)
    original = fb.generate_sentiment_window_features

    def perturbed(*a, **k):
        out = original(*a, **k)
        out.loc[out.index[-1], "sw_count_1"] += 1
        return out
    monkeypatch.setattr(fb, "generate_sentiment_window_features", perturbed)
    with pytest.raises(fb.FeatureBuildError, match="F duplicate check failed"):
        build(env)
    assert nothing_written(env)


def test_nan_in_features_fails_before_anything_is_written(tmp_path):
    env = build_env(tmp_path)
    frames, cals = env.prices()
    frames["AAPL"].loc[3, "RSI"] = np.nan
    env.prices = lambda: (frames, cals)
    with pytest.raises(fb.FeatureBuildError, match="non-finite"):
        build(env)
    assert nothing_written(env)


def test_prefix_invariance_violation_is_detected(tmp_path, monkeypatch):
    env = build_env(tmp_path)
    original = fb.generate_sentiment_features

    def leaky(items, dates, calendar, symbols):
        """Simulated leak: the first row depends on how many articles exist in total (incl. later ones)."""
        import dataclasses
        items = list(items)
        rows = original(items, dates, calendar, symbols)
        return [dataclasses.replace(rows[0], news_count=rows[0].news_count + len(items))] + rows[1:]
    monkeypatch.setattr(fb, "generate_sentiment_features", leaky)
    with pytest.raises(fb.FeatureBuildError, match="prefix invariance failed"):
        build(env)
    assert nothing_written(env)


def test_build_is_deterministic(tmp_path):
    a, b = build_env(tmp_path / "a"), build_env(tmp_path / "b")
    va, vb = build(a), build(b)
    for sym in UNIVERSE:
        assert (a.features / a.run_id / f"{sym}.csv").read_bytes() == (b.features / b.run_id / f"{sym}.csv").read_bytes()
    assert {k: v["sha256"] for k, v in va["frame_files"].items()} == {k: v["sha256"] for k, v in vb["frame_files"].items()}


def test_completed_build_is_write_once(tmp_path):
    env = build_env(tmp_path)
    build(env)
    before = env.verification.read_bytes()
    with pytest.raises(fb.FeatureBuildError, match="already exists"):
        build(env)
    assert env.verification.read_bytes() == before


def test_existing_frame_with_different_content_is_not_overwritten(tmp_path):
    env = build_env(tmp_path)
    target = env.features / env.run_id / "AAPL.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"something else\n")
    with pytest.raises(fb.FeatureBuildError, match="exists with different content"):
        build(env)
    assert target.read_bytes() == b"something else\n" and not env.verification.exists()


def test_verification_record_carries_the_gate_c_contract(tmp_path, monkeypatch):
    import tests.unit.test_phase11b_leakage as leakage
    env = build_env(tmp_path)
    v = build(env)
    for key in ("frame_paths", "row_count_per_stock", "shared_columns_match_11a", "f_duplicate_check_passed",
                "nans_in_development_features", "prefix_invariance_check_passed", "horizon_independence_check_passed",
                "panel_oos_rows_all29_h1", "final_panel_family_size", "final_per_stock_family_size",
                "phase_11a_p_values_hash_verified", "news_file_hashes_match_records",
                "sentiment_file_hashes_match_records", "event_file_hashes_match_records"):
        assert key in v, key
    assert set(v["frame_paths"]) == {"B", "C", "D", "F"} and all(set(m) == set(UNIVERSE) for m in v["frame_paths"].values())
    assert v["nans_in_development_features"] == 0 and v["panel_oos_rows_all29_h1"] == env.spec.panel_oos_rows_h1
    assert (v["final_panel_family_size"], v["final_per_stock_family_size"]) == (198, 5742)
    for sym, info in v["frame_files"].items():
        assert info["sha256"] == hashlib.sha256((env.features / env.run_id / f"{sym}.csv").read_bytes()).hexdigest()
    monkeypatch.setattr(leakage, "FEATURE_VERIFICATION_PATH", env.verification)       # Gate C helpers, synthetic record
    leakage._assert_t1_prefix_invariance_real()
    leakage._assert_t8_horizon_independence()
    leakage._assert_t12_holm_families()
    leakage._assert_t13_f_duplicate()


@pytest.mark.parametrize("extra", [["--start", "2017-01-01"], ["--symbols", "AAPL"], ["--include-sealed"]])
def test_cli_accepts_no_paths_dates_or_selection(extra):
    with pytest.raises(SystemExit) as exc:
        fb.main(extra)
    assert exc.value.code == 2
