"""Phase 11.1 price integrity: normalization, validation, events, exclusions, eligibility, artifacts (offline)."""

import json

import numpy as np
import pandas as pd
import pytest

import training.phase10a_research as p10a
import training.phase11_design as p11d
import training.phase11_prices as pp
import training.phase9_research as p9

SESSIONS = pd.bdate_range("2016-01-04", "2026-09-29")


def provider_frame(dates=SESSIONS, seed=0, split_at=None, split=0.0, div_at=None, div=0.0, drop_at=None, drop=0.0):
    """Fake yfinance responses (adjusted + unadjusted) on a NY-tz DatetimeIndex."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.01, len(dates))
    if drop_at is not None:
        r[list(dates).index(pd.Timestamp(drop_at))] += drop
    close = 100 * np.cumprod(1 + r)
    idx = pd.DatetimeIndex(dates).tz_localize("America/New_York")
    adj = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                        "Volume": np.full(len(dates), 1e6), "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    for d in pd.bdate_range(dates[0], dates[-1], freq="QS")[:-1]:
        if d in adj.index.tz_localize(None):
            adj.loc[adj.index.tz_localize(None) == d, "Dividends"] = 0.5
    if split_at is not None:
        adj.loc[adj.index.tz_localize(None) == pd.Timestamp(split_at), "Stock Splits"] = split
    if div_at is not None:
        adj.loc[adj.index.tz_localize(None) == pd.Timestamp(div_at), "Dividends"] = div
    raw = pd.DataFrame({"Close": close * 1.02, "Adj Close": close, "Dividends": adj["Dividends"],
                        "Stock Splits": adj["Stock Splits"]}, index=idx)
    return adj, raw


def frame(**kw):
    return pp.normalize(*provider_frame(**kw))


SPY = frame(seed=99)


def test_universe_frozen_29_dd_absent_rtx_present_no_substitution():
    assert pp.UNIVERSE == tuple(p11d.UNIVERSE) and len(pp.UNIVERSE) == 29
    assert "DD" not in pp.UNIVERSE and "RTX" in pp.UNIVERSE and "UTX" not in pp.UNIVERSE
    assert pp.FROZEN_DESIGN_SHA == p11d.design_sha256(p11d.design())
    assert {e["symbol"] for e in pp.EVENTS} <= set(pp.UNIVERSE)


def test_normalization_deterministic_sorted_and_no_repair():
    adj, raw = provider_frame()
    a, b = pp.normalize(adj, raw), pp.normalize(adj.iloc[::-1], raw.iloc[::-1])
    assert pp.serialize(a) == pp.serialize(b)                                     # order-independent
    assert list(a["Date"]) == sorted(a["Date"]) and list(a.columns) == pp.COLUMNS
    np.testing.assert_array_equal(a["Close"].to_numpy(), adj["Close"].to_numpy())  # values untouched
    assert pp.serialize(pp.normalize_roundtrip(a)) == pp.serialize(a)            # float round trip stable
    assert len(a) == len(adj)                                                     # nothing filled / dropped


def test_duplicate_and_missing_session_detection():
    adj, raw = provider_frame()
    dup = pd.concat([adj, adj.iloc[[100]]])
    dupr = pd.concat([raw, raw.iloc[[100]]])
    v = pp.validate(pp.normalize(dup, dupr), list(SPY["Date"]))
    assert v["duplicate_dates"] == 1
    gone = [SESSIONS[300], SESSIONS[301]]
    f = frame(dates=SESSIONS.drop(gone))
    v = pp.validate(f, list(SPY["Date"]))
    assert v["n_missing_sessions"] == 2 and v["missing_sessions"] == [d.strftime("%Y-%m-%d") for d in gone]
    assert v["chronological"] and v["nan_values"] == 0


def test_spin_off_with_provider_record_is_properly_adjusted():
    f = frame(split_at="2020-11-16", split=1.054)
    ev = {"id": "x", "symbol": "PFE", "type": "spin_off", "public_date": "2020-11-16", "evidence": "test"}
    r = pp.check_event(f, SPY, ev)
    assert r["E"] == "2020-11-16" and r["determination"] == "PROPERLY_ADJUSTED" and r["exclusion_sessions"] == []


def test_spin_off_without_record_or_with_abnormal_move_is_not_adjusted():
    ev = {"id": "x", "symbol": "IBM", "type": "spin_off", "public_date": "2021-11-04", "evidence": "test"}
    r = pp.check_event(frame(drop_at="2021-11-04", drop=-0.12), SPY, ev)
    assert r["determination"] == "NOT_PROPERLY_ADJUSTED" and r["reason"] == "no provider record in window"
    assert r["exclusion_sessions"] == ["2021-11-04"]
    r = pp.check_event(frame(split_at="2021-11-04", split=1.1, drop_at="2021-11-04", drop=-0.12), SPY, ev)
    assert r["determination"] == "NOT_PROPERLY_ADJUSTED" and "abnormal" in r["reason"]


def test_reverse_split_and_public_date_on_weekend():
    f = frame(split_at="2021-08-02", split=0.125)
    ev = {"id": "x", "symbol": "GE", "type": "reverse_split", "public_date": "2021-07-31", "evidence": "test"}
    r = pp.check_event(f, SPY, ev)                                                # Saturday -> next session
    assert r["E"] == "2021-08-02" and r["determination"] == "PROPERLY_ADJUSTED"
    r = pp.check_event(frame(), SPY, ev)
    assert r["determination"] == "NOT_PROPERLY_ADJUSTED"


def test_merger_continuity_and_exchange_offer():
    ev = {"id": "m", "symbol": "RTX", "type": "merger_continuity", "public_date": "2020-04-03", "evidence": "t"}
    assert pp.check_event(frame(), SPY, ev)["determination"] == "CONTINUOUS"
    gap = frame(dates=SESSIONS.drop([pd.Timestamp("2020-04-06")]))
    assert pp.check_event(gap, SPY, ev)["determination"] == "NOT_CONTINUOUS"
    ex = {"id": "k", "symbol": "JNJ", "type": "exchange_offer", "public_date": None, "evidence": "t"}
    r = pp.check_event(frame(), SPY, ex)
    assert r["determination"] == "NO_ADJUSTMENT_REQUIRED" and r["exclusion_sessions"] == []


def test_large_dividend_counts_as_provider_record():
    f = frame(div_at="2023-01-04", div=20.0)
    recs = pp.provider_records(f)
    assert any(r["kind"] == "large_dividend" and r["date"] == "2023-01-04" for r in recs)


@pytest.mark.parametrize("h", [1, 3, 5])
def test_exclusion_windows_and_eligibility(h):
    ref = list(SPY["Date"])
    e = "2020-06-01"
    elig = pp.eligibility(ref, ref, [], [e])
    dev = elig[str(h)]["development"]
    assert dev["lost_corporate_action"] == h + pp.LOOKBACK                         # D in [E-h, E+49]
    assert dev["potential_labelled_rows"] == len([s for s in ref if "2017-02-01" <= s <= "2024-09-30"]) - h
    assert dev["final_rows"] == dev["potential_labelled_rows"] - h - 50 and dev["eligible"]
    clean = pp.eligibility(ref, ref, [], [])[str(h)]["development"]
    assert clean["lost_corporate_action"] == 0 and clean["lost_missing_data"] == 0 and clean["lost_warm_up"] == 0


def test_missing_session_exclusion_and_min_rows_after_exclusions():
    ref = list(SPY["Date"])
    m = "2019-05-01"
    own = [s for s in ref if s != m]
    d = pp.eligibility(own, ref, [m], [])["1"]["development"]
    assert d["lost_missing_data"] == 1 + 50                                      # [m-1, m+49] incl. the absent row
    many = [s for s in ref if "2018-01-02" <= s <= "2024-09-30"][::40]
    d = pp.eligibility(ref, ref, [], many)["1"]["development"]
    assert d["final_rows"] < pp.MIN_DEV_ROWS and not d["eligible"]                # fails only after exclusions


def test_artifact_write_once_and_hash(tmp_path):
    f = frame()
    a = pp.write_artifact("AAPL", f, {"note": "test"}, tmp_path)
    path = next(tmp_path.glob("AAPL_1d_*.csv"))
    assert pp.sha256_bytes(path.read_bytes()) == a["sha256"]
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == a["sha256"] and meta["columns"] == pp.COLUMNS
    assert pp.write_artifact("AAPL", f, {"note": "test"}, tmp_path)["sha256"] == a["sha256"]   # identical: accepted
    g = f.copy()
    g.loc[5, "Close"] *= 1.01
    with pytest.raises(pp.Phase11PriceError, match="refusing to overwrite"):
        pp.write_artifact("AAPL", g, {"note": "test"}, tmp_path)


def test_run_offline_end_to_end(tmp_path, monkeypatch):
    def fake_fetch(sym):
        return provider_frame(seed=sum(map(ord, sym)), **({"split_at": "2021-08-02", "split": 0.125} if sym == "GE" else {}))
    rep = pp.run(report_path=tmp_path / "rep.json", market_dir=tmp_path / "m", fetcher=fake_fetch, log=lambda *_: None)
    o = rep["PHASE_11_1_OBSERVATIONS"]
    assert set(o["per_stock"]) == set(pp.UNIVERSE) and len(rep["artifacts"]) == 31
    assert o["determinism"]["all_identical_across_passes"] and o["determinism"]["normalization_idempotent"]
    ge = {e["id"]: e for e in o["per_stock"]["GE"]["events"]}
    assert ge["GE_reverse_split_2021"]["determination"] == "PROPERLY_ADJUSTED"
    assert ge["GE_healthcare_2023"]["determination"] == "NOT_PROPERLY_ADJUSTED"     # fake data has no record
    assert rep["experiments_run"] is False and rep["FROZEN_DESIGN"]["design_sha256"] == pp.FROZEN_DESIGN_SHA
    with pytest.raises(pp.Phase11PriceError, match="written once"):
        pp.run(report_path=tmp_path / "rep.json", market_dir=tmp_path / "m", fetcher=fake_fetch, log=lambda *_: None)


def test_earlier_phases_unchanged():
    assert p9.matrix_sha256(p9.OFFICIAL.matrix()) == "12e16f11ba886d1a61d7f91bf912478dc6850653373e3c1094afd26ae9b12545"
    assert p10a.matrix_sha256(p10a.OFFICIAL.matrix()) == "fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45"
    assert p11d.load_registry()["design_sha256"] == pp.FROZEN_DESIGN_SHA
