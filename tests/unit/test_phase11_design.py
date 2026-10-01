"""Phase 11.0 design freeze: universe, corporate-action semantics, matrix counts, families, write-once registry."""

import json

import numpy as np
import pytest

import training.phase11_design as p11

DJIA_2016 = {"AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM",
             "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM"}


def test_universe_29_dd_excluded_rtx_continues_utx():
    u = set(p11.UNIVERSE)
    assert len(u) == 29 and "DD" not in u and "UTX" not in u and "RTX" in u and "AAPL" in u
    assert u == (DJIA_2016 - {"DD", "UTX"}) | {"RTX"}
    assert set(p11.DJIA_MEMBERS_2016) == DJIA_2016
    assert "single continuous listed security series" in p11.EXCLUDED["DD"]
    assert p11.CONTINUATIONS["UTX"]["research_ticker"] == "RTX"
    for removed_later in ("GE", "XOM", "PFE", "INTC", "VZ"):          # historical membership, not current
        assert removed_later in u


def test_matrix_counts_and_families():
    d = p11.design()
    c = d["counts"]
    assert (c["universe"], c["feature_sets"], c["horizons"], c["models"]) == (29, 6, 3, 11)
    assert c["per_stock"] == 29 * 6 * 3 * 11 == 5742
    assert c["panel"] == 198 and c["aapl_excluded_sensitivity"] == 198
    ids = d["comparison_ids"]
    assert len(set(ids["per_stock"])) == 5742 and len(set(ids["panel"])) == 198
    assert len(set(ids["aapl_excluded_sensitivity"])) == 198
    assert not set(ids["panel"]) & set(ids["aapl_excluded_sensitivity"])
    mc = d["multiple_comparisons"]
    assert mc["primary_panel_family"]["size"] == 198 and mc["secondary_per_stock_family"]["size"] == 5742
    assert "descriptive only" in mc["aapl_excluded_sensitivity"]["role"]


def test_feature_sets_models_and_panel_spec():
    d = p11.design()
    assert {k: v["n_features"] for k, v in d["feature_sets"].items()} == \
        {"A": 20, "B": 35, "C": 43, "D": 58, "E": 29, "F": 64}
    cols = {c for v in d["feature_sets"].values() for c in v["columns"]}
    assert "stock_minus_spy_return_1" in cols and not any(c.startswith("aapl_") for c in cols)
    assert not cols & {"symbol", "ticker", "stock_id"}                       # no ticker identity feature
    panel = d["evaluation"]["panel"]
    assert panel["ticker_identity_feature"] is False and "unique development trading DATES" in panel["folds"]
    assert "gap = h" in panel["folds"] and "TimeSeriesSplit(20)" in panel["folds"]
    assert "per-date cross-sectional mean loss" in panel["dm"]
    assert sum(not m["is_baseline"] for m in d["models"]) == 11
    assert d["horizons"] == [1, 3, 5] and d["evaluation"]["gate_version"] == "gate_v1"
    assert d["intervals"]["development"] == ["2017-02-01", "2024-09-30"]
    assert d["intervals"]["confirmation"] == ["2024-10-01", "2026-09-25"]
    assert d["staging"]["11A"]["feature_sets"] == ["A", "E"]


@pytest.mark.parametrize("h", [1, 3, 5])
def test_affected_row_semantics(h):
    e, n = 100, 300
    mask = p11.affected_row_mask(n, e, h)
    expected = np.zeros(n, bool)
    expected[e - h:e + p11.MAX_FEATURE_LOOKBACK] = True                      # D in [E-h, E+49]
    np.testing.assert_array_equal(mask, expected)
    assert not mask[e - h - 1] and mask[e - h] and mask[e - 1] and mask[e] and mask[e + 49] and not mask[e + 50]
    assert mask.sum() == h + p11.MAX_FEATURE_LOOKBACK


def test_minimum_rows_applied_after_exclusions():
    assert p11.development_eligibility(1927, 0)["eligible"]
    r = p11.development_eligibility(1927, 151)
    assert r["remaining"] == 1776 and not r["eligible"]                    # fails only after the exclusions
    assert p11.development_eligibility(1850, 50)["eligible"]               # exactly 1,800 remain
    with pytest.raises(p11.Phase11DesignError):
        p11.development_eligibility(100, 200)


def test_corporate_action_events_have_no_invented_details():
    ev = p11.CORPORATE_ACTION_EVENTS
    assert {e["symbol"] for e in ev} >= {"GE", "MMM", "MRK", "IBM", "PFE", "JNJ", "RTX"}
    assert all(e["status"] == p11.VERIFY for e in ev)
    dated = [e for e in ev if e["date"] is not None]
    assert len(dated) == 1 and dated[0]["date"] == "2020-04-03" and "date_basis" in dated[0]
    assert not any("ratio" in e for e in ev)


def test_design_hash_deterministic_and_registry_write_once(tmp_path):
    assert p11.design_sha256(p11.design()) == p11.design_sha256(p11.design())
    path = tmp_path / "design_registry.json"
    reg = p11.freeze(path)
    assert reg["status"] == "DESIGN_FROZEN" and reg["data_acquired"] is False and reg["experiments_run"] is False
    assert p11.load_registry(path)["design_sha256"] == reg["design_sha256"]
    with pytest.raises(p11.Phase11DesignError, match="frozen once"):
        p11.freeze(path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["design"]["universe"]["size"] = 30
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(p11.Phase11DesignError, match="edited after freezing"):
        p11.load_registry(path)
