"""Phase 11 amendment 01: frozen numerical reproducibility tolerance (offline)."""

import numpy as np
import pandas as pd
import pytest

import training.phase11_design as p11d
import training.phase11_reproducibility as rp

DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2017-01-03", periods=300)]


def closes(scale=1.0, bump_at=None, bump=0.0):
    c = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, len(DATES))) * scale
    if bump_at is not None:
        c[bump_at] *= 1 + bump
    return pd.DataFrame({"Date": DATES, "Close": c})


def test_tolerance_is_frozen_as_specified():
    t1, t2 = rp.TOLERANCE["tier1_data"], rp.TOLERANCE["tier2_results"]
    assert (t1["max_rel_close_diff"], t1["max_abs_return_diff"]) == (1e-5, 1e-5)
    assert set(t1["symbols"]) == {"AAPL", "SPY", "QQQ"}
    assert (t2["max_rel_primary_metric_diff"], t2["max_abs_dm_p_diff"]) == (5e-3, 0.02)
    assert rp.FROZEN_DESIGN_SHA == p11d.load_registry()["design_sha256"]       # frozen design untouched


def test_tier1_passes_for_provider_rounding_and_fails_beyond_tolerance():
    old = closes()
    ok = rp.tier1_compare(closes(scale=1 + 7.6e-7), old)                        # Phase 11.1-sized difference
    assert ok["pass"] and ok["max_rel_close_diff"] == pytest.approx(7.6e-7, rel=1e-3)
    bad = rp.tier1_compare(closes(bump_at=150, bump=5e-5), old)                  # one close 5e-5 off
    assert not bad["pass"] and bad["max_rel_close_diff"] > 1e-5
    with pytest.raises(rp.Phase11AmendmentError):
        rp.tier1_compare(closes(), pd.DataFrame({"Date": ["1990-01-02"], "Close": [1.0]}))


def test_tier2_rules():
    ref = {("A", 1, "Ridge"): {"status": "NOT_QUALIFIED", "primary_value": 1.0, "p_value": 0.90}}
    near = {("A", 1, "Ridge"): {"status": "NOT_QUALIFIED", "primary_value": 1.004, "p_value": 0.91}}
    assert rp.tier2_compare(near, ref)["pass"]
    for change in ({"status": "QUALIFIED"}, {"primary_value": 1.006}, {"p_value": 0.93}):
        assert not rp.tier2_compare({k: {**v, **change} for k, v in near.items()}, ref)["pass"]
    with pytest.raises(rp.Phase11AmendmentError, match="missing"):
        rp.tier2_compare({}, ref)


def test_outcome_and_write_once(tmp_path, monkeypatch):
    t1 = {"AAPL": {"pass": True}, "SPY": {"pass": True}}
    assert rp.outcome(t1, None) == "TIER1_PASS_TIER2_PENDING"
    assert rp.outcome(t1, {"pass": True}) == "REPRODUCED_WITHIN_TOLERANCE"
    assert rp.outcome({"AAPL": {"pass": False}}, {"pass": True}) == "REPRODUCIBILITY_CHECK_FAILED"
    monkeypatch.setattr(rp, "_canonical_frame", lambda s: closes(scale=1 + 1e-7))
    monkeypatch.setattr(rp, "_phase10_frame", lambda s: closes())
    monkeypatch.setattr(rp, "file_sha256", lambda p: rp.CANONICAL_INPUT["verification_report_sha256"])
    rec = rp.freeze(tmp_path / "a.json")
    assert rec["status"] == "TIER1_PASS_TIER2_PENDING" and rec["amends"]["frozen_registry_edited"] is False
    assert "models" in rec["unchanged"] and rec["frozen_before_any_phase11a_result"]
    with pytest.raises(rp.Phase11AmendmentError, match="written once"):
        rp.freeze(tmp_path / "a.json")
