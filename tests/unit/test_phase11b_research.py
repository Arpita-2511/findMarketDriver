"""
Phase 11B Stage 11B.4 research runner - synthetic tests.

Feature frames are fabricated (real ticker names, random values) and written with a
synthetic feature_verification.json; the leakage step uses a fake pytest runner; the
Amendment 02 data comparison uses synthetic canonical files. No real feature frame,
real article, confirmation-period row or network resource is used.

Real model fitting (about 6-7 s per run) is exercised where it proves something: the synthetic
smoke step, the determinism test, and one cached real per-stock and one cached real panel run.
Orchestration tests reuse those cached real reports through recording stand-ins, which also
assert exactly which columns, stocks, horizons and splits each step requests.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

import training.phase11a_research as a11
import training.phase11b_feature_build as fb
import training.phase11b_research as r
from services.news_schema import NewsArticle
from tests.sentiment_fakes import news
from tests.unit.test_phase11b_sentiment_scoring import CUT, H, US, make_env

UNIVERSE = ("AAPL", "AXP", "BA")
ROWS = 80
GOOD_PYTEST = (0, "....................................\n35 passed in 3.21s\n")


# ============================================================
# Synthetic 11B.3 outputs
# ============================================================


def _oos(n_labelled: int, n_splits: int = 3, gap: int = 1) -> int:
    return sum(len(te) for _, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.zeros((n_labelled, 1))))


def write_fixture(root: Path, *, dates=None, mutate_frame=None, mutate_verification=None, seed=3):
    """Synthetic frames + feature_verification.json; returns (spec, paths)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = dates if dates is not None else pd.bdate_range("2020-01-02", periods=ROWS)
    frame_files = {}
    for i, sym in enumerate(UNIVERSE):
        f = pd.DataFrame({"Date": pd.to_datetime(dates).astype("datetime64[ns]"),
                          "Close": 80 + 5 * i + np.cumsum(rng.normal(0, 0.5, len(dates))),
                          fb.SPY_CLOSE: 300 + np.cumsum(rng.normal(0, 0.5, len(dates)))})
        for c in fb.FRAME_COLUMNS[3:]:
            f[c] = rng.normal(0, 1, len(dates))
        if mutate_frame:
            f = mutate_frame(sym, f)
        p = root / "frames" / f"{sym}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(fb._frame_csv(f))
        frame_files[sym] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    oos1 = _oos(ROWS - 1)
    v = {"document_type": "phase_11b_feature_verification", "stage": "11B.3", "run_id": "synthetic_run",
         "shared_columns_match_11a": True, "f_duplicate_check_passed": True, "prefix_invariance_check_passed": True,
         "horizon_independence_check_passed": True, "phase_11a_p_values_hash_verified": True,
         "news_file_hashes_match_records": True, "sentiment_file_hashes_match_records": True,
         "event_file_hashes_match_records": True, "partition_reproduces_integrity_report": True,
         "nans_in_development_features": 0, "row_count_per_stock": ROWS, "panel_oos_rows_all29_h1": oos1 * len(UNIVERSE),
         "final_panel_family_size": 198, "final_per_stock_family_size": 5742, "sealed_content_used": False,
         "phase_8_score_reuse": False,
         "feature_sets": {k: list(r.FEATURE_SETS[k]["columns"]) for k in r.STAGE_FEATURE_SETS},
         "frame_columns": list(fb.FRAME_COLUMNS), "frame_files": frame_files,
         "frame_paths": {fs: {s: frame_files[s]["path"] for s in UNIVERSE} for fs in r.STAGE_FEATURE_SETS}}
    if mutate_verification:
        mutate_verification(v)
    vp = root / "feature_verification.json"
    vp.write_text(json.dumps(v), encoding="utf-8")
    spec = r.Spec(universe=UNIVERSE, rows_per_stock=ROWS, oos_rows_per_stock_h1=oos1,
                  panel_oos_rows_h1=oos1 * len(UNIVERSE), n_splits=3,
                  expected_counts=(len(UNIVERSE) * 4 * 3 * 11, 132, 132),
                  feature_verification_path=vp, feature_verification_sha256=hashlib.sha256(vp.read_bytes()).hexdigest())
    return spec, r.Paths(out_dir=root / "11B")


def fake_runner(rc=0, out="35 passed in 3.21s"):
    return lambda: (rc, out)


def data_comparison_stub():
    return {"scope": "development articles only (synthetic stub)", "pass_fail": False, "affects_results": False}


def run_steps(spec, paths, upto="panel"):
    r.validate_leakage(spec, paths, runner=fake_runner())
    if upto == "leakage":
        return
    r.validate_smoke(spec, paths)
    if upto == "smoke":
        return
    r.validate_stock("aapl", spec, paths)
    if upto == "aapl":
        return
    r.validate_stock("stock", spec, paths)
    if upto == "stock":
        return
    r.validate_panel(spec, paths)


@pytest.fixture(scope="module")
def cached(tmp_path_factory):
    """One REAL per-stock run and one REAL panel run (F, h=1) on the synthetic fixture frames."""
    spec, _ = write_fixture(tmp_path_factory.mktemp("cached"))
    frames = r.load_frames(r.verified_inputs(spec), spec)
    cols = list(r.FEATURE_SETS["F"]["columns"])
    return {"stock": r.p10a.evaluate_excess(frames["AAPL"], cols, a11.candidates(), horizon=1, n_splits=3),
            "panel": a11.evaluate_panel(frames, cols, a11.panel_candidates(), horizon=1, n_splits=3)}


def make_fakes(cached, calls):
    def fake_excess(frame, cols, cands, *, horizon, n_splits, splits=None):
        calls.append(("stock", tuple(cols), horizon, n_splits, len(frame)))
        rep = copy.deepcopy(cached["stock"])
        rep["horizon_days"] = horizon
        return rep

    def fake_panel(frames, cols, cands, *, horizon, n_splits=a11.N_SPLITS):
        calls.append(("panel", tuple(sorted(frames)), tuple(cols), horizon, n_splits))
        rep = copy.deepcopy(cached["panel"])
        rep["n_symbols"] = len(frames)
        return rep
    return fake_excess, fake_panel


@pytest.fixture
def fast(monkeypatch, cached):
    calls = []
    fake_excess, fake_panel = make_fakes(cached, calls)
    monkeypatch.setattr(r.p10a, "evaluate_excess", fake_excess)
    monkeypatch.setattr(a11, "evaluate_panel", fake_panel)
    return calls


@pytest.fixture(scope="module")
def completed(tmp_path_factory, cached):
    """Full synthetic chain: leakage (fake pytest), smoke (REAL models), aapl/stock/panel, freeze, develop."""
    root = tmp_path_factory.mktemp("chain")
    spec, paths = write_fixture(root)
    r.validate_leakage(spec, paths, runner=fake_runner())
    r.validate_smoke(spec, paths)                                            # real model fitting, synthetic data
    calls = []
    fake_excess, fake_panel = make_fakes(cached, calls)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(r.p10a, "evaluate_excess", fake_excess)
        mp.setattr(a11, "evaluate_panel", fake_panel)
        r.validate_stock("aapl", spec, paths)
        r.validate_stock("stock", spec, paths)
        r.validate_panel(spec, paths)
        n_before = len(calls)
        reg = r.freeze(spec, paths)
        res = r.develop(spec, paths, workers=1, log=lambda *_: None, data_comparison=data_comparison_stub)
    return spec, paths, reg, res, calls[n_before:]


# ============================================================
# Frozen matrix and specification
# ============================================================


def test_frozen_matrix_has_exactly_the_frozen_counts():
    m = r.matrix()
    assert m["counts"] == {"per_stock": 3828, "panel": 132, "aapl_excluded_sensitivity": 132}
    assert {k: v["total"] for k, v in m["families"]["final"].items()} == {"panel": 198, "per_stock": 5742}
    assert m["qualification"]["status"] == r.DEFERRED and m["evaluation"]["confirmation"] == "NOT_EVALUATED"
    assert list(m["feature_sets"]) == ["B", "C", "D", "F"] and m["horizons"] == [1, 3, 5] and len(m["models"]) == 11


def test_matrix_with_wrong_expected_counts_stops():
    spec = r.Spec(universe=UNIVERSE, expected_counts=(3828, 132, 132))
    with pytest.raises(r.Phase11BError, match="comparison counts"):
        r.matrix(spec)


def test_synthetic_specification_is_never_real():
    spec, _ = write_fixture(Path(__import__("tempfile").mkdtemp()))
    assert r.is_real(r.FROZEN_SPEC) and not r.is_real(spec)
    assert r.spec_sha256(spec) != r.spec_sha256(r.FROZEN_SPEC)


# ============================================================
# Verified inputs: hashes, schema, dates, confirmation period
# ============================================================


def test_verified_inputs_accept_a_consistent_fixture(tmp_path):
    spec, _ = write_fixture(tmp_path)
    inputs = r.verified_inputs(spec)
    assert sorted(inputs["frames"]) == sorted(UNIVERSE) and inputs["phase11a_registry_sha256"] == r.PHASE_11A_REGISTRY_SHA256
    frames = r.load_frames(inputs, spec)
    assert all(len(f) == ROWS and list(f.columns) == list(fb.FRAME_COLUMNS) for f in frames.values())


def test_pinned_feature_verification_hash_mismatch_stops(tmp_path):
    spec, _ = write_fixture(tmp_path)
    spec = r.Spec(**{**spec.__dict__, "feature_verification_sha256": "0" * 64})
    with pytest.raises(r.Phase11BError, match="pinned SHA256"):
        r.verified_inputs(spec)


@pytest.mark.parametrize("mutate, match", [
    (lambda v: v.update(f_duplicate_check_passed=False), "f_duplicate_check_passed"),
    (lambda v: v.update(nans_in_development_features=1), "nans_in_development_features"),
    (lambda v: v.update(sealed_content_used=True), "sealed_content_used"),
    (lambda v: v.update(final_panel_family_size=199), "final_families"),
    (lambda v: v["feature_sets"]["F"].reverse(), "feature_sets"),
    (lambda v: v.update(frame_columns=list(reversed(v["frame_columns"]))), "frame_columns"),
    (lambda v: v["frame_files"].pop("BA"), "universe"),
    (lambda v: v.update(stage="11B.2"), "stage"),
])
def test_verification_record_failing_any_check_stops(tmp_path, mutate, match):
    spec, _ = write_fixture(tmp_path, mutate_verification=mutate)
    with pytest.raises(r.Phase11BError, match=match):
        r.verified_inputs(spec)


def test_tampered_feature_frame_stops(tmp_path):
    spec, _ = write_fixture(tmp_path)
    p = tmp_path / "frames" / "AXP.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(r.Phase11BError, match="AXP: feature frame differs"):
        r.verified_inputs(spec)


def test_frame_paths_disagreeing_with_frame_files_stops(tmp_path):
    spec, _ = write_fixture(tmp_path, mutate_verification=lambda v: v["frame_paths"]["C"].update(BA="elsewhere.csv"))
    with pytest.raises(r.Phase11BError, match="frame_paths disagree"):
        r.verified_inputs(spec)


def test_confirmation_period_rows_are_rejected(tmp_path):
    dates = pd.bdate_range(end="2024-10-01", periods=ROWS)                   # last row is in the confirmation period
    spec, _ = write_fixture(tmp_path, dates=dates)
    inputs = r.verified_inputs(spec)
    with pytest.raises(r.Phase11BError, match="confirmation period is NOT_EVALUATED"):
        r.load_frames(inputs, spec)


@pytest.mark.parametrize("mutate, match", [
    (lambda s, f: f[[c for c in f.columns if c != "RSI"] + ["RSI"]], "frame columns"),
    (lambda s, f: f.iloc[:-1], "rows"),
    (lambda s, f: f.assign(RSI=np.where(np.arange(len(f)) == 4, np.nan, f["RSI"])) if s == "AXP" else f, "non-finite"),
    (lambda s, f: f.assign(Date=f["Date"] + pd.Timedelta(days=1)) if s == "BA" else f, "same development sessions"),
])
def test_frame_schema_rows_values_and_sessions_are_enforced(tmp_path, mutate, match):
    spec, _ = write_fixture(tmp_path, mutate_frame=mutate)
    inputs = r.verified_inputs(spec)
    with pytest.raises(r.Phase11BError, match=match):
        r.load_frames(inputs, spec)


# ============================================================
# Pre-development steps: order, evidence, masquerading
# ============================================================


def test_steps_must_run_in_order(tmp_path, fast):
    spec, paths = write_fixture(tmp_path)
    with pytest.raises(r.Phase11BError, match="'leakage' is not complete"):
        r.validate_smoke(spec, paths)
    r.validate_leakage(spec, paths, runner=fake_runner())
    with pytest.raises(r.Phase11BError, match="'smoke' is not complete"):
        r.validate_stock("aapl", spec, paths)
    r.validate_smoke(spec, paths)
    with pytest.raises(r.Phase11BError, match="'aapl' is not complete"):
        r.validate_stock("stock", spec, paths)
    with pytest.raises(r.Phase11BError, match="'aapl' is not complete"):
        r.freeze(spec, paths)
    with pytest.raises(r.Phase11BError, match="not found"):
        r.develop(spec, paths, workers=1, data_comparison=data_comparison_stub)


@pytest.mark.parametrize("rc, out", [
    (1, "1 failed, 34 passed in 3s"), (0, "34 passed, 1 skipped in 3s"), (0, "33 passed, 2 deselected in 3s"),
    (0, "1 error in 3s"), (0, ""), (5, "no tests ran in 0.1s"),
])
def test_leakage_step_requires_every_test_to_run_and_pass(tmp_path, rc, out):
    spec, paths = write_fixture(tmp_path)
    with pytest.raises(r.Phase11BError, match="did not all pass"):
        r.validate_leakage(spec, paths, runner=fake_runner(rc, out))
    assert not paths.evidence("leakage").exists()


def test_leakage_runner_selects_every_test_including_gate_authorization(monkeypatch):
    seen = {}
    monkeypatch.setattr(r.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd) or
                        type("P", (), {"returncode": 0, "stdout": "35 passed", "stderr": ""})())
    r.run_leakage_tests()
    assert r.LEAKAGE_TEST_FILE in seen["cmd"] and "gate_authorization or not gate_authorization" in seen["cmd"]


def test_evidence_is_write_once(tmp_path):
    spec, paths = write_fixture(tmp_path)
    r.validate_leakage(spec, paths, runner=fake_runner())
    with pytest.raises(r.Phase11BError, match="write-once"):
        r.validate_leakage(spec, paths, runner=fake_runner())


def test_smoke_is_labelled_synthetic_and_real_steps_are_labelled_by_specification(completed):
    spec, paths, _, _, _ = completed
    kinds = {s: json.loads(paths.evidence(s).read_text(encoding="utf-8"))["data_kind"] for s in r.STEPS}
    assert kinds["smoke"] == "synthetic"
    assert kinds["aapl"] == kinds["stock"] == kinds["panel"] == "synthetic_fixture"      # never "real" here
    for s in r.REAL_STEPS:                                                              # cannot satisfy the real run
        assert r.evidence_status(s, r.FROZEN_SPEC, None, paths)[0] is False


def test_validation_evidence_records_structure_not_results(completed):
    _, paths, _, _, _ = completed
    for s in ("aapl", "stock", "panel"):
        text = paths.evidence(s).read_text(encoding="utf-8")
        for forbidden in ("p_value", "QUALIFIED", "MSE", "Log_Loss", "Accuracy"):
            assert forbidden not in text, (s, forbidden)
    stock = json.loads(paths.evidence("stock").read_text(encoding="utf-8"))["details"]
    assert stock["symbol"] == "AXP" and set(stock["runs"]) == {"B", "C", "D", "F"}


def test_changed_inputs_invalidate_earlier_evidence(tmp_path, fast):
    spec, paths = write_fixture(tmp_path)
    run_steps(spec, paths, upto="aapl")
    p = tmp_path / "frames" / "BA.csv"
    f = pd.read_csv(p)
    f.loc[0, "RSI"] += 1.0
    p.write_bytes(fb._frame_csv(f.assign(Date=pd.to_datetime(f["Date"]))))
    v = json.loads(spec.feature_verification_path.read_text(encoding="utf-8"))
    v["frame_files"]["BA"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    spec.feature_verification_path.write_text(json.dumps(v), encoding="utf-8")
    spec = r.Spec(**{**spec.__dict__, "feature_verification_sha256": hashlib.sha256(spec.feature_verification_path.read_bytes()).hexdigest()})
    with pytest.raises(r.Phase11BError, match="'leakage' is not complete \\(recorded for a different specification\\)"):
        r.validate_stock("stock", spec, paths)


# ============================================================
# Registry freeze and integrity
# ============================================================


def test_registry_is_frozen_with_exact_ids_and_prerequisites(completed):
    spec, paths, reg, _, _ = completed
    assert reg["status"] == "FROZEN" and reg["data_kind"] == "synthetic_fixture"
    assert reg["matrix"]["counts"] == {"per_stock": 396, "panel": 132, "aapl_excluded_sensitivity": 132}
    assert set(reg["prerequisites"]) == set(r.STEPS) and reg["confirmation"] == "NOT_EVALUATED"
    assert r.load_registry(spec, paths)["matrix_sha256"] == reg["matrix_sha256"]


def test_registry_is_write_once(completed):
    spec, paths, _, _, _ = completed
    with pytest.raises(r.Phase11BError, match="frozen once"):
        r.freeze(spec, paths)


def test_edited_registry_or_evidence_is_rejected(tmp_path, fast):
    spec, paths = write_fixture(tmp_path)
    run_steps(spec, paths)
    r.freeze(spec, paths)
    original = paths.registry.read_bytes()
    reg = json.loads(original)
    reg["matrix"]["horizons"] = [1, 3]
    paths.registry.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(r.Phase11BError, match="matrix was edited"):
        r.load_registry(spec, paths)
    paths.registry.write_bytes(original)
    ev = paths.evidence("panel")
    ev.write_text(ev.read_text(encoding="utf-8").replace('"PASS"', '"PASS" '), encoding="utf-8")
    with pytest.raises(r.Phase11BError, match="'panel' changed after the registry was frozen"):
        r.load_registry(spec, paths)


# ============================================================
# Development run
# ============================================================


def test_develop_covers_exactly_the_matrix_and_defers_qualification(completed):
    spec, paths, reg, res, _ = completed
    out = res["results"]
    assert res["counts"] == {"per_stock": 396, "panel": 132, "aapl_excluded_sensitivity": 132}
    assert sorted(e["id"] for e in out["per_stock"]) == sorted(reg["matrix"]["comparison_ids"]["per_stock"])
    assert sorted(e["id"] for e in out["panel"]) == sorted(reg["matrix"]["comparison_ids"]["panel"])
    for fam in ("per_stock", "panel"):
        for e in out[fam]:
            assert e["final_qualification"] == r.DEFERRED and "qualified" not in e
            assert 0.0 <= e["p_value_holm_11b_stage_descriptive"] <= 1.0
            assert e["p_value_holm_11b_stage_descriptive"] >= e["p_value_raw"]
    assert all(e["p_value_holm_11b_stage_descriptive"] is None for e in out["aapl_excluded_sensitivity"])
    assert res["qualification"]["status"] == r.DEFERRED and res["confirmation"] == "NOT_EVALUATED"
    assert {e["symbol"] for e in out["per_stock"]} == set(UNIVERSE)
    assert all(e["n_symbols"] == 2 for e in out["aapl_excluded_sensitivity"])                    # AAPL excluded


def test_develop_artifacts_are_published_whole_and_hash_consistent(completed):
    _, paths, _, res, _ = completed
    assert paths.results.exists() and paths.development.is_dir() and not paths.staging.exists()
    for k, a in res["artifacts"].items():
        p = Path(a["path"]) if Path(a["path"]).is_absolute() else r.PROJECT_ROOT / a["path"]
        assert p.is_relative_to(paths.development) and hashlib.sha256(p.read_bytes()).hexdigest() == a["sha256"], k
    assert res["registry_sha256"] == hashlib.sha256(paths.registry.read_bytes()).hexdigest()


def test_amendment02_results_comparison_is_descriptive(completed):
    _, _, _, res, _ = completed
    rc = res["amendment_02"]["results_comparison"]
    assert rc["n_comparisons"] == 132 and rc["pass_fail"] is False and rc["affects_results"] is False
    assert res["amendment_02"]["data_comparison"]["affects_results"] is False


def test_develop_runs_once(completed):
    spec, paths, _, _, _ = completed
    with pytest.raises(r.Phase11BError, match="development results already exist"):
        r.develop(spec, paths, workers=1, data_comparison=data_comparison_stub)


def test_failed_develop_leaves_only_a_labelled_incomplete_folder(tmp_path, fast, monkeypatch):
    spec, paths = write_fixture(tmp_path)
    run_steps(spec, paths)
    r.freeze(spec, paths)

    def boom(*_a, **_k):
        raise RuntimeError("simulated failure after the models ran")
    monkeypatch.setattr(r, "amendment02_results_comparison", boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        r.develop(spec, paths, workers=1, log=lambda *_: None, data_comparison=data_comparison_stub)
    assert paths.staging.is_dir() and not paths.development.exists() and not paths.results.exists()
    with pytest.raises(r.Phase11BError, match="incomplete development run exists"):
        r.develop(spec, paths, workers=1, data_comparison=data_comparison_stub)


def test_data_comparison_failure_stops_before_any_model_runs(tmp_path, fast):
    spec, paths = write_fixture(tmp_path)
    run_steps(spec, paths)
    r.freeze(spec, paths)
    n_before = len(fast)

    def boom():
        raise r.Phase11BError("Phase 8 AAPL dataset metadata does not match its verified hash")
    with pytest.raises(r.Phase11BError, match="Phase 8"):
        r.develop(spec, paths, workers=1, log=lambda *_: None, data_comparison=boom)
    assert len(fast) == n_before and not paths.staging.exists() and not paths.results.exists()


def test_develop_refuses_changed_code(tmp_path, monkeypatch, fast):
    spec, paths = write_fixture(tmp_path)
    run_steps(spec, paths)
    r.freeze(spec, paths)
    real = r.code_sha256()
    monkeypatch.setattr(r, "code_sha256", lambda: {**real, "training/phase11b_research.py": "0" * 64})
    with pytest.raises(r.Phase11BError, match="prerequisite|code"):
        r.develop(spec, paths, workers=1, data_comparison=data_comparison_stub)


def test_real_per_stock_and_panel_runs_are_deterministic():
    """Two identical REAL runs (all 11 models, seed 42) give identical metrics and gate outcomes."""
    frames = r.synthetic_frames(n_stocks=2, n_rows=80, seed=5)
    cols = list(r.FEATURE_SETS["D"]["columns"])
    runs = [r.p10a.evaluate_excess(frames["SYN0"], cols, a11.candidates(), horizon=3, n_splits=3) for _ in range(2)]
    assert [(m["model"], m["metrics"], m["qualification"]) for m in runs[0]["models"]] == \
        [(m["model"], m["metrics"], m["qualification"]) for m in runs[1]["models"]]
    panels = [a11.evaluate_panel(frames, cols, a11.panel_candidates(), horizon=3, n_splits=3) for _ in range(2)]
    assert [(m["model"], m["metrics"]) for m in panels[0]["models"]] == [(m["model"], m["metrics"]) for m in panels[1]["models"]]


def test_stage_holm_is_descriptive_and_size_checked():
    entries = [{"p_value_raw": p} for p in (0.01, 0.04, 0.5)]
    r.stage_holm_descriptive(entries, 3)
    assert [e["p_value_holm_11b_stage_descriptive"] for e in entries] == [0.03, 0.08, 0.5]
    assert all(e["final_qualification"] == r.DEFERRED and "qualified" not in e for e in entries)
    with pytest.raises(r.Phase11BError, match="expected 4"):
        r.stage_holm_descriptive([{"p_value_raw": 0.1}] * 3, 4)


# ============================================================
# Amendment 02 data comparison: development only on both sides
# ============================================================


def _write_raw_canonical(path: Path, dicts: list[dict]):
    payload = "".join(json.dumps(d, sort_keys=True) + "\n" for d in dicts).encode("utf-8")
    path.write_bytes(payload)
    path.with_suffix(".meta.json").write_text(json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")
    return path, hashlib.sha256(payload).hexdigest()


def test_amendment02_data_comparison_uses_development_articles_only(tmp_path, monkeypatch):
    sentinel = "SEALED_AMENDMENT02_SENTINEL"
    shared_dev = news(1, CUT - 2 * H, headline="Apple beats estimates")
    sealed_11b = news(9, CUT + H, symbols=("AAPL",)).to_dict()
    sealed_11b.update(headline=123, summary=sentinel)                              # a full parse would raise
    env = make_env(tmp_path, dev=[shared_dev, news(2, CUT - H, symbols=("AAPL",)), news(3, CUT - H, symbols=("MSFT",))],
                   sealed=[sealed_11b])
    p8_sealed = news(9, CUT + H).to_dict()
    p8_sealed.update(headline=123, summary=sentinel)
    p8 = _write_raw_canonical(tmp_path / "phase8.jsonl",
                              [shared_dev.to_dict(), news(4, CUT - 3 * H).to_dict(), p8_sealed])
    parsed = []
    original = NewsArticle.from_dict
    monkeypatch.setattr(NewsArticle, "from_dict",
                        classmethod(lambda cls, d: (parsed.append(d.get("provider_article_id")), original(d))[1]))
    out = r.amendment02_data_comparison(report_path=env.report, phase8=lambda: p8)
    assert "9" not in parsed                                                         # sealed never parsed, either side
    assert (out["n_11b_dev_aapl"], out["n_phase8_dev_aapl"], out["n_shared"]) == (2, 2, 1)
    assert out["share_shared_identical_text_hash"] == 1.0 and out["share_shared_identical_information_available_at"] == 1.0
    assert out["affects_results"] is False and "development articles only" in out["scope"]
    assert sentinel not in json.dumps(out)


# ============================================================
# CLI and leakage surface
# ============================================================


@pytest.mark.parametrize("argv", [["validate", "--what", "confirmation"], ["develop", "--confirm"],
                                  ["holdout"], ["validate"], ["develop", "--include-sealed"]])
def test_cli_rejects_unknown_steps_confirmation_and_sealed_options(argv):
    with pytest.raises(SystemExit) as exc:
        r.main(argv)
    assert exc.value.code == 2


def test_module_never_references_full_canonical_loaders():
    tree = ast.parse(inspect.getsource(r))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"load_canonical_dataset", "load_sentiment_dataset", "load_event_dataset"}


def test_phase11a_functions_are_reused_unchanged():
    src = inspect.getsource(r)
    for fn in ("a11.evaluate_panel", "p10a.evaluate_excess", "a11.candidates", "a11.panel_candidates", "holm_adjust"):
        assert fn in src
    assert not any(name in src for name in ("def evaluate_panel", "def panel_walk_forward", "def evaluate_excess"))


def test_develop_requests_exactly_the_frozen_runs(completed):
    spec, _, _, _, calls = completed
    stock = [c for c in calls if c[0] == "stock"]
    panel = [c for c in calls if c[0] == "panel"]
    assert len(stock) == len(UNIVERSE) * 4 * 3 and len(panel) == 2 * 4 * 3
    want_cols = {tuple(r.FEATURE_SETS[fs]["columns"]) for fs in r.STAGE_FEATURE_SETS}
    assert {c[1] for c in stock} == want_cols and {c[2] for c in stock} == {1, 3, 5}
    assert all(c[3] == spec.n_splits and c[4] == ROWS for c in stock)
    assert {c[1] for c in panel} == {tuple(sorted(UNIVERSE)), tuple(sorted(set(UNIVERSE) - {"AAPL"}))}
    assert {c[2] for c in panel} == want_cols and {c[3] for c in panel} == {1, 3, 5} and {c[4] for c in panel} == {3}


def test_cached_real_runs_satisfy_the_real_structure_checks(cached):
    spec, _ = write_fixture(Path(__import__("tempfile").mkdtemp()))
    assert r._check_stock_run(cached["stock"], spec)["n_oos_rows"] == spec.oos_rows_per_stock_h1
    assert cached["panel"]["n_oos_rows"] == spec.panel_oos_rows_h1 and cached["panel"]["n_symbols"] == 3
