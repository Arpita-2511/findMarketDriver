"""
Phase 11B Stage 11B.5 (Amendment 06) - tests.

Synthetic fixtures build small families, a synthetic 11B registry / development record with run
artifacts, a synthetic Phase 11A registry, a synthetic Phase 11.0 design registry and a synthetic
amendment, all in a temporary directory; the audit, frame loading and model re-runs are replaced
by stand-ins. No model is fitted and no confirmation-period or sealed data is touched.

Two tests read the committed real evidence (JSON only, no model fitting): the real families
(F17) and the exact re-derivation of every 11B entry from the published artifacts (F18).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pytest

import training.phase10a_research as p10a
import training.phase11_design as p11d
import training.phase11b_final as f
import training.phase11b_research as r
from training.phase9_research import holm_adjust

UNIVERSE = ("AAPL", "AXP", "BA")
COUNTS = (("per_stock", 3, 6), ("panel", 2, 3), ("aapl_excluded_sensitivity", 2, 3))
MODELS = ("Linear Regression", "Random Forest")
ENV = {"python": "3.14.6", "packages": {"numpy": "2.5.1", "pandas": "3.0.3"}}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return p


def _entry(i, fam, phase, p, gate="NOT_QUALIFIED", sym="AAPL", fs="B", h=1, model="Linear Regression"):
    e = {"id": i, "feature_set": fs, "horizon": h, "model": model, "task": "regression",
         "gate_status": gate, "p_value_raw": p}
    if fam == "per_stock":
        e["symbol"] = sym
    return e


def _run_obj(tag: str, warnings: int = 0) -> dict:
    return {"models": [{"model": "Linear Regression", "metrics": {"MSE": 0.25, "tag": tag}}],
            "fold_metrics": [{"fold": 1, "MSE": 0.125}], "warnings": {"total": warnings, "by_message": {}}}


def make_world(tmp: Path, *, b_per_stock=None, a_per_stock=None):
    """Synthetic sources; returns (Sources, spec, dev, reg)."""
    out = tmp / "11B"
    a_ps = a_per_stock or [_entry(f"a_ps_{k}", "per_stock", "11A", 0.5 + k / 10, fs="A") for k in range(3)]
    b_ps = b_per_stock or [
        _entry("b_ps_0", "per_stock", "11B", 0.30, sym="AXP", fs="C", h=1),
        _entry("b_ps_1", "per_stock", "11B", 0.06, sym="BA", fs="F", h=3),
        _entry("b_ps_2", "per_stock", "11B", 0.90, sym="AAPL", fs="B", h=5),
        _entry("b_ps_3", "per_stock", "11B", 0.07, sym="AXP", fs="D", h=1),
        _entry("b_ps_4", "per_stock", "11B", 0.08, sym="BA", fs="F", h=3, model="Random Forest"),
        _entry("b_ps_5", "per_stock", "11B", 0.99, sym="AAPL", fs="C", h=3)]
    a_pn = [_entry(f"a_pn_{k}", "panel", "11A", 0.8, fs="E") for k in range(2)]
    b_pn = [_entry("b_pn_0", "panel", "11B", 0.70, fs="D", h=3), _entry("b_pn_1", "panel", "11B", 0.40, fs="B", h=5),
            _entry("b_pn_2", "panel", "11B", 0.95, fs="F", h=1)]
    a_se = [_entry(f"a_se_{k}", "sens", "11A", 0.6) for k in range(2)]
    b_se = [_entry(f"b_se_{k}", "sens", "11B", 0.03 + k) if k == 0 else _entry(f"b_se_{k}", "sens", "11B", 0.5)
            for k in range(3)]
    ids = {"per_stock": [e["id"] for e in a_ps + b_ps][::-1], "panel": [e["id"] for e in b_pn + a_pn],
           "aapl_excluded_sensitivity": [e["id"] for e in a_se + b_se]}
    design = {"comparison_ids": ids}
    design_path = _write(tmp / "design_registry.json", {"registry_version": p11d.REGISTRY_VERSION, "design": design,
                                                        "design_sha256": p11d.design_sha256(design)})
    a11_path = _write(tmp / "11A" / "registry.json", {"results": {"per_stock": a_ps, "panel": a_pn,
                                                                  "aapl_excluded_sensitivity": a_se}})
    artifacts = {}
    for s in UNIVERSE:
        p = _write(out / "development" / "experiments" / f"stock_{s}.json",
                   {"symbol": s, "runs": {f"{fs}_h{h}": _run_obj(f"{s}{fs}{h}", warnings=3 if s == "AXP" else 0)
                                          for fs, h in f.CELLS}})
        artifacts[f"stock_{s}"] = {"path": str(p), "sha256": _sha(p)}
    for kind in ("panel", "panel_exAAPL"):
        for fs, h in f.CELLS:
            p = _write(out / "development" / kind / f"{kind}_{fs}_h{h}.json",
                       {"kind": kind, "feature_set": fs, "horizon": h, "phase": "11B", "matrix_sha256": "m",
                        **_run_obj(f"{kind}{fs}{h}")})
            artifacts[f"{kind}_{fs}_h{h}"] = {"path": str(p), "sha256": _sha(p)}
    reg = {"status": "FROZEN", "environment": ENV, "matrix_sha256": "m", "data_kind": "synthetic_fixture"}
    reg_path = _write(out / "registry.json", reg)
    dev = {"document_type": "phase_11b_development_results", "status": "COMPLETE", "registry_sha256": _sha(reg_path),
           "artifacts": artifacts, "confirmation": "NOT_EVALUATED",
           "results": {"per_stock": b_ps, "panel": b_pn, "aapl_excluded_sensitivity": b_se}}
    _write(out / "development_results.json", dev)
    amend = _write(out / "amendment_06_final_audit_procedure.json", {"amendment_id": "06", "status": "ADOPTED"})
    src = f.Sources(phase11a_registry=a11_path, phase11a_sha256=_sha(a11_path), design_registry=design_path,
                    research=r.Paths(out_dir=out), amendment=amend, amendment_sha256=_sha(amend),
                    matrix_sha256="m", family_counts=COUNTS, protected_dirs=(tmp / "11A", out))
    spec = r.Spec(universe=UNIVERSE)
    return src, spec, json.loads((out / "development_results.json").read_text()), reg


def fake_hooks(dev, reg, *, rerun_mutation=None, env=None, audit_passed=True, during=None):
    calls = []

    def audit(src, spec):
        return {"passed": audit_passed, "checks": {"A1_frozen_inputs": {"pass": audit_passed}},
                "context": {"dev": dev, "registry": reg, "inputs": {}}}

    def rerun(frames, run, n_splits, matrix_sha):
        calls.append((run["kind"], run["symbol"], run["feature_set"], run["horizon"], n_splits))
        if during:
            during()
        obj = copy.deepcopy(f.stored_run(dev, run))
        obj["warnings"] = {"total": 99, "by_message": {"x": 99}}           # never part of the comparison
        if rerun_mutation:
            rerun_mutation(run, obj)
        return obj

    hooks = f.Hooks(audit=audit, load_frames=lambda inputs, spec: {}, rerun_stock=rerun, rerun_panel=rerun,
                    environment=lambda: env or ENV, log=lambda m: None)
    return hooks, calls


# ============================================================
# F1-F4 families
# ============================================================


def _fam(n_a, n_b, p=0.5):
    a = [_entry(f"a{k}", "panel", "11A", p) for k in range(n_a)]
    b = [_entry(f"b{k}", "panel", "11B", p) for k in range(n_b)]
    return a, b, [e["id"] for e in a + b]


def test_f1_exact_family_sizes():
    a, b, ids = _fam(2, 3)
    assert len(f.assemble_family("panel", a, b, ids, (2, 3))) == 5
    with pytest.raises(f.Phase11B5Error, match="FAMILY_INTEGRITY_FAILED"):
        f.assemble_family("panel", a, b[:2], ids, (2, 3))                     # one missing
    with pytest.raises(f.Phase11B5Error, match="FAMILY_INTEGRITY_FAILED"):
        f.assemble_family("panel", a, b + [_entry("b9", "panel", "11B", 0.5)], ids + ["b9"], (2, 3))   # one extra
    with pytest.raises(f.Phase11B5Error, match="distinct"):
        f.assemble_family("panel", a, b, ids[:-1] + [ids[0]], (2, 3))         # duplicate in the frozen list


def test_f2_disjoint_ids():
    a, b, ids = _fam(2, 3)
    b[0]["id"] = "a0"
    with pytest.raises(f.Phase11B5Error, match="appears twice"):
        f.assemble_family("panel", a, b, ["a0", "a1", "a0x", "b1", "b2"], (2, 3))


def test_f3_membership_against_frozen_design():
    a, b, ids = _fam(2, 3)
    with pytest.raises(f.Phase11B5Error, match="differs from the frozen design"):
        f.assemble_family("panel", a, b, ids[:-1] + ["not_in_results"], (2, 3))


def test_f4_order_is_frozen_design_order_and_output_is_byte_identical():
    a, b, ids = _fam(3, 4)
    for i, e in enumerate(a + b):
        e["p_value_raw"] = round(0.1 * (i + 1), 3)
    design = list(reversed(ids))
    out1 = f.assemble_family("panel", a, b, design, (3, 4))
    rnd = random.Random(7)
    a2, b2 = a[:], b[:]
    rnd.shuffle(a2), rnd.shuffle(b2)
    out2 = f.assemble_family("panel", a2, b2, design, (3, 4))
    assert [x["id"] for x in out1] == design
    f.combined_holm(out1), f.combined_holm(out2)
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


# ============================================================
# F5-F10 Holm and qualification
# ============================================================


@pytest.mark.parametrize("p, expected", [
    ([0.01, 0.04, 0.03, 0.005], [0.03, 0.06, 0.06, 0.02]),
    ([0.5], [0.5]),
    ([0.3, 0.4], [0.6, 0.6]),
    ([0.6, 0.9], [1.0, 1.0]),
])
def test_f5_holm_known_values(p, expected):
    assert holm_adjust(p) == pytest.approx(expected, abs=1e-12)
    rows = [{"gate_status": "NOT_QUALIFIED", "p_value_raw": x} for x in p]
    f.combined_holm(rows)
    assert [x["p_value_holm_final"] for x in rows] == pytest.approx(expected, abs=1e-12)


def test_f6_ties_are_order_independent():
    assert holm_adjust([0.02, 0.02, 0.5]) == pytest.approx([0.06, 0.06, 0.5])
    assert holm_adjust([0.5, 0.02, 0.02]) == pytest.approx([0.5, 0.06, 0.06])


def test_f7_boundaries_are_strict():
    assert f.qualify_final("QUALIFIED", 0.05, 0.01)["raw_ok"] is False
    assert f.qualify_final("QUALIFIED", 0.01, 0.05)["holm_ok"] is False
    assert f.qualify_final("QUALIFIED", 0.0, 0.0)["qualified"] is True
    a, b, ids = _fam(1, 1, p=0.0)
    assert len(f.assemble_family("panel", a, b, ids, (1, 1))) == 2       # p = 0 is a valid p-value


@pytest.mark.parametrize("bad", [float("nan"), None, -0.1, 1.1, "0.1", True, float("inf")])
def test_f8_invalid_p_values_stop_the_run(bad):
    a, b, ids = _fam(1, 2)
    b[1]["p_value_raw"] = bad
    with pytest.raises(f.Phase11B5Error, match="invalid raw p-value"):
        f.assemble_family("panel", a, b, ids, (1, 2))


def test_f8_unexpected_gate_status_stops_the_run():
    a, b, ids = _fam(1, 2)
    b[0]["gate_status"] = "BASELINE"
    with pytest.raises(f.Phase11B5Error, match="gate status"):
        f.assemble_family("panel", a, b, ids, (1, 2))


@pytest.mark.parametrize("gate, raw, holm, qualified", [
    ("QUALIFIED", 0.01, 0.04, True),
    ("QUALIFIED", 0.01, 0.20, False),
    ("NOT_QUALIFIED", 0.001, 0.01, False),
    ("QUALIFIED", 0.06, 0.06, False),
])
def test_f9_gate_raw_and_holm_must_all_pass(gate, raw, holm, qualified):
    assert f.qualify_final(gate, raw, holm)["qualified"] is qualified


def test_f9_holm_is_computed_for_every_comparison_and_never_overrides_the_gate():
    rows = [{"gate_status": "NOT_QUALIFIED", "p_value_raw": 0.0001}, {"gate_status": "QUALIFIED", "p_value_raw": 0.001}]
    f.combined_holm(rows)
    assert all(x["p_value_holm_final"] is not None for x in rows)
    assert [x["qualified"] for x in rows] == [False, True]


def test_f10_sensitivity_family_never_gets_holm_or_qualifies(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    fams = f.build_families(f.load_phase11a(src), dev, f.load_design_ids(src), src)
    sens = fams["aapl_excluded_sensitivity"]
    assert all(x["p_value_holm_final"] is None and x["qualified"] is None and x["holm_ok"] is None for x in sens)
    assert any(x["raw_ok"] for x in sens)                                   # a raw p < 0.05 still never qualifies
    s = f.family_summary(sens, "aapl_excluded_sensitivity")
    assert s["qualified"] is None and s["method"] is None and not s["can_qualify"]
    assert all(x["p_value_holm_final"] is not None for fam in f.HOLM_FAMILIES for x in fams[fam])


# ============================================================
# F11-F12 determinism sample and comparison
# ============================================================


def _expected_hash_symbol(fs, h, universe):
    return universe[int(hashlib.sha256(f"phase11b5|determinism_v1|{fs}|{h}".encode()).hexdigest(), 16) % len(universe)]


def test_f11_sample_selection_is_pure_stable_and_follows_amendment_06(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    ps, pn = dev["results"]["per_stock"], dev["results"]["panel"]
    s1 = f.select_determinism_sample(ps, pn, UNIVERSE)
    assert s1 == f.select_determinism_sample(copy.deepcopy(ps), copy.deepcopy(pn), UNIVERSE)
    cov = [x for x in s1 if x["stratum"] == "S1_coverage"]
    assert [(x["feature_set"], x["horizon"]) for x in cov] == list(f.CELLS)
    assert all(x["symbol"] == _expected_hash_symbol(x["feature_set"], x["horizon"], UNIVERSE) for x in cov)
    # S2: three smallest 11B per-stock p-values 0.06 (BA F h3), 0.07 (AXP D h1), 0.08 (BA F h3, same run)
    s2 = [(x["symbol"], x["feature_set"], x["horizon"]) for x in s1 if x["stratum"] == "S2_boundary"]
    expected_s2 = []
    for key in (("BA", "F", 3), ("AXP", "D", 1)):
        if key[0] != _expected_hash_symbol(key[1], key[2], UNIVERSE) and key not in expected_s2:
            expected_s2.append(key)
    assert s2 == expected_s2                                                # duplicates skipped, never replaced
    s3 = [x for x in s1 if x["stratum"] == "S3_panel"]
    assert [(x["kind"], x["feature_set"], x["horizon"]) for x in s3] == [
        ("panel", "B", 5),
        ("panel_exAAPL", *f.CELLS[int(hashlib.sha256(b"phase11b5|determinism_v1|panel_exAAPL").hexdigest(), 16) % 12])]


def test_f11_ties_in_boundary_stratum_use_frozen_order():
    ps = [_entry("x1", "per_stock", "11B", 0.1, sym="BA", fs="B", h=1),
          _entry("x2", "per_stock", "11B", 0.1, sym="AXP", fs="C", h=1)]
    pn = [_entry("p1", "panel", "11B", 0.5, fs="F", h=5)]
    picked = [x for x in f.select_determinism_sample(ps, pn, ("ZZZ",)) if x["stratum"] == "S2_boundary"]
    assert [x["symbol"] for x in picked] == ["BA", "AXP"]


def test_f12_compare_run_ignores_only_warnings():
    a = _run_obj("t", warnings=0)
    b = copy.deepcopy(a)
    b["warnings"] = {"total": 10527, "by_message": {"UserWarning": 10527}}
    res = f.compare_run(a, b)
    assert res["identical"] and res["stored_sha256"] == res["rerun_sha256"]
    assert res["warnings_total"] == {"stored": 0, "rerun": 10527}
    c = copy.deepcopy(a)
    c["fold_metrics"][0]["MSE"] = 0.125 + 1e-15
    res = f.compare_run(a, c)
    assert not res["identical"] and res["n_differences"] == 1
    assert res["differences"][0]["path"] == "/fold_metrics[0]/MSE" and res["differences"][0]["abs_diff"] > 0
    d = copy.deepcopy(a)
    del d["models"][0]["metrics"]["tag"]
    assert f.compare_run(a, d)["differences"][0]["rerun"] == "<missing>"
    e = copy.deepcopy(a)
    e["models"].append({"model": "extra"})
    assert any(x["path"].endswith("#len") for x in f.compare_run(a, e)["differences"])


def test_f12_rerun_stock_calls_evaluate_excess_exactly_as_the_11b4_job(monkeypatch):
    seen = []
    monkeypatch.setattr(p10a, "evaluate_excess", lambda frame, cols, cands, horizon, n_splits:
                        seen.append((frame, tuple(cols), tuple(c.name for c in cands), horizon, n_splits)) or {})
    r._stock_job("AXP", "FRAME", 20)
    job = {(x[3], x[1]): x for x in seen}
    seen.clear()
    f.rerun_stock({"AXP": "FRAME"}, {"kind": "stock", "symbol": "AXP", "feature_set": "D", "horizon": 3}, 20, "m")
    assert seen[0] == job[(3, tuple(r.FEATURE_SETS["D"]["columns"]))]


def test_f12_rerun_panel_wraps_like_the_11b4_artifact(monkeypatch):
    monkeypatch.setattr(r, "_panel_job", lambda kind, fs, h, frames, n: {"result": {"models": [], "n_rows": 1}})
    out = f.rerun_panel({}, {"kind": "panel_exAAPL", "feature_set": "C", "horizon": 5}, 20, "msha")
    assert out == {"kind": "panel_exAAPL", "feature_set": "C", "horizon": 5, "phase": "11B",
                   "matrix_sha256": "msha", "models": [], "n_rows": 1}


# ============================================================
# Orchestration F13-F16 and happy paths
# ============================================================


def test_happy_path_publishes_write_once_final(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    hooks, calls = fake_hooks(dev, reg)
    rep = f.run_final(src, spec, hooks)
    assert sorted(p.name for p in src.final.iterdir()) == sorted(f.OUTPUT_FILES + ("manifest.json",))
    assert not src.staging.exists()
    f.verify_manifest(src)
    assert rep["qualification"]["total_qualified"] == 0
    assert rep["determinism"]["outcome"] == "DETERMINISM_VERIFIED"
    assert rep["phase_11b_acceptance"] == "PENDING_OWNER_REVIEW"
    assert rep["confirmation"] == {"status": "NOT_EVALUATED", "applicability": "NOT_APPLICABLE",
                                   "authorization": None, "sealed_content_read": False}
    proto = json.loads((src.final / "protocol.json").read_text())
    assert len(calls) == proto["determinism"]["n_runs"] == len(f.select_determinism_sample(
        dev["results"]["per_stock"], dev["results"]["panel"], UNIVERSE))
    holm = json.loads((src.final / "combined_holm.json").read_text())
    assert [x["id"] for x in holm["families"]["per_stock"]] == json.loads(
        src.design_registry.read_text())["design"]["comparison_ids"]["per_stock"]
    assert (src.final / "final_report.md").read_text().startswith("# Phase 11B.5 Final Report")
    assert any("final            : COMPLETE" in x for x in f.status_lines(src))


def test_qualifier_present_marks_confirmation_pending_authorization(tmp_path):
    b_ps = [_entry(f"b_ps_{k}", "per_stock", "11B", 0.9, sym="AXP", fs="C") for k in range(6)]
    b_ps[0].update(gate_status="QUALIFIED", p_value_raw=1e-6)
    src, spec, dev, reg = make_world(tmp_path, b_per_stock=b_ps)
    f.write_protocol(src, spec)
    rep = f.run_final(src, spec, fake_hooks(dev, reg)[0])
    assert rep["qualification"]["families"]["per_stock"]["qualified_ids"] == ["b_ps_0"]
    assert rep["confirmation"]["applicability"] == "PENDING_OWNER_AUTHORIZATION"
    assert rep["confirmation"]["status"] == "NOT_EVALUATED"


def test_determinism_failure_preserves_results_and_withholds_acceptance(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    before = f.snapshot(src)

    def mutate(run, obj):
        if run["stratum"] == "S1_coverage" and run["feature_set"] == "C" and run["horizon"] == 3:
            obj["fold_metrics"][0]["MSE"] += 1e-12

    rep = f.run_final(src, spec, fake_hooks(dev, reg, rerun_mutation=mutate)[0])
    assert rep["determinism"]["outcome"] == "DETERMINISM_CHECK_FAILED"
    assert rep["phase_11b_acceptance"] == "NOT_ACCEPTED_DETERMINISM_FAILED"
    det = json.loads((src.final / "determinism.json").read_text())
    bad = [x for x in det["runs"] if not x["identical"]]
    assert len(bad) == 1 and bad[0]["differences"][0]["path"] == "/fold_metrics[0]/MSE"
    assert f.snapshot(src) == before                                        # frozen evidence untouched


def test_f13_environment_mismatch_refuses_the_rerun(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    hooks, calls = fake_hooks(dev, reg, env={"python": "3.14.6", "packages": {"numpy": "2.5.2", "pandas": "3.0.3"}})
    with pytest.raises(f.Phase11B5Error, match="ENVIRONMENT_MISMATCH"):
        f.run_final(src, spec, hooks)
    assert calls == [] and not src.final.exists()
    det = json.loads((src.staging / "determinism.json").read_text())
    assert det["outcome"] == "ENVIRONMENT_MISMATCH" and det["mismatches"] == ["numpy 2.5.2 != 2.5.1"]


def test_environment_mismatches_lists_every_difference():
    assert f.environment_mismatches(ENV, ENV) == []
    got = f.environment_mismatches({"python": "3.13.0", "packages": {"numpy": "2.5.1"}}, ENV)
    assert got == ["python 3.13.0 != 3.14.6", "pandas None != 3.0.3"]


def test_audit_failure_writes_only_failure_evidence(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    hooks, calls = fake_hooks(dev, reg, audit_passed=False)
    with pytest.raises(f.Phase11B5Error, match="AUDIT_FAILED"):
        f.run_final(src, spec, hooks)
    assert sorted(p.name for p in src.staging.iterdir()) == ["audit.json", "protocol.json"]
    assert json.loads((src.staging / "audit.json").read_text())["outcome"] == "AUDIT_FAILED"
    assert calls == [] and not src.final.exists()


def test_f14_write_once_final_and_protected_evidence(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    f.run_final(src, spec, fake_hooks(dev, reg)[0])
    with pytest.raises(f.Phase11B5Error, match="runs once"):
        f.write_protocol(src, spec)
    with pytest.raises(f.Phase11B5Error, match="runs once"):
        f.run_final(src, spec, fake_hooks(dev, reg)[0])


def test_f14_a8_detects_a_changed_pre_existing_file(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    victim = src.out_dir / "development_results.json"
    hooks, _ = fake_hooks(dev, reg, during=lambda: victim.write_text(victim.read_text() + " "))
    with pytest.raises(f.Phase11B5Error, match="pre-existing evidence changed"):
        f.run_final(src, spec, hooks)
    assert not src.final.exists()
    audit = json.loads((src.staging / "audit.json").read_text())
    assert audit["checks"]["A8_write_once"]["pass"] is False


def test_f15_failure_leaves_staging_and_status_reports_incomplete(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    f.write_protocol(src, spec)
    assert any("protocol written, not run" in x for x in f.status_lines(src))

    def boom(*a):
        raise RuntimeError("worker crashed")

    hooks, _ = fake_hooks(dev, reg)
    hooks.rerun_stock = boom
    with pytest.raises(RuntimeError):
        f.run_final(src, spec, hooks)
    assert src.staging.exists() and not src.final.exists()
    assert any("INCOMPLETE run present" in x for x in f.status_lines(src))
    with pytest.raises(f.Phase11B5Error, match="earlier attempt"):
        f.run_final(src, spec, fake_hooks(dev, reg)[0])
    with pytest.raises(f.Phase11B5Error, match="move it aside"):
        f.write_protocol(src, spec)


def test_f16_protocol_is_required_and_cannot_be_edited(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    with pytest.raises(f.Phase11B5Error, match="run `protocol` first"):
        f.run_final(src, spec, fake_hooks(dev, reg)[0])
    f.write_protocol(src, spec)
    p = src.staging / "protocol.json"
    proto = json.loads(p.read_text())
    proto["determinism"]["sample"] = proto["determinism"]["sample"][:-1]
    p.write_text(json.dumps(proto, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(f.Phase11B5Error, match="no longer matches Amendment 06"):
        f.run_final(src, spec, fake_hooks(dev, reg)[0])


def test_protocol_refuses_an_unpinned_amendment_or_11a_registry(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    bad = f.Sources(**{**src.__dict__, "amendment_sha256": "0" * 64})
    with pytest.raises(f.Phase11B5Error, match="Amendment 06"):
        f.write_protocol(bad, spec)
    bad = f.Sources(**{**src.__dict__, "phase11a_sha256": "0" * 64})
    with pytest.raises(f.Phase11B5Error, match="11A registry"):
        f.write_protocol(bad, spec)
    assert not src.staging.exists()


def test_protocol_refuses_results_from_another_registry(tmp_path):
    src, spec, dev, reg = make_world(tmp_path)
    reg_path = src.research.registry
    reg_path.write_text(reg_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(f.Phase11B5Error, match="does not belong to the frozen registry"):
        f.write_protocol(src, spec)


# ============================================================
# Real pins and real evidence (JSON only; no model fitting)
# ============================================================


def test_amendment_06_pin_and_content():
    assert _sha(f.AMENDMENT_06_PATH) == f.AMENDMENT_06_SHA256
    a = json.loads(f.AMENDMENT_06_PATH.read_text(encoding="utf-8"))
    assert a["status"] == "ADOPTED" and a["weakens_thresholds"] is False
    assert sorted(a["owner_decisions"]) == ["D1", "D2", "D3", "D4", "D5", "D6"]
    assert all(v.startswith("APPROVED") for v in a["owner_decisions"].values())
    design = json.loads((r.PHASE_11B_DIR / "design_freeze.json").read_text(encoding="utf-8"))
    final = design["multiple_testing"]["final_families"]
    assert (a["families"]["primary_panel"]["size"], a["families"]["secondary_per_stock"]["size"]) == \
        (final["panel"]["total"], final["per_stock"]["total"])
    assert f.CODE_FILES[1:] == r.CODE_FILES


def test_f17_real_families_match_the_frozen_design():
    src = f.DEFAULT_SOURCES
    dev, ids, reg = f._verify_sources(src)
    fams = f.build_families(f.load_phase11a(src), dev, ids, src)
    assert {k: len(v) for k, v in fams.items()} == {"per_stock": 5742, "panel": 198, "aapl_excluded_sensitivity": 198}
    for fam, rows in fams.items():
        assert [x["id"] for x in rows] == ids[fam]
        a = {x["id"] for x in rows if x["phase"] == "11A"}
        b = {x["id"] for x in rows if x["phase"] == "11B"}
        assert not a & b and (len(a), len(b)) == src.counts(fam)
        assert all(math.isfinite(x["p_value_raw"]) for x in rows)


def test_f18_real_entries_rederive_exactly_from_the_published_artifacts():
    dev = f.load_development(f.DEFAULT_SOURCES)
    got = f.rederive_entries(dev, r.FROZEN_SPEC)
    stored = [dev["results"][k] for k in f.FAMILIES]
    assert f._plain(list(got)) == f._plain(stored)
