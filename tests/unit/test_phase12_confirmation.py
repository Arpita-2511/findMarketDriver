"""
Phase 12 Stage 3A finalist freeze and sealed-confirmation gate - synthetic tests.

A synthetic Stage 2 directory holds a finalist fitted with the frozen Stage 2 config on fabricated development
frames; "sealed" frames are fabricated too (dates in the confirmation window). No real sealed path exists here.
The real Stage 2 artifact is only hash-checked and inspected.
"""

from __future__ import annotations

import json
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

import training.phase12_confirmation as c
import training.phase12_magnitude as m2
from tests.unit.test_phase12_magnitude import planted
from tests.unit.test_phase12_signal_diagnostics import make_frame
from training.build_dataset import file_sha256

UNIVERSE = ("AAA", "BBB", "CCC")


def sealed_frames(n=450, seed=100, strength=0.9) -> dict:
    """Confirmation-window frames with the same planted Volume_Change_1 -> |excess return| link."""
    out = {}
    for i, sym in enumerate(UNIVERSE):
        rng = np.random.default_rng(seed + i)
        f = make_frame(n, seed=seed + i, start="2024-10-01")
        vc = rng.normal(size=n)
        sigma = 0.01 * np.exp(strength * vc)
        f["Volume_Change_1"] = vc
        f["Close"] = 100 * np.exp(np.cumsum(np.concatenate([[0.0], sigma[:-1] * rng.normal(size=n - 1)])))
        f["SPY_Close"] = 300 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        out[sym] = f
    return out


@pytest.fixture(scope="module")
def dev():
    return planted(n=520, k=3)


@pytest.fixture
def env(tmp_path, monkeypatch, dev):
    stage2 = tmp_path / "stage2"
    (stage2 / "results" / "finalists").mkdir(parents=True)
    joblib.dump(m2.fit_finalist(dev, "lightgbm", 1, m2.FROZEN_CONFIG),
                stage2 / "results" / "finalists" / c.FINALIST_FILE)
    for rel in ("protocol.json", "results/manifest.json", "results/results.json"):
        (stage2 / rel).write_text(f'{{"synthetic": "{rel}"}}\n')
    pins = {rel: file_sha256(stage2 / rel.removeprefix("stage2/")) for rel in c.STAGE2_PINS}
    monkeypatch.setattr(c, "STAGE2_PINS", pins)
    monkeypatch.setattr(c, "FINALIST_SHA256", pins["stage2/results/finalists/pooled_lightgbm_h1.joblib"])
    builder = tmp_path / "phase12_sealed_features.py"
    builder.write_text("# synthetic sealed feature builder\n")
    monkeypatch.setattr(c, "SEALED_BUILDER_FILE", str(builder))
    out = tmp_path / "stage3"
    c.freeze(out, stage2, frames=dev, inputs_sha="synthetic")
    sha = file_sha256(out / c.AMENDMENT_FILE)
    return types.SimpleNamespace(stage2=stage2, out=out, sha=sha, builder=builder, dev=dev)


def authorize(e, **over) -> None:
    rec = {"amendment_sha256": e.sha, "runner_code_sha256": file_sha256(c.PROJECT_ROOT / "training/phase12_confirmation.py"),
           "sealed_builder_code_sha256": file_sha256(e.builder), "authorized_by": "owner",
           "authorized_at": "2026-11-01T00:00:00Z", "statement": c.AUTHORIZATION_STATEMENT, **over}
    (e.out / c.AUTHORIZATION_FILE).write_text(json.dumps(rec))


def run_confirm(e, loader, **kw):
    return c.confirm(e.out, e.stage2, frames=e.dev, sealed_loader=loader, pinned_sha=kw.pop("pinned_sha", e.sha),
                     universe=UNIVERSE, **kw)


def never_called():
    raise AssertionError("sealed loader was called")


# ============================================================
# Frozen rules and the real Stage 2 evidence
# ============================================================


def test_frozen_rules_reuse_stage2_thresholds():
    rl, cfg = c.rules(), m2.FROZEN_CONFIG
    assert rl["finalist"]["features_in_order"] == list(m2.FEATURES)
    assert (cfg.min_rel_mse_improvement, cfg.min_rel_mae_improvement, cfg.alpha, cfg.min_blocks_improved,
            cfg.n_blocks, cfg.min_share_stocks_improved) == (0.01, 0.0, 0.05, 3, 4, 0.6)
    assert rl["config"]["stage2_config"] == m2.dataclasses.asdict(cfg)
    assert rl["statistics"]["family"].startswith("1 comparison")
    assert {"pooled Ridge h=1", "pooled Random Forest h=1", "every h=5 model", "every per-stock model"} <= set(rl["excluded"])
    assert (c.CONFIRMATION_START.isoformat(), c.CONFIRMATION_END.isoformat()) == ("2024-10-01", "2026-09-25")
    assert set(rl["criteria_review"]) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}


def test_real_stage2_finalist_hash_and_structure():
    path = m2.OUT_DIR / "results" / "finalists" / c.FINALIST_FILE
    if not path.is_file():
        pytest.skip("Stage 2 finalist artifact not present")
    c.verify_stage2()
    art = c.load_artifact(path, c.FINALIST_SHA256)
    c.check_artifact(art, "lightgbm")
    assert art["features"] == list(m2.FEATURES) and art["horizon"] == 1
    assert art["trained_on"][1] <= "2024-09-30" and len(art["per_stock_scaling"]) == 29
    assert art["protocol_sha256"] == c.STAGE2_PROTOCOL_SHA256 == m2.protocol_sha256()


def test_real_amendment_matches_its_pin_and_the_rules_in_code():
    if not (c.OUT_DIR / c.AMENDMENT_FILE).is_file():
        pytest.skip("Amendment 07 not frozen")
    doc = c.load_amendment()
    assert doc["sealed_data_read"] is False
    assert doc["pins"]["finalist"]["file_sha256"] == c.FINALIST_SHA256
    assert doc["pins"]["inputs_sha256"] == c.INPUTS_SHA256
    assert doc["pins"]["stage2_files_sha256"] == c.STAGE2_PINS
    assert file_sha256(c.OUT_DIR / c.BASELINE_FILE) == doc["pins"]["baseline"]["file_sha256"]


# ============================================================
# Artifacts, preprocessing, target, baseline
# ============================================================


def test_load_artifact_checks_hash_before_unpickling(env, monkeypatch):
    p = env.stage2 / "results" / "finalists" / c.FINALIST_FILE
    good = file_sha256(p)
    monkeypatch.setattr(joblib, "load", lambda *_: (_ for _ in ()).throw(AssertionError("unpickled")))
    with pytest.raises(c.Phase12S3Error, match="pinned SHA256"):
        c.load_artifact(p, "0" * 64)
    with pytest.raises(c.Phase12S3Error, match="not found"):
        c.load_artifact(p.with_name("missing.joblib"), good)


def test_check_artifact_refuses_feature_order_and_hyperparameter_changes(env):
    art = c.load_artifact(env.stage2 / "results" / "finalists" / c.FINALIST_FILE, c.FINALIST_SHA256)
    c.check_artifact(art, "lightgbm")
    with pytest.raises(c.Phase12S3Error, match="feature order"):
        c.check_artifact({**art, "features": list(reversed(art["features"]))}, "lightgbm")
    est = m2.make_model("lightgbm", "pooled", m2.FROZEN_CONFIG).set_params(learning_rate=0.1)
    with pytest.raises(c.Phase12S3Error, match="hyperparameters"):
        c.check_artifact({**art, "model": est}, "lightgbm")
    with pytest.raises(c.Phase12S3Error, match="horizon"):
        c.check_artifact({**art, "horizon": 5}, "lightgbm")


def test_inference_reproduces_the_stage2_training_transform(env, dev):
    art = c.load_artifact(env.stage2 / "results" / "finalists" / c.FINALIST_FILE, c.FINALIST_SHA256)
    panel = m2.build_panel(dev, 1)
    scaled, _, _ = m2.pooled_scale(panel, panel.iloc[:0])
    x = c.design_matrix(art, panel)
    assert np.array_equal(x, scaled[list(m2.FEATURES)].to_numpy())
    assert np.array_equal(c.predict(art, panel), np.maximum(0.0, art["model"].predict(x)))
    # the stored development statistics are used, never statistics of the rows being predicted
    shifted = panel.copy()
    shifted["Volume_Change_1"] += 5.0
    assert not np.allclose(c.design_matrix(art, shifted)[:, 2], x[:, 2])
    with pytest.raises(c.Phase12S3Error, match="no development scaling"):
        c.design_matrix(art, panel.assign(symbol="ZZZ"))


def test_target_is_absolute_one_day_excess_return():
    fr = sealed_frames()
    panel, val = c.sealed_panel(fr, UNIVERSE)
    assert val["valid"]
    f = fr["BBB"]
    want = np.abs((f["Close"].shift(-1) / f["Close"] - 1) - (f["SPY_Close"].shift(-1) / f["SPY_Close"] - 1)).iloc[:-1]
    got = panel[panel["symbol"] == "BBB"]["y"].to_numpy()
    assert np.allclose(got, want.to_numpy(), rtol=0, atol=1e-15)
    assert len(got) == len(f) - 1                       # the last session has no target inside the window


def test_baseline_is_the_stage2_reference_on_volatility_only(env, dev):
    base = c.load_artifact(env.out / c.BASELINE_FILE, file_sha256(env.out / c.BASELINE_FILE))
    c.check_artifact(base, m2.REFERENCE)
    assert base["features"] == list(m2.VOL_FEATURES)
    panel = m2.build_panel(dev, 1)
    ref = m2.make_model(m2.REFERENCE, "pooled", m2.FROZEN_CONFIG).fit(panel[list(m2.VOL_FEATURES)].to_numpy(),
                                                                       panel["y"].to_numpy())
    # same estimator and data; bit-equality is not expected across array layouts (BLAS summation order)
    assert np.allclose(c.predict(base, panel), np.maximum(0.0, ref.predict(panel[list(m2.VOL_FEATURES)].to_numpy())),
                       rtol=0, atol=1e-12)
    noisy = panel.copy()
    noisy[list(m2.STOCK_SCALED)] = 99.0
    assert np.array_equal(c.predict(base, noisy), c.predict(base, panel))
    assert c.model_sha256(c.fit_baseline(dev)) == c.model_sha256(base)


# ============================================================
# Amendment integrity
# ============================================================


def test_freeze_is_write_once_and_pins_everything(env):
    doc = json.loads((env.out / c.AMENDMENT_FILE).read_text())
    assert doc["sealed_data_read"] is False
    assert doc["pins"]["finalist"]["file_sha256"] == c.FINALIST_SHA256
    assert doc["pins"]["baseline"]["file_sha256"] == file_sha256(env.out / c.BASELINE_FILE)
    assert set(doc["pins"]["generator_code_sha256"]) == set(c.GENERATOR_FILES)
    with pytest.raises(c.Phase12S3Error, match="written once"):
        c.freeze(env.out, env.stage2, frames=env.dev, inputs_sha="synthetic")
    c.verify(env.out, env.stage2, frames=env.dev, pinned_sha=env.sha)


def test_amendment_must_match_its_pin_and_the_rules_in_code(env, monkeypatch):
    c.load_amendment(env.out, env.sha)
    monkeypatch.setattr(c, "AMENDMENT_SHA256", None)
    with pytest.raises(c.Phase12S3Error, match="not pinned"):
        c.load_amendment(env.out)
    p = env.out / c.AMENDMENT_FILE
    doc = json.loads(p.read_text())
    doc["qualification"]["Q1_improvement"] = "relative MSE improvement >= 0.0"
    p.write_text(json.dumps(doc))
    with pytest.raises(c.Phase12S3Error, match="cannot change"):
        c.load_amendment(env.out, env.sha)
    with pytest.raises(c.Phase12S3Error, match="rules differ"):
        c.load_amendment(env.out, file_sha256(p))       # re-pinned, but the rules no longer equal the code


def test_verify_refuses_changed_stage2_evidence(env):
    (env.stage2 / "results" / "results.json").write_text("{}\n")
    with pytest.raises(c.Phase12S3Error, match="missing or changed"):
        c.verify(env.out, env.stage2, frames=env.dev, pinned_sha=env.sha)


def test_verify_refuses_a_replaced_baseline(env):
    joblib.dump(c.fit_baseline(planted(n=520, k=3, seed=7)), env.out / c.BASELINE_FILE)
    with pytest.raises(c.Phase12S3Error, match="pinned SHA256"):
        c.verify(env.out, env.stage2, frames=env.dev, pinned_sha=env.sha)


def test_verify_refuses_model_fingerprint_drift(env, monkeypatch):
    monkeypatch.setattr(c, "predictions_sha256", lambda p: "drift")
    with pytest.raises(c.Phase12S3Error, match="no longer reproduce"):
        c.verify(env.out, env.stage2, frames=env.dev, pinned_sha=env.sha)


# ============================================================
# Authorization gate: nothing sealed is touched before it passes
# ============================================================


def test_confirm_without_authorization_never_calls_the_sealed_loader(env):
    with pytest.raises(c.Phase12S3Error, match="not authorized"):
        run_confirm(env, never_called)
    assert not (env.out / c.CONFIRMATION_DIR).exists()
    assert not (env.out / (c.CONFIRMATION_DIR + ".incomplete")).exists()


@pytest.mark.parametrize("field,value", [("amendment_sha256", "0" * 64), ("runner_code_sha256", "0" * 64),
                                         ("sealed_builder_code_sha256", "0" * 64), ("authorized_by", ""),
                                         ("statement", "ok go")])
def test_confirm_refuses_a_mismatched_authorization(env, field, value):
    authorize(env, **{field: value})
    with pytest.raises(c.Phase12S3Error, match=field):
        run_confirm(env, never_called)


def test_confirm_refuses_when_the_amendment_or_model_changed(env):
    authorize(env)
    with pytest.raises(c.Phase12S3Error, match="cannot change"):
        run_confirm(env, never_called, pinned_sha="1" * 64)
    joblib.dump(c.fit_baseline(planted(n=520, k=3, seed=7)), env.out / c.BASELINE_FILE)
    with pytest.raises(c.Phase12S3Error, match="pinned SHA256"):
        run_confirm(env, never_called)


def test_default_loader_refuses_until_the_sealed_builder_exists(env):
    authorize(env)
    with pytest.raises(c.Phase12S3Error, match="has not been written"):
        run_confirm(env, c.sealed_feature_build_unavailable)
    assert (env.out / (c.CONFIRMATION_DIR + ".incomplete") / "attempt.json").is_file()   # the attempt is recorded
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        run_confirm(env, never_called)


def test_authorized_confirmation_runs_once_without_fitting(env, monkeypatch):
    authorize(env)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("fit called"))   # noqa: E731
    monkeypatch.setattr(LGBMRegressor, "fit", boom)
    monkeypatch.setattr(Pipeline, "fit", boom)
    res = run_confirm(env, sealed_frames)
    assert res["outcome"] in ("CONFIRMED", "NOT_CONFIRMED")
    q = res["qualification"]
    assert q["Q4_separation"] is True and q["Q6_reproducibility"] is True
    assert set(res["ranking"]) == {"lightgbm", m2.REFERENCE}
    saved = json.loads((env.out / c.CONFIRMATION_DIR / "confirmation.json").read_text())
    assert saved["outcome"] == res["outcome"] and saved["amendment_sha256"] == env.sha
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        run_confirm(env, sealed_frames)


def test_planted_signal_is_confirmed_and_noise_is_not(env):
    fin = c.load_artifact(env.stage2 / "results" / "finalists" / c.FINALIST_FILE, c.FINALIST_SHA256)
    base = c.load_artifact(env.out / c.BASELINE_FILE, file_sha256(env.out / c.BASELINE_FILE))
    assert c.evaluate(fin, base, sealed_frames(), UNIVERSE)["outcome"] == "CONFIRMED"
    assert c.evaluate(fin, base, sealed_frames(strength=0.0), UNIVERSE)["outcome"] == "NOT_CONFIRMED"


# ============================================================
# Order: A preflight (no attempt) -> B register the attempt -> C sealed read
# ============================================================


def marker(e):
    return e.out / (c.CONFIRMATION_DIR + ".incomplete") / "attempt.json"


class Source:
    """A two-step sealed loader double; records whether the attempt marker existed at each step."""

    def __init__(self, e, fail=None, sealed_fail=False):
        self.e, self.fail, self.sealed_fail, self.events = e, fail, sealed_fail, []

    def preflight(self):
        self.events.append(("preflight", marker(self.e).exists()))
        if self.fail:
            raise c.Phase12S3Error(self.fail)
        return {"development_reproduction": "EXACT_MATCH", "equivalence": {"matches": 500}}

    def __call__(self):
        self.events.append(("sealed", marker(self.e).is_file()))
        if self.sealed_fail:
            raise RuntimeError("sealed build crashed")
        return sealed_frames()


@pytest.mark.parametrize("failure", ["HF_HUB_OFFLINE=1 required", "TRANSFORMERS_OFFLINE=1 required",
                                     "Erratum 01 is missing or differs from its pinned SHA256",
                                     "cached FinBERT files differ from the Erratum 01 fingerprint",
                                     "development reproduction FAILED", "FinBERT label equivalence FAILED: 499/500"])
def test_a_preflight_failure_consumes_no_attempt(env, failure):
    authorize(env)
    bad = Source(env, fail=failure)
    with pytest.raises(c.Phase12S3Error, match=failure.split(":")[0][:20]):
        run_confirm(env, bad)
    assert bad.events == [("preflight", False)]                         # never reached the sealed step
    assert not (env.out / (c.CONFIRMATION_DIR + ".incomplete")).exists()
    good = Source(env)                                                  # the attempt is still available
    assert run_confirm(env, good)["outcome"] in ("CONFIRMED", "NOT_CONFIRMED")


def test_invalid_authorization_fails_before_preflight(env):
    src = Source(env)
    with pytest.raises(c.Phase12S3Error, match="not authorized"):
        run_confirm(env, src)
    authorize(env, statement="go")
    with pytest.raises(c.Phase12S3Error, match="statement"):
        run_confirm(env, src)
    assert src.events == [] and not marker(env).parent.exists()


@pytest.mark.parametrize("existing", [c.CONFIRMATION_DIR, c.CONFIRMATION_DIR + ".incomplete"])
def test_an_existing_attempt_or_confirmation_is_refused_before_preflight(env, existing):
    authorize(env)
    (env.out / existing).mkdir()
    src = Source(env)
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        run_confirm(env, src)
    assert src.events == []


def test_marker_is_written_once_after_preflight_and_before_the_first_sealed_read(env):
    authorize(env)
    src = Source(env)
    run_confirm(env, src)
    assert src.events == [("preflight", False), ("sealed", True)]
    done = env.out / c.CONFIRMATION_DIR
    attempt = json.loads((done / "attempt.json").read_text())
    assert attempt["preflight"]["equivalence"]["matches"] == 500 and attempt["amendment_sha256"] == env.sha
    assert sorted(p.name for p in done.iterdir()) == ["attempt.json", "confirmation.json"]
    again = Source(env)
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        run_confirm(env, again)
    assert again.events == []


def test_a_failure_after_registration_keeps_the_marker_and_refuses_retries(env):
    authorize(env)
    src = Source(env, sealed_fail=True)
    with pytest.raises(RuntimeError, match="crashed"):
        run_confirm(env, src)
    assert src.events == [("preflight", False), ("sealed", True)] and marker(env).is_file()
    retry = Source(env)
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        run_confirm(env, retry)
    assert retry.events == [] and marker(env).is_file()


def test_concurrent_registrations_cannot_both_succeed(tmp_path):
    import threading
    n, start, wins, losses = 12, threading.Barrier(12), [], []

    def go(i):
        start.wait()
        try:
            wins.append(c.register_attempt(tmp_path, "a" * 64, {"by": i}))
        except c.Phase12S3Error:
            losses.append(i)
    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(losses) == n - 1
    assert [p.name for p in (tmp_path / (c.CONFIRMATION_DIR + ".incomplete")).iterdir()] == ["attempt.json"]


def test_registration_after_a_completed_run_is_refused_without_leaving_a_marker(tmp_path, monkeypatch):
    (tmp_path / c.CONFIRMATION_DIR).mkdir()                             # a run finished in the meantime
    monkeypatch.setattr(c, "_refuse_existing_attempt", lambda out_dir: None)
    with pytest.raises(c.Phase12S3Error, match="runs once"):
        c.register_attempt(tmp_path, "a" * 64, {})
    assert not (tmp_path / (c.CONFIRMATION_DIR + ".incomplete")).exists()


def test_the_real_two_step_builder_reads_sealed_data_only_after_registration(env, tmp_path, monkeypatch):
    import inspect

    import training.phase12_sealed_features as s
    assert "SealedBuild()" in inspect.getsource(c.main)                 # the CLI uses the two-step builder
    authorize(env)
    order = []
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(s, "check_sealed_authorization",
                        lambda: order.append("A:gates") or {"authorization": {}, "erratum": {"model_files_sha256": {}}})
    monkeypatch.setattr(s, "check_finbert", lambda *a, **k: order.append("A:finbert") or {})
    monkeypatch.setattr(s, "development_news", lambda *a, **k: order.append("A:news") or ([], [], {}))
    monkeypatch.setattr(s, "reproduce", lambda *a, **k: order.append("A:reproduce") or
                        {"result": "EXACT_MATCH", "frame_sha256": {}})
    monkeypatch.setattr(s, "require_equivalence", lambda *a, **k: order.append("A:labels") or
                        {"result": "PASS", "checked": 500, "matches": 500, "article_keys_sha256": "k"})

    def sealed_read(ready, out_dir):
        order.append(("C:sealed", marker(env).is_file()))
        return sealed_frames()
    monkeypatch.setattr(s, "build_sealed_frames", sealed_read)
    build = s.SealedBuild(service_factory=lambda: None, out_dir=tmp_path / "sealed", attempt_marker=marker(env))
    assert run_confirm(env, build)["outcome"] in ("CONFIRMED", "NOT_CONFIRMED")
    assert order == ["A:gates", "A:finbert", "A:news", "A:reproduce", "A:labels", ("C:sealed", True)]


# ============================================================
# Sealed eligibility and missing data
# ============================================================


def test_eligibility_rules_decide_validity_before_any_comparison(env):
    fin = c.load_artifact(env.stage2 / "results" / "finalists" / c.FINALIST_FILE, c.FINALIST_SHA256)
    base = c.load_artifact(env.out / c.BASELINE_FILE, file_sha256(env.out / c.BASELINE_FILE))
    fr = sealed_frames()
    fr["AAA"].loc[10:12, "sw_count_surprise_1_20"] = np.nan          # 3 of ~1,350 rows: excluded, still valid
    panel, val = c.sealed_panel(fr, UNIVERSE)
    assert val["valid"] and val["excluded_rows"] == 3 and len(panel) == val["rows"] - 3
    fr["AAA"].loc[20:60, "event__event_count"] = np.nan              # > 5% of one stock: INVALID, no metrics
    res = c.evaluate(fin, base, fr, UNIVERSE)
    assert res["outcome"] == "INVALID" and "comparison" not in res
    missing = {k: v for k, v in sealed_frames().items() if k != "CCC"}
    assert c.evaluate(fin, base, missing, UNIVERSE)["outcome"] == "INVALID"
    short = {k: v.iloc[:300] for k, v in sealed_frames().items()}
    assert "300 common sealed sessions" in c.sealed_panel(short, UNIVERSE)[1]["reasons"][0]
    bad_price = sealed_frames()
    bad_price["BBB"].loc[5, "Close"] = np.nan
    assert c.evaluate(fin, base, bad_price, UNIVERSE)["outcome"] == "INVALID"


def test_sealed_frames_outside_the_window_are_refused():
    early = sealed_frames()
    early["AAA"] = make_frame(450, seed=1, start="2024-09-02")
    with pytest.raises(c.Phase12S3Error, match="outside"):
        c.sealed_panel(early, UNIVERSE)
    late = {k: make_frame(600, seed=i, start="2024-10-01") for i, k in enumerate(UNIVERSE)}   # runs past 2026-09-25
    with pytest.raises(c.Phase12S3Error, match="outside"):
        c.sealed_panel(late, UNIVERSE)


def test_quarters_match_the_stage2_fold_length_and_merge_rule():
    q = c.quarters(500)
    assert (q == 0).sum() == 63 and q.max() == 7 and (q == 7).sum() == 500 - 7 * 63
    q = c.quarters(63 * 7 + 10)                         # a 10-session tail joins the previous quarter
    assert q.max() == 6 and (q == 6).sum() == 73
