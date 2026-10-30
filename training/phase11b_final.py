"""
Phase 11B Stage 11B.5: final families, audit, determinism re-run and report (Amendment 06).

Frozen procedure (data/results/research/phase11/11B/amendment_06_final_audit_procedure.json):
    families      primary panel 198 (66 11A + 132 11B) and secondary per-stock 5,742 (1,914 + 3,828),
                  Holm at 0.05, members and order = design_registry.json design.comparison_ids;
                  AAPL-excluded sensitivity 198: descriptive only, no Holm, never qualifies
    qualification gate_status == QUALIFIED AND raw p < 0.05 AND Holm p < 0.05 (strict); gate status and
                  raw p are taken as recorded - Phase 11A p-values are loaded by hash, never recomputed
    audit         A1-A8: frozen inputs, registry, development results, artifacts, exact re-derivation of
                  every 11B entry, code, descriptive stage Holm, unchanged pre-existing evidence
    determinism   14-17 runs chosen by a pure hash rule (coverage, smallest p-values, two panel runs),
                  re-executed sequentially with the 11B.4 functions and compared bit for bit
                  (warnings excluded); refused if Python / package versions differ from the registry
    confirmation  NOT_EVALUATED; no sealed path is opened

Commands (outputs go to data/results/research/phase11/11B/final/, write-once, built in final.incomplete/):

    python -m training.phase11b_final status     # read-only
    python -m training.phase11b_final protocol   # final.incomplete/protocol.json only, no computation
    python -m training.phase11b_final run        # audit -> Holm -> determinism -> report -> manifest -> publish

The outcome of this stage never alters Phase 11A or 11B.4 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase10a_research as p10a  # noqa: E402
import training.phase11_design as p11d  # noqa: E402
import training.phase11a_research as a11  # noqa: E402
import training.phase11b_research as r  # noqa: E402
import training.phase9_research as p9  # noqa: E402
from training.build_dataset import file_sha256  # noqa: E402
from training.evaluation_harness import NOT_QUALIFIED, QUALIFIED, _json_safe, git_info  # noqa: E402
from training.phase11b_sentiment_scoring import PHASE_11B_DIR  # noqa: E402
from training.phase9_research import ResearchError, _write_new_json, canonical_json, environment, holm_adjust  # noqa: E402

PHASE = "11B"
STAGE = "11B.5"
PROTOCOL_VERSION = "phase11b5_protocol_v1"
ALPHA = 0.05

# ---- pins ----
AMENDMENT_06_PATH = PHASE_11B_DIR / "amendment_06_final_audit_procedure.json"
AMENDMENT_06_SHA256 = "f10127438e16991ae6d6cccf92dd057e3a1d4136aa078b4f275d5e1e56476c6a"
FROZEN_MATRIX_SHA256 = "169c369f89056749754f8324edccee1cd85eacaa4d80a8f9e8be995118f87b08"
CODE_FILES = ("training/phase11b_final.py",) + r.CODE_FILES

FAMILIES = ("per_stock", "panel", "aapl_excluded_sensitivity")
HOLM_FAMILIES = ("per_stock", "panel")
FAMILY_LABELS = {"per_stock": "secondary_per_stock", "panel": "primary_panel",
                 "aapl_excluded_sensitivity": "aapl_excluded_sensitivity"}
FROZEN_FAMILY_COUNTS = (("per_stock", 1914, 3828), ("panel", 66, 132), ("aapl_excluded_sensitivity", 66, 132))
GATE_STATUSES = (QUALIFIED, NOT_QUALIFIED)
SAMPLE_SALT = "phase11b5|determinism_v1"
CELLS = tuple((fs, h) for fs in r.STAGE_FEATURE_SETS for h in r.HORIZONS)
STAGE_HOLM_KEYS = ("p_value_holm_11b_stage_descriptive", "final_qualification")
OUTPUT_FILES = ("protocol.json", "audit.json", "combined_holm.json", "determinism.json",
                "final_report.json", "final_report.md")


class Phase11B5Error(ResearchError):
    """Phase 11B.5 protocol violation; the run stops."""


# ============================================================
# Sources and paths
# ============================================================


@dataclass(frozen=True)
class Sources:
    """Where 11B.5 reads from and writes to. Defaults are the real, pinned locations."""
    phase11a_registry: Path = a11.REGISTRY_PATH
    phase11a_sha256: str = r.PHASE_11A_REGISTRY_SHA256
    design_registry: Path = p11d.REGISTRY_PATH
    research: r.Paths = r.DEFAULT_PATHS
    amendment: Path = AMENDMENT_06_PATH
    amendment_sha256: str = AMENDMENT_06_SHA256
    matrix_sha256: str = FROZEN_MATRIX_SHA256
    family_counts: tuple = FROZEN_FAMILY_COUNTS
    protected_dirs: tuple = (a11.OUT_DIR, PHASE_11B_DIR)

    @property
    def out_dir(self) -> Path:
        return self.research.out_dir

    @property
    def final(self) -> Path:
        return self.out_dir / "final"

    @property
    def staging(self) -> Path:
        return self.out_dir / "final.incomplete"

    def counts(self, family: str) -> tuple[int, int]:
        for name, na, nb in self.family_counts:
            if name == family:
                return na, nb
        raise Phase11B5Error(f"unknown family {family}")


DEFAULT_SOURCES = Sources()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> dict:
    return {f: file_sha256(PROJECT_ROOT / f) for f in CODE_FILES}


def verify_amendment(src: Sources) -> None:
    if not Path(src.amendment).is_file() or file_sha256(src.amendment) != src.amendment_sha256:
        raise Phase11B5Error("Amendment 06 is missing or differs from its pinned SHA256")
    a = _load_json(src.amendment)
    if a.get("amendment_id") != "06" or a.get("status") != "ADOPTED":
        raise Phase11B5Error("Amendment 06 is not ADOPTED")


def load_phase11a(src: Sources) -> dict:
    if file_sha256(src.phase11a_registry) != src.phase11a_sha256:
        raise Phase11B5Error("Phase 11A registry differs from its pinned SHA256 (11A p-values are loaded by hash)")
    return _load_json(src.phase11a_registry)["results"]


def load_design_ids(src: Sources) -> dict:
    return p11d.load_registry(src.design_registry)["design"]["comparison_ids"]


def load_development(src: Sources) -> dict:
    dev = _load_json(src.research.results)
    if dev.get("document_type") != "phase_11b_development_results" or dev.get("status") != "COMPLETE":
        raise Phase11B5Error("development_results.json is not a COMPLETE Phase 11B development record")
    return dev


# ============================================================
# Families, Holm, qualification
# ============================================================


def _valid_p(p) -> bool:
    return isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0.0 <= p <= 1.0


def assemble_family(family: str, a_entries: list[dict], b_entries: list[dict], design_ids: list[str],
                    counts: tuple[int, int]) -> list[dict]:
    """11A + 11B entries in frozen design order; any size, membership or value problem stops the run."""
    na, nb = counts
    if (len(a_entries), len(b_entries)) != (na, nb):
        raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: {family} has {len(a_entries)} 11A / {len(b_entries)} 11B "
                             f"comparisons, expected {na} / {nb}")
    if len(design_ids) != na + nb or len(set(design_ids)) != len(design_ids):
        raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: frozen {family} id list is not {na + nb} distinct ids")
    by_id = {}
    for phase, entries in (("11A", a_entries), ("11B", b_entries)):
        for e in entries:
            if e["id"] in by_id:
                raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: {family} id {e['id']} appears twice "
                                     f"({by_id[e['id']][0]} and {phase})")
            by_id[e["id"]] = (phase, e)
    missing, extra = set(design_ids) - set(by_id), set(by_id) - set(design_ids)
    if missing or extra:
        raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: {family} differs from the frozen design "
                             f"(missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]})")
    rows = []
    for i in design_ids:
        phase, e = by_id[i]
        if not _valid_p(e.get("p_value_raw")):
            raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: {i} has an invalid raw p-value {e.get('p_value_raw')!r}")
        if e.get("gate_status") not in GATE_STATUSES:
            raise Phase11B5Error(f"FAMILY_INTEGRITY_FAILED: {i} has an unexpected gate status {e.get('gate_status')!r}")
        row = {"id": i, "phase": phase, "feature_set": e["feature_set"], "horizon": e["horizon"],
               "model": e["model"], "task": e["task"], "gate_status": e["gate_status"],
               "p_value_raw": float(e["p_value_raw"])}
        if "symbol" in e:
            row["symbol"] = e["symbol"]
        rows.append(row)
    return rows


def qualify_final(gate_status: str, p_raw: float, p_holm: float) -> dict:
    gate_ok, raw_ok, holm_ok = gate_status == QUALIFIED, p_raw < ALPHA, p_holm < ALPHA
    return {"gate_ok": gate_ok, "raw_ok": raw_ok, "holm_ok": holm_ok, "qualified": gate_ok and raw_ok and holm_ok}


def combined_holm(rows: list[dict]) -> None:
    """Holm over the whole family (every comparison, whatever its gate status); never overrides the gate."""
    for row, p in zip(rows, holm_adjust([row["p_value_raw"] for row in rows])):
        row["p_value_holm_final"] = p
        row.update(qualify_final(row["gate_status"], row["p_value_raw"], p))


def descriptive_only(rows: list[dict]) -> None:
    for row in rows:
        row.update({"p_value_holm_final": None, "gate_ok": row["gate_status"] == QUALIFIED,
                    "raw_ok": row["p_value_raw"] < ALPHA, "holm_ok": None, "qualified": None})


def build_families(a11_results: dict, dev: dict, design_ids: dict, src: Sources) -> dict:
    out = {}
    for fam in FAMILIES:
        rows = assemble_family(fam, a11_results[fam], dev["results"][fam], design_ids[fam], src.counts(fam))
        (combined_holm if fam in HOLM_FAMILIES else descriptive_only)(rows)
        out[fam] = rows
    return out


def family_summary(rows: list[dict], fam: str) -> dict:
    holm = fam in HOLM_FAMILIES
    s = {"family": FAMILY_LABELS[fam], "size": len(rows), "method": "Holm" if holm else None,
         "alpha": ALPHA if holm else None, "can_qualify": holm,
         "by_phase": {ph: sum(x["phase"] == ph for x in rows) for ph in ("11A", "11B")},
         "gate_passes": sum(x["gate_ok"] for x in rows), "raw_p_below_alpha": sum(x["raw_ok"] for x in rows),
         "min_p_raw": min(x["p_value_raw"] for x in rows)}
    if holm:
        s.update({"holm_p_below_alpha": sum(x["holm_ok"] for x in rows),
                  "min_p_holm": min(x["p_value_holm_final"] for x in rows),
                  "qualified": sum(x["qualified"] for x in rows),
                  "qualified_ids": [x["id"] for x in rows if x["qualified"]],
                  "qualified_by_phase": {ph: sum(x["qualified"] and x["phase"] == ph for x in rows)
                                         for ph in ("11A", "11B")}})
    else:
        s.update({"role": "descriptive only; no Holm; never qualifies", "qualified": None})
    return s


# ============================================================
# Determinism sample and comparison
# ============================================================


def _hash_index(label: str, n: int) -> int:
    return int(hashlib.sha256(f"{SAMPLE_SALT}|{label}".encode("utf-8")).hexdigest(), 16) % n


def _smallest(entries: list[dict], k: int) -> list[dict]:
    """k entries with the smallest raw p; ties broken by position (frozen order)."""
    return [e for _, _, e in sorted((e["p_value_raw"], i, e) for i, e in enumerate(entries))[:k]]


def select_determinism_sample(per_stock_11b: list[dict], panel_11b: list[dict], universe: tuple) -> list[dict]:
    """Amendment 06 sample: S1 coverage (12 per-stock), S2 boundary (<= 3 per-stock), S3 panel (2). Pure."""
    runs, seen = [], set()

    def add(stratum, kind, fs, h, symbol=None):
        key = (kind, symbol, fs, h)
        if key in seen:
            return
        seen.add(key)
        runs.append({"stratum": stratum, "kind": kind, "symbol": symbol, "feature_set": fs, "horizon": h})

    for fs, h in CELLS:
        add("S1_coverage", "stock", fs, h, universe[_hash_index(f"{fs}|{h}", len(universe))])
    for e in _smallest(per_stock_11b, 3):
        add("S2_boundary", "stock", e["feature_set"], e["horizon"], e["symbol"])
    e = _smallest(panel_11b, 1)[0]
    add("S3_panel", "panel", e["feature_set"], e["horizon"])
    fs, h = CELLS[_hash_index("panel_exAAPL", len(CELLS))]
    add("S3_panel", "panel_exAAPL", fs, h)
    return runs


def run_key(run: dict) -> str:
    return f"stock_{run['symbol']}" if run["kind"] == "stock" else f"{run['kind']}_{run['feature_set']}_h{run['horizon']}"


def _canonical_run(obj: dict) -> dict:
    """The 11B.4 writer's JSON view of a run object, without the non-reproducible warnings key."""
    plain = json.loads(json.dumps(_json_safe(obj), sort_keys=True, allow_nan=False))
    plain.pop("warnings", None)
    return plain


def _diff(a, b, path: str, out: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "stored": a.get(k, "<missing>"), "rerun": b.get(k, "<missing>")})
            else:
                _diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": f"{path}#len", "stored": len(a), "rerun": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b or type(a) is not type(b):
        d = {"path": path, "stored": a, "rerun": b}
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            d["abs_diff"] = abs(a - b)
            d["rel_diff"] = abs(a - b) / abs(a) if a else None
        out.append(d)


def compare_run(stored: dict, rerun: dict, max_reported: int = 200) -> dict:
    s, n = _canonical_run(stored), _canonical_run(rerun)
    diffs: list = []
    _diff(s, n, "", diffs)
    return {"identical": not diffs, "stored_sha256": _sha(s), "rerun_sha256": _sha(n), "n_differences": len(diffs),
            "differences": diffs[:max_reported],
            "warnings_total": {"stored": (stored.get("warnings") or {}).get("total"),
                               "rerun": (rerun.get("warnings") or {}).get("total")}}


def environment_mismatches(current: dict, frozen: dict) -> list[str]:
    out = [] if current.get("python") == frozen.get("python") else \
        [f"python {current.get('python')} != {frozen.get('python')}"]
    cp, fp = current.get("packages", {}), frozen.get("packages", {})
    out += [f"{k} {cp.get(k)} != {fp.get(k)}" for k in sorted(set(cp) | set(fp)) if cp.get(k) != fp.get(k)]
    return out


def rerun_stock(frames: dict, run: dict, n_splits: int, matrix_sha256: str) -> dict:
    """One per-stock run exactly as 11B.4 _stock_job computes it."""
    fs, h = run["feature_set"], run["horizon"]
    return p10a.evaluate_excess(frames[run["symbol"]], list(r.FEATURE_SETS[fs]["columns"]), a11.candidates(),
                                horizon=h, n_splits=n_splits)


def rerun_panel(frames: dict, run: dict, n_splits: int, matrix_sha256: str) -> dict:
    """One panel run exactly as 11B.4 _panel_job, wrapped as 11B.4 wrote the artifact."""
    kind, fs, h = run["kind"], run["feature_set"], run["horizon"]
    res = r._panel_job(kind, fs, h, frames, n_splits)["result"]
    return {"kind": kind, "feature_set": fs, "horizon": h, "phase": r.PHASE, "matrix_sha256": matrix_sha256, **res}


def stored_run(dev: dict, run: dict) -> dict:
    art = dev["artifacts"][run_key(run)]
    path = r._resolve(art["path"])
    if file_sha256(path) != art["sha256"]:
        raise Phase11B5Error(f"artifact {run_key(run)} differs from its recorded SHA256")
    obj = _load_json(path)
    return obj["runs"][f"{run['feature_set']}_h{run['horizon']}"] if run["kind"] == "stock" else obj


# ============================================================
# Audit (A1-A7; A8 at the end of the run)
# ============================================================


def rederive_entries(dev: dict, spec: r.Spec) -> tuple[list, list, list]:
    """Entries rebuilt from the published artifacts with the unchanged 11B.4 logic (descriptive stage Holm included)."""
    art = dev["artifacts"]
    stock_res = {s: _load_json(r._resolve(art[f"stock_{s}"]["path"]))["runs"] for s in spec.universe}
    panel_res = {(k, fs, h): _load_json(r._resolve(art[f"{k}_{fs}_h{h}"]["path"]))
                 for k in ("panel", "panel_exAAPL") for fs, h in CELLS}
    per_stock, panel, sens = r._entries(stock_res, panel_res, spec, art)
    r.stage_holm_descriptive(per_stock, len(per_stock))
    r.stage_holm_descriptive(panel, len(panel))
    for e in sens:
        e["p_value_holm_11b_stage_descriptive"] = None
        e["final_qualification"] = "NEVER (descriptive sensitivity)"
    return per_stock, panel, sens


def _plain(x):
    return json.loads(json.dumps(_json_safe(x), sort_keys=True, allow_nan=False))


def _strip(entries: list[dict], keys) -> list[dict]:
    return [{k: v for k, v in e.items() if k not in keys} for e in entries]


def _only(entries: list[dict], keys) -> list[dict]:
    return [{k: e.get(k) for k in ("id", *keys)} for e in entries]


def real_audit(src: Sources, spec: r.Spec) -> dict:
    """A1-A7. Returns {"checks": {...}, "passed": bool, "context": {...}}; stops at the first failed check."""
    checks, ctx = {}, {}

    def record(name, fn) -> bool:
        try:
            checks[name] = {"pass": True, **(fn() or {})}
        except (ResearchError, OSError, KeyError, ValueError, TypeError) as e:
            checks[name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        return checks[name]["pass"]

    def a1():
        ctx["inputs"] = r.verified_inputs(spec)
        return {"inputs_sha256": r.inputs_sha256(ctx["inputs"]), "frames": len(ctx["inputs"]["frames"])}

    def a2():
        reg = r.load_registry(spec, src.research)
        if reg["matrix_sha256"] != src.matrix_sha256:
            raise Phase11B5Error(f"registry matrix sha256 {reg['matrix_sha256']} != {src.matrix_sha256}")
        if reg["inputs"] != ctx["inputs"]:
            raise Phase11B5Error("verified inputs differ from the registry inputs")
        bad = {s: why for s in r.STEPS for ok, why in [r.evidence_status(
            s, spec, ctx["inputs"] if s in r.REAL_STEPS else None, src.research)] if not ok}
        if bad:
            raise Phase11B5Error(f"prerequisite evidence not PASS: {bad}")
        ctx["registry"] = reg
        return {"registry_sha256": file_sha256(src.research.registry), "matrix_sha256": reg["matrix_sha256"],
                "prerequisites": "5 / 5 PASS"}

    def a3():
        dev = load_development(src)
        want = dict(zip(("per_stock", "panel", "aapl_excluded_sensitivity"), spec.expected_counts))
        problems = {k: v for k, v in {
            "counts": dev.get("counts") == want,
            "result_lengths": {k: len(v) for k, v in dev["results"].items()} == want,
            "registry_sha256": dev.get("registry_sha256") == file_sha256(src.research.registry),
            "matrix_sha256": dev.get("matrix_sha256") == ctx["registry"]["matrix_sha256"],
            "confirmation": dev.get("confirmation") == "NOT_EVALUATED",
            "data_kind": dev.get("data_kind") == ctx["registry"]["data_kind"],
        }.items() if not v}
        if problems:
            raise Phase11B5Error(f"development_results.json fails: {sorted(problems)}")
        ctx["dev"] = dev
        return {"development_results_sha256": file_sha256(src.research.results), "counts": dev["counts"]}

    def a4():
        bad = [k for k, a in ctx["dev"]["artifacts"].items() if file_sha256(r._resolve(a["path"])) != a["sha256"]]
        if bad:
            raise Phase11B5Error(f"artifact hashes differ: {bad}")
        if src.research.staging.exists():
            raise Phase11B5Error("development.incomplete exists")
        return {"artifacts_verified": len(ctx["dev"]["artifacts"])}

    def a5():
        ctx["rederived"] = rederive_entries(ctx["dev"], spec)
        stored = [ctx["dev"]["results"][k] for k in FAMILIES]
        if _plain([_strip(x, STAGE_HOLM_KEYS) for x in ctx["rederived"]]) != _plain([_strip(x, STAGE_HOLM_KEYS) for x in stored]):
            raise Phase11B5Error("entries re-derived from the artifacts differ from development_results.json")
        return {"entries_compared": sum(len(x) for x in stored)}

    def a6():
        if r.code_sha256() != ctx["registry"]["code_sha256"]:
            raise Phase11B5Error("research code differs from the code recorded at registry freeze")
        return {"code_files": len(r.CODE_FILES)}

    def a7():
        stored = [ctx["dev"]["results"][k] for k in FAMILIES]
        if _plain([_only(x, STAGE_HOLM_KEYS) for x in ctx["rederived"]]) != _plain([_only(x, STAGE_HOLM_KEYS) for x in stored]):
            raise Phase11B5Error("recomputed descriptive stage Holm differs from the stored values")
        return {"stage_holm_compared": len(stored[0]) + len(stored[1])}

    for name, fn in (("A1_frozen_inputs", a1), ("A2_registry", a2), ("A3_development_results", a3),
                     ("A4_artifacts", a4), ("A5_rederivation", a5), ("A6_code", a6), ("A7_stage_holm", a7)):
        if not record(name, fn):
            return {"checks": checks, "passed": False, "context": ctx}
    return {"checks": checks, "passed": True, "context": ctx}


def snapshot(src: Sources) -> dict:
    """sha256 of every pre-existing file in the protected directories, excluding the 11B.5 outputs."""
    skip = (src.final.resolve(), src.staging.resolve())
    out = {}
    for d in src.protected_dirs:
        for p in sorted(Path(d).rglob("*")):
            rp = p.resolve()
            if p.is_file() and not any(rp == s or s in rp.parents for s in skip):
                out[rp.as_posix()] = file_sha256(p)
    return out


# ============================================================
# Hooks (real by default; tests substitute them)
# ============================================================


@dataclass
class Hooks:
    audit: Callable[[Sources, r.Spec], dict] = real_audit
    load_frames: Callable[[dict, r.Spec], dict] = r.load_frames
    rerun_stock: Callable[..., dict] = rerun_stock
    rerun_panel: Callable[..., dict] = rerun_panel
    environment: Callable[[], dict] = environment
    log: Callable[[str], None] = field(default=print)


# ============================================================
# Protocol
# ============================================================


def _protocol_body(src: Sources, spec: r.Spec, dev: dict, design_ids: dict, reg: dict) -> dict:
    sample = select_determinism_sample(dev["results"]["per_stock"], dev["results"]["panel"], tuple(spec.universe))
    return {
        "protocol_version": PROTOCOL_VERSION, "phase": PHASE, "stage": STAGE,
        "amendment_06": {"file": Path(src.amendment).name, "sha256": src.amendment_sha256},
        "families": {fam: {"label": FAMILY_LABELS[fam], "size": sum(src.counts(fam)),
                           "phase_11a": src.counts(fam)[0], "phase_11b": src.counts(fam)[1],
                           "holm": fam in HOLM_FAMILIES, "order": "design_registry.json design.comparison_ids",
                           "ids_sha256": _sha(design_ids[fam])} for fam in FAMILIES},
        "qualification_rule": "gate_status == QUALIFIED AND p_value_raw < 0.05 AND p_value_holm_final < 0.05 (strict)",
        "determinism": {"sample": sample, "n_runs": len(sample),
                        "comparison": "canonical JSON without top-level warnings; bit-identical",
                        "mode": "sequential, in-process (workers = 1)",
                        "environment_rule": "refuse unless Python and package versions equal the registry environment"},
        "inputs": {"phase11a_registry_sha256": file_sha256(src.phase11a_registry),
                   "design_registry_sha256": file_sha256(src.design_registry),
                   "registry_sha256": file_sha256(src.research.registry),
                   "development_results_sha256": file_sha256(src.research.results),
                   "artifacts": {k: a["sha256"] for k, a in sorted(dev["artifacts"].items())}},
        "frozen_environment": reg.get("environment"),
        "confirmation": "NOT_EVALUATED",
    }


def _verify_sources(src: Sources) -> tuple[dict, dict, dict]:
    verify_amendment(src)
    load_phase11a(src)
    dev = load_development(src)
    if file_sha256(src.research.registry) != dev.get("registry_sha256"):
        raise Phase11B5Error("development_results.json does not belong to the frozen registry")
    reg = _load_json(src.research.registry)
    return dev, load_design_ids(src), reg


def write_protocol(src: Sources = DEFAULT_SOURCES, spec: r.Spec = r.FROZEN_SPEC) -> dict:
    if src.final.exists():
        raise Phase11B5Error("final/ already exists; 11B.5 runs once")
    if src.staging.exists():
        raise Phase11B5Error("final.incomplete/ exists; inspect it and move it aside manually")
    dev, design_ids, reg = _verify_sources(src)
    body = _protocol_body(src, spec, dev, design_ids, reg)
    protocol = {**body, "code_sha256": code_sha256(), "environment": environment(),
                "created_at": _now(), "git": git_info()}
    src.staging.mkdir(parents=True)
    _write_new_json(src.staging / "protocol.json", protocol)
    return protocol


def load_protocol(src: Sources, spec: r.Spec) -> dict:
    path = src.staging / "protocol.json"
    if not path.is_file():
        raise Phase11B5Error("final.incomplete/protocol.json not found (run `protocol` first)")
    extra = sorted(p.name for p in src.staging.iterdir() if p.name != "protocol.json")
    if extra:
        raise Phase11B5Error(f"final.incomplete/ holds files from an earlier attempt {extra}; move it aside manually")
    protocol = _load_json(path)
    dev, design_ids, reg = _verify_sources(src)
    body = _protocol_body(src, spec, dev, design_ids, reg)
    if {k: protocol.get(k) for k in body} != _plain(body):
        raise Phase11B5Error("protocol.json no longer matches Amendment 06 applied to the current inputs")
    if protocol.get("code_sha256") != code_sha256():
        raise Phase11B5Error("code changed after protocol.json was written")
    return protocol


# ============================================================
# Report
# ============================================================


def render_markdown(rep: dict) -> str:
    q, d, a = rep["qualification"], rep["determinism"], rep["audit"]
    lines = [f"# Phase 11B.5 Final Report", "",
             f"- Generated: {rep['created_at']} (from final_report.json; not hand-edited)",
             f"- Protocol sha256: `{rep['protocol_sha256']}`; Amendment 06 sha256: `{rep['amendment_06_sha256']}`",
             f"- Phase 11B acceptance: **{rep['phase_11b_acceptance']}**", "",
             "## Qualification (gate_v1 AND raw p < 0.05 AND Holm p < 0.05)", "",
             "| Family | Size | 11A / 11B | Gate passes | Raw p < 0.05 | Holm p < 0.05 | Min raw p | Min Holm p | Qualified |",
             "|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        s = q["families"][fam]
        lines.append(f"| {s['family']} | {s['size']} | {s['by_phase']['11A']} / {s['by_phase']['11B']} | "
                     f"{s['gate_passes']} | {s['raw_p_below_alpha']} | {s.get('holm_p_below_alpha', 'n/a')} | "
                     f"{s['min_p_raw']:.4g} | {('%.4g' % s['min_p_holm']) if s.get('min_p_holm') is not None else 'n/a'} | "
                     f"{s['qualified'] if s['qualified'] is not None else 'never (descriptive)'} |")
    lines += ["", f"Development qualifiers: **{q['total_qualified']}**.", "",
              f"## Audit: {'PASS' if a['passed'] else 'FAIL'}", ""]
    lines += [f"- {k}: {'PASS' if v['pass'] else 'FAIL'}" for k, v in a["checks"].items()]
    lines += ["", f"## Determinism: {d['outcome']}", "",
              f"{d['n_identical']} / {d['n_runs']} sampled runs bit-identical (warnings excluded).", ""]
    lines += [f"- {x['stratum']} {run_key(x)} {x['feature_set']}_h{x['horizon']}: "
              f"{'identical' if x['identical'] else 'DIFFERENT (%d fields)' % x['n_differences']}" for x in d["runs"]]
    lines += ["", "## Confirmation period", "",
              f"- Status: {rep['confirmation']['status']}; applicability: {rep['confirmation']['applicability']}",
              "- No sealed content was read.", "", "## Known limitations", ""]
    lines += [f"- {x}" for x in rep["known_limitations"]]
    return "\n".join(lines) + "\n"


# ============================================================
# Run
# ============================================================


def run_final(src: Sources = DEFAULT_SOURCES, spec: r.Spec = r.FROZEN_SPEC, hooks: Hooks | None = None) -> dict:
    hooks = hooks or Hooks()
    if src.final.exists():
        raise Phase11B5Error("final/ already exists; 11B.5 runs once")
    protocol = load_protocol(src, spec)
    protocol_sha = file_sha256(src.staging / "protocol.json")
    started = time.time()
    before = snapshot(src)
    stamp = {"phase": PHASE, "stage": STAGE, "protocol_sha256": protocol_sha}

    def write(name: str, obj: dict) -> dict:
        return _write_new_json(src.staging / name, {**obj, **stamp, "created_at": _now()})

    # ---- audit A1-A7 ----
    aud = hooks.audit(src, spec)
    ctx = aud["context"]
    if not aud["passed"]:
        write("audit.json", {"document_type": "phase_11b5_audit", "checks": aud["checks"], "passed": False,
                             "outcome": "AUDIT_FAILED"})
        raise Phase11B5Error(f"AUDIT_FAILED: {[k for k, v in aud['checks'].items() if not v['pass']]}")
    dev, reg = ctx["dev"], ctx["registry"]
    hooks.log("[11B.5] audit A1-A7 PASS")

    # ---- combined Holm ----
    fams = build_families(load_phase11a(src), dev, load_design_ids(src), src)
    summaries = {fam: family_summary(rows, fam) for fam, rows in fams.items()}
    holm_art = write("combined_holm.json", {"document_type": "phase_11b5_combined_holm", "alpha": ALPHA,
                                            "qualification_rule": protocol["qualification_rule"],
                                            "summary": summaries, "families": fams})
    total_q = sum(summaries[f]["qualified"] for f in HOLM_FAMILIES)
    hooks.log(f"[11B.5] combined Holm: qualified per-stock {summaries['per_stock']['qualified']}, "
              f"panel {summaries['panel']['qualified']}")

    # ---- determinism ----
    sample = protocol["determinism"]["sample"]
    mism = environment_mismatches(hooks.environment(), reg["environment"])
    if mism:
        write("determinism.json", {"document_type": "phase_11b5_determinism", "outcome": "ENVIRONMENT_MISMATCH",
                                   "mismatches": mism, "runs": [], "sample": sample})
        raise Phase11B5Error(f"ENVIRONMENT_MISMATCH: determinism rerun refused ({mism})")
    frames = hooks.load_frames(ctx["inputs"], spec)
    det_runs = []
    for run in sample:
        t0 = time.time()
        fn = hooks.rerun_stock if run["kind"] == "stock" else hooks.rerun_panel
        res = compare_run(stored_run(dev, run), fn(frames, run, spec.n_splits, reg["matrix_sha256"]))
        det_runs.append({**run, **res, "runtime_seconds": round(time.time() - t0, 1)})
        hooks.log(f"[11B.5] determinism {run_key(run)} {run['feature_set']}_h{run['horizon']}: "
                  f"{'identical' if res['identical'] else 'DIFFERENT'}")
    n_ident = sum(x["identical"] for x in det_runs)
    det_outcome = "DETERMINISM_VERIFIED" if n_ident == len(det_runs) else "DETERMINISM_CHECK_FAILED"
    det_art = write("determinism.json", {"document_type": "phase_11b5_determinism", "outcome": det_outcome,
                                         "n_runs": len(det_runs), "n_identical": n_ident, "runs": det_runs,
                                         "comparison": protocol["determinism"]["comparison"],
                                         "environment": hooks.environment()})

    # ---- A8: pre-existing evidence unchanged ----
    after = snapshot(src)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    checks = {**aud["checks"], "A8_write_once": {"pass": not changed, "files_checked": len(before),
                                                 "changed": changed[:50]}}
    audit_art = write("audit.json", {"document_type": "phase_11b5_audit", "checks": checks,
                                     "passed": not changed, "outcome": "PASS" if not changed else "AUDIT_FAILED"})
    if changed:
        raise Phase11B5Error(f"AUDIT_FAILED: pre-existing evidence changed during 11B.5: {changed[:5]}")

    # ---- report ----
    freeze_path = src.out_dir / "design_freeze.json"
    design = _load_json(freeze_path) if freeze_path.is_file() else {}
    report = {
        "document_type": "phase_11b_final_report", "status": "COMPLETE",
        "amendment_06_sha256": src.amendment_sha256,
        "qualification": {"rule": protocol["qualification_rule"], "families": summaries,
                          "total_qualified": total_q},
        "audit": {"passed": True, "checks": {k: {"pass": v["pass"]} for k, v in checks.items()}},
        "determinism": {"outcome": det_outcome, "n_runs": len(det_runs), "n_identical": n_ident,
                        "runs": [{k: x[k] for k in ("stratum", "kind", "symbol", "feature_set", "horizon",
                                                    "identical", "n_differences")} for x in det_runs]},
        "phase_11b_acceptance": ("PENDING_OWNER_REVIEW" if det_outcome == "DETERMINISM_VERIFIED"
                                 else "NOT_ACCEPTED_DETERMINISM_FAILED"),
        "confirmation": {"status": "NOT_EVALUATED",
                         "applicability": "NOT_APPLICABLE" if total_q == 0 else "PENDING_OWNER_AUTHORIZATION",
                         "authorization": None, "sealed_content_read": False},
        "phase_11a_stage_holm": "11A registry p_value_holm / qualified (11A-only families) are superseded by this report",
        "known_limitations": design.get("known_limitations", []),
        "files": {"protocol.json": protocol_sha, "audit.json": audit_art["sha256"],
                  "combined_holm.json": holm_art["sha256"], "determinism.json": det_art["sha256"]},
        "runtime_seconds": round(time.time() - started, 1), "git": git_info(), "environment": hooks.environment(),
    }
    write("final_report.json", report)
    report = _load_json(src.staging / "final_report.json")
    md = src.staging / "final_report.md"
    if md.exists():
        raise Phase11B5Error("final_report.md already exists")
    with open(md, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(report))

    # ---- manifest (last) and publish ----
    manifest = {"document_type": "phase_11b5_manifest",
                "files": {n: {"sha256": file_sha256(src.staging / n), "bytes": (src.staging / n).stat().st_size}
                          for n in OUTPUT_FILES},
                "git": git_info(), "environment": hooks.environment(), "command": " ".join(sys.argv)}
    write("manifest.json", manifest)
    os.replace(src.staging, src.final)
    verify_manifest(src)
    return report


def verify_manifest(src: Sources = DEFAULT_SOURCES) -> dict:
    m = _load_json(src.final / "manifest.json")
    bad = [n for n, info in m["files"].items() if file_sha256(src.final / n) != info["sha256"]]
    if bad or set(m["files"]) != set(OUTPUT_FILES):
        raise Phase11B5Error(f"final/ does not match its manifest: {bad or 'file set differs'}")
    return m


# ============================================================
# Status / CLI
# ============================================================


def status_lines(src: Sources = DEFAULT_SOURCES) -> list[str]:
    lines = []
    try:
        verify_amendment(src)
        lines.append("amendment 06     : ADOPTED, hash verified")
    except (ResearchError, OSError) as e:
        lines.append(f"amendment 06     : NOT VERIFIED ({e})")
    if src.final.exists():
        try:
            verify_manifest(src)
            rep = _load_json(src.final / "final_report.json")
            q = rep["qualification"]["families"]
            lines += ["final            : COMPLETE, manifest verified",
                      f"qualified        : per-stock {q['per_stock']['qualified']}, panel {q['panel']['qualified']}",
                      f"determinism      : {rep['determinism']['outcome']} "
                      f"({rep['determinism']['n_identical']} / {rep['determinism']['n_runs']})",
                      f"acceptance       : {rep['phase_11b_acceptance']}"]
        except (ResearchError, OSError, KeyError, ValueError) as e:
            lines.append(f"final            : INVALID ({e})")
    elif src.staging.exists():
        names = sorted(p.name for p in src.staging.iterdir())
        lines.append("final            : protocol written, not run" if names == ["protocol.json"]
                     else f"final            : INCOMPLETE run present ({names})")
    else:
        lines.append("final            : not started")
    lines.append("confirmation     : NOT_EVALUATED")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11B Stage 11B.5 final families, audit, determinism and report")
    parser.add_argument("command", choices=["status", "protocol", "run"])
    args = parser.parse_args(argv)
    if args.command == "status":
        print("\n".join(status_lines()))
    elif args.command == "protocol":
        p = write_protocol()
        sizes = ", ".join(f"{k} {v['size']}" for k, v in p["families"].items())
        print(f"protocol.json written: {p['determinism']['n_runs']} determinism runs; families {sizes}")
    else:
        rep = run_final()
        print("\n".join(status_lines()))
        print(f"Development qualifiers: {rep['qualification']['total_qualified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
