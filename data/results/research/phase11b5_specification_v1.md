# Phase 11B.5: Final Families, Audit, Determinism and Report — Specification v1

**Status:** DRAFT FOR APPROVAL. No code written; nothing run.
**Date:** 2026-10-26
**Repository state:** `main` @ `f5b91e9`, in sync with `origin/main`. Untracked: `.kilo/`, the project audit, and this file.

**Labels used in this document**
- **[CONFIRMED]** — required by a frozen artifact; the source is cited.
- **[PROPOSED]** — a design decision this specification makes. Needs your approval.
- **[DECISION]** — the frozen design is silent or ambiguous. Needs your explicit choice (§9).

---

## 1. Purpose and scope

**[CONFIRMED]** `design_freeze.json` lists the stage `stages.11B_5_final_families_audit_report` with four steps and `stop_after: true`:
1. "Holm correction over 198 panel and 5742 per-stock comparisons"
2. "audit"
3. "determinism sample rerun"
4. "report"

The confirmation period is "not opened without explicit authorization".

**In scope**
- Combined Holm correction over Phase 11A + Phase 11B.
- Final qualification decision under the frozen rule.
- Audit that the frozen 11B.4 evidence matches its artifacts.
- Bounded determinism re-run of 11B development runs.
- Write-once final evidence and report.

**Out of scope**
- Confirmation period: stays sealed and NOT_EVALUATED.
- Recomputing any Phase 11A result.
- Any new features, targets, models, thresholds or families.
- Sealed article content.
- Documentation updates.

---

## 2. Evidence verified for this specification (read-only, 2026-10-26)

| Fact | Evidence |
|---|---|
| Final families fixed before results | `phase11/design_registry.json → design.multiple_comparisons`: `primary_panel_family` Holm, size 198; `secondary_per_stock_family` Holm, size 5,742; `aapl_excluded_sensitivity` "descriptive only; no Holm, no qualification", size 198; rule "families fixed now; never merged or re-defined after results" |
| Family membership | `design.comparison_ids` lists 5,742 / 198 / 198 IDs. **The union of 11A + 11B result IDs equals the frozen design ID set exactly for all three families.** |
| No overlap | 11A ∩ 11B = ∅ in every family; 0 duplicate IDs within either phase |
| Feature-set split | 11A uses A and E; 11B uses B, C, D, F |
| Ordering | 11A and 11B result lists are each in frozen-design order. The design list is not sorted alphabetically and is not "all 11A then all 11B". |
| p-values | All 5,742 + 198 + 198 raw p-values are floats in [0, 1]; none missing or NaN; none exactly 0.05 |
| Ties | Per-stock family: 142 tied raw p-values (91 equal to 1.0). Panel family: 0 ties. |
| Gate statuses (existing evidence) | Per-stock 5,742 / 5,742 `NOT_QUALIFIED`; panel 198 / 198 `NOT_QUALIFIED`; sensitivity 198 / 198 `NOT_QUALIFIED` |
| Qualification rule | `design_freeze.json → multiple_testing.qualification_requires_all_of = ["gate_QUALIFIED", "raw_p < 0.05", "holm_p < 0.05"]`; `design_registry.json → qualification.panel / per_stock` say the same |
| 11A p-values | `design_freeze.json → multiple_testing.phase_11a_p_values`: source `11A/registry.json`, `loaded_by_hash: true`, `never_recomputed: true`. Pinned SHA `d5cdef22…` (`PHASE_11A_REGISTRY_SHA256`, `training/phase11b_research.py`) |
| Stop condition | `design_freeze.json → stop_conditions`: "Holm families are not exactly 198 / 5742" |
| Existing Holm code | `holm_adjust`, `training/phase9_research.py:327`: step-down, input-order output, ties keep input order, rejects p outside [0, 1]. Used by `qualify_family` (`training/phase11a_research.py:443`). |
| Determinism controls | Models use `n_jobs=1`, `random_state=42`; LightGBM `deterministic=True` (`training/phase9_research.py:141–160`). 11B.4 ran with `--workers 1` (`development_results.json → workers`). |
| Earlier determinism precedent | Phase 8: A/B/C "bit-identical". Phase 10B: 90 classification results "bit-identical to Phase 10A". Phase 11 Amendment 01 used *tolerances*, but for **cross-phase** reproduction. |
| Run artifacts | Per-stock files hold, per run: `models` (metrics, qualification), `fold_metrics`, `folds`, `warnings`. Panel files also hold `per_date_losses`. **No row-level predictions are stored.** |
| Non-deterministic field | `warnings`: 18,308 identical scikit-learn UserWarnings appeared in only 3 of 372 runs (NKE D_h1, AXP C_h5, AAPL D_h1), so the warning *count* is not reproducible |

**Consequence:**
- Every comparison in both final families already has gate status `NOT_QUALIFIED`.
- Under the frozen AND rule, **the 11B.5 qualification outcome is 0 qualifiers in both families, whatever the Holm values are.**
- The Holm step must still be run and recorded as specified. Holm cannot change this outcome.

---

## 3. Statistical families and Holm procedure

### 3.1 Family definitions [CONFIRMED]

| Family | Size | Members | Holm | Can qualify |
|---|---|---|---|---|
| `primary_panel` | 198 | `design.comparison_ids.panel` = 66 (11A, A/E) + 132 (11B, B/C/D/F) | Yes, α = 0.05 | Yes |
| `secondary_per_stock` | 5,742 | `design.comparison_ids.per_stock` = 1,914 (11A) + 3,828 (11B) | Yes, α = 0.05 | Yes |
| `aapl_excluded_sensitivity` | 198 | `design.comparison_ids.aapl_excluded_sensitivity` = 66 + 132 | **No** | **Never**. Raw p and gate status reported, descriptive only. |

The 132 11B sensitivity comparisons are **not** merged into any confirmatory family (design rule: "never merged").

### 3.2 Inputs [PROPOSED]

**11A entries**
- Source: `11A/registry.json → results.{per_stock, panel, aapl_excluded_sensitivity}`.
- The file is hashed first and must equal `d5cdef22…`; on any mismatch, stop.
- Fields used: `id`, `gate_status`, `p_value_raw`.
- The existing 11A-family `p_value_holm` and `qualified` are left untouched and reported only as "11A stage, superseded".

**11B entries**
- Source: `11B/development_results.json → results.*`.
- Its file hash is recorded at 11B.5 start.
- Its `registry_sha256` must equal the frozen registry hash `dd0b47f9…`, and the registry's `matrix_sha256` must equal `169c369f…`.
- Every one of its 53 `artifacts` hashes must verify.

**Gate status**
- Taken **as recorded** in the frozen evidence. Never re-evaluated in the Holm step.
- The audit step (§5) confirms that the recorded gate status and p-value match the per-run artifacts.

### 3.3 Ordering [PROPOSED]

- Each family is assembled in the order of the frozen list `design_registry.json → design.comparison_ids.<family>`.
- Holm numbers don't depend on input order, even with ties (proof below).
- The fixed order guarantees byte-identical output files.

**Why ties don't matter:** tied p-values occupy consecutive ranks *r* and *r+1*. The first gets (m−r)·p. The second gets max(running, (m−r−1)·p) = (m−r)·p. Both receive the same adjusted value in either order.

### 3.4 Integrity checks before Holm [PROPOSED; any failure stops the run]

1. Family size is exactly 198 / 5,742 (sensitivity 198).
2. The set of assembled IDs equals the frozen design ID set. No missing, extra or duplicate IDs.
3. 11A IDs ∩ 11B IDs = ∅.
4. Every `p_value_raw` is a finite float in [0, 1].
5. Every `gate_status` is one of the gate_v1 statuses that appear in the evidence (`QUALIFIED`, `NOT_QUALIFIED`).
6. 11A per-family counts are 1,914 / 66 / 66 and 11B counts are 3,828 / 132 / 132.

**Failed or missing runs:** there are none (4,092 / 4,092 11B, 2,046 / 2,046 11A). Procedure if one is ever found:
- The family is **not** shrunk.
- 11B.5 stops with `FAMILY_INTEGRITY_FAILED`.
- Imputing p = 1 or dropping hypotheses would change a frozen family, so neither is allowed.

### 3.5 Holm algorithm [CONFIRMED method; PROPOSED implementation]

- Reuse `training.phase9_research.holm_adjust` unchanged, which was already used for the 11A families.
- For m hypotheses, sort by (p, frozen position). The adjusted p at rank k (0-based) is the running maximum of min(1, (m−k)·p_(k)), returned in input order.

### 3.6 Qualification (per comparison, in this order) [CONFIRMED rule; PROPOSED evaluation order]

1. `gate_ok` = (`gate_status == "QUALIFIED"`). gate_v1 already contains, depending on task:
   - main-metric improvement over the reference baseline
   - DM p < 0.05
   - for classification, accuracy ≥ Always UP
2. `raw_ok` = `p_value_raw < 0.05` (strict).
3. `holm_ok` = `p_value_holm_final < 0.05` (strict).
4. `qualified` = `gate_ok AND raw_ok AND holm_ok`.

- All three flags are reported for every comparison.
- Holm is computed for every comparison in a confirmatory family, including those whose gate already failed. It never overrides the gate.
- Boundary values exactly equal to 0.05 fail (strict `<`, matching `qualify` in `training/evaluation_harness.py`).

---

## 4. Confirmation-period handling

- **[CONFIRMED]** `design_registry.json → qualification.confirmation`: "with no qualifier the confirmation holdout is NOT evaluated (NOT_APPLICABLE)".
- **[PROPOSED]** 11B.5 records `confirmation.status = "NOT_EVALUATED"` and `confirmation.applicability = "NOT_APPLICABLE"` (0 development qualifiers), plus `"authorization": null`.
- Sealed data is not touched. No sealed path is opened; only `news_integrity_report.json` is hashed.
- **[DECISION D3]** — see §9.

---

## 5. Audit step [PROPOSED]

This is a read-only reconciliation of the frozen 11B.4 evidence. No model fitting.

| # | Check | Pass condition |
|---|---|---|
| A1 | Frozen inputs | `verified_inputs()` passes: design freeze, Amendments 03–05, 11.0 design, Amendment 01, price report, 11A registry, feature verification, 29 feature CSVs |
| A2 | Registry | `load_registry()` passes: FROZEN, matrix SHA `169c369f…`, 5 prerequisites PASS with pinned hashes |
| A3 | Results file | `status == COMPLETE`; counts 3,828 / 132 / 132; `registry_sha256` matches; `confirmation == NOT_EVALUATED`; `data_kind == real` |
| A4 | Artifacts | All 53 artifact SHA-256s match; no `development.incomplete/` directory |
| A5 | Re-derivation | Re-derive the 4,092 entries from the 53 artifacts using the existing `_entries` logic. Every `id`, `gate_status`, `p_value_raw`, `dm_stat` and `metrics` must equal `development_results.json` **exactly**. |
| A6 | Code | Current `code_sha256()` equals the registry's `code_sha256` |
| A7 | 11B stage Holm | Recompute descriptive stage Holm over 3,828 / 132; must equal the stored `p_value_holm_11b_stage_descriptive` exactly |
| A8 | Write-once | Hashes of every pre-existing file under `phase11/11B/` and `phase11/11A/` are recorded before and after 11B.5; they must be identical |

Any failure → `AUDIT_FAILED`. The run stops before Holm and writes only the failure evidence.

---

## 6. Determinism re-run

### 6.1 Unit of re-run [PROPOSED]

- **Per-stock run:** one (symbol, feature set, horizon). It produces 11 candidate comparisons plus 4 baselines. Re-executed exactly as `_stock_job` does for one run: `p10a.evaluate_excess(frame, FEATURE_SETS[fs].columns, a11.candidates(), horizon=h, n_splits=20)`.
- **Panel run:** one (kind, feature set, horizon). It produces 11 comparisons. Re-executed exactly as `_panel_job`.

### 6.2 Sample selection [PROPOSED; fixed before any re-run]

**Stratum S1: coverage**, 12 per-stock runs.
- For each (fs, h) in frozen order (B, C, D, F × 1, 3, 5), choose
  `symbol = universe[int(sha256(f"phase11b5|determinism_v1|{fs}|{h}").hexdigest(), 16) % 29]`, using the frozen universe order.
- Covers all 11 models × 4 feature sets × 3 horizons (132 model / feature-set / horizon cells) across up to 12 symbols.
- No random seed is involved; the rule is a pure hash.

**Stratum S2: decision boundary**, at most 3 per-stock runs.
- The runs containing the 3 smallest 11B per-stock raw p-values, ties broken by frozen order. Currently XOM C_h1 (p = 0.0577), MSFT F_h3 (0.0585) and V F_h3 (0.0792).
- Duplicates of S1 are skipped and **not** replaced.
- Determinism matters most near the significance threshold.

**Stratum S3: panel**, 2 panel runs.
- (a) The primary-panel run containing the smallest 11B panel raw p (currently `panel` D_h1).
- (b) One AAPL-excluded run chosen by `sha256("phase11b5|determinism_v1|panel_exAAPL")`, taken modulo the 12 (fs, h) cells.

**Total:** 14–17 runs.
- Approximately 132–165 per-stock comparisons (3.4–4.3% of 3,828).
- 22 panel and sensitivity comparisons (≈ 8% of 264).
- The exact list is written into the protocol file (§7) **before** any re-run starts.

### 6.3 Execution conditions [PROPOSED]

- Same code: `code_sha256()` must equal the registry.
- Same inputs: `inputs_sha256` must equal the registry.
- **Same environment:** Python and package versions must equal `registry.environment`. On any mismatch, 11B.5 refuses to re-run and reports `ENVIRONMENT_MISMATCH`. It doesn't compare across environments.
- Same mode as 11B.4: sequential, in-process (`workers=1`).

### 6.4 Comparison rule [PROPOSED: bit-identical]

- For each re-run, the new run report is serialized with the same JSON writer used for 11B.4 artifacts. It is then compared with the stored run object (`stock_<SYM>.json → runs.<fs>_h<h>` or the panel file).
- Comparison uses canonical JSON (sorted keys, the writer's float representation) **after removing the `warnings` key** (non-deterministic, §2).

**Required: exact equality of**
- `models[*].metrics`
- `models[*].qualification`: gate status, reasons, DM statistic, p-value, model and baseline values
- `fold_metrics`, `folds`, `evaluation_period`, `training_period`, `target_summary`
- for panel runs, `per_date_losses`

**Recorded per run:** SHA-256 of the canonical stored object and of the canonical re-run object. These must be equal.

**Reported separately (never a failure):** stored vs re-run `warnings.total`.

**Why bit-identical, not a tolerance:**
- Same machine, same pinned versions, fixed seeds, `n_jobs=1`, LightGBM `deterministic=True`.
- Phases 8 and 10B already achieved bit-identity.
- Amendment 01's tolerance (5e-3 relative, 0.02 on p) applies only to *cross-phase* reproduction, not to re-running the same code on the same inputs.
- **[DECISION D2]** confirms this choice.

### 6.5 Outcome [PROPOSED]

- `DETERMINISM_VERIFIED`: every sampled run is byte-identical after removing `warnings`.
- `DETERMINISM_CHECK_FAILED`: any difference. **Every** differing field path is reported, with stored value, re-run value and absolute/relative difference.
- Following the Amendment 01 principle, the outcome **never alters** the frozen 11B results or the qualification decision.
- A failure sets `phase_11b_acceptance = "NOT_ACCEPTED_DETERMINISM_FAILED"` (see **[DECISION D4]**).

---

## 7. Output artifacts

**[PROPOSED] New write-once directory:** `data/results/research/phase11/11B/final/`
- Built in `final.incomplete/` and renamed on success, the same pattern as 11B.4.
- Already covered by the `data/results/research/** -text` rule in `.gitattributes`, so its bytes are preserved on checkout.
- Nothing in `phase11/11A/`, `phase11/11B/*.json`, `prerequisites/` or `development/` is written or modified.

| File | Content |
|---|---|
| `protocol.json` | Written **first**, before any computation. Families (IDs in order), qualification rule, the exact determinism sample list from the §6.2 algorithm, comparison rule, input hashes (11A registry, 11B registry, development results, 53 artifacts), `code_sha256` (11B.5 module + reused modules), environment. Its own SHA-256 is recorded in every later file. |
| `audit.json` | Checks A1–A8: each with `pass`, observed values, expected values |
| `combined_holm.json` | Per family: `family`, `size`, `alpha`, `method: "Holm"`, `order_source`. Per entry: `id`, `phase` (11A / 11B), `feature_set`, `horizon`, `model`, `symbol` (per-stock), `gate_status`, `p_value_raw`, `p_value_holm_final`, `gate_ok`, `raw_ok`, `holm_ok`, `qualified`. Summary counts. Sensitivity family: same entries **without** Holm and with `qualified: null`, `role: "descriptive only"`. |
| `determinism.json` | Sample list with stratum, per-run hashes (stored / re-run), `identical`, differences, `warnings.total` stored vs re-run, runtime per run, outcome |
| `final_report.json` | `document_type: "phase_11b_final_report"`, `phase: "11B"`, `stage: "11B.5"`, `status: "COMPLETE"`; qualification outcome per family (counts and IDs of qualifiers); 11A vs 11B breakdown; audit and determinism outcomes; `phase_11b_acceptance`; confirmation block (§4); limitations copied from `design_freeze.json → known_limitations`; hashes of all sibling files |
| `final_report.md` | Human-readable rendering of `final_report.json`. Generated, never hand-edited. |
| `manifest.json` | Written **last**. SHA-256 and byte size of every file in `final/`, plus `protocol_sha256`, git commit, `uncommitted_changes`, `created_at` (UTC ISO-8601), Python and package versions, command line |

**Common rules**
- JSON uses UTF-8, LF line endings, sorted keys and indent 1, matching the existing evidence.
- Every file carries `phase`, `stage`, `protocol_sha256`, `created_at`.
- Runtimes are recorded but excluded from any identity comparison.

---

## 8. Module and command design [PROPOSED]

**New module:** `training/phase11b_final.py`. Existing modules are unchanged and only imported.

| Function | Responsibility |
|---|---|
| `load_final_inputs()` | Verifies hashes (A1–A4) and returns 11A results, 11B results, the design ID lists and the registry |
| `assemble_family(name, a11_entries, b11_entries, design_ids)` | Orders entries by frozen design list; integrity checks §3.4; raises `Phase11B5Error` |
| `combined_holm(family_entries)` | Calls `holm_adjust`; adds `p_value_holm_final` |
| `qualify_final(entry)` | Gate AND raw AND Holm; returns the flags |
| `rederive_entries()` | Re-derives entries from artifacts with `phase11b_research._entries`; compares exactly (A5) |
| `select_determinism_sample(dev_results, universe)` | Pure function implementing §6.2 |
| `rerun_stock(sym, fs, h, frames)` / `rerun_panel(kind, fs, h, frames)` | Thin wrappers around the existing evaluate functions |
| `compare_run(stored, rerun)` | Canonical JSON diff without `warnings`; returns hashes and field-level differences |
| `write_protocol()`, `run_final()`, `status_lines()` | Orchestration, staging, publishing |

**Commands**
```
python -m training.phase11b_final status    # read-only
python -m training.phase11b_final protocol  # writes final.incomplete/protocol.json only (no computation)
python -m training.phase11b_final run       # audit → Holm → determinism → report → manifest → publish
```

- `run` refuses unless `protocol.json` exists and verifies.
- `run` refuses if `final/` already exists (write-once).
- After `final/` exists, `status` prints the outcome and verifies the manifest hashes.

**Follow-ups this requires (also code)**
- `.gitattributes`: add `training/phase11b_final.py -text`, because its hash is recorded in `protocol.json` (the P1-01 rule for pinned code).
- `pyproject.toml`: no change.

---

## 9. Decisions requiring your approval

| ID | Question | Recommendation |
|---|---|---|
| **D1** | Determinism sample: S1 (12 hash-chosen per-stock runs covering every model × feature set × horizon) + S2 (≤ 3 runs at the smallest p-values) + S3 (2 panel runs) = 14–17 runs. | Approve as written. The alternative is S1 + S3 only (14 runs). |
| **D2** | Comparison: bit-identical, excluding the `warnings` field. | Approve. Tolerance only if you prefer cross-machine portability, which would need a written amendment with numeric limits. |
| **D3** | Confirmation record: `NOT_EVALUATED` with `applicability: NOT_APPLICABLE`, no authorization field set. | Approve. |
| **D4** | On determinism failure: results and qualification unchanged; `phase_11b_acceptance = NOT_ACCEPTED_DETERMINISM_FAILED` until investigated. | Approve. |
| **D5** | Where this procedure is frozen: `final/protocol.json`, written before computation, as a procedural record. The alternative is a formal **Amendment 06**, a separate JSON next to Amendments 03–05. | Amendment 06 is the more conservative choice, because the frozen design only names the 11B.5 steps. A protocol file is lighter. **Your call.** |
| **D6** | Re-run environment: refuse on any version mismatch versus `registry.environment`. | Approve. |

---

## 10. Test plan [PROPOSED]

New file: `tests/unit/test_phase11b_final.py`. Synthetic inputs except where marked "real".

| # | Test | Asserts |
|---|---|---|
| F1 | Exact family sizes | Synthetic 11A / 11B families of the right sizes pass; one extra, one missing or one duplicate → `Phase11B5Error` |
| F2 | Disjoint IDs | Overlapping 11A / 11B ID → error |
| F3 | Membership | ID not in the frozen design list → error; design ID absent from results → error |
| F4 | Deterministic order | Shuffled input gives the same output order and byte-identical JSON |
| F5 | Holm known values | `[0.01, 0.04, 0.03, 0.005]` → `[0.03, 0.06, 0.06, 0.02]`; `[0.5]` → `[0.5]`; `[0.3, 0.4]` → `[0.6, 0.6]`; capping: `[0.6, 0.9]` → `[1.0, 1.0]` |
| F6 | Ties | `[0.02, 0.02, 0.5]` → `[0.06, 0.06, 0.5]` in either input order |
| F7 | Boundaries | raw p = 0.05 → `raw_ok` false; Holm p = 0.05 → `holm_ok` false; p = 0 accepted |
| F8 | Invalid p | NaN, None, −0.1, 1.1, string → error (no imputation) |
| F9 | Gate interaction | (QUALIFIED, raw 0.01, Holm 0.04) → qualified. (QUALIFIED, 0.01, Holm 0.2) → not. (NOT_QUALIFIED, 0.001, 0.01) → not. (QUALIFIED, 0.06, —) → not. |
| F10 | Sensitivity family | Never receives Holm; `qualified` is null |
| F11 | Sample selection | Pure and stable: same input gives the same list; S1 covers all 12 (fs, h) cells; S2 dedupes against S1; S3 has exactly 2 runs; known hash fixture → known symbols |
| F12 | Run comparison | Identical objects → identical; `warnings` differences ignored; one changed float in `fold_metrics` → failure with the field path; missing key → failure |
| F13 | Environment guard | Mismatched package version → `ENVIRONMENT_MISMATCH`, no re-run |
| F14 | Write-once | Existing `final/` → refuse. Pre-existing 11A / 11B files are hashed before and after the run → identical (temporary directory). |
| F15 | Staging | Failure mid-run leaves `final.incomplete/` and no `final/`; `status` reports INCOMPLETE |
| F16 | Protocol first | `run` without a valid `protocol.json` → refuse; edited protocol → hash mismatch → refuse |
| F17 (real) | Families from real evidence | Assembled real families: 5,742 / 198 / 198; sets equal the design IDs; 11A ∩ 11B = ∅. Reads committed JSON only. |
| F18 (real) | Re-derivation | `rederive_entries()` on real artifacts equals `development_results.json`. No model fitting. |

**Not a unit test (manual, user-run):**
- Fresh-clone reproducibility. Clone with `core.autocrlf=true` (and `core.longpaths=true` if the path is deep), copy the 29 development-only feature CSVs, then run `phase11b_final status`, which must verify the manifest.
- The default `pytest` suite must stay green.

---

## 11. Acceptance criteria

1. F1–F18 pass; the full default suite passes.
2. `protocol.json` written and hashed before any computation; your approvals D1–D6 are recorded in it.
3. Audit A1–A8 pass.
4. Combined Holm computed over exactly 198 and 5,742 comparisons in frozen order; sensitivity reported without Holm.
5. Qualification counts reported per family and per phase (expected from existing evidence: 0 and 0).
6. Determinism outcome recorded with per-run hashes.
7. All pre-existing evidence hashes unchanged (A8).
8. `final/` published atomically; manifest verifies; `status` passes from a fresh clone that has the feature CSVs.
9. Confirmation NOT_EVALUATED; no sealed path opened.
10. You review, commit and push.

---

## 12. Risks and open questions

| Risk | Mitigation |
|---|---|
| Determinism fails because of BLAS threading. In the 11B.4 `workers=1` path, the thread limit from `a11._init_worker` was **not** applied; linear models may use multi-threaded BLAS. | Re-run in the same mode on the same machine. If it fails, the field-level diff will show whether only linear models differ. Don't "fix" this by changing 11B.4. |
| Panel re-runs are costly | Only 2 in the sample |
| The outcome is fixed in advance (0 qualifiers) | Stated openly. The value of 11B.5 is formal closure, the audit and determinism evidence, not new inference. |
| Same development period reused across 9,108 comparisons | Documented limitation. 11B.5 makes no further claims. |
| Long filenames in a fresh clone on deep paths | Documented; `core.longpaths` |
| `status` crashes with `FileNotFoundError` when local feature CSVs are missing (from P1-01 testing) | The new `status` catches missing inputs and reports NOT VERIFIED. Existing `phase11b_research` left unchanged. |

---

## 13. Effort estimate

| Item | Estimate | Basis |
|---|---|---|
| Implementation + tests | 1.5–2.5 days | Size of `phase11b_research.py` and its 54 tests (INFERRED) |
| Audit + Holm runtime | < 2 min | JSON reads and hashing only |
| Determinism runtime | ≈ 20–60 min (INFERRED, not measured) | 11B.4 took 18,892 s for 348 per-stock + 24 panel runs, sequentially. Panel runs dominate. 14–17 runs are ≈ 4–5% of the per-stock work plus 2 of 24 panel runs. Actual runtime is recorded in `determinism.json`. |
| Your review | Spec + D1–D6; later the result files | — |

---

## 14. Review checklist

| Item | Status |
|---|---|
| Family sizes and membership traced to frozen IDs (5,742 / 198; sensitivity 198 descriptive) | ✅ verified |
| 11A p-values loaded by hash, never recomputed | ✅ specified |
| Ordering, ties, boundaries, invalid inputs specified | ✅ |
| Qualification rule unchanged (gate AND raw AND Holm, strict < 0.05) | ✅ |
| No threshold, model, target or family changes | ✅ |
| Determinism sample algorithm defined, frozen before running | ✅ pending D1 |
| Comparison rule (bit-identical minus `warnings`) | ✅ pending D2 |
| Write-once location separate from frozen evidence | ✅ |
| Confirmation untouched | ✅ pending D3 wording |
| Tests and acceptance criteria | ✅ |
| Formal freezing vehicle (protocol vs Amendment 06) | ⚠ D5 open |

**Verdict:** the specification is complete and **ready for approval, subject to decisions D1–D6**. D5 (protocol file vs formal Amendment 06) is the only one without a clear default. No code has been written; nothing has been run, committed or pushed.
