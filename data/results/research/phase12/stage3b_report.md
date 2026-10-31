# Phase 12 Stage 3B — Sealed Feature Builder (development-only validation)

**Result:** `training/phase12_sealed_features.py` rebuilds the 29 development frames (2017-02-01 .. 2024-09-30, 1,928 rows each) **exactly** from the frozen 11B inputs, using one assembly path that the sealed build also uses.

**Blocking issue:** the sealed build cannot run as Amendment 07 is written. Amendment 07 says no FinBERT run is needed, but `event__negative_event_count` is defined by each article's FinBERT label (§4). The sealed path therefore refuses until you approve an erratum. No sealed record was parsed and no sealed row was generated.

## 1. Generators reused (all unchanged; hashes equal the Amendment 07 pins)

| Feature | Generator | Definition at row D |
|---|---|---|
| `Volatility` | `features.feature_engineering.engineer_features` | sample std of the 20 daily returns ending at D |
| `Volume_Change_1`, `Volume_Change_5` | same | `Volume[D] / Volume[D-k] - 1` |
| `spy_volatility_20` | `services.market_context.compute_market_context` | std (ddof 1) of 20 SPY log returns ending at D; SPY sessions must equal the stock's sessions, with no forward fill |
| `event__event_count`, `event__earnings_event_count` | `services.event_features.generate_event_features` | articles with an event label from rules_v1 (`KeywordEventClassifier`), counted in the row window |
| `event__negative_event_count` | same | events whose **FinBERT label is negative** (`impact_from_sentiment_label`) |
| `sw_count_surprise_1_20` | `services.sentiment_window.generate_sentiment_window_features` | `count_1 - count_20 / 20`; NaN when the 20-session window is not fully covered by the news interval |

The builder adds only assembly glue: `price_features` mirrors `phase11a_research.stock_frame`, and `news_features` mirrors `phase11b_feature_build.news_features`. Tests show each produces the same values as the frozen function.

## 2. Timestamp rules

- **Prediction time.** Row D is predicted at ts(D) = D 16:30 America/New_York.
- **Prices.** Bars are cut at the period end before anything is computed, and every generator is backward-looking. A test multiplies all later prices and volumes and confirms earlier rows are unchanged.
- **News.** An article counts for row D only if ts(previous session) < `information_available_at` ≤ ts(D), where `information_available_at` = max(created_at, updated_at) of the stored latest revision.
  - Tests confirm the boundary: an article at exactly ts(D) counts for D; one microsecond later counts for D+1.
  - Adding a later article never changes an earlier row.
- **Revisions.** An article revised after the cutoff becomes available only at its revision time, which is the conservative choice.
  - Known frozen limitation: snapshot prices are dividend-adjusted using later dividends. The development frames share this, so it is unchanged.
- **Missing values.**
  - Event counts are 0 for empty windows, never NaN.
  - `sw_count_surprise_1_20` stays NaN when its window is uncovered (no fill); the confirmation runner then excludes that row.
  - A non-finite price feature makes the generator stop (fail closed). The run then aborts with no metrics.
- **Scaling.** None in the builder. The confirmation runner standardizes with the finalist's stored development statistics.

## 3. Development reproduction gate

- **Exact match** for all 29 stocks: symbols, dates, row counts, column order, float64 dtypes, NaN locations, and the values of `Close`, `SPY_Close` and the 8 features.
- The comparison is `np.array_equal`; no tolerance was needed.
- Record: `phase12/stage3b/development_reproduction.json`, written once. SHA-256 `c2e32dce…`. It holds per-stock frame hashes, input hashes and generator hashes.
- **Inputs:**
  - the Stage 2 input frames (inputs sha `0fb601fc…`);
  - the 11A-verified price snapshots;
  - Gate B's canonical news file (`74fc87d8…`);
  - the 11B.2 scores and events, verified by the 11B.3 functions, with rules_v1 labels re-derived and equal.
- **No sealed record is parsed for development:**
  - **Prices:** each CSV is parsed only up to 2024-09-30. Full files are hashed (bytes only) by the 11A verifier.
  - **News:** the file is sorted by created_at, so parsing stopped at line 97,792, the first article created after the cutoff. Only that line's partition fields were decoded. The remaining ~31.7k lines were only hashed.
  - 3 articles created before the cutoff but revised after it were routed by timestamp alone and dropped.
  - The 97,788 development articles reproduce Gate B's count and hash.
- **On any mismatch,** the gate stops with the stock, column, first date and both values, and writes nothing. Tests cover a 1-ulp value change, a moved NaN, a missing row, shifted dates, a changed dtype and an extra symbol.

## 4. Unresolved issue: Amendment 07 needs an erratum

> **Resolved in Stage 3C.** Erratum 01 was approved and written: `phase12/stage3/erratum_01.json`, SHA-256 `0dd61b0b…`, pinned in the builder. The development label check passed 500/500 (`phase12/stage3b/finbert_label_equivalence.json`). The final builder hash is `211c264a…`; the `5a0c2718…` hash in §6 was the Stage 3B version. The label-check record was written by `02046472…`, which differs from the final builder only in how it finds the local model cache.

Amendment 07's `sealed_feature_build` section says:

> "no FinBERT or any other model is run"

and

> "sentiment scores are not an input of any of the 8 features; where a generator needs one a constant placeholder is passed"

That is wrong for `event__negative_event_count`: a placeholder label would set it to a constant on sealed data and break the frozen feature definition. Stage 3A missed this.

**Proposed erratum 01** (not created; needs your approval). The amendment's rules and criteria stay unchanged:

1. Sealed articles are scored with the frozen 11B.2 FinBERT: `ProsusAI/finbert` revision `4556d130…`, CPU, default batch size, offline (`HF_HUB_OFFLINE=1`). The revision is already in the local cache.
2. The labels are used only for `event__negative_event_count`.
3. Before sealed scoring, 500 hash-chosen development articles are re-scored. Their labels must equal the stored 11B.2 labels, or the build stops.

Until the erratum file exists and its hash is pinned in `SCORING_ERRATUM_SHA256`, `build_sealed` refuses. No credentials or external services are needed.

## 5. Sealed path (implemented, never run)

`build_sealed` checks, in order:

1. Amendment 07 hash and rules.
2. Your `authorization.json`, which must name the amendment, runner and builder hashes and the exact statement.
3. The erratum pin.
4. Output directory not inside frozen evidence, and not already present.
5. A fresh development reproduction in the same process.
6. The FinBERT label check.
7. Only then: sealed articles are scored and classified, bars are cut at 2026-09-25, and frames for 2024-10-01 .. 2026-09-25 are written once with hashes to `data/processed/features/phase12_sealed/`.

Tests replace every data loader with one that raises if called. They show that each refusal happens before any data is read.

## 6. Hashes and runner integration

| Item | SHA-256 |
|---|---|
| Builder `training/phase12_sealed_features.py` | `5a0c2718fa6ffc119a7b5d0d154c07dab73a8b0d05518e2f2c2da9d2afceaf45` |
| Runner `training/phase12_confirmation.py` (after wiring) | `e3a35ded73db5ad3193e77e8e0293dc29da466a65300a2f1b3dcf6d697d5da31` |
| Amendment 07 (unchanged) | `54aa14a8…` |
| Finalist (unchanged) | `01f3c7e4…` |
| Baseline (unchanged) | `9e30150f…` |

- The runner's `confirm` command now passes the builder's `sealed_loader`. This is the only runner change; `rules()` and the amendment are unchanged, and `verify` still reproduces every pin.
- The runner accepts an authorization only if its `sealed_builder_code_sha256` equals the builder file's current hash (tested).
- `authorization.json` was not created.
- If the erratum is approved, pinning its hash changes the builder hash; the authorization must name the final hash.

## 7. Tests

- New builder tests: 28 passed.
- Full default suite: 1123 passed, 12 deselected (previously 1095).
- `git diff --check`: clean.

## 8. Files

- **New:** `training/phase12_sealed_features.py`, `tests/unit/test_phase12_sealed_features.py` (28 tests), `phase12/stage3b/development_reproduction.json`, this report.
- **Modified:** `training/phase12_confirmation.py` (CLI wiring only); `.gitattributes` (one `-text` rule for the builder).
- **Unchanged:** Stage 1, Stage 2, Amendment 07, the finalist, the baseline, and Phases 1–11B.5. Their status and manifest checks verify.
