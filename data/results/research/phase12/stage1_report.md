# Phase 12 Stage 1 — Development-Data Signal Diagnostics

**Status:** EXPLORATORY. No model was trained, and no association below is a qualified or confirmed predictor.
**Data:** the 29 hash-verified Phase 11B development frames (2017-02-01 .. 2024-09-30; inputs sha256 `0fb601fc…`, the same inputs as the frozen 11B registry). No sealed-period row was read.
**Evidence:** `phase12/stage1/` — `tests.csv` (sha256 `d4ac4715…`), `summary.json` (`67213b81…`), `manifest.json`.
**Code:** `training/phase12_signal_diagnostics.py` (`ed102d7d…`).

## 1. Bottom line

1. **Direction and return: no usable signal.** No feature family shows a robust association with future excess returns or their direction, at either horizon, per stock or pooled.
   - Pooled over all days: 0 calibrated Holm survivors out of 520 tests. The largest |ρ| is 0.026.
   - The only survivors are 2 news-volume counts on news days at h = 5, with ρ ≈ −0.035. That is within the range of chance survivors seen on placebo-shifted data.
2. **Magnitude (|future excess return|): clear, robust associations that survive placebo calibration.**
   - Strongest: **volume changes** (`Volume_Change_1/2/5`, calibrated t up to 12.9, even after removing the stock's own `Volatility`).
   - **Event counts** come next: earnings, negative, guidance, capital-action and total events, with calibrated t ≈ 4–7.3. Earnings-event magnitude has a positive sign in 29 of 29 stocks.
   - **Volatility features** (`Volatility`, `spy_volatility_20`) have the largest ρ (0.18–0.20). Calibration discounts them heavily because they are very persistent; they remain significant.
   - Associations are strongest at h = 1 and weaker at h = 5.
3. **Windowed sentiment (`sw_*`):** sentiment **tone** (`sw_mean_*`, positive and negative ratios) shows no association with any target. Only news **volume and surprise** features (`sw_count_surprise_1_20`, `sw_count_1`) show a modest magnitude association.
4. **Cumulative sentiment (known-bad control):** behaves exactly as predicted.
   - Raw HAC tests flag it massively: 57 pooled Holm survivors, and 45–74% placebo rejection.
   - Placebo calibration removes all of it: 0 calibrated survivors.
   - This confirms both the earlier feature defect (L1) and that the calibration works.

## 2. Method (what was tested and how)

| Item | Definition |
|---|---|
| Targets | `return` = future_excess_return_h (frozen `excess_return_v1`); `direction` = 1 if > 0 else 0; `magnitude` = \|future_excess_return_h\| (the only new definition). h = 1, 5. Last h rows of each frame unlabelled. |
| Features | 13 `sw_*`, 20 technical (set A), 9 market context (E − A), 23 `event__*` = 65 candidates; 15 cumulative `sentiment__*` = controls only. No feature recomputed. `sw_count_1` ≡ `event__article_count` by construction. |
| Samples | All labelled days (1,927 / 1,923 per stock) and news days (`event__article_count` > 0: median 1,070 per stock, minimum 328 for TRV; all ≥ 250). |
| Statistic | Spearman ρ = OLS slope of standardized ranks. Per stock: Newey–West SE. Pooled: within-stock standardized ranks (removes stock identity and scale), Driscoll–Kraay SE (scores summed per date, then Newey–West), robust to any cross-stock correlation on a date. Bartlett kernel, lag L = max(2h, ⌈n^⅓⌉) = 13; t with 2L also reported. |
| Placebo | Circular shift of the target relative to the features by 100 fixed offsets (10–90% of the series), the same offset for every stock. Keeps each series' own autocorrelation and the cross-stock dependence; breaks alignment. |
| Calibration | Raw HAC p-values over-reject for persistent series (see §3). Each test is also judged against its own 100 placebo statistics: t_cal = t / max(1, RMS of placebo t); p_empirical = (1 + #\|t_placebo\| ≥ \|t\|) / 101 (floor 0.0099). |
| Multiplicity | Holm within families fixed in code beforehand: (level, scope, target, candidate/control), each spanning both horizons, plus all stocks for per-stock families. **Primary: Holm on p_cal.** Holm on raw p also reported. |
| Tests | 29,116 tests, 29,092 eligible. Eligibility depends only on n ≥ 250 and the feature varying, never on results. |

## 3. Placebo results: why calibration was necessary

Placebo rejection rates of the raw HAC p-values, at nominal 5% (100 shifts):

| Family | Per stock | Pooled |
|---|---|---|
| return / direction, candidates | 6.3–6.5% | 6.4–7.5% |
| magnitude, candidates | 10–12% | 19–24% |
| cumulative sentiment controls | 8–51% | 16–74% |

- For return and direction, the raw p-values are only slightly liberal.
- For magnitude, and for anything trending, a 13-lag HAC under-covers badly. Volatility regimes and trending counts create spurious correlations over multi-year windows.
- A first (uncalibrated, 20-shift) run exposed this. It was set aside before review and kept outside the repository, and the procedure was changed to the calibrated one above.
- All conclusions in this report use calibrated statistics.

## 4. Findings by question

**Q1. Does windowed sentiment contain useful signal?**
- For direction or return: **no**. Pooled tone features have |ρ| ≤ 0.026 and calibrated |t| ≤ 2.8. There are no survivors in any pooled or per-stock family, and signs are mixed across stocks.
- For magnitude: only the **news-volume** members:
  - `sw_count_surprise_1_20` is a pooled survivor after removing volatility (ρ = 0.058, t_cal = 5.2).
  - `sw_count_1` / article count is a survivor on news days.
- Sentiment tone (`sw_mean_*`) has no magnitude association either.

**Q2. Is the signal more apparent for magnitude than direction?**
- **Yes, decisively.**

| Target | Pooled calibrated Holm survivors (candidates, both horizons) | Calibrated per-stock rejection rate |
|---|---|---|
| Magnitude | 17 all days, 20 news days | 15–25% |
| Return | 0 all days, 0 news days | ≈ 5–8% (at the nominal rate) |
| Direction | 0 all days, 2 news days | ≈ 5–8% (at the nominal rate) |

**Q3. Do technical features offer stronger evidence?**
- **For magnitude, yes:** `Volume_Change_*` are the strongest effects in the study, and `Volatility` / `BB_Width` carry the largest correlations. They also have the most per-stock calibrated survivors (18 of 36).
- **For direction or return, no.** The only per-stock survivors are 3 tests for V at h = 5 (short-term reversal: `Close_to_SMA_7`, `Close_to_EMA_12`, `Return_5`, ρ ≈ −0.16). That's a single stock and isolated.

**Q4. Are event features useful enough to justify further work?**
- **For magnitude, yes.** Earnings, negative, guidance, capital-action and total event counts survive calibration pooled, both all days and news days, also net of volatility. Earnings, event-count and negative-event magnitude correlations are positive in **29 of 29 stocks**.
- **For direction, no.**
- The keyword labels are still unvalidated (42.8% OTHER), so the event types should be treated as noisy proxies. Counts are more trustworthy than specific types.

**Q5. Does pooling improve the evidence?**
- **Yes.** Per-stock effects of ρ ≈ 0.05–0.08 rarely clear a 3,768-test Holm family (36 per-stock magnitude survivors). The same effects pooled across 29 stocks reach calibrated t of 4–13.
- The sign consistency across stocks (29 / 29 for the main event counts) shows the pooled result isn't driven by a few stocks.

**Q6. Which combinations deserve a small pre-registered experiment?**
- **Magnitude, h = 1** (primary) and **magnitude, h = 5** (secondary), pooled panel and per stock, compared against a **volatility-persistence baseline**.

**Q7. Which families should be excluded?**
- Cumulative `sentiment__*`: defective, trend proxies.
- Sentiment tone (`sw_mean_*`, `sw_*_ratio_*`): no evidence for any target.
- All features for **direction / return** as a primary objective.
- Fine-grained event types with few events (legal, regulatory, m_and_a, partnership, product, ownership, management): no evidence, and the labels are unvalidated.
- Market-context features beyond SPY volatility.

## 5. Recommended Phase 12 Stage 2 candidate set (for approval)

**Primary hypothesis H1 (magnitude, h = 1):** adding volume and news-event information improves out-of-sample forecasts of next-day |excess return| over a volatility-only baseline.
- **Baseline model:** `Volatility`, `spy_volatility_20` (persistence; plus naïve 20-day mean |excess return|).
- **Candidate features (8):** baseline + `Volume_Change_1`, `Volume_Change_5`, `event__earnings_event_count`, `event__event_count`, `event__negative_event_count`, `sw_count_surprise_1_20`.
- **Models:** standardized ridge and one shallow LightGBM. No sequence models.
- **Evaluation:** pooled panel (primary) and per stock (secondary), walk-forward with gap = h and a minimum training window.
- **Metrics:** QLIKE / MSE of |r| and out-of-sample R² versus the baseline, with a DM test.
- **Family:** about 8–12 hypotheses in total, Holm.

**Secondary hypothesis H2 (magnitude, h = 5):** same design.

**Optional H3 (direction, h = 5, news days):** a single pre-registered test of news-volume reversal (`sw_count_5` / `event__recent_event_count`, ρ ≈ −0.035).
- Low prior: the effect is within placebo-survivor noise.
- **Recommendation: leave it out of the primary family.**

Final evidence for any Stage 2 finalist must come from the sealed confirmation window, used once, as already decided.

## 6. Limitations

1. **Development data reuse.** This is the same 2017–2024 period used by Phases 9–11B. Everything here is exploratory and hypothesis-generating.
2. **Association is not prediction.** Full-sample ranks were used. ρ is an in-sample association, not an out-of-sample forecast, and implies nothing causal.
3. **Calibration is approximate.**
   - The calibrated t assumes a scaled-normal null.
   - The 100 circular shifts overlap and aren't independent.
   - p_empirical cannot go below 0.0099.
   - Per-stock news-day HAC runs over an irregular observation index.
4. **Overlapping families.** Families overlap (horizons, scopes, partial-volatility), so they are not independent evidence.
5. **Event labels.** Keyword `rules_v1` labels, unvalidated, with an Apple-specific PRODUCT rule.
6. **Magnitude ≠ direction for the product.** Magnitude predicts how much a stock may move, not which way.
7. **Effect sizes are small.** Magnitude correlations of 0.04–0.10 for news and volume features, 0.20 for volatility. Their economic value has to be shown by an out-of-sample model, not assumed.

## 7. Go / no-go

| Target / horizon | Decision |
|---|---|
| Magnitude h = 1 | **GO** — Stage 2 primary |
| Magnitude h = 5 | **GO** — Stage 2 secondary |
| Direction / return h = 1 | **NO-GO** |
| Direction / return h = 5 | **NO-GO** as primary; optional single exploratory test |
