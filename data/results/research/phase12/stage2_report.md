# Phase 12 Stage 2 — Stock Movement Magnitude Prediction (development period)

**Result:** under the pre-registered rule, **three pooled 1-day models qualify against the volatility-only baseline: LightGBM (best), Ridge and Random Forest.** No 5-day model qualifies and no per-stock model qualifies.

This is a **development qualification only**. It is not yet confirmed, because the features were chosen in Stage 1 on the same 2017–2024 data (see §4). The sealed holdout is untouched.

**Evidence**
- Protocol: `phase12/stage2/protocol.json`, written before any model fit. Protocol sha256 `fb9e9e38…`, file sha256 `19d0ace0…`.
- Results: `phase12/stage2/results/` — `results.json`, `per_stock_metrics.csv`, `per_stock_comparisons.csv`, `manifest.json`, `finalists/`.
- Inputs sha256 `0fb601fc…`, the frozen 11B development frames.
- Code: `training/phase12_magnitude.py`.

## 1. What was tested

| Item | Pre-registered setting |
|---|---|
| Target | \|future excess return over SPY\|. h = 1 primary, h = 5 secondary. |
| Features | `Volatility`, `spy_volatility_20`, `Volume_Change_1`, `Volume_Change_5`, `event__earnings_event_count`, `event__event_count`, `event__negative_event_count`, `sw_count_surprise_1_20` |
| Reference baseline | Ridge on the 2 volatility features only |
| Candidates | Ridge, Random Forest (depth 4), LightGBM (3-level, 7 leaves). Fixed parameters, no tuning. |
| Benchmarks and control | Training median; scaled volatility; LightGBM on the volatility features only (control) |
| Scopes | Pooled across 29 stocks (primary). Non-volatility features z-scored per stock using training-fold statistics; no stock identity feature. Per-stock models (secondary). |
| Validation | Expanding walk-forward: 504-session minimum training window, 63-session test folds, gap = h. 23 folds. Test dates 2019-02 .. 2024-09, identical for every model. About 41,000 pooled predictions per horizon. |
| Statistics | One-sided Diebold–Mariano on the per-date mean squared-error difference. Newey–West lag 12 (24 as sensitivity). Holm: primary family of 6, secondary family of 174. |
| Qualification | All of the following must hold: **Q1** MSE ≥ 1% better and MAE not worse; **Q2** Holm p < 0.05 and raw p (2L lag) < 0.05; **Q3** better in ≥ 3 of 4 time blocks; **Q4** gap and leakage checks; **Q5** ≥ 60% of stocks improved and still better without the best stock and without the best fold; **Q6** bit-identical rerun. |

## 2. Results

### Pooled, h = 1 (primary)

| Model | MSE gain vs baseline | MAE gain | R² vs median | DM t | Holm p | Blocks better | Stocks improved | Without best stock | Without best fold | Qualified |
|---|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** | **+5.04%** | +1.53% | 0.199 | 8.16 | < 1e-6 | 4 / 4 | 26 / 29 | +4.67% | +4.67% | **YES** |
| Ridge | +2.57% | +0.66% | 0.178 | 6.37 | < 1e-6 | 4 / 4 | 23 / 29 | +2.33% | +2.36% | **YES** |
| Random Forest | +2.02% | +0.98% | 0.174 | 3.41 | 0.00096 | 3 / 4 | 19 / 29 | +1.73% | +1.71% | **YES** |
| *Baseline (Ridge, volatility only)* | — | — | 0.157 | — | — | — | — | — | — | — |
| *LightGBM, volatility only (control)* | −0.25% | −0.55% | 0.155 | −0.95 | — | 1 / 4 | 11 / 29 | — | — | (control) |

**The control shows the gain comes from the volume and news features.** The same LightGBM using only the two volatility features is slightly *worse* than the baseline.

LightGBM's improvement grows across the four time blocks: +2.5%, +4.0%, +6.5%, +8.2%.

### Pooled, h = 5 (secondary)

| Model | MSE gain | MAE gain | Holm p | Blocks | Stocks improved | Failed criteria |
|---|---|---|---|---|---|---|
| LightGBM | +1.10% | +0.20% | 0.0018 | 3 / 4 | 13 / 29 | Q5 (breadth) |
| Ridge | +0.51% | +0.36% | 0.0004 | 3 / 4 | 17 / 29 | Q1 (< 1%), Q5 |
| Random Forest | +0.19% | −0.04% | 0.27 | 2 / 4 | 13 / 29 | Q1, Q2, Q3, Q5 |

### Per-stock models (secondary family, 174 comparisons)

- **0 qualify.** 0 survive Holm; 27 have raw p < 0.05.
- Stacked across stocks, per-stock models are no better than the baseline. LightGBM is worse, which looks like overfitting on about 500–1,800 rows per stock.

## 3. Answers

1. **Did any model outperform the volatility-only baseline?**
   - Yes, at h = 1, all three candidates, by 2.0–5.0% MSE.
   - At h = 5 the gains are small (0.2–1.1%) and narrow.
2. **Best at h = 1:**
   - **Pooled LightGBM.** MSE −5.0%, MAE −1.5%, Spearman correlation between predicted and actual magnitude 0.239 against 0.205 for the baseline.
3. **Best at h = 5:**
   - Pooled LightGBM (+1.1% MSE), but it fails breadth: only 13 of 29 stocks improve. Nothing qualifies at h = 5.
4. **Did pooled models beat per-stock models?**
   - **Yes, clearly.** Pooled LightGBM improves 26 of 29 stocks. Per-stock LightGBM, stacked across stocks, is worse than the baseline.
   - Pooling is what makes the signal usable.
5. **Which stocks showed reliable performance?**
   - With the pooled LightGBM at h = 1, the largest gains are CSCO +18%, IBM +17%, NKE +16%, INTC +13%, V +10%, MSFT +9%, DIS +8%.
   - Small losses: JPM −1.3%, MCD −0.4%, JNJ −0.2%.
   - No stock-specific model survives the 174-test Holm correction. The best is CSCO ridge h = 5, raw p 0.0003, Holm p 0.058.
6. **Did improvements survive statistical correction?**
   - Yes, for all three pooled h = 1 candidates (Holm p ≤ 0.001, also with the doubled lag) and for LightGBM and Ridge at h = 5.
   - None of the per-stock comparisons survive.
7. **Did any model satisfy the qualification criteria?**
   - Yes: pooled LightGBM, Ridge and Random Forest at h = 1, all six criteria.
   - They were refit on all development rows and saved as finalists.
8. **What prevented the others?**
   - h = 5: breadth (Q5), and for Ridge and Random Forest also the 1% minimum gain.
   - Per stock: statistical correction (Q2), given small per-stock samples.

## 4. Error analysis and limitations

**Prediction behaviour**
- Predictions rise with actual magnitude across all ten deciles.
- They are strongly **shrunk toward the average**:
  - On the 5% largest moves, the best model predicts only about 27% of the actual size (mean 1.2% vs 4.5%).
  - On the smallest half of moves, it predicts 0.85% against an actual 0.34%.
- Residuals are right-skewed (skew 3.8).
- **The models rank days by expected move size reasonably well, but they systematically underestimate large moves and overestimate quiet days.** Calibration is needed before any user-facing number.

**Limitations**
1. **Optimistic development result.** The 8 features were selected in Stage 1 using associations over 2017–2024, which includes these test dates. Training never saw test targets, but feature selection did. Treat the gains as an upper bound until the sealed holdout confirms them.
2. **Only magnitude, not direction.** A qualified model says how much a stock is likely to move relative to SPY, not which way.
3. **Modest effect size.** R² vs the median benchmark is 0.199 against 0.157 for the baseline. Economic value (for example for risk sizing) has not been tested.
4. **Uneven gains.** They are concentrated in tech and consumer names (CSCO, IBM, INTC, NKE, MSFT, V) and close to zero for financials and staples.
5. **Unvalidated event labels.** Event counts come from keyword rules (`rules_v1`).
6. **Untracked artifacts.** Finalist `.joblib` files are excluded by the repository `.gitignore` (`*.joblib`). They are hashed in the manifest but won't be committed unless that rule changes.

## 5. Recommended next step (for approval)

Confirm on the sealed holdout, 2024-10-01 .. 2026-09-25, **once**.
- Evaluate the **single pre-designated finalist**, pooled LightGBM h = 1, against the same volatility baseline.
- Reuse this Stage 2 code and the saved model, with no refitting and no changes.
- Use the same criteria Q1–Q3 and Q5.
- Freeze an amendment designating the finalist and the holdout procedure **before** unsealing.
- Ridge h = 1 can be named as a secondary finalist, with a Holm correction over the two.
