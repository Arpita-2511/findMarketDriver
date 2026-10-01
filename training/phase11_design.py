"""
Phase 11.0 - cross-stock generalization research: DESIGN FREEZE ONLY.

Question: is the absence of a qualified predictive signal specific to AAPL, or
does the FindMarketDriver methodology fail to generalize across equities?

This module defines and freezes the Phase 11 design (universe, corporate-action
integrity rule, intervals, sample rule, target, feature sets, models, panel and
per-stock evaluation, baselines, DM procedure, multiple-comparison families,
qualification, staging) as a write-once registry with a content hash.
It downloads nothing, reads no market or news data and fits no model; the
pure helpers below (affected-row exclusion, eligibility) are the frozen
semantics that Phase 11.1+ must use.

Usage (project root):
    python -m training.phase11_design freeze
    python -m training.phase11_design status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase10a_research as p10a  # noqa: E402
from training.evaluation_harness import GATE_VERSION, HARNESS_VERSION, HarnessConfig, git_info  # noqa: E402
from training.excess_targets import HORIZONS, target_definition  # noqa: E402
from training.phase9_research import ResearchError, _slug, _write_json_atomic, canonical_json, environment  # noqa: E402

PHASE = "11.0"
DESIGN_VERSION = "phase11_design_v1"
REGISTRY_VERSION = "phase11_design_registry_v1"
RESEARCH_DIR = PROJECT_ROOT / "data" / "results" / "research" / "phase11"
REGISTRY_PATH = RESEARCH_DIR / "design_registry.json"

# ---------------- universe ----------------
MEMBERSHIP_DATE = "2016-12-31"
DJIA_MEMBERS_2016 = ("AAPL", "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ",
                     "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "V", "VZ",
                     "WMT", "XOM")
EXCLUDED = {"DD": ("the 2016 DJIA member DD (E.I. du Pont de Nemours) does not have a single continuous listed "
                   "security series covering the full research interval; today's DD successor security is NOT "
                   "substituted and NO synthetic shareholder-return series is constructed")}
CONTINUATIONS = {"UTX": {"research_ticker": "RTX",
                         "basis": ("legal/security continuation: United Technologies was the legal survivor of the "
                                   "Raytheon merger completed 2020-04-03 and was renamed Raytheon Technologies "
                                   "(ticker RTX, later RTX Corporation)"),
                         "condition": ("Phase 11.1 must verify whether the provider's adjusted RTX history correctly "
                                       "accounts for the Carrier/Otis distributions and related corporate actions; "
                                       "if not, the general event-span exclusion rule applies; no other ticker is "
                                       "substituted")}}
UNIVERSE = tuple(sorted(CONTINUATIONS.get(t, {}).get("research_ticker", t) for t in DJIA_MEMBERS_2016 if t not in EXCLUDED))
MEMBERSHIP_SOURCE = {
    "source": 'Wikipedia, "Historical components of the Dow Jones Industrial Average", section "March 19, 2015"',
    "url": "https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average",
    "in_force": "2015-03-19 until the next component change on 2017-09-01 (so in force on 2016-12-31)",
    "accessed": "2026-10-01",
    "scope": ("used only to establish the historical membership universe; it does not independently verify the "
              "subsequent corporate-action mechanics"),
    "membership_rule": ("historical membership on 2016-12-31; members later removed from the index are NOT replaced"),
}

# ---------------- corporate actions (to verify in Phase 11.1; no invented dates or ratios) ----------------
VERIFY = "REQUIRES_PHASE_11_1_VERIFICATION"
CORPORATE_ACTION_EVENTS = [
    {"symbol": "RTX", "member": "UTX", "event": "merger with Raytheon Company; UTC legal survivor renamed Raytheon "
     "Technologies / RTX", "type": "merger / security continuity", "date": "2020-04-03",
     "date_basis": "completion date stated on the Wikipedia RTX Corporation page (accessed 2026-10-01)", "status": VERIFY},
    {"symbol": "RTX", "member": "UTX", "event": "Carrier and Otis distributions to UTC shareholders",
     "type": "spin-off / distribution", "date": None, "status": VERIFY},
    {"symbol": "GE", "event": "transportation-business transaction with Wabtec", "type": "separation", "date": None, "status": VERIFY},
    {"symbol": "GE", "event": "GE HealthCare spin-off", "type": "spin-off", "date": None, "status": VERIFY},
    {"symbol": "GE", "event": "GE Vernova spin-off", "type": "spin-off", "date": None, "status": VERIFY},
    {"symbol": "GE", "event": "reverse stock split", "type": "reverse split", "date": None, "status": VERIFY},
    {"symbol": "MMM", "event": "Solventum spin-off", "type": "spin-off", "date": None, "status": VERIFY},
    {"symbol": "MRK", "event": "Organon spin-off", "type": "spin-off", "date": None, "status": VERIFY},
    {"symbol": "IBM", "event": "Kyndryl spin-off", "type": "spin-off", "date": None, "status": VERIFY},
    {"symbol": "PFE", "event": "Upjohn separation and combination with Mylan (Viatris)", "type": "spin-off / combination",
     "date": None, "status": VERIFY},
    {"symbol": "JNJ", "event": "Kenvue separation / exchange offer", "type": "separation / exchange offer", "date": None,
     "status": VERIFY},
]
CORPORATE_ACTION_RULE = {
    "prices": "use the market-data provider's adjusted history where available",
    "scope": ("every pre-registered material split, spin-off, distribution, merger/security-continuity event or "
              "similar corporate action of every included security; Phase 11.1 also records any further material "
              "action found in the provider's action records for any of the 29 securities"),
    "steps": {
        "A": "verify the event against the provider's recorded action/adjustment information (Phase 11.1)",
        "B": "determine whether the downloaded adjusted price history properly incorporates the event",
        "C": "properly adjusted -> retain the affected observations",
        "D": ("NOT properly adjusted -> exclude prediction rows whose feature lookback or target horizon spans the "
              "event; count and report every excluded row; never repair prices, never splice securities, never "
              "exclude the whole stock unless a separately frozen rule requires it"),
    },
}

# ---------------- affected-row semantics ----------------
MAX_FEATURE_LOOKBACK = 50          # sessions (spy_close_to_sma_50 / SMA-type windows); see AFFECTED_ROW_SEMANTICS
AFFECTED_ROW_SEMANTICS = {
    "event_session": ("E = the first trading session (of the stock's own calendar) whose price reflects the event, "
                      "i.e. the discontinuity sits in the session-E return Close[E]/Close[E-1]-1"),
    "lookback": (f"row D's features span the event iff E lies in the last {MAX_FEATURE_LOOKBACK} sessions ending at D: "
                 f"D in [E, E+{MAX_FEATURE_LOOKBACK - 1}] (session indices)"),
    "target": "row D's h-session target Close[D+h]/Close[D]-1 spans the event iff D < E <= D+h: D in [E-h, E-1]",
    "excluded_rows": f"D in [E-h, E+{MAX_FEATURE_LOOKBACK - 1}] (session indices), per horizon h",
    "timing": "session-based semantics of Phases 9-10: prediction timestamp D 16:30 America/New_York, gap = h",
    "ema_note": ("EMA-based features (Close_to_EMA_12/26, MACD_*) have exponentially decaying memory beyond 50 "
                 "sessions; the residual influence of an unadjusted event older than 50 sessions is a recorded "
                 "limitation, not an exclusion criterion"),
}
MIN_DEVELOPMENT_ROWS = 1800


class Phase11DesignError(ResearchError):
    """The Phase 11 design would be violated."""


def affected_row_mask(n_rows: int, event_session: int, horizon: int, lookback: int = MAX_FEATURE_LOOKBACK) -> np.ndarray:
    """
    Boolean mask over rows 0..n_rows-1 (session indices of one stock's calendar):
    True where the feature lookback or the h-session target spans an unadjusted event at session E.
    """
    if horizon < 1 or lookback < 1:
        raise Phase11DesignError("horizon and lookback must be >= 1")
    d = np.arange(n_rows)
    spans_lookback = (d >= event_session) & (d <= event_session + lookback - 1)
    spans_target = (d >= event_session - horizon) & (d <= event_session - 1)
    return spans_lookback | spans_target


def development_eligibility(n_labelled_dev_rows: int, excluded_dev_rows: int, minimum: int = MIN_DEVELOPMENT_ROWS) -> dict:
    """Minimum-row rule evaluated AFTER all frozen data-integrity / corporate-action exclusions."""
    remaining = n_labelled_dev_rows - excluded_dev_rows
    if excluded_dev_rows < 0 or remaining < 0:
        raise Phase11DesignError("invalid row counts")
    return {"labelled_dev_rows": n_labelled_dev_rows, "excluded_dev_rows": excluded_dev_rows,
            "remaining": remaining, "minimum": minimum, "eligible": remaining >= minimum}


# ---------------- feature sets, models ----------------
COLUMN_RENAMES = {"aapl_minus_spy_return_1": "stock_minus_spy_return_1",
                  "aapl_minus_spy_return_5": "stock_minus_spy_return_5"}


def feature_sets() -> dict:
    """Phase 10A feature sets A-F (F = 64 distinct), with the AAPL-named relative returns generalized per stock."""
    out = {}
    for k, v in p10a.FEATURE_SETS.items():
        cols = [COLUMN_RENAMES.get(c, c) for c in v["columns"]]
        out[k] = {"families": list(v["families"]), "columns": cols, "n_features": len(cols)}
    return out


FEATURE_SETS = feature_sets()
STAGES = {"11A": {"feature_sets": ["A", "E"], "inputs": "prices only (stock, SPY, QQQ)"},
          "11B": {"feature_sets": ["B", "C", "D", "F"],
                  "inputs": "prices + multi-symbol news, FinBERT sentiment, rules_v1 events",
                  "commitment": "runs regardless of the 11A results"}}


def per_stock_id(symbol: str, fs: str, h: int, model: str) -> str:
    return f"p11_stock_{symbol}_{fs}_h{h}_{_slug(model)}"


def panel_id(fs: str, h: int, model: str, sensitivity: bool = False) -> str:
    return f"p11_{'panel_exAAPL' if sensitivity else 'panel'}_{fs}_h{h}_{_slug(model)}"


def design() -> dict:
    cands = p10a.Spec10A().candidates()
    models = [c for c in cands if not c.is_baseline]
    fs_list, horizons = list(FEATURE_SETS), list(HORIZONS)
    per_stock = [per_stock_id(s, f, h, c.name) for s in UNIVERSE for f in fs_list for h in horizons for c in models]
    panel = [panel_id(f, h, c.name) for f in fs_list for h in horizons for c in models]
    sens = [panel_id(f, h, c.name, True) for f in fs_list for h in horizons for c in models]
    alpha = HarnessConfig().significance_level
    return {
        "phase": PHASE, "design_version": DESIGN_VERSION,
        "research_question": ("Is the absence of a qualified predictive signal specific to AAPL, or does the current "
                              "FindMarketDriver methodology fail to generalize across equities?"),
        "universe": {"membership_date": MEMBERSHIP_DATE, "djia_members": list(DJIA_MEMBERS_2016),
                     "excluded": EXCLUDED, "continuations": CONTINUATIONS, "symbols": list(UNIVERSE),
                     "size": len(UNIVERSE), "source": MEMBERSHIP_SOURCE,
                     "inclusion_rule": ("a security may fail inclusion only through an explicitly frozen "
                                        "data-integrity, continuity, availability or minimum-development-row rule; "
                                        "never because of predictive performance or low news coverage")},
        "benchmarks": {"target_benchmark": "SPY", "market_context": ["SPY", "QQQ"]},
        "intervals": {"research_frame": ["2017-02-01", "2026-09-28"], "development": ["2017-02-01", "2024-09-30"],
                      "confirmation": ["2024-10-01", "2026-09-25"],
                      "confirmation_note": ("confirmation holdout, not pristine for AAPL; evaluated only for "
                                            "development qualifiers with explicit authorization; otherwise never read")},
        "sample_rule": {"min_labelled_development_rows": MIN_DEVELOPMENT_ROWS,
                        "evaluated": "after all frozen data-integrity / corporate-action exclusions",
                        "missing_sessions": "never filled; rows needing a missing session are excluded and counted",
                        "news_coverage": "not an exclusion criterion"},
        "corporate_actions": {"rule": CORPORATE_ACTION_RULE, "affected_row_semantics": AFFECTED_ROW_SEMANTICS,
                              "max_feature_lookback_sessions": MAX_FEATURE_LOOKBACK,
                              "events_to_verify": CORPORATE_ACTION_EVENTS},
        "target": {"definition": {str(h): target_definition(h) for h in horizons},
                   "note": "Phase 10A excess return vs SPY, with each stock's own adjusted close"},
        "horizons": horizons,
        "feature_sets": FEATURE_SETS,
        "feature_notes": {"renamed": COLUMN_RENAMES, "rename_basis": "same formula with the stock's own close (AAPL values unchanged)",
                          "events": "event_features_v1 / rules_v1 unchanged; its product rule contains Apple-specific "
                                    "keywords, so product events are under-detected for other companies (recorded limitation)",
                          "f_duplicate_resolution": p10a.DUPLICATE_RESOLUTION},
        "models": [{"name": c.name, "task": c.task, "is_baseline": c.is_baseline, "description": c.description}
                   for c in cands],
        "model_settings": "Phase 9 fixed parameters, seed 42 where supported, no tuning, no feature selection",
        "staging": STAGES,
        "evaluation": {
            "harness_version": HARNESS_VERSION, "gate_version": GATE_VERSION, "significance_level": alpha,
            "per_stock": ("secondary: the existing per-symbol harness walk-forward for each stock (TimeSeriesSplit(20), "
                          "gap = h, pooled OOS, Phase 10A excess targets, existing baselines and gate_v1)"),
            "panel": {
                "role": "primary",
                "rows": "all eligible stocks pooled; one row per (stock, session)",
                "folds": ("TimeSeriesSplit(20) over the unique development trading DATES with gap = h dates; every "
                          "stock's rows on a date belong to the same fold"),
                "model": "one model per walk-forward fold trained on the pooled training observations",
                "ticker_identity_feature": False,
                "baselines": {"regression": ["Zero Return", "Mean Return (pooled training-fold mean)",
                                             "Per-Stock Mean Return (training-fold mean of each stock)"],
                              "classification": ["Always UP", "Base Rate (pooled training-fold share)",
                                                 "Per-Stock Base Rate (training-fold share of each stock)"]},
                "reference": ("regression: the baseline with the lowest pooled OOS MSE; classification: the lower "
                              "pooled OOS log loss of the two base rates; Always UP is the accuracy floor"),
                "dm": ("per-date cross-sectional mean loss: for every OOS date the model and the reference losses "
                       "are averaged over the stocks present on that date; the existing one-sided diebold_mariano "
                       "is applied to the per-date series with Newey-West (h - 1) lags"),
                "gate": ("gate_v1 decision logic unchanged on the pooled OOS population: regression MSE < reference "
                         "AND DM p < 0.05; classification log loss < reference AND DM p < 0.05 AND accuracy >= Always UP"),
            },
            "aapl_excluded_sensitivity": "the panel without AAPL; 198 comparisons; descriptive only, never qualifies",
        },
        "multiple_comparisons": {
            "primary_panel_family": {"size": len(panel), "method": "Holm", "alpha": 0.05},
            "secondary_per_stock_family": {"size": len(per_stock), "method": "Holm", "alpha": 0.05},
            "aapl_excluded_sensitivity": {"size": len(sens), "role": "descriptive only; no Holm, no qualification"},
            "rule": "families fixed now; never merged or re-defined after results",
        },
        "qualification": {
            "panel": ("gate_v1 QUALIFIED on the pooled OOS population AND raw DM p < 0.05 AND Holm-adjusted p < 0.05 "
                      "within the 198-comparison panel family -> development qualifier"),
            "per_stock": ("gate_v1 QUALIFIED AND raw p < 0.05 AND Holm-adjusted p < 0.05 within the per-stock family "
                          "-> development qualifier"),
            "confirmation": ("only development qualifiers, only with explicit authorization, each exactly once; "
                             "with no qualifier the confirmation holdout is NOT evaluated (NOT_APPLICABLE)"),
            "generalization_claim": "requires a confirmed primary-panel qualifier",
        },
        "aapl_reproducibility": ("AAPL per-stock A/E results must be checked for equivalence against Phase 10A where "
                                 "data and design are mathematically identical; a reproducibility check only"),
        "counts": {"universe": len(UNIVERSE), "feature_sets": len(fs_list), "horizons": len(horizons),
                   "models": len(models), "per_stock": len(per_stock), "panel": len(panel),
                   "aapl_excluded_sensitivity": len(sens)},
        "comparison_ids": {"per_stock": per_stock, "panel": panel, "aapl_excluded_sensitivity": sens},
        "deferred_to_phase_11_1": ["price downloads (one retrieval batch) and session/coverage checks",
                                   "verification of every corporate-action event (dates, types, provider adjustment)",
                                   "affected-row counts and minimum-row eligibility per stock",
                                   "any further material actions in the provider's action records"],
    }


def design_sha256(d: dict) -> str:
    return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def freeze(path: Path = REGISTRY_PATH) -> dict:
    path = Path(path)
    if path.exists():
        raise Phase11DesignError(f"{path} already exists; the Phase 11 design is frozen once")
    d = design()
    reg = {"registry_version": REGISTRY_VERSION, "phase": PHASE, "status": "DESIGN_FROZEN", "design": d,
           "design_sha256": design_sha256(d), "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "frozen_git": git_info(), "frozen_environment": environment(),
           "data_acquired": False, "experiments_run": False}
    _write_json_atomic(path, reg)
    return reg


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if reg.get("registry_version") != REGISTRY_VERSION or design_sha256(reg["design"]) != reg["design_sha256"]:
        raise Phase11DesignError("Phase 11 design registry invalid or edited after freezing")
    return reg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11.0 design freeze")
    parser.add_argument("command", choices=["freeze", "status"])
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args(argv)
    reg = freeze(args.registry) if args.command == "freeze" else load_registry(args.registry)
    c = reg["design"]["counts"]
    print(f"Registry : {reg['status']}  design sha256 {reg['design_sha256']}")
    print(f"Universe : {c['universe']} securities ({', '.join(reg['design']['universe']['symbols'])})")
    print(f"Matrix   : per-stock {c['per_stock']} | panel {c['panel']} | AAPL-excluded sensitivity "
          f"{c['aapl_excluded_sensitivity']} (descriptive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
