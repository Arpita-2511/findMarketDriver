"""
Phase 11.1 - cross-stock price data and corporate-action integrity verification.

DATA + INTEGRITY VERIFICATION ONLY: no model, no prediction, no news, no
FinBERT. Implements the frozen Phase 11.0 design (training/phase11_design.py,
design sha256 266daaddba0e8cb18163b4ec9db7eadabe2d5ab574bf410e043a7d28d3247650)
and never changes it.

    python -m training.phase11_prices run      # download batch (twice), verify, write artifacts + report
    python -m training.phase11_prices status

Download (one retrieval batch): the 29 frozen securities + SPY + QQQ from
yfinance (the existing provider), daily bars 2016-01-01 .. 2026-09-29
(end exclusive 2026-09-30), two calls per symbol:
    history(auto_adjust=True,  actions=True)   -> research prices (Phase 10 convention) + actions
    history(auto_adjust=False, actions=True)   -> Yahoo "Close" (split-adjusted only) and "Adj Close"
The whole batch is downloaded twice to test provider determinism.

Normalized artifact (what is hashed): CSV, UTF-8, "\\n", columns
    Date, Open, High, Low, Close, Volume, Dividends, Stock_Splits, Close_unadjusted, Adj_Close
sorted by Date (America/New_York exchange date, ISO), pandas default float repr,
no row added, removed or modified (duplicates are kept and reported, never repaired).

Adjustment determination rule (fixed in code BEFORE any download; see
ADJUSTMENT_RULE): provider action record within +-3 sessions AND no abnormal
adjusted move (|z| <= 4, z = excess return over SPY / sample std of the 60
prior sessions' excess returns) -> PROPERLY_ADJUSTED; anything else for a
distribution or split -> NOT_PROPERLY_ADJUSTED -> the frozen affected-row
exclusion D in [E-h, E+49] (session indices) applies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training.phase11_design as p11d  # noqa: E402
from features.feature_engineering import MIN_HISTORY_ROWS  # noqa: E402
from training.evaluation_harness import git_info  # noqa: E402
from training.phase9_research import ResearchError, _write_json_atomic  # noqa: E402

FROZEN_DESIGN_SHA = "266daaddba0e8cb18163b4ec9db7eadabe2d5ab574bf410e043a7d28d3247650"
UNIVERSE = tuple(p11d.UNIVERSE)
BENCHMARKS = ("SPY", "QQQ")
REFERENCE_CALENDAR = "SPY"                                    # trades every NYSE session
DOWNLOAD = {"provider": "yfinance (Yahoo Finance)", "start": "2016-01-01", "end_exclusive": "2026-09-30",
            "interval": "1d", "calls": ["Ticker.history(auto_adjust=True, actions=True)",
                                        "Ticker.history(auto_adjust=False, actions=True)"],
            "passes": 2, "repair": False}
FRAME_END = date(2026, 9, 28)
DEV = (date(2017, 2, 1), date(2024, 9, 30))
CONF = (date(2024, 10, 1), date(2026, 9, 25))
HORIZONS = (1, 3, 5)
LOOKBACK = p11d.MAX_FEATURE_LOOKBACK                          # 50
REQUIRED_PRIOR_SESSIONS = max(MIN_HISTORY_ROWS - 1, LOOKBACK)  # warm-up before a row can have every feature
MIN_DEV_ROWS = p11d.MIN_DEVELOPMENT_ROWS                       # 1,800
MARKET_DIR = PROJECT_ROOT / "data" / "processed" / "phase11" / "market"
REPORT_PATH = PROJECT_ROOT / "data" / "results" / "research" / "phase11" / "price_verification.json"
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock_Splits", "Close_unadjusted", "Adj_Close"]
COLUMN_NOTES = {
    "Open/High/Low/Close": "yfinance auto_adjust=True (split- and dividend-adjusted as of retrieval) - research prices",
    "Volume": "yfinance auto_adjust=True",
    "Dividends": "provider-recorded cash distribution per share on that session (0 if none)",
    "Stock_Splits": "provider-recorded split factor on that session (0 if none; <1 = reverse split)",
    "Close_unadjusted": "yfinance auto_adjust=False 'Close' (Yahoo convention: split-adjusted, NOT dividend-adjusted)",
    "Adj_Close": "yfinance auto_adjust=False 'Adj Close'",
}

ADJUSTMENT_RULE = {
    "event_session": "E = first session of the stock on or after the documented public date",
    "provider_record": ("a Stock_Splits value not in {0, 1}, or a Dividends value > 3 x the median of the stock's "
                        "positive dividends, within sessions [E-3, E+3]"),
    "abnormal_move": ("z(s) = (r_adj(s) - r_SPY(s)) / sample std (ddof=1) of the stock's excess returns over the 60 "
                      "sessions before s; abnormal iff |z| > 4; r = Close[s]/Close[s-1] - 1 on adjusted closes"),
    "spin_off / distribution": "PROPERLY_ADJUSTED iff a provider record exists AND |z| <= 4 at E and at every record session",
    "split / reverse split": "PROPERLY_ADJUSTED iff a split record exists in the window AND |z| <= 4 at E and the record session",
    "merger_continuity": "CONTINUOUS iff every reference session in [E-60, E+60] is present (no adjustment needed for the survivor)",
    "exchange_offer (non-pro-rata)": "NO_ADJUSTMENT_REQUIRED (no distribution to all holders); z reported if a date is known",
    "otherwise": ("NOT_PROPERLY_ADJUSTED -> frozen exclusion D in [E-h, E+49] (session indices) at E and at every "
                  "record session; prices never repaired, no splicing, no substitution"),
    "additional_provider_actions": "every other provider split record is tested with the split rule",
}

# Documented events (public evidence; no invented ratios). Wikipedia pages accessed 2026-10-01/02.
EVENTS = [
    {"id": "RTX_merger_2020", "symbol": "RTX", "member": "UTX", "type": "merger_continuity", "public_date": "2020-04-03",
     "evidence": "Wikipedia 'RTX Corporation': merger completed April 3, 2020; UTC the legal survivor, ticker RTX"},
    {"id": "RTX_carrier_otis_2020", "symbol": "RTX", "member": "UTX", "type": "spin_off", "public_date": "2020-04-03",
     "evidence": ("Wikipedia 'Carrier Global': separation completed April 2020; 'Otis Worldwide': Otis common stock "
                  "began trading on the NYSE April 3, 2020")},
    {"id": "GE_wabtec_2019", "symbol": "GE", "type": "spin_off", "public_date": "2019-02-25",
     "evidence": ("Wikipedia 'Wabtec': GE Transportation merger completed February 25, 2019; GE shareholders own "
                  "24.3% of the merged company (distribution to GE shareholders)")},
    {"id": "GE_reverse_split_2021", "symbol": "GE", "type": "reverse_split", "public_date": "2021-07-30",
     "evidence": "Wikipedia 'General Electric': reverse stock split 1-for-8 (announced / completed July 30, 2021)"},
    {"id": "GE_healthcare_2023", "symbol": "GE", "type": "spin_off", "public_date": "2023-01-04",
     "evidence": "Wikipedia 'GE HealthCare': spin-off completed and trading began January 4, 2023"},
    {"id": "GE_vernova_2024", "symbol": "GE", "type": "spin_off", "public_date": "2024-04-02",
     "evidence": "Wikipedia 'GE Vernova' / 'General Electric': spin-off, GE Vernova listed April 2, 2024"},
    {"id": "MMM_solventum_2024", "symbol": "MMM", "type": "spin_off", "public_date": "2024-04-01",
     "evidence": "Wikipedia 'Solventum': spin-off completed April 1, 2024; 3M retained about 19.9%"},
    {"id": "MRK_organon_2021", "symbol": "MRK", "type": "spin_off", "public_date": "2021-06-03",
     "evidence": "Wikipedia 'Organon & Co.': spinoff completed, Organon publicly traded June 3, 2021"},
    {"id": "IBM_kyndryl_2021", "symbol": "IBM", "type": "spin_off", "public_date": "2021-11-04",
     "evidence": "Wikipedia 'Kyndryl': separation completed and NYSE trading began November 4, 2021"},
    {"id": "PFE_upjohn_viatris_2020", "symbol": "PFE", "type": "spin_off", "public_date": "2020-11-16",
     "evidence": ("Wikipedia 'Viatris': on November 16, 2020 Upjohn merged with Mylan in a Reverse Morris Trust "
                  "transaction (RMT structure implies a distribution of Upjohn to Pfizer shareholders - inference)")},
    {"id": "JNJ_kenvue_2023", "symbol": "JNJ", "type": "exchange_offer", "public_date": None,
     "evidence": ("Wikipedia 'Kenvue': IPO May 4, 2023; exchange offer launched July 24, 2023 letting J&J holders "
                  "exchange all, some or none of their shares (non-pro-rata split-off); completion date not stated")},
]


class Phase11PriceError(ResearchError):
    """Phase 11.1 data / integrity protocol violation."""


# ==========================================
# Download + normalization
# ==========================================


def fetch(symbol: str, retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf  # network dependency

    last = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(symbol)
            adj = t.history(start=DOWNLOAD["start"], end=DOWNLOAD["end_exclusive"], interval="1d",
                            auto_adjust=True, actions=True)
            raw = t.history(start=DOWNLOAD["start"], end=DOWNLOAD["end_exclusive"], interval="1d",
                            auto_adjust=False, actions=True)
            if adj.empty or raw.empty:
                raise Phase11PriceError(f"{symbol}: empty provider response")
            return adj, raw
        except Exception as e:  # recorded and retried
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise Phase11PriceError(f"{symbol}: download failed after {retries} attempts: {last}")


def _exchange_dates(index) -> list[str]:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return [d.strftime("%Y-%m-%d") for d in idx.normalize()]


def normalize(adj: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Deterministic representation; nothing filled, dropped or repaired."""
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in adj.columns:
            raise Phase11PriceError(f"adjusted response lacks '{col}'")
    if "Close" not in raw.columns or "Adj Close" not in raw.columns:
        raise Phase11PriceError("unadjusted response lacks 'Close' / 'Adj Close'")
    a = pd.DataFrame({"Date": _exchange_dates(adj.index)})
    for col in ("Open", "High", "Low", "Close", "Volume"):
        a[col] = adj[col].to_numpy(dtype="float64")
    a["Dividends"] = adj["Dividends"].to_numpy(dtype="float64") if "Dividends" in adj.columns else 0.0
    a["Stock_Splits"] = adj["Stock Splits"].to_numpy(dtype="float64") if "Stock Splits" in adj.columns else 0.0
    r = pd.DataFrame({"Date": _exchange_dates(raw.index), "Close_unadjusted": raw["Close"].to_numpy(dtype="float64"),
                      "Adj_Close": raw["Adj Close"].to_numpy(dtype="float64")})
    if len(r) != len(a) or list(r["Date"]) != list(a["Date"]):
        # keep every row from both calls; a mismatch is a finding, never silently aligned away
        out = a.merge(r, on="Date", how="outer", sort=False)
    else:
        out = a.copy()
        out["Close_unadjusted"] = r["Close_unadjusted"].to_numpy()
        out["Adj_Close"] = r["Adj_Close"].to_numpy()
    out = out.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return out[COLUMNS]


def serialize(df: pd.DataFrame) -> bytes:
    return df[COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ==========================================
# Validation
# ==========================================


def validate(df: pd.DataFrame, reference_sessions: list[str]) -> dict:
    dates = list(df["Date"])
    num = df[COLUMNS[1:]].to_numpy(dtype="float64")
    prices = df[["Open", "High", "Low", "Close"]].to_numpy(dtype="float64")
    first, last = (dates[0], dates[-1]) if dates else (None, None)
    ref_in = [s for s in reference_sessions if first <= s <= min(last, FRAME_END.isoformat())] if dates else []
    have = set(dates)
    missing = [s for s in ref_in if s not in have]
    extra = sorted(s for s in have if s not in set(reference_sessions))
    hl = df[["Open", "High", "Low", "Close"]]
    return {
        "rows": len(df), "first_date": first, "last_date": last,
        "chronological": dates == sorted(dates), "duplicate_dates": int(len(dates) - len(have)),
        "duplicate_ohlc_rows": int(df.duplicated(subset=["Open", "High", "Low", "Close", "Volume"]).sum()),
        "nan_values": int(np.isnan(num).sum()), "inf_values": int(np.isinf(num).sum()),
        "non_positive_prices": int((prices <= 0).sum()), "zero_volume_rows": int((df["Volume"] <= 0).sum()),
        "high_low_inconsistent_rows": int(((hl["High"] < hl[["Open", "Close"]].max(axis=1) - 1e-9) |
                                           (hl["Low"] > hl[["Open", "Close"]].min(axis=1) + 1e-9)).sum()),
        "adjusted_close_available": bool(df["Close"].notna().all()),
        "auto_adjust_close_vs_adj_close_max_rel_diff": float(np.nanmax(np.abs(df["Close"] / df["Adj_Close"] - 1)))
        if len(df) else None,
        "reference_sessions_in_range": len(ref_in), "missing_sessions": missing, "n_missing_sessions": len(missing),
        "extra_sessions_not_in_reference": extra,
    }


# ==========================================
# Corporate actions
# ==========================================


def _returns(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(df["Close"].to_numpy(dtype="float64"), index=list(df["Date"]))
    return s / s.shift(1) - 1


def _z(r_stock: pd.Series, r_spy: pd.Series, session: str) -> float | None:
    dates = list(r_stock.index)
    if session not in r_stock.index:
        return None
    i = dates.index(session)
    if i < 61:
        return None
    window = dates[i - 60:i]
    ex = np.array([r_stock[d] - r_spy.get(d, np.nan) for d in window], dtype="float64")
    ex = ex[np.isfinite(ex)]
    if len(ex) < 30:
        return None
    sd = float(np.std(ex, ddof=1))
    v = r_stock[session] - r_spy.get(session, np.nan)
    return float(v / sd) if sd > 0 and np.isfinite(v) else None


def provider_records(df: pd.DataFrame) -> list[dict]:
    """Every provider split record and every 'large' dividend (> 3x median positive dividend)."""
    out = []
    pos = df.loc[df["Dividends"] > 0, "Dividends"]
    med = float(pos.median()) if len(pos) else None
    for _, row in df.iterrows():
        sp = row["Stock_Splits"]
        if np.isfinite(sp) and sp not in (0.0, 1.0):
            out.append({"date": row["Date"], "kind": "split", "value": float(sp)})
        if med is not None and row["Dividends"] > 3 * med:
            out.append({"date": row["Date"], "kind": "large_dividend", "value": float(row["Dividends"]),
                        "median_positive_dividend": med})
    return out


def check_event(df: pd.DataFrame, spy: pd.DataFrame, event: dict) -> dict:
    dates = list(df["Date"])
    res = {**event, "E": None, "provider_records_in_window": [], "z": {}, "returns": {}}
    if event["type"] == "exchange_offer" or event.get("public_date") is None:
        res["determination"] = "NO_ADJUSTMENT_REQUIRED" if event["type"] == "exchange_offer" else "NOT_VERIFIABLE"
        res["reason"] = "non-pro-rata exchange offer: no distribution to all holders, no price adjustment expected"
        res["exclusion_sessions"] = []
        return res
    after = [d for d in dates if d >= event["public_date"]]
    if not after:
        res["determination"] = "NOT_VERIFIABLE"
        res["reason"] = "no session on/after the public date"
        res["exclusion_sessions"] = []
        return res
    E = after[0]
    i = dates.index(E)
    res["E"] = E
    window = set(dates[max(0, i - 3): i + 4])
    recs = [r for r in provider_records(df) if r["date"] in window]
    res["provider_records_in_window"] = recs
    r_s, r_spy = _returns(df), _returns(spy)
    sessions = sorted({E, *[r["date"] for r in recs]})
    for s in sessions:
        res["z"][s] = _z(r_s, r_spy, s)
        j = dates.index(s)
        res["returns"][s] = {"adjusted": float(r_s[s]) if np.isfinite(r_s[s]) else None,
                             "unadjusted_close": float(df["Close_unadjusted"].iloc[j] / df["Close_unadjusted"].iloc[j - 1] - 1)
                             if j > 0 else None,
                             "spy": float(r_spy.get(s, np.nan)) if np.isfinite(r_spy.get(s, np.nan)) else None}
    zs_ok = all(z is not None and abs(z) <= 4 for z in res["z"].values())
    t = event["type"]
    if t == "merger_continuity":
        ref = list(spy["Date"])
        k = ref.index(E) if E in ref else None
        win = ref[max(0, k - 60): k + 61] if k is not None else []
        missing = [s for s in win if s not in set(dates)]
        res["determination"] = "CONTINUOUS" if win and not missing else "NOT_CONTINUOUS"
        res["reason"] = f"{len(win)} reference sessions around E, {len(missing)} missing"
        res["exclusion_sessions"] = [] if not missing else [E]
        return res
    if t == "spin_off":
        has = bool(recs)
    elif t in ("reverse_split", "split"):
        has = any(r["kind"] == "split" for r in recs)
    else:
        raise Phase11PriceError(f"unknown event type {t}")
    if has and zs_ok:
        res["determination"] = "PROPERLY_ADJUSTED"
        res["reason"] = "provider record in window and no abnormal adjusted move"
        res["exclusion_sessions"] = []
    else:
        res["determination"] = "NOT_PROPERLY_ADJUSTED"
        res["reason"] = ("no provider record in window" if not has else "abnormal adjusted move (|z| > 4) at E or record")
        res["exclusion_sessions"] = sessions
    return res


def additional_actions(df: pd.DataFrame, spy: pd.DataFrame, symbol: str, linked_dates: set[str]) -> list[dict]:
    out = []
    for r in provider_records(df):
        if r["date"] in linked_dates or r["kind"] != "split" or not (DOWNLOAD["start"] <= r["date"] <= FRAME_END.isoformat()):
            continue
        ev = {"id": f"{symbol}_provider_split_{r['date']}", "symbol": symbol, "type": "split", "public_date": r["date"],
              "evidence": f"provider Stock_Splits record {r['value']} on {r['date']} (not linked to a documented event)"}
        out.append(check_event(df, spy, ev))
    return out


def extreme_moves(df: pd.DataFrame, spy: pd.DataFrame, threshold_abs: float = 0.10, threshold_z: float = 6.0) -> list[dict]:
    r_s, r_spy = _returns(df), _returns(spy)
    out = []
    for s in r_s.index:
        if not (DEV[0].isoformat() <= s <= FRAME_END.isoformat()):
            continue
        ex = r_s[s] - r_spy.get(s, np.nan)
        if np.isfinite(ex) and abs(ex) > threshold_abs:
            z = _z(r_s, r_spy, s)
            if z is not None and abs(z) > threshold_z:
                out.append({"date": s, "adjusted_return": float(r_s[s]), "excess_return": float(ex), "z": z})
    return out


# ==========================================
# Exclusion accounting and eligibility
# ==========================================


def eligibility(symbol_dates: list[str], reference: list[str], missing: list[str], exclusion_sessions: list[str]) -> dict:
    """
    Rows are reference-calendar sessions. Per horizon h:
      dev labelled rows = dev sessions minus the last h (no close after the development end)
      warm-up loss      = rows with < REQUIRED_PRIOR_SESSIONS own sessions before them
      missing-data loss = rows absent from the stock + rows within [m-h, m+49] of a missing reference session m
      corporate-action  = rows within [E-h, E+49] of a NOT_PROPERLY_ADJUSTED event session (stock's own index)
    Losses are counted in that order without double counting.
    """
    ref = [s for s in reference if s <= FRAME_END.isoformat()]
    ref_idx = {s: k for k, s in enumerate(ref)}
    own = [s for s in symbol_dates if s <= FRAME_END.isoformat()]
    own_idx = {s: k for k, s in enumerate(own)}
    out = {}
    for h in HORIZONS:
        res = {}
        for name, (lo, hi) in (("development", DEV), ("confirmation", CONF)):
            rows = [s for s in ref if lo.isoformat() <= s <= hi.isoformat()]
            if name == "development":
                labelled = rows[:-h]
            else:
                labelled = [s for s in rows if ref_idx[s] + h < len(ref)]
            lost = {"warm_up": set(), "missing_data": set(), "corporate_action": set()}
            for s in labelled:
                if s in own_idx and own_idx[s] < REQUIRED_PRIOR_SESSIONS:
                    lost["warm_up"].add(s)
            miss_rows = {s for s in labelled if s not in own_idx}
            for m in missing:
                k = ref_idx.get(m)
                if k is None:
                    continue
                miss_rows |= {s for s in labelled if k - h <= ref_idx[s] <= k + LOOKBACK - 1}
            lost["missing_data"] = miss_rows - lost["warm_up"]
            ca = set()
            for e in exclusion_sessions:
                if e not in own_idx:
                    continue
                mask = p11d.affected_row_mask(len(own), own_idx[e], h)
                ca |= {own[j] for j in np.flatnonzero(mask)}
            lost["corporate_action"] = (ca & set(labelled)) - lost["warm_up"] - lost["missing_data"]
            final = len(labelled) - sum(len(v) for v in lost.values())
            res[name] = {"potential_labelled_rows": len(labelled), **{f"lost_{k}": len(v) for k, v in lost.items()},
                         "final_rows": final}
        res["development"]["minimum"] = MIN_DEV_ROWS
        res["development"]["eligible"] = res["development"]["final_rows"] >= MIN_DEV_ROWS
        out[str(h)] = res
    out["eligible_all_horizons"] = all(out[str(h)]["development"]["eligible"] for h in HORIZONS)
    return out


# ==========================================
# Artifacts and run
# ==========================================


def write_artifact(symbol: str, df: pd.DataFrame, meta_extra: dict, directory: Path = MARKET_DIR) -> dict:
    payload = serialize(df)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}_1d_{df['Date'].iloc[0].replace('-', '')}_{df['Date'].iloc[-1].replace('-', '')}.csv"
    path = directory / name
    if path.exists() and path.read_bytes() != payload:
        raise Phase11PriceError(f"{name} already exists with different content; refusing to overwrite")
    path.write_bytes(payload)
    meta = {"symbol": symbol, "file": name, "sha256": sha256_bytes(payload), "rows": len(df),
            "first_date": df["Date"].iloc[0], "last_date": df["Date"].iloc[-1], "columns": COLUMNS,
            "column_notes": COLUMN_NOTES, "hashed": "the CSV bytes exactly as written (normalized representation)",
            **meta_extra}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path),
            "sha256": meta["sha256"], "rows": len(df)}


def verify(frames: dict[str, pd.DataFrame]) -> dict:
    spy = frames[REFERENCE_CALENDAR]
    reference = [s for s in spy["Date"] if s <= FRAME_END.isoformat()]
    per_stock, events = {}, []
    for sym in UNIVERSE:
        df = frames[sym]
        v = validate(df, reference)
        ev = [check_event(df, spy, e) for e in EVENTS if e["symbol"] == sym]
        linked = {r["date"] for e in ev for r in e["provider_records_in_window"]}
        ev += additional_actions(df, spy, sym, linked)
        excl = sorted({s for e in ev if e["determination"] in ("NOT_PROPERLY_ADJUSTED", "NOT_CONTINUOUS")
                       for s in e["exclusion_sessions"]})
        elig = eligibility(list(df["Date"]), reference, v["missing_sessions"], excl)
        per_stock[sym] = {"validation": v, "provider_records": provider_records(df), "events": ev,
                          "exclusion_event_sessions": excl, "extreme_moves": extreme_moves(df, spy),
                          "eligibility": elig}
        events += ev
    return {"per_stock": per_stock, "events": events,
            "benchmarks": {b: validate(frames[b], reference) for b in BENCHMARKS},
            "reference_calendar": {"symbol": REFERENCE_CALENDAR, "sessions_to_frame_end": len(reference),
                                   "first": reference[0], "last": reference[-1]}}


def run(report_path: Path = REPORT_PATH, market_dir: Path = MARKET_DIR, fetcher=fetch, log=print) -> dict:
    report_path = Path(report_path)
    if report_path.exists():
        raise Phase11PriceError(f"{report_path} already exists; Phase 11.1 verification is written once")
    design_reg = p11d.load_registry()
    if design_reg["design_sha256"] != FROZEN_DESIGN_SHA or tuple(design_reg["design"]["universe"]["symbols"]) != UNIVERSE:
        raise Phase11PriceError("frozen Phase 11 design mismatch")
    symbols = list(UNIVERSE) + list(BENCHMARKS)
    passes, retrieved = [], []
    for p in range(DOWNLOAD["passes"]):
        retrieved.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
        frames = {}
        for sym in symbols:
            log(f"[pass {p + 1}] {sym}")
            frames[sym] = normalize(*fetcher(sym))
        passes.append(frames)
    frames = passes[0]
    det = {s: {"pass1_sha256": sha256_bytes(serialize(passes[0][s])), "pass2_sha256": sha256_bytes(serialize(passes[1][s]))}
           for s in symbols}
    for s in det.values():
        s["identical"] = s["pass1_sha256"] == s["pass2_sha256"]
    norm_twice = all(serialize(frames[s]) == serialize(normalize_roundtrip(frames[s])) for s in symbols)
    obs = verify(frames)
    meta_extra = {"provider": DOWNLOAD["provider"], "yfinance_version": version("yfinance"),
                  "download": DOWNLOAD, "retrieved_at_utc": retrieved[0], "frozen_design_sha256": FROZEN_DESIGN_SHA}
    artifacts = {s: write_artifact(s, frames[s], meta_extra, market_dir) for s in symbols}
    fails = [s for s, v in obs["per_stock"].items() if not v["eligibility"]["eligible_all_horizons"]]
    report = {
        "phase": "11.1", "report_version": "phase11_price_verification_v1",
        "FROZEN_DESIGN": {"design_sha256": FROZEN_DESIGN_SHA, "commit": "38dfd1f", "universe": list(UNIVERSE),
                          "excluded": p11d.EXCLUDED, "continuations": p11d.CONTINUATIONS,
                          "affected_row_semantics": p11d.AFFECTED_ROW_SEMANTICS, "min_development_rows": MIN_DEV_ROWS,
                          "intervals": design_reg["design"]["intervals"]},
        "PHASE_11_1_IMPLEMENTATION": {
            "download": DOWNLOAD, "yfinance_version": version("yfinance"), "retrieved_at_utc": retrieved,
            "frame_end": FRAME_END.isoformat(), "warm_up": {"download_start": DOWNLOAD["start"],
                                                            "required_prior_sessions": REQUIRED_PRIOR_SESSIONS,
                                                            "basis": f"max(MIN_HISTORY_ROWS - 1 = {MIN_HISTORY_ROWS - 1}, lookback {LOOKBACK})"},
            "adjustment_rule": ADJUSTMENT_RULE, "documented_events": EVENTS,
            "missing_session_rule": ("missing reference session m -> rows with reference index in [m-h, m+49] excluded "
                                     "(implementation of the frozen 'rows needing a missing session are excluded')"),
            "reference_calendar": REFERENCE_CALENDAR},
        "PHASE_11_1_OBSERVATIONS": {**obs, "determinism": {"per_symbol": det,
                                                           "all_identical_across_passes": all(d["identical"] for d in det.values()),
                                                           "normalization_idempotent": norm_twice},
                                    "stocks_failing_min_rows": fails},
        "artifacts": artifacts,
        "git": git_info(), "experiments_run": False, "models_trained": False, "news_downloaded": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json_atomic(report_path, report)
    return report


def normalize_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Re-read the serialized CSV (float round trip) - used to prove the representation is stable."""
    import io

    back = pd.read_csv(io.BytesIO(serialize(df)), float_precision="round_trip", dtype={"Date": str})
    return back[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 11.1 price data + corporate-action verification")
    parser.add_argument("command", choices=["run", "status"])
    args = parser.parse_args(argv)
    if args.command == "run":
        rep = run()
    else:
        rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    o = rep["PHASE_11_1_OBSERVATIONS"]
    print(f"stocks failing >= {MIN_DEV_ROWS} rows: {o['stocks_failing_min_rows']}")
    print(f"determinism identical across passes: {o['determinism']['all_identical_across_passes']}")
    for e in o["events"]:
        print(f"  {e['id']:32s} E={e['E']}  {e['determination']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
