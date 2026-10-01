"""
Offline Phase 7 fixtures: real-format inputs for the feature store, built with the
actual Phase 5B / Phase 6 generators and Phase 5C / 6 serializers.

Sessions: 2024-06-03 .. 2024-08-30 (Juneteenth, July 4 excluded)
Technical spine: 2024-06-17 .. 2024-08-29   News coverage: July 2024
"""

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from features.feature_engineering import TECHNICAL_FEATURES
from services.event_features import generate_event_features
from services.historical_events import serialize_daily_events
from services.historical_sentiment import serialize_daily_rows
from services.market_calendar_service import TradingCalendar
from services.sentiment_features import generate_sentiment_features
from tests.event_fakes import NEG, NEU, POS, classified
from tests.sentiment_fakes import bars_for
from training.build_dataset import save_raw_snapshot

NY = ZoneInfo("America/New_York")
HOLIDAYS = {date(2024, 6, 19), date(2024, 7, 4)}
SESSIONS = [d for d in pd.bdate_range("2024-06-03", "2024-08-30").date if d not in HOLIDAYS]
TECH_DATES = [d for d in SESSIONS if date(2024, 6, 17) <= d <= date(2024, 8, 29)]
NEWS_DATES = [d for d in SESSIONS if date(2024, 7, 1) <= d <= date(2024, 7, 31)]
CALENDAR = TradingCalendar(SESSIONS)

EARN, LEGAL, ANALYST = "Apple Reports Q1 EPS beat", "Apple Says It Is Suing Qualcomm", "Wells Fargo Cuts Apple Estimates"
NOISE = "Inside The Hidden Economy Of Pawn Shops"


def ny(*args):
    return datetime(*args, tzinfo=NY)


def default_items():
    """No article before 07-02 (07-01 has zero news); no event on 07-09."""
    return [classified(1, ny(2024, 7, 2, 10, 0), EARN, POS), classified(2, ny(2024, 7, 2, 11, 0), LEGAL, NEG),
            classified(3, ny(2024, 7, 3, 17, 0), ANALYST), classified(4, ny(2024, 7, 8, 12, 0), NOISE),
            classified(5, ny(2024, 7, 10, 9, 0), EARN + " again", POS), classified(6, ny(2024, 7, 15, 14, 0), LEGAL + " #2")]


def _write_with_meta(path: Path, payload: bytes, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    rows = payload.decode("utf-8").count("\n") - (1 if path.suffix == ".csv" else 0)
    meta = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": rows, **extra}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return path


def write_technical(directory: Path, dates=TECH_DATES, seed=0, symbol="AAPL") -> Path:
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates) + 1))       # one extra close for the last Target
    df = pd.DataFrame({"Date": [d.isoformat() for d in dates], "Close": closes[:-1]})
    for f in TECHNICAL_FEATURES:
        df[f] = rng.normal(0, 1, len(dates))
    df["Target"] = closes[1:] / closes[:-1] - 1
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _write_with_meta(Path(directory) / "final_stock_dataset.csv", payload,
                            feature_version="technical_v2", raw_snapshot={"ticker": symbol})


def write_daily(directory: Path, items, dates=NEWS_DATES, symbols=("AAPL",)) -> tuple[Path, Path]:
    sentiment_rows = generate_sentiment_features([c.scored for c in items], dates, CALENDAR, symbols)
    event_rows = generate_event_features(items, dates, CALENDAR, symbols)
    s = _write_with_meta(Path(directory) / "news.finbert-test.daily.jsonl", serialize_daily_rows(sentiment_rows))
    e = _write_with_meta(Path(directory) / "news.finbert-test.events-rules_v1.daily.jsonl",
                         serialize_daily_events(event_rows))
    return s, e


def write_snapshot(directory: Path) -> Path:
    return save_raw_snapshot(bars_for(SESSIONS), "AAPL", "1y", datetime(2024, 9, 30, 22, tzinfo=timezone.utc),
                             raw_dir=Path(directory))


def write_inputs(directory: Path, items=None) -> dict:
    directory = Path(directory)
    s, e = write_daily(directory / "daily", items if items is not None else default_items())
    return {"technical": write_technical(directory / "tech"), "sentiment": s, "events": e,
            "snapshot": write_snapshot(directory / "raw")}


def rewrite_jsonl(path: Path, mutate) -> None:
    """Edit JSONL rows and re-hash consistently (simulates a well-formed but wrong input)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    _write_with_meta(path, payload)
