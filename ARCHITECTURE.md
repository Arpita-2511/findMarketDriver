FindMarketDriver — Master Architecture & Implementation Plan

Version: 1.0
Date: 2026-09-30

1. PROJECT OVERVIEW

FindMarketDriver is an evidence-based stock analysis platform designed to answer two related questions:

What is the model's forecast for the next trading day?

Which observable market/news factors contributed most to that forecast?

The system should combine:

Historical stock market data

Technical indicators

Historical financial/news data

FinBERT sentiment

Event classification

Machine-learning classification

Probability calibration

SHAP model attribution

A backend API

A React dashboard

Scheduled live evaluation

IMPORTANT:
"Market driver" means model attribution, not proof of causal market movement.

The system must never present a model explanation as proof that a particular news article or event caused a stock movement.

2. CURRENT PROJECT STATE

The existing repository already contains:

Python data ingestion

yfinance-based stock fetching

Technical feature engineering

A next-day-return target

Chronological model training

Multiple model experiments

CLI prediction

Live prediction logging

CSV datasets

Evaluation/diagnostic scripts

The current production model is NOT considered a valid predictive model.

Current audit finding:

Lasso collapses to a constant prediction.

Technical-only features do not beat the mean/always-UP baselines.

Tree models overfit.

Historical news/NLP is missing.

Flask backend is missing.

React frontend is missing.

Database/feature store is missing.

SHAP is missing.

Deployment is missing.

Therefore:

DO NOT build the dashboard around the current constant model.

The first implementation gate is the evaluation harness.

3. TARGET END-TO-END ARCHITECTURE

                                USER
                                 |
                                 v
                    +-------------------------+
                    |     React Frontend      |
                    |-------------------------|
                    | Stock Search            |
                    | Price / Candlestick     |
                    | Prediction               |
                    | Probability             |
                    | Sentiment               |
                    | News Timeline            |
                    | Events                  |
                    | Model Drivers           |
                    | Historical Accuracy      |
                    +------------+------------+
                                 |
                              HTTPS/JSON
                                 |
                                 v
                    +-------------------------+
                    |      Flask Backend      |
                    |-------------------------|
                    | /api/stocks              |
                    | /api/news                |
                    | /api/predictions         |
                    | /api/drivers             |
                    | /api/evaluation          |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       Stock Service       News Service        ML Service
              |                  |                  |
              v                  v                  v
          yfinance          News Dataset       Model Registry
              |                  |                  |
              |                  v                  v
              |             FinBERT             Classifier
              |                  |                  |
              |             Event Model            |
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                       Feature Engineering
                                 |
                                 v
                   Combined Feature Store
                                 |
                                 v
                    Prediction + Calibration
                                 |
                                 v
                              SHAP
                                 |
                                 v
                         Driver Attribution
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
          PostgreSQL          Redis             Scheduler
              |                  |                  |
              |                  |                  v
              |                  |          Live Evaluation
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                         Docker / Deployment

4. REPOSITORY STRUCTURE

Target repository:

FindMarketDriver/
│
├── README.md
├── ARCHITECTURE.md
├── REQUIREMENTS.md
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── app.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── raw/
│   │   ├── stocks/
│   │   └── news/
│   │
│   ├── processed/
│   │   ├── technical/
│   │   ├── sentiment/
│   │   ├── events/
│   │   └── combined/
│   │
│   ├── datasets/
│   │   └── versioned/
│   │
│   └── results/
│       ├── evaluation/
│       ├── predictions/
│       └── reports/
│
├── features/
│   ├── __init__.py
│   ├── technical_features.py
│   ├── news_features.py
│   ├── sentiment_features.py
│   ├── event_features.py
│   └── feature_pipeline.py
│
├── services/
│   ├── __init__.py
│   ├── live_stock_service.py
│   ├── stock_service.py
│   ├── news_service.py
│   └── market_calendar_service.py
│
├── training/
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── evaluation_harness.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── train_sentiment.py
│   ├── train_event_classifier.py
│   ├── calibrate_probability.py
│   └── save_model.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── walk_forward.py
│   ├── feature_ablation.py
│   └── reports.py
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── registry.py
│   └── artifacts/
│       ├── metadata/
│       └── versions/
│
├── nlp/
│   ├── __init__.py
│   ├── finbert.py
│   ├── sentiment_pipeline.py
│   ├── event_classifier.py
│   └── news_aggregator.py
│
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py
│   ├── feature_attribution.py
│   └── driver_generator.py
│
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── stocks.py
│   │   ├── news.py
│   │   ├── predictions.py
│   │   ├── drivers.py
│   │   └── evaluation.py
│   ├── schemas/
│   └── errors.py
│
├── database/
│   ├── schema.sql
│   ├── migrations/
│   └── repositories/
│
├── jobs/
│   ├── fetch_market_data.py
│   ├── fetch_news.py
│   ├── generate_features.py
│   ├── generate_prediction.py
│   └── evaluate_predictions.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   └── ml/
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── charts/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── types/
│   └── public/
│
└── docs/
    ├── data-contracts.md
    ├── api-contracts.md
    ├── ml-pipeline.md
    ├── news-pipeline.md
    ├── explainability.md
    └── deployment.md

5. ARCHITECTURAL PRINCIPLES

5.1 Evaluation before UI

No frontend feature should be built around an unvalidated prediction.

5.2 Every model must beat a baseline

A model is not considered useful merely because it produces predictions.

Regression must be compared with:

Mean prediction

Zero-return baseline

Classification must be compared with:

Always-UP baseline

Base-rate probability

5.3 Time-aware validation

Financial data must never use random train/test splitting for the primary evaluation.

Use:

TimeSeriesSplit

with:

n_splits = 20
gap = prediction horizon

5.4 Feature selection inside folds

Feature selection must never see future/test rows.

Incorrect:

all data -> feature selection -> cross validation

Correct:

training fold -> feature selection -> model
test fold -> evaluation

5.5 Reproducibility

Every model artifact must record:

Model name

Model version

Training date

Dataset version

Feature version

Target definition

Training date range

Validation configuration

Metrics

Python version

Dependency versions

6. DATA LAYER

6.1 Stock data

Primary source currently:

yfinance

Data:

Open

High

Low

Close

Volume

Do not use unfinished intraday bars as daily closes.

Only completed trading-day bars should enter the daily prediction pipeline.

6.2 Implemented data contract (Phase 2)

Canonical schema (services/market_data.py):

Date      timezone-naive, = America/New_York trading date
Open, High, Low, Close, Volume   float64

Both yfinance paths (Ticker.history and the yf.download fallback) are
normalized to exactly this schema; Dividends / Stock Splits are dropped.
Invalid data (missing columns, NaN/inf, duplicate or unsorted dates,
non-positive prices or volume) raises MarketDataError; it is never
silently repaired.

Completed-bar policy (services/market_calendar_service.py):

A bar dated D is complete iff now >= D 16:00 America/New_York + 30 min.
The in-progress bar is dropped by fetch_latest_stock_data().
A bar dated after today (exchange time) is an error.
Limitation: early-close days count as complete only from 16:30.

Minimum-history policy (features/feature_engineering.py):

MIN_HISTORY_ROWS is derived from the feature windows:
max(longest rolling window = 30,
    EMA warm-up = rows until the start value of EMA_26 and then the
    MACD signal EMA_9 carries < 1% weight = 60 + 21) = 81 bars.
Fewer bars -> InsufficientHistoryError. The first 80 rows are
removed; every returned row must be finite, otherwise
FeatureGenerationError (no silent row dropping).

Price semantics:

auto_adjust=True on every path: split- and dividend-adjusted prices,
so returns and the Target are total returns. Yahoo back-adjusts
history after each dividend, so prices from different retrievals must
never be mixed; live outcome scoring computes returns from one
retrieval (models/evaluate_live_tracking.py).

Reproducibility (training/build_dataset.py):

yfinance -> data/raw/stocks/<TICKER>_1d_<period>_<UTC>.csv + .meta.json
(ticker, period, source method, retrieval time, parameters, sha256)
-> features -> data/final_stock_dataset.csv + .meta.json (snapshot hash,
feature version, dataset sha256). A download is not reproducible; a
stored snapshot is: `python -m training.build_dataset --from-snapshot
<file>` rebuilds a byte-identical dataset (same library versions).
The evaluation harness copies the dataset metadata into its report.
.gitattributes marks these data files -text so git never rewrites
their line endings (which would break the stored hashes).

7. TECHNICAL FEATURE PIPELINE

Initial technical feature groups:

Price/return features

1-day return

2-day return

3-day return

5-day return

10-day return

Log return

Trend

SMA 7

SMA 30

EMA 12

EMA 26

Momentum

RSI

MACD

MACD signal

MACD histogram

Volatility

Bollinger position

Bollinger width

Rolling volatility

Volume

Volume change 1

Volume change 2

Volume change 5

Do NOT include:

Dividends as a technical predictor

Stock Splits as a technical predictor

Duplicate Daily_Return

8. TARGET DEFINITION

The intended product target is:

UP / DOWN

For day t:

future_return = Close[t+1] / Close[t] - 1

Then:

target = 1 if future_return > 0 else 0

The regression target may remain useful for experiments, but the product-facing prediction should eventually be:

Direction + Probability

Example:

{
  "ticker": "AAPL",
  "prediction": "UP",
  "probability": 0.64
}

The probability must be calibrated and evaluated.

8.1 Implemented classification target (Phase 3, training/targets.py)

For the feature row dated t (features use completed bars up to t):

future_return[t] = Close[t+h] / Close[t] - 1      h = horizon (default 1)
direction[t]     = 1 (UP) if future_return[t] > 0 else 0 (DOWN)

target_version = direction_v1. Close is adjusted, so this is a total return.

Boundary decision:

- future_return == 0 exactly -> DOWN. "UP" asserts a rise; this is the
  rule above and the one used by the Phase 1-2 baselines.
- No dead zone. Excluding "small" moves would choose evaluation rows
  using the future outcome, unknown at prediction time.
- No epsilon. Identical adjusted closes give exactly 0.0; any non-zero
  total return (e.g. a dividend on a flat day) is a real move.
- The last h rows have no future close and get no label.

Every report records target_summary, including the number of
zero-return rows labelled DOWN.

Temporal safety: the latest close used by any training label is
Close[train_end + h]; TimeSeriesSplit gap = h puts the first test row at
train_end + h + 1, so every training label is known before the first
test prediction.

Benchmark: python -m training.train_classification runs the central
harness on the classification track only (Always UP, Base Rate,
Logistic Regression, Random Forest, XGBoost, LightGBM; fixed, untuned
hyperparameters) and saves classification_technical_YYYYMMDD.json.
Reports include a 10-bin reliability table per model (REQ-PROB-002).

9. EVALUATION HARNESS

This is the FIRST major implementation task.

File:

training/evaluation_harness.py

Responsibilities:

Load dataset

Validate schema

Construct target

Split chronologically

Run 20-fold walk-forward evaluation

Apply feature selection inside each fold

Train baselines

Train candidate models

Generate out-of-sample predictions

Calculate metrics

Save predictions

Save metrics

Generate a comparison report

Regression metrics:

MAE
RMSE
R²

Classification metrics:

Accuracy
Balanced Accuracy
ROC-AUC
Brier Score
Log Loss

Required baselines:

Mean Return
Zero Return

Always UP
Base Rate Probability

Model selection rule:

IF model does not beat baseline:
    model_status = "NOT QUALIFIED"
ELSE:
    model_status = "QUALIFIED"

Never crown a model simply because it has the highest R².

"Beat the baseline" is defined as (gate_v1, training/evaluation_harness.py):

Regression:
    pooled OOS MSE lower than the better of Mean / Zero Return
    AND one-sided Diebold-Mariano p < 0.05

Classification:
    pooled OOS Log Loss lower than Base Rate
    AND one-sided Diebold-Mariano p < 0.05
    AND Accuracy >= Always UP

If no model passes, the result is "NO QUALIFIED MODEL".

The gate measures skill, not correctness: a look-ahead feature would
pass it. Leakage is guarded by the ML tests (tests/ml/).

10. MODEL EXPERIMENT PIPELINE

Initial candidates:

Logistic Regression
Ridge
Random Forest
XGBoost
LightGBM

Start with Logistic Regression as the classification baseline.

Then compare tree models.

The goal is not maximum complexity.

The goal is:

Does the model generalize out of sample?

11. NEWS DATA PIPELINE

Historical news is required before implementing the final news model.

Pipeline:

News Source
    |
    v
Raw Articles
    |
    v
Deduplication
    |
    v
Timestamp normalization
    |
    v
Trading-day assignment
    |
    v
Ticker mapping
    |
    v
Sentiment
    |
    v
Event classification
    |
    v
Daily aggregation

Each article should contain:

article_id
ticker
headline
source
url
published_at
assigned_trading_date
text

11.1 Provider architecture (Phase 4A)

NewsProvider                      services/news_provider.py (interface)
├── AlpacaNewsProvider            services/alpaca_news_provider.py
│                                 PRIMARY: canonical historical provider
│                                 (Benzinga via Alpaca; future real-time via
│                                 Alpaca WebSocket - not in Phase 4A)
├── AlphaVantageNewsProvider      FUTURE: secondary / reference only
└── NewsDataProvider              FUTURE: optional expansion

Rules:

- The training dataset is built from ONE canonical provider (Alpaca).
  Providers are never merged into training data.
- Provider-specific code stays inside its provider module. Everything
  downstream consumes only the canonical NewsArticle schema.
- No provider-specific ML features. NewsArticle.ml_view() is the only
  ML-facing representation and contains no provider fields.

11.2 Canonical news schema (services/news_schema.py, news_v1)

provider                   str       e.g. "alpaca"
provider_article_id        str       provider's id, as text
headline                   str       required, non-empty
summary                    str|None
content                    str|None  as delivered (Benzinga: HTML)
symbols                    tuple     upper-case tickers, sorted, unique
source                     str|None  publisher reported by the provider
source_url                 str|None
created_at                 UTC datetime  provider publication time
updated_at                 UTC datetime|None  provider last-update time
information_available_at   UTC datetime  earliest time THIS version of
                                     the record may be used (see 12)
fetched_at                 UTC datetime  retrieval time
provider_metadata          dict      raw provider fields not mapped above
                                     (provenance/debugging only)

Mapping to the article list above: article_id = provider +
provider_article_id; ticker -> symbols (an article can mention several);
text -> content; url -> source_url; published_at -> created_at;
assigned_trading_date is derived later from information_available_at
(section 12), not stored by the provider layer.

All timestamps must be timezone-aware; they are normalized to UTC.
Naive timestamps are rejected. updated_at is never substituted for
created_at.

Deduplication key: (provider, provider_article_id). First occurrence
wins; duplicates and conflicting duplicates are counted in provenance.

11.3 Historical ingestion and provenance

services/news_service.py fetches a query (symbols, UTC start/end,
sort, include_content, page size) through a provider, deduplicates,
and writes a raw snapshot, reusing the Phase 2 snapshot pattern:

data/raw/news/<provider>_<SYMBOLS>_<start>_<end>_<UTC stamp>.jsonl
data/raw/news/...meta.json   provider, endpoint, request parameters
                             (never credentials), fetched_at, pages,
                             raw/duplicate counts, availability rule,
                             schema version, sha256

Snapshots are loaded only after their sha256 is verified.
data/raw/news/ is git-ignored: full-text articles are large and
provider content is licensed; snapshots stay local.

Alpaca specifics (isolated in its module): GET
https://data.alpaca.markets/v1beta1/news, page_token pagination, page
size 1-50 (Alpaca's documented maximum), headers APCA-API-KEY-ID /
APCA-API-SECRET-KEY read from ALPACA_API_KEY / ALPACA_API_SECRET.
Pagination stops when next_page_token is empty, and fails on a repeated
token or when a max_pages cap is reached (no infinite loops). 429 /
5xx / timeouts are retried with backoff; 401/403 and other 4xx are not.

11.4 Historical ingestion runs and canonical dataset (Phase 4C)

Purpose: turn multi-year provider history into ONE deterministic,
reproducible canonical dataset that later phases (FinBERT, events,
feature store) read. No NLP happens here.

Flow:

IngestionConfig (provider, symbols, UTC start/end, chunk_days,
                 include_content, page_size, sort, max_pages)
  -> chunk_intervals          consecutive half-open [a, b) chunks
  -> per chunk: NewsProvider.fetch_historical (provider paginates until
                next_page_token is absent; bounded as in 11.3)
              save_news_snapshot (11.3; hash + meta, never overwritten)
  -> run manifest             data/raw/news/<run_id>.manifest.json
  -> build_canonical_dataset  data/processed/news/<run_id>.jsonl + .meta.json

services/historical_news_ingestion.py orchestrates through the
NewsProvider interface only (no HTTP / provider formats).
services/news_dataset.py builds the dataset.

All-or-nothing: the manifest is written only after every chunk
succeeded. Provider, auth, pagination and validation errors propagate;
a failed run has no manifest and cannot become a dataset.

Canonicalization (pure, order-independent):

1. Deduplicate by (provider, provider_article_id). Within one fetch the
   provider keeps the first occurrence (11.3). Across chunk snapshots
   the dataset keeps the version with the latest
   (information_available_at, fetched_at), then the smallest serialized
   record - independent of arrival order, and leakage-conservative.
   Headline similarity is never used.
2. Interval: start <= created_at < end (publication time, half-open so
   chunks tile exactly). updated_at / information_available_at may lie
   after end and are kept unchanged.
3. Symbols: keep an article iff its canonical symbols contain a
   requested symbol; its symbols stay exactly as provided.
4. Sort by (created_at, provider, provider_article_id); serialize with
   sorted keys; sha256 over the bytes.

Ingestion interval vs prediction eligibility: the interval decides which
published articles are IN the dataset; eligibility for a prediction is
still only information_available_at <= prediction_timestamp (12.1). An
article published inside the interval but revised after it is in the
dataset yet not eligible until its information_available_at.

Reproducibility: the same manifest + snapshots always yield a
byte-identical dataset. Dataset metadata records the manifest hash,
every source snapshot hash, all rules, the counts (input, duplicates,
conflicting duplicates, outside interval, without requested symbol,
rows) and the dataset sha256. The manifest records the run parameters,
per-chunk pages / counts / snapshot hash, and totals. No credentials.

Storage: data/raw/news/ and data/processed/news/ are git-ignored (full
licensed text). Datasets are rebuilt locally from snapshots.

11.5 Historical backfill (Phase 8)

Purpose: expand the news sample (2017-01..02) to the technical spine's
full range with the EXISTING stages; no new provider, model, schema or
feature definition. Pipeline per run:

  4C  python -m services.historical_news_ingestion ... --resume
      chunked (default 30 days), --resume reuses a chunk snapshot only if
      its metadata records the same provider + request (symbols, exact
      start/end, sort, include_content, page_size) AND its sha256 verifies;
      a corrupt snapshot raises; a snapshot without metadata (interrupted
      write) is ignored; the manifest marks reused chunks.
  audit  python -m services.news_audit --news <canonical>   (read-only:
      duplicate ids (must be 0), duplicate URLs / same headline+time under
      different ids (reported, never deleted), per-year counts, empty
      months, after-close/weekend availability, revised articles)
  5C  python -m services.historical_sentiment ... --checkpoint <file>
      --revision defaults to the pinned FinBERT commit
      4556d13015211d73dccd3fdd39d39232506f3e43; --checkpoint appends
      validated sentiment_records_v1 every 512 articles (flushed) and a
      rerun reuses entries whose (article key, text_v1 hash) and full
      provenance match (a torn last line is repaired; other provenance
      raises). The final dataset is identical with or without resuming.
  6   python -m services.historical_events ...           (unchanged)
  7   python -m training.feature_store ... / training.incremental_evaluation
      (unchanged; the 252-row guard decides whether A/B/C run)

Determinism: provider responses are not reproducible (Yahoo/Benzinga
revise); RAW chunk snapshots are the frozen record. Everything after them
is hash-verified. Rebuilding events, daily event features, the feature
store and the A/B/C evaluation from the stored sentiment records was
verified byte-identical (Phase 8 run record below). FinBERT model-level
determinism (re-scoring the same texts reproduces the same records) is
NOT verified: the stored sentiment records are the authoritative FinBERT
output for this backfill.

--no-content is safe for this project: text_v1 uses content only when
headline and summary are both empty, and canonical articles always have a
headline; it keeps backfill snapshots small. include_content is part of
the request identity, so --resume never mixes the two modes.

Phase 8 run record (AAPL, 2026-10-01; artifacts git-ignored, local only)

  Ingestion (4C, --no-content --resume, 30-day chunks)
    request interval   2017-01-01T00:00Z .. 2026-09-29T00:00Z (end exclusive)
    chunks             119
    canonical dataset  alpaca_AAPL_20170101T000000Z_20260929T000000Z_
                       20261001T022146Z.jsonl
    articles           28,992 (all tagged AAPL by the provider)
    duplicates / conflicting duplicates / without requested symbol: 0 / 0 / 0
  Audit (read-only)
    duplicate article ids 0; same headline + created_at under different
    ids 7 (kept, reported); months without articles: none;
    available after 16:00 New York or on a weekend 5,122 (kept: they
    reach the next eligible row, see below); revised after creation
    (information_available_at > created_at) 16,388 - one stored version
    per id, availability = max(created_at, updated_at); the size of the
    revision lag was not measured.
    Other tickers in the audit's symbol counts are the provider's
    per-article tags of multi-ticker stories, not dataset membership.
  FinBERT (5C)       ProsusAI/finbert @ 4556d13015211d73dccd3fdd39d39232506f3e43,
                     text_v1 (headline + summary), CPU
    28,992 scored, 0 collapsed; 2,447 daily rows 2017-01-03 .. 2026-09-28
    (all with news - sentiment_features_v1 is cumulative)
    daily sha256 e25e7d9d8923cb6a463f18c595821c1bb9c3f3083b76eb4f5846c36ab2a6be3a
    independent verification 22/22: record count, unique ids == canonical
    ids, finite probabilities summing to 1, AAPL only, pinned revision in
    every record and both metadata files, rows == every session in the
    interval, prediction_timestamp == D 16:30 New York, cumulative counts
    re-counted as #articles with information_available_at <= ts(D)
  Events (6)         KeywordEventClassifier rules_v1, taxonomy_v1
    17,686 events, 11,306 OTHER; 2,447 daily rows, 2,364 with events
    records sha256 072287327265e7a0...; daily sha256
    50a44ae00df0a9a0051c6870afc86043b2f03991d9b487e8421142aed99e6bb0
  Feature store (7)  section 15.2

Eligibility is unchanged: an article contributes to row D only if
information_available_at <= D 16:30 America/New_York. After-close and
weekend articles are therefore not leakage; they first count in the next
row whose prediction timestamp is at or after their availability
(articles available 16:00-16:30 on D count in row D, whose target starts
from the already-fixed Close[D]).

Determinism verification (step 17): a full re-run of FinBERT on the
28,992 articles was started and stopped after roughly 2-3 hours of CPU
inference without completing; it was not repeated. The downstream
rebuild from the verified sentiment records passed 46/46 checks:

  event records, event daily, feature store   byte-identical to the originals
  A/B/C evaluation on the rebuilt store        bit-identical metrics, periods,
                                               statuses (in memory, not saved)
  preserved step 16 report                     unchanged (sha256 997e6010d224...)
  Phase 7 feature store + its 2017 inputs      unchanged

  NOT verified: FinBERT model-level determinism; sentiment records and
  sentiment daily were not rebuilt.

12. NEWS TIME ALIGNMENT

Critical rule:

News must be assigned to the trading day on which it was actually available.

Example:

Article published:
2026-09-29 15:30 ET

Assigned trading date:
2026-09-29

But:

Article published:
2026-09-29 17:00 ET

should normally influence:

2026-09-30

Weekend/holiday news rolls forward to the next trading session.

This prevents future information from leaking into the prediction.

Timestamp used for alignment (Phase 4A decision):

Trading-date assignment uses information_available_at, never created_at
or updated_at directly.

Four times are kept separate:

created_at                 when the provider says the article was published
updated_at                 when the provider last changed it
fetched_at                 when we retrieved it
information_available_at   when the version WE STORED may be used

For historical backfill (rule historical_backfill_v1):

information_available_at = max(created_at, updated_at)

A historical request returns the latest version of each article; its
text is not proven to have existed before updated_at, so the stored
version is treated as available only from then. created_at is still
kept unchanged. Future real-time ingestion will use the receipt time.

12.1 Implemented temporal alignment (Phase 4B, services/news_alignment.py)

Eligibility - the ONLY rule deciding whether news may inform a prediction:

    information_available_at <= prediction_timestamp

information_available_at is the authoritative eligibility timestamp;
prediction_timestamp is the boundary. created_at, updated_at, calendar
dates and session labels are never used to decide eligibility.

Two separate concepts - never conflated:

    news availability     information_available_at of the article
    daily-bar completion  D 16:00 New York + 30 min (section 6.2); defines
                          only WHEN a daily row's prediction is made

News at 10:00 is available at 10:00; it is not delayed to 16:30. The
16:30 bar-completion time is simply the prediction_timestamp of a daily
row, so that 10:00 article is eligible for that row.

direction_v1 integration:

    row t prediction_timestamp = completion_time(bar t) = D_t 16:30 New York
    news eligible for row t     : information_available_at <= that timestamp
    label                        : Close[t+h] / Close[t] - 1

The horizon h moves only the label's close, never the prediction
timestamp, so news arriving in (t, t+h] can never be eligible for row t.
(row_prediction_timestamp; tested for h = 1 and h = 5.)

Trading sessions (services/market_calendar_service.TradingCalendar):

Sessions are the dates of completed daily bars (the Phase 2 principle:
holidays and weekends produce no bar). A weekday without a bar inside
the calendar's range is a non-trading day. Dates outside the range
raise OutsideCalendarError - they are never guessed from weekdays.

Session assignment (descriptive label, REQ-NEWS-005, cutoff 16:00):

    phase            exchange-local time of information_available_at   session
    pre_market       trading day D, before 09:30                        D
    regular          trading day D, 09:30 .. 16:00 inclusive            D
    post_market      trading day D, after 16:00                         next session after D
    non_trading_day  weekend / holiday                                  next session after that date

Friday post-market and weekend news -> Monday (or the next session if
Monday is a holiday). The label says which session an article falls
into; it does NOT make the article eligible for that session's earlier
predictions (10:30 news is in session D but not eligible at D 09:30).

Time handling:

- UTC internally; America/New_York only to read calendar dates and
  phases. zoneinfo applies EST/EDT, so the same UTC instant can be
  pre-market in winter and regular session in summer (tested around
  2024-03-10).
- Naive datetimes are rejected everywhere.

Future information:

- information_available_at > fetched_at is impossible (the stored
  version cannot appear after we retrieved it) -> FutureInformationError.
- In live use, eligible_articles(..., now=...) rejects a prediction
  timestamp later than now.

Multi-ticker: an article is available to every symbol in its canonical
`symbols` and to no other; symbols are never inferred. Weighting of
multi-symbol articles is left to the feature layer.

Lookback windows: eligible_articles(..., not_before=x) selects
x < information_available_at <= prediction_timestamp. Window features
(pre-market, intraday, previous session, rolling) are built later on
top of this; Phase 4B implements none of them.

13. FINBERT PIPELINE

Use a pretrained financial sentiment model.

Pipeline:

Headline / Article
       |
       v
Tokenizer
       |
       v
FinBERT
       |
       +---- Positive
       +---- Neutral
       +---- Negative
       |
       v
Sentiment Score

Store:

positive_probability
neutral_probability
negative_probability
sentiment_score

Do not fine-tune initially.

Use the pretrained model first.

13.1 Implemented article-level sentiment (Phase 5A, services/finbert_sentiment.py)

Model: ProsusAI/finbert via transformers + torch, inference only (no
fine-tuning). Loaded ONCE per service; eval mode; torch.inference_mode()
(no gradients, no sampling). Weights stay in the Hugging Face cache and
are never committed. The public model needs no token; any token or API
key values found in the environment are redacted from error messages.

Device: CPU by default. "cuda" is optional and only accepted if
torch.cuda.is_available(); nothing assumes a GPU.

Text policy (text_v1, compose_sentiment_text / build_sentiment_text):

    headline + " " + summary      each whitespace-collapsed; summary dropped
                                  if it repeats the headline (case-insensitive)
    content (HTML stripped)       only if headline and summary are both empty
    empty                         SentimentTextError - never a fabricated score

URLs, ids, symbols, timestamps and source are never model input. The
article is not modified.

Tokenization: truncation by TOKENS at the model's own maximum
(min(tokenizer.model_max_length, max_position_embeddings) = 512 for
FinBERT); characters are never cut. A single text is not padded; a
batch is padded to its longest member (attention mask applied).

Output - SentimentResult (inference_version finbert_sentiment_v1):

    positive_probability, negative_probability, neutral_probability
                          softmax of the logits in float64, each in [0, 1],
                          sum = 1 (validated)
    label                 most probable class; ties -> positive, negative, neutral
    sentiment_score       positive_probability - negative_probability, in [-1, 1]
                          (never the label itself)
    model_name, model_revision (resolved Hugging Face commit hash),
    input_text_hash (sha256 of the exact input text), text_policy

Logits are mapped to labels via the model's own id2label, so column order
cannot silently swap classes. Structurally invalid output (wrong shape,
non-finite values, wrong label set, inconsistent probabilities) raises
SentimentModelError.

Batching: configurable batch size; output order == input order; empty
input -> empty output. The service never persists results.

Sentiment is an attribute of an article's text. It does not change when
the article may be used: eligibility is still only
information_available_at <= prediction_timestamp (12.1).

Testing: unit tests use a deterministic fake backend (offline, no
torch/transformers needed); the real model runs only in the opt-in
integration test tests/integration/test_live_finbert.py.

Not in 5A: corpus-wide scoring, daily aggregation, sentiment features,
model integration.

14. DAILY NEWS AGGREGATION

For each ticker and trading day:

news_count
mean_sentiment
positive_news_count
negative_news_count
max_positive_sentiment
max_negative_sentiment
recent_news_count

Potential event counts:

earnings
product
legal
management
macro
analyst
regulatory
M&A

The exact event taxonomy should be documented before implementation.

14.1 Implemented sentiment features (Phase 5B, services/sentiment_features.py)

Version: sentiment_features_v1. Pure, in-memory; no network, no model
inference, no persistence (no feature store yet).

Input: ScoredArticle = canonical NewsArticle + the SentimentResult for
ITS text. Validated on construction: the Phase 5A result contract is
re-checked (probabilities, sum = 1, label = argmax, score); the result's
input_text_hash must equal the article's text_v1 hash (sentiment cannot
be attached to the wrong article); impossible-future records rejected.

Rows: (symbol, trading_date) for completed-bar session dates of the
TradingCalendar. Weekend / holiday dates raise - no artificial rows.

    prediction_timestamp(D) = row_prediction_timestamp(D) = D 16:30 America/New_York
                              (stored timezone-aware, UTC)

Eligibility (reused from 12.1, the only rule):

    symbol in article.symbols  and  is_eligible(article, prediction_timestamp(D))
    i.e. information_available_at <= prediction_timestamp(D)

created_at, updated_at, fetched_at, dates and session labels are never
used. The generator has no horizon input: sentiment features for a date
are identical for h = 1 and h = 5 (tested through build_targets).
News at 17:00 on D reaches the row of the next session, not D.

Semantics (v1): CUMULATIVE - all eligible articles for the symbol in the
supplied news (i.e. since the start of the ingestion interval), as of
prediction_timestamp. No rolling windows, no decay; windows are a later,
explicit feature-design decision.

Features per row (labels are the validated argmax labels):

    news_count                  eligible articles
    positive/negative/neutral_count
    positive/negative/neutral_ratio      count / news_count
    mean_sentiment              mean of sentiment_score (= p_pos - p_neg)
    sentiment_std               POPULATION std of sentiment_score (Welford)
    mean_positive/negative/neutral_probability
    max_positive/negative/neutral_probability

Zero-news convention: when news_count == 0 every feature is 0 (never
NaN). A model therefore sees "no news" as neutral-looking zeros;
news_count == 0 is what distinguishes it and must be kept as a feature.

Duplicates: identity (provider, provider_article_id); identical records
collapse; conflicting records (article or sentiment differ) raise.
Headline/timestamp similarity is never used.

Determinism: articles sorted by (information_available_at, provider,
provider_article_id), rows by prediction timestamp; one chronological
pointer pass per symbol admits each article once (O(articles + rows)).
Output sorted by (symbol, trading_date); shuffled input -> identical
output. features_as_of() computes one row directly with the same
accumulator (used to cross-check the single pass).

Relation to section 14 above: v1 implements news_count,
positive_count (= positive_news_count), negative_count
(= negative_news_count), mean_sentiment and the probability maxima
(max_positive_probability ~ max_positive_sentiment). recent_news_count
requires a time window and is NOT implemented in v1.

Known limitation: cumulative counts grow with elapsed time (non-
stationary) and cumulative means are dominated by old news; v1 is
leakage-safe but probably weak as a model input until windowed
features are designed and evaluated.

14.2 Historical sentiment scoring and daily dataset (Phase 5C,
     services/historical_sentiment.py)

Purpose: score a whole canonical news dataset once, store auditable
article-level sentiment, and derive the daily sentiment_features_v1
dataset - reproducibly and without leakage. Orchestration only: text
policy + inference = Phase 5A, validation + aggregation = Phase 5B.

Flow:

canonical news .jsonl (4C, sha256 verified)
  -> unique_articles           identical duplicates collapse (scored once),
                               conflicting records raise; no headline merging
  -> impossible-future check   before any model work
  -> FinBertSentimentService   batched (batch size never changes results)
  -> ScoredArticle             5A result contract + text_v1 hash match (5B)
  -> <source>.finbert-<rev>.sentiment.jsonl + .meta.json
  -> reload                    file sha256, source dataset sha256, provenance,
                               re-join each record to its canonical article
                               (same key, information_available_at, symbols)
                               and re-validate
  -> generate_sentiment_features (5B, unchanged) on completed-bar sessions
     from a verified Phase 2 market snapshot
  -> <source>.finbert-<rev>.daily.jsonl + .meta.json

Article-level record (sentiment_records_v1), one per unique article,
ordered by (information_available_at, provider, provider_article_id):
record_version, provider, provider_article_id, symbols,
information_available_at, inference_version, model_name, model_revision,
text_policy, label, positive/negative/neutral_probability,
sentiment_score, input_text_hash. No article text is duplicated; records
are re-joined to the canonical dataset on load.

Provenance contract: one (model_name, model_revision, inference_version,
text_policy) per dataset, taken from the scoring backend; every record
must match it and none may be missing - mixing fails. The model revision
is part of the file name. Metadata: source dataset file/sha256/run_id,
provenance, input/scored/collapsed counts, rows, ordering, sha256,
generated_at.

Determinism: sorted input, sorted output, sorted JSON keys, shortest
round-trip floats. The file (and its sha256) contains content only;
generated_at is metadata. Identical re-runs rewrite identical bytes;
DIFFERENT content for an existing file is refused (never silently
overwritten). Bit-identical floats are not guaranteed across hardware /
torch builds - such a run is refused rather than mixed.

Daily dataset (daily_sentiment_v1): one row per (symbol, completed-bar
trading date), start <= date < end (default: the news interval's UTC
dates), columns = sentiment_features_v1 output (symbol, trading_date,
prediction_timestamp as aware UTC ISO, version, 15 features). Eligibility
remains only information_available_at <= prediction_timestamp
(D 16:30 New York); zero-news rows are all 0; no windows. Metadata links
the sentiment file hash, source dataset, provenance and the market
snapshot (file + sha256) that defined the sessions.

Storage: data/processed/sentiment/ is git-ignored (derived from licensed
news; reproducible from the canonical dataset + cached model). No API
keys are needed.

FinBERT sentiment is an NLP signal about article text. It does not
establish that any news caused a price movement.

14.3 Financial event classification (Phase 6)

Modules:

services/event_taxonomy.py     the ONLY definition of event types/impacts
services/event_classifier.py   EventClassifier interface; KeywordEventClassifier
services/event_features.py     daily event_features_v1 (pure)
services/historical_events.py  article-level + daily event datasets, CLI
services/event_evaluation.py   human-label validation tooling

Taxonomy (taxonomy_v1, derived from the 372-headline AAPL sample):
EARNINGS, GUIDANCE, M_AND_A, LEGAL, REGULATORY, CAPITAL_ACTION,
ANALYST_RATING, MANAGEMENT_GOVERNANCE, PARTNERSHIP, OWNERSHIP, PRODUCT,
OPERATIONS, MACRO, MARKET_ACTIVITY, OTHER. Changes vs the suggested list:
PRODUCT_LAUNCH/UPDATE merged (headlines rarely distinguish them);
MANAGEMENT_CHANGE widened to governance (shareholder meetings, proxy);
OPERATIONS = supply chain/manufacturing; OWNERSHIP (13F, investor
stakes) and MARKET_ACTIVITY (option/technical alerts, block trades,
market wraps - price-action reports, kept apart so they are never read
as fundamental events) added. Order = tie-break priority.

CURRENT BASELINE classifier - KeywordEventClassifier (rules_v1):
weighted, case-insensitive regex rules per type on text_v1 (headline +
summary; the exact text FinBERT scored, same input_text_hash).
score(type) = sum of matched rule weights; event_type = argmax (ties ->
taxonomy order); no match -> OTHER. event_confidence =
score(event_type) / sum(scores): the share of matched rule evidence, a
heuristic - NOT a probability and NOT accuracy. Every record lists the
rules that fired. Chosen because no labelled event data exists (no model
could be shown to be better), it is deterministic, instant on CPU, needs
no download/dependency, and is fully explainable. Rejected for now:
zero-shot NLI (large download, slow on CPU, unverifiable without labels)
and FinBERT embeddings (not trained for similarity).

Impact is separate from type: event_impact = the article's FinBERT label
(POSITIVE/NEGATIVE/NEUTRAL), event_impact_score = its sentiment_score,
read from the Phase 5C dataset (no FinBERT re-run).

Article-level dataset (event_records_v1), one record per unique article,
ordered by (information_available_at, provider, provider_article_id):
provider, provider_article_id, symbols, published_at (= created_at),
information_available_at (= the eligibility timestamp), input_text_hash,
event_type, event_confidence, matched_rules, event_impact,
event_impact_score, classifier_name, classifier_version,
taxonomy_version. No article text. Built from the verified 5C sentiment
dataset; reload verifies hashes and lineage, re-joins each record to its
article/sentiment and can re-classify to prove labels reproduce.
Metadata: classifier, impact source (sentiment file + sha + model
revision), source dataset, data-quality report (counts, duplicates,
type/impact distribution, confidence stats, low-confidence count,
symbols, availability range), labels_are_ground_truth = false.

Daily dataset (event_features_v1), per (symbol, completed-bar date D):

    window  ts(previous session) < information_available_at <= ts(D),  ts(D) = D 16:30 New York
    article_count, event_count (non-OTHER), unique_event_type_count,
    <type>_event_count for every non-OTHER type (generated from the taxonomy),
    positive/negative/neutral_event_count, event_impact_score (mean
    sentiment_score of events), mean_event_confidence, dominant_event_type
    (ties by taxonomy order; NONE if no events), recent_event_count
    (same rule over the last 5 sessions)

The upper bound is the Phase 4B eligibility rule; window bounds come from
the TradingCalendar (previous_session_before), never from which dates
were requested, so each article lands in exactly one row and features
do not depend on the horizon. Per-session windows were chosen (unlike the
cumulative sentiment_features_v1) so counts do not grow with time. Zero
convention: counts/scores 0, dominant NONE. Weekend/holiday news falls
into the next session's window; DST via zoneinfo.

Storage: data/processed/events/ (git-ignored, includes validation
templates). Duplicates: identical collapse, conflicting raise; identical
re-runs rewrite identical bytes, different content is never overwritten.

Evaluation: no labelled event dataset exists, so FORMAL SUPERVISED
CLASSIFICATION ACCURACY CANNOT BE ESTABLISHED from the current data.
Classifier output and event_confidence are not accuracy.
event_evaluation exports a human-labelling template (predictions hidden)
and, once a person fills it, reports precision/recall/F1 per class,
macro/weighted F1 and a confusion matrix against baselines "always
OTHER" and "majority human label"; under 100 labels it is descriptive
only.

FUTURE IMPROVEMENT (not implemented): a human-labelled financial event
dataset, then a zero-shot or supervised fine-tuned classifier evaluated
against rules_v1 on held-out labels; incremental evaluation of event
features through the central harness (REQ-EVENT-003).

Event classification describes what an article is about; like sentiment
it does not establish that the event caused a price movement.

15. COMBINED FEATURE STORE

Final daily row:

ticker
trading_date

technical features
+
news features
+
sentiment features
+
event features

Example:

AAPL
2026-09-29

RSI
MACD
Volatility
Return_1
Return_5
Volume_Change_1

news_count
mean_sentiment
negative_news_count
earnings_event_count
legal_event_count
...

Version the feature set.

Example:

feature_version = v1.0

15.1 Implemented feature store (Phase 7, training/feature_store.py)

CURRENT IMPLEMENTATION - feature_store_v1 / feature_store_schema_v1.
Joins EXISTING, hash-verified datasets; no feature is recomputed:

technical  data/final_stock_dataset.csv               Phase 2, technical_v2 (20)
sentiment  data/processed/sentiment/*.daily.jsonl     Phase 5C, sentiment_features_v1 (15)
events     data/processed/events/*.daily.jsonl        Phase 6, event_features_v1 (24 = 23 numeric + 1 categorical)
sessions   data/raw/stocks/<snapshot>.csv             Phase 2 raw snapshot (calendar)

Grain: (symbol, trading_date) - one prediction observation per row,
predicted at row_prediction_timestamp(D) = D 16:30 America/New_York (the
existing convention; no new timestamp system). The technical dataset is
the spine; its symbol comes from its metadata (raw_snapshot.ticker).

Join contract: keys normalised (upper-case symbol, ISO date) and checked
for nulls/duplicates BEFORE joining (error, never silent aggregation);
every key must be a calendar session; every sentiment/event row's
prediction_timestamp must equal row_prediction_timestamp(D); 1:1 left
joins (pandas validate="one_to_one"). News rows without a technical row
(e.g. before the technical warm-up) are counted in the join report.

Namespaces: technical names unchanged (the harness uses them);
sentiment__*, event__* for the news families; version columns moved to
metadata; event__dominant_event_type kept as a categorical column outside
the numeric families; sentiment__covered / event__covered flags.

Missing-data semantics:
  inside a family's coverage (its first..last daily row): the family's
    own zero convention applies (no news / no events -> 0, dominant NONE);
    every covered session must have a row - a gap is an ERROR (missing
    data is never read as "no news")
  outside coverage: values NaN (unknown), <family>__covered = False
  no blanket fill with 0

Targets (unchanged, direction_v1): target_return_1d = stored Target =
Close[t+1]/Close[t] - 1 (re-verified against Close), target_direction_1d
= 1 if > 0. Close is kept as a reference column (needed to rebuild
targets in the harness); targets and Close are never in a feature family.
Feature columns only contain information available at prediction time:
technical = completed bar D, sentiment/events = articles with
information_available_at <= D 16:30 (inherited from 5B/6).

Output: data/processed/features/feature_store_v1_<SYM>_<first>_<last>.csv
+ .meta.json (git-ignored). The CSV is deterministic (fixed columns,
sorted rows, ISO dates, full-precision floats); its sha256 is the CONTENT
hash. generated_at and git commit are build metadata only. Identical
rebuilds rewrite identical bytes; different content is never
overwritten. Metadata: versions, grain, every source file + sha256 +
version/provenance, target definitions, family counts, row/column
counts, join report, data-quality report (date range, symbols,
duplicate/null keys, NaN per column, infinities, coverage, constant
features, |z| > 6 counts, unexpected categories, target distribution).

Chronological order is preserved (rows sorted by symbol, trading_date);
downstream evaluation must stay walk-forward (no shuffling).

Incremental evaluation (training/incremental_evaluation.py), through the
CENTRAL harness only:

  A technical (20)   B + sentiment (35)   C + events (58)

Same rows (sessions where every family is covered), same targets, same
default_candidates, same TimeSeriesSplit(20, gap = horizon), same
metrics and gate_v1 for all three. Guard: fewer than 252 evaluable rows
-> INSUFFICIENT_DATA, nothing is fitted or reported.

Phase 7 result (historical; 2017-01/02 AAPL news sample): the families
overlap on 23 sessions (2017-01-26 .. 2017-02-28; the technical dataset
starts after its warm-up) -> INSUFFICIENT_DATA. This is a
pipeline/integration validation dataset, not sufficient evidence of
generalizable model performance. That store
(data/processed/features/feature_store_v1_AAPL_20170126_20260928.csv,
sha256 5af804bd42bd...) is kept unchanged. Phase 8 result: section 15.2.

FUTURE IMPROVEMENTS (not implemented): multi-symbol evaluation; windowed
sentiment features (sentiment_features_v1 is cumulative and
non-stationary); human-validated event labels; a database-backed store.
(The full 2017-present news backfill was done in Phase 8.)

15.2 Phase 8 feature store and A/B/C result (AAPL, 2026-10-01)

Feature store (unchanged feature_store_v1 code; separate output directory
because the file name depends only on the technical date range):

  data/processed/features/backfill_2017_2026/
      feature_store_v1_AAPL_20170126_20260928.csv
  sha256   73966443a9223b344752c2a40a3fdb185d4b561b9138538c4a3ca72ec37ca12d
  rows 2,431 (2017-01-26 .. 2026-09-28), columns 67
  families technical 20 / sentiment 15 / event 23 numeric
           + 1 categorical (event__dominant_event_type) + 2 targets
  coverage technical, sentiment, events, all three: 2,431 each
  sessions with events 2,348 (the other 16 of the 2,364 fall before the
           technical spine starts; join report 16 sentiment + 16 event rows
           without a technical row)
  quality  0 duplicate keys, 0 NaN / infinite feature values, 0 constant
           features, no unexpected categories; up share 0.536
  inputs   technical final_stock_dataset.csv (sha256 e5fef0d2...), raw
           snapshot AAPL_1d_10y_20260929T205051Z.csv (141aba8f...), sentiment
           and event daily files of section 11.5
  verification 37/37 (step 15): hash == metadata; schema/version/family
           lists == code contract; dates, Close, technical features and
           target_return_1d exactly equal to the technical spine; every row
           an AAPL session, none skipped; prediction_timestamp == D 16:30
           New York; store values == source daily rows of the SAME date;
           sentiment news_count and event counts re-counted from the
           article-level records (0 mismatches); categorical values valid;
           full hash chain store -> daily -> records -> canonical news;
           Phase 7 artifact unchanged.

A/B/C evaluation (training.incremental_evaluation, unchanged; canonical
report data/results/evaluation/feature_store_incremental_20261001.json,
sha256 997e6010d22451654c5a615d498a983fc73abf4bcabc63a64051a5b91c5f00c2;
a temporary _phase8_backfill copy of the same file was not kept):

  rows in store 2,431; evaluable 2,431; out-of-sample 2,300 rows,
  2017-08-02 .. 2026-09-25, identical for A, B and C (the harness rebuilds
  the target from Close, so 2026-09-28 has no next close; 2,430 // 21 =
  115 rows per fold x 20 folds)
  TimeSeriesSplit(20), gap 1, horizon 1, min_rows 252, gate_v1,
  default_candidates, no tuning; features A 20, B 35, C 58

  Baselines (feature-independent, identical in A, B and C)
    Mean Return   MSE 3.572769e-4  MAE 0.0131249  RMSE 0.0189018  R2 -0.000937
    Zero Return   MSE 3.582601e-4  MAE 0.0131515  RMSE 0.0189278  R2 -0.003691
    Always UP     Acc 0.534783  BalAcc 0.5  AUC 0.5  Brier 0.465217
                  LogLoss 16.0684  share UP 1.0
    Base Rate     Acc 0.534783  BalAcc 0.5  AUC 0.482349  Brier 0.249044
                  LogLoss 0.691237  share UP 1.0

  Regression (reference = Mean Return, lowest baseline MSE)
    exp  model    MSE          vs Mean   MAE        RMSE       R2
    A    Linear   3.743109e-4  +4.8%     0.0135642  0.0193471  -0.048659
    A    Ridge    3.658838e-4  +2.4%     0.0134358  0.0191281  -0.025050
    B    Linear   4.338441e-4  +21.4%    0.0151676  0.0208289  -0.215445
    B    Ridge    3.984617e-4  +11.5%    0.0144362  0.0199615  -0.116319
    C    Linear   4.905051e-4  +37.3%    0.0160604  0.0221473  -0.374185
    C    Ridge    4.271430e-4  +19.6%    0.0148485  0.0206674  -0.196672
    gate_v1 reason (all six): "MSE not better than Mean Return"

  Classification (reference = Base Rate; accuracy floor = Always UP)
    exp  Logistic  LogLoss   Acc       BalAcc    AUC       Brier     share UP
    A              0.712826  0.516522  0.496239  0.484048  0.257692  0.791304
    B              0.809571  0.492174  0.485389  0.483540  0.273988  0.596522
    C              0.863365  0.485652  0.480385  0.480345  0.280054  0.574348
    gate_v1 reasons (all three): "Log_Loss not better than Base Rate";
    "Accuracy below Always UP"

  Qualification: A, B and C - regression NO QUALIFIED MODEL,
  classification NO QUALIFIED MODEL.

  Reasons are derived from the gate code because the report stores only
  the status: every candidate's primary metric is worse than its
  reference, so the Diebold-Mariano reason is never reached (its p-value is
  not stored; with a worse mean loss the one-sided statistic is negative,
  p > 0.5).

  B and C did not improve on A on any reported metric: every candidate
  degrades A -> B -> C (MSE, MAE, log loss, accuracy, AUC, Brier). AUC < 0.5
  everywhere: no ranking ability. No sign of leakage (leakage would
  inflate B/C, not degrade them).

  Consistency checks on the report: baselines byte-identical across
  experiments; Always UP Brier = share of DOWN days and log loss =
  0.465217 x -ln(1e-15); R2 consistent with MSE (same target variance for
  both baselines); every accuracy x 2,300 is an integer; scalers fitted
  inside each training fold.

  Determinism: re-running A/B/C in memory on the byte-identical rebuilt
  store reproduced every metric, period and status exactly (section 11.5).

Limitations of this result:

  1. No model qualified in any experiment or track; adding sentiment and
     events made every candidate worse. This extends, and does not
     overturn, the earlier technical-only finding of no signal.
  2. sentiment_features_v1 is cumulative: its counts grow from ~0 to
     28,992, so in every expanding walk-forward test fold they lie above
     the training range (extrapolation). Likely contributor to the B/C
     degradation (inferred, not separately tested). B therefore tests
     these cumulative features, not news sentiment in general; windowed
     or per-session sentiment features are untested.
  3. Early folds train on ~130 rows; with 35 (B) or 58 (C) features that
     is ~3.7 / ~2.2 rows per feature for an unregularised linear model
     (inferred contributor).
  4. Event labels are keyword rules_v1 output, not ground truth
     (labels_are_ground_truth = false); 39% of articles are OTHER.
  5. News relevance: an AAPL tag includes multi-ticker stories; only
     headline + summary are used (--no-content).
  6. Revision timing: 16,388 articles were revised after creation; the
     lag was not measured. Availability = max(created_at, updated_at) is
     leakage-safe but may delay late-revised articles (signal dilution).
  7. Scope: one symbol (the harness is single-series), one-day horizon,
     the harness's fixed linear candidates only (tree models are installed
     but not default candidates), no tuning by design.
  8. Traceability: the incremental report stores neither the feature-store
     path/sha256 nor gate reasons or Diebold-Mariano p-values; reports are
     named by UTC date, so a same-day rerun overwrites the file (the
     committed report is identified by its sha256 above).
  9. Multiple comparisons: 3 experiments x 3 non-baseline candidates; moot
     here because nothing qualified.
 10. FinBERT model-level determinism not verified (section 11.5).

15.3 Phase 9 predictive signal research - development stage (AAPL, 2026-10-01)

Design: pre-registered, two-stage (training/phase9_research.py). The
complete experiment matrix was frozen BEFORE any result existed; the
development stage runs it once through the UNCHANGED central harness
(run_evaluation, gate_v1); only development qualifiers may be evaluated,
once each, on the Phase-9 confirmation holdout.

  registry   data/results/research/phase9/registry.json
             matrix sha256   12e16f11ba886d1a61d7f91bf912478dc6850653373e3c1094afd26ae9b12545
             registry sha256 3f7ffd7fb2c5aec36a4fd04460cc24497a346bc69333ed80aa10f92b531b14f1
             (after development; status DEVELOPMENT_COMPLETE)
  run files  data/results/research/phase9/experiments/dev_{A..F}_h{1,3,5}.json
             (18 harness reports, write-once, sha256 recorded in the registry)
  commands   python -m training.phase9_research freeze | validate-context |
             develop | holdout --confirm | status

New modules (no existing module changed):
  training/research_targets.py   documents h = 1/3/5 use of the EXISTING
                                 targets (future_return, direction_v1, gap = h)
  services/sentiment_window.py   sentiment_window_v1 (13 features), read-only
                                 from the verified Phase 8 sentiment records
  services/market_context.py     market_context_v1 (9 features) from raw
                                 SPY / QQQ snapshots
  training/phase9_research.py    freeze / develop / holdout / registry

sentiment_window_v1: W_k(D) = AAPL articles with ts(prev_k(D)) <
information_available_at <= ts(D), ts(D) = D 16:30 New York, k = 1, 5, 20
sessions. sw_count_k, sw_mean_k (mean FinBERT sentiment_score),
sw_positive_ratio_k / sw_negative_ratio_k (k = 5, 20), sw_mean_change_5_20
= sw_mean_5 - sw_mean_20, sw_count_surprise_1_20 = sw_count_1 -
sw_count_20 / 20, sw_mean_surprise_1_20 = sw_mean_1 - sw_mean_20; empty
window -> 0. A row is covered only if its 20-session window lies inside the
news interval (otherwise NaN).

market_context_v1 (S = SPY, Q = QQQ, A = AAPL adjusted Close; k sessions
back on each series): spy_return_1/5/20, qqq_return_1/5 = X[D]/X[D-k] - 1;
aapl_minus_spy_return_1/5; spy_volatility_20 = sample std (ddof 1) of 20
SPY log returns ending D; spy_close_to_sma_50 = S[D] / mean(S[D-49..D]).
Only bars dated <= D (complete at D 16:30 New York = prediction time).
SPY/QQQ sessions must equal AAPL sessions over the needed range; a missing
or extra session raises (no forward fill, no dropped rows).
  SPY data/raw/stocks/SPY_1d_10y_20261001T104943Z.csv  sha256 bc827c76bc61bee2...
  QQQ data/raw/stocks/QQQ_1d_10y_20261001T104948Z.csv  sha256 106509bae648328d...
  (2,512 rows each, 2016-10-03 .. 2026-09-30, yfinance 1.5.1; created with
  training.build_dataset.fetch_and_snapshot only - the AAPL dataset was not
  rebuilt). Over the needed range 2016-11-18 .. 2026-09-28: 2,476 AAPL
  sessions, 0 missing / 0 extra in SPY and QQQ.

Pre-registered matrix: 6 feature sets x 3 horizons x (6 regression + 5
classification models) = 198 experiments (108 regression, 90
classification); baselines (Mean Return, Zero Return, Always UP, Base
Rate) are references in every run, not experiments.
  A technical (20)                         B + sentiment_features_v1 (35)
  C technical + events (43)                D technical + sentiment + events (58)
  E technical + market context (29)
  F technical + sentiment window + events + market context (65 registered
    columns, 64 distinct - see limitations)
  horizons 1, 3, 5 trading days; targets future_return_{h}d (regression) and
    direction_{h}d (classification), existing formulas, gap = h
  regression: Linear Regression, Ridge, RF Regressor, HistGradientBoosting
    Regressor, XGBoost Regressor, LightGBM Regressor
  classification: Logistic Regression, Random Forest, HistGradientBoosting,
    XGBoost, LightGBM
  fixed parameters (Phase 3 tree settings, seed 42; HistGradientBoosting
  without early stopping); no tuning.

Data: research frame = Phase 8 store rows covered by every family, 2,427
rows 2017-02-01 .. 2026-09-28 (the first 4 store rows lack a full 20-session
news window). Development = rows <= 2024-09-30: 1,928 rows; targets are
built inside this frame, so the last h rows are unlabelled and no price
after 2024-09-30 is used. Labelled rows 1,927 / 1,925 / 1,923 (h = 1/3/5);
TimeSeriesSplit(20), gap = h, test size 91, 1,820 pooled out-of-sample rows
per run (h = 1: 2017-07-06 .. 2024-09-27); first training fold 106 / 102 /
98 rows. Inputs verified against the frozen hashes (store 73966443...,
sentiment records d3f40991..., sentiment daily e25e7d9d..., events
50a44ae0..., canonical news cc974e83..., AAPL snapshot 141aba8f...).

Development result (18 runs, 27 min, exit 0):
  0 development qualifiers under gate_v1 (0 / 108 regression, 0 / 90
  classification); raw p < 0.05: 0 (smallest raw p 0.390; none < 0.10);
  Holm-adjusted p: 1.0000 for all 198 (informational only).
  Baselines (identical across feature sets):
    h  Mean Return MSE   Zero Return MSE   Base Rate log loss   Always UP accuracy
    1  3.62967e-4        3.64138e-4        0.691610             0.5368
    3  9.70034e-4        9.80392e-4        0.684821             0.5736
    5  1.56097e-3        1.59044e-3        0.678512             0.5918
  Descriptive only (no ranking, no recommendation): 4 of 198 experiments
  beat their reference baseline on the primary metric, none qualified:
    p9_A_h1_rf_regressor    MSE -0.102% vs Mean Return, DM 0.251, p 0.401
    p9_C_h1_rf_regressor    MSE -0.103% vs Mean Return, DM 0.279, p 0.390
    p9_C_h5_random_forest   log loss -0.052% vs Base Rate, DM 0.077, p 0.469;
                            accuracy below Always UP
    p9_F_h5_random_forest   log loss -0.064% vs Base Rate, DM 0.099, p 0.461;
                            accuracy below Always UP
  The other 194 experiments did not beat their reference on the primary
  metric. Full per-experiment metrics, baseline metrics, DM statistics,
  raw/Holm p-values, gate reasons and fold boundaries are in the registry
  and run files.

Phase-9 confirmation holdout (2024-10-01 .. 2026-09-25): NOT EVALUATED.
With no development qualifiers it is not applicable; no holdout file
exists. (Recording NOT_APPLICABLE in the registry is done by `holdout
--confirm`, which reads no data in this case; it has not been run, so the
registry status is still DEVELOPMENT_COMPLETE.) The holdout is in any case
not pristine: Phase 3 (technical trees, h = 1) and Phase 8 (linear A/B/C,
h = 1) evaluated parts of it. Phase 3 / Phase 8 results are separate
evaluations on different rows and are not part of the Phase 9 matrix.

Conclusion (scoped): no candidate passed gate_v1 during Phase-9
development - i.e. none of these 198 pre-registered AAPL configurations
(these features, horizons, fixed models and this walk-forward design)
showed significant out-of-sample improvement over its baseline. This is
not a statement about predictive signal in general.

Audit: an independent read-only audit passed 32/32 checks (completeness,
registry == run files, gate_v1 re-applied from stored metrics reproduces
all 198 statuses, raw p == 1 - Phi(DM), Holm recomputed, cutoff and gap per
fold, input and artifact hashes) and found no non-finite values in 5,400
fold-metric rows.

Limitations:
  1. No candidate passed gate_v1 during Phase-9 development; the
     confirmation holdout was therefore not evaluated.
  2. Feature set F contains an exact duplicate: event__article_count ==
     sw_count_1 (both count AAPL articles in the 1-session window). F has
     65 registered columns but 64 distinct; kept unchanged because the
     matrix was frozen. Its effect on predictions was not assessed.
  3. SPY/QQQ hashes are stored in registry.development.context_inputs, not
     in the individual run files (which are harness reports of in-memory
     data); the registry links each run file by sha256.
  4. 101,097 identical scikit-learn UserWarnings ("sklearn.utils.parallel
     .delayed should be used with sklearn.utils.parallel.Parallel ...")
     were emitted. They produced no errors and no non-finite metrics; the
     triggering estimator was not identified.
  5. Development rerun determinism has NOT been independently verified.
  6. Inherited: sentiment_features_v1 is cumulative (non-stationary);
     event labels are keyword rules, not ground truth; single symbol; fixed
     parameters only (no tuning by design); FinBERT model-level determinism
     not verified (section 11.5).

15.4 Phase 10B volatility-normalized excess-return research - development stage (AAPL, 2026-10-01)

Objective: test whether normalizing AAPL's future excess return over SPY by
its prediction-time realized volatility gives a target on which the
existing features show significant out-of-sample improvement over the
baselines. A target-reformulation experiment; every other element is the
frozen Phase 10A design (Phase 10A: excess-return research, matrix
fbd763e7980232bb1bf7409616f8dbbb0af9e8b0c70857a680cb89aad1da6a45, 0 development
qualifiers, holdout NOT_APPLICABLE; commit 91998ab).

  code       training/normalized_targets.py, training/phase10b_research.py
             (no existing module changed; reuses Phase 9 / 10A components)
  registry   data/results/research/phase10b/registry.json
             matrix sha256   101dd0fd888b28b101a510adecf4d4ef11ea25500e3540cefa457900548bffd3
             registry sha256 b1372fe6f8faff2edac38fa6192a67efde74fdbbafbf51f71bd37833f892f812
             (status COMPLETE)
  run files  data/results/research/phase10b/experiments/dev_{A..F}_h{1,3,5}.json
             (18, write-once, sha256 in the registry; each also records the
             input hashes, matrix hash and recorded warnings)
  targets    data/results/research/phase10b/targets/target_diagnostics_development.json
  commands   python -m training.phase10b_research freeze | target-diagnostics |
             develop | holdout --confirm | status

Target (fixed before any Phase 10B result):
  future_excess_return_h(D) = (Close[D+h]/Close[D] - 1) - (SPY_Close[D+h]/SPY_Close[D] - 1)
                              (Phase 10A numerator, training/excess_targets.py)
  x_j          = (Close[j]/Close[j-1] - 1) - (SPY_Close[j]/SPY_Close[j-1] - 1)
                 daily excess simple return of AAPL session j
  volatility_D = sample std (ddof = 1) of x_j for the 20 sessions j = D-19 .. D
                 (20-session window and ddof = 1 as the technical `Volatility`
                 and spy_volatility_20; simple returns as the Phase 10A target;
                 computed on the full hash-verified AAPL and SPY snapshots on
                 AAPL sessions; no horizon scaling - a constant factor per
                 horizon cannot change a gate_v1 decision)
  normalized_excess_return_h(D)    = future_excess_return_h(D) / volatility_D
  normalized_excess_direction_h(D) = 1 if normalized_excess_return_h(D) > 0 else 0
  Insufficient history (< 20 daily excess returns ending at D) or a
  non-positive / non-finite volatility raises; rows are never dropped or
  filled. The volatility is a target denominator only, never a feature.

Matrix: the frozen Phase 10A feature sets A technical (20), B + sentiment
(35), C technical + events (43), D technical + sentiment + events (58),
E technical + market context (29), F technical + sentiment window + events
+ market context (64 distinct; sw_count_1 dropped, event__article_count kept)
x horizons 1, 3, 5 x 11 models (6 regression: Linear Regression, Ridge, RF
Regressor, HistGradientBoosting Regressor, XGBoost Regressor, LightGBM
Regressor; 5 classification: Logistic Regression, Random Forest,
HistGradientBoosting, XGBoost, LightGBM; Phase 9 fixed parameters, seed 42,
no tuning) = 198 pre-registered experiments (108 regression, 90
classification). Baselines keep their harness names (Mean Return, Zero
Return, Base Rate, Always UP), applied to the normalized targets.

Evaluation: development 2017-02-01 .. 2024-09-30 (1,928 rows; labelled
1,927 / 1,925 / 1,923 for h = 1/3/5); TimeSeriesSplit(20), gap = horizon,
1,820 pooled out-of-sample rows per run (h = 1: 2017-07-06 .. 2024-09-27);
gate_v1 unchanged (evaluation_harness.validate_dataset, walk_forward,
pooled_metrics, qualify; only the target construction differs); Holm
adjustment informational. Confirmation holdout 2024-10-01 .. 2026-09-25
(not pristine) - only for development qualifiers with --confirm.

Leakage controls: volatility uses closes dated <= D only (unit tests:
changing later prices leaves it unchanged; the last row's volatility never
reaches a label); development frame cut at 2024-09-30 before targets are
built; every fold has exactly h sessions between training end and test
start; the last development label uses the 2024-09-30 close; no target,
price or volatility column in any feature set; all 8 input hashes (store
73966443..., sentiment records d3f40991..., sentiment daily e25e7d9d...,
events 50a44ae0..., news cc974e83..., AAPL 141aba8f..., SPY bc827c76...,
QQQ 106509ba...) verified at freeze and at development.

Target diagnostics (development rows only): volatility mean 0.0114 (range
0.0028 .. 0.0269); normalized target std 1.18 / 2.14 / 2.75 (h = 1/3/5), all
finite; positive direction 51.84 / 53.56 / 54.24 %; all spot checks match.

Development result (18 runs, 16 min, exit 0; 0 warnings recorded):
  0 development qualifiers under gate_v1 (0 / 108 regression, 0 / 90
  classification); raw p < 0.05: 0 (smallest raw p 0.528; none < 0.10);
  Holm-adjusted p: 1.0000 for all 198.
  Baselines (identical across feature sets):
    h  Mean MSE   Zero MSE   Base Rate log loss   Always UP accuracy
    1  1.40152    1.40264    0.693401             0.5198
    3  4.53294    4.54220    0.692616             0.5346
    5  7.50522    7.53820    0.693789             0.5374
  No experiment beat its reference baseline on the primary metric (0 / 198).
  Descriptive only: the RF Regressor came closest at h = 1 (C +0.03%,
  A +0.10% MSE vs Mean Return; p 0.528 / 0.580). Compared descriptively with
  Phase 10A's regression experiments (same features, models, folds), the
  relative MSE gap was smaller in 54 / 108 pairs (median 15.37% vs 15.50%):
  normalization produced no improvement that passes the gate.
  Holdout: NOT_APPLICABLE - no development qualifier; the Phase-10B
  confirmation holdout was not read and no holdout file exists.

Classification experiments: volatility > 0, so normalized_excess_direction_h
equals Phase 10A's excess_direction_h; with the same rows, folds, features
and fixed models the 90 classification results are label-identical to
Phase 10A and add no new evidence. The audit confirmed they reproduced Phase
10A bit for bit (this verifies deterministic reruns for those 90
configurations only).

Conclusion (scoped): for these 198 pre-registered AAPL configurations,
normalizing the excess return by its 20-session prediction-time volatility
produced no significant out-of-sample improvement under gate_v1. This is not
a statement about predictive signal in general.

Audit: read-only audit 23 / 23 checks (completeness, registry == run files,
gate_v1 re-applied reproduces all 198 statuses, raw p == 1 - Phi(DM), Holm
recomputed, cutoff and gap per fold, input and artifact hashes,
classification identical to Phase 10A).

Limitations:
  1. The 90 classification experiments are label-identical to Phase 10A by
     construction and add no new evidence; only the 108 regression
     experiments test the normalization.
  2. The development period has now been used by Phase 9, Phase 10A and
     Phase 10B (594 pre-registered tests); Holm is applied within Phase 10B
     only.
  3. One volatility estimator (20-session excess-return realized volatility);
     alternatives (AAPL-only volatility, EWMA, other windows) are untested.
  4. Inherited: sentiment_features_v1 is cumulative; event labels are keyword
     rules, not ground truth; single symbol; fixed parameters only; FinBERT
     model-level determinism not verified (section 11.5).
  5. DEVELOPMENT RERUN DETERMINISM NOT VERIFIED for the full Phase 10B run
     (frame, matrix and development-target construction were verified
     identical at freeze).
  6. 0 warnings were recorded; Phase 9's scikit-learn UserWarning did not
     occur inside the recording context (inferred: it fires only when the
     captured warning-filter list is empty; not tested directly).

15.5 Phase 10C market-regime-conditioned research - development stage (AAPL, 2026-10-01)

Research question: do predictive relationships emerge conditionally under
observable market regimes even though no unconditional AAPL model qualified
in Phases 9, 10A or 10B? Hypothesis (tested, not assumed): explicitly
pre-registered regime conditioning may reveal statistically reliable
out-of-sample performance that is absent in the unconditional evaluation.

  code       training/regimes.py, training/phase10c_research.py (no existing
             module changed; reuses Phase 9 / 10A components)
  registry   data/results/research/phase10c/registry.json
             matrix sha256   ffd3591d69927f5f275e9e1e5ef583ab6866a8d8a056d0ecfe25a81e2422783d
             registry sha256 184c5efe8e2e6637a602154ea1ae1225b459a922a6137e1921d6864639248c27
             (status COMPLETE)
  run files  data/results/research/phase10c/experiments/dev_{A..F}_h{1,3,5}.json
             (18, write-once, sha256 in the registry; folds, fold metrics,
             per-fold regime thresholds and state counts, every state's
             metrics and gate results for all candidates and baselines, the
             unconditional reference, inputs, matrix hash, warnings)
  regimes    data/results/research/phase10c/regimes/regime_diagnostics_development.json
  commands   python -m training.phase10c_research freeze | regime-diagnostics |
             develop | holdout --confirm | status

Design (everything except regime conditioning is the frozen Phase 10A design):
  target     Phase 10A excess return: future_excess_return_h and
             excess_direction_h (training/excess_targets.py)
  matrix     6 feature sets (A 20, B 35, C 43, D 58, E 29, F 64) x horizons
             1, 3, 5 x 11 models (Phase 9 fixed parameters, seed 42, no
             tuning) = 198 base experiments (108 regression, 90
             classification); 18 walk-forward runs
  regimes    (all computed from information available at D 16:30 New York)
    R1 AAPL volatility   regime_aapl_vol_20 = sample std (ddof 1) of 20 AAPL
                         simple daily returns ending at D (equals the technical
                         `Volatility` feature, max diff 1.2e-16);
                         LOW_VOL if <= training-fold median else HIGH_VOL
    R2 SPY volatility    regime_spy_vol_20 = sample std (ddof 1) of 20 SPY
                         simple daily returns ending at D (not the log-return
                         market_context feature); training-fold median ->
                         LOW_SPY_VOL / HIGH_SPY_VOL
    R3 SPY trend         spy_close_to_sma_50 (SPY close / 50-session mean)
                         >= 1.0 -> POSITIVE_TREND else NEGATIVE_TREND (fixed)
    R4 SPY 20d return    spy_return_20 >= 0 -> POSITIVE_20D_RETURN else
                         NEGATIVE_20D_RETURN (fixed)
    R5 SPY vol x trend   R2 state x R3 state (4 states)
    = 12 states. Fold-fitted thresholds: median over the TRAINING rows of each
    walk-forward fold only; validation rows labelled with that frozen
    threshold; ties (== median) -> LOW.
  evaluation ONE model per training fold (harness walk_forward,
             TimeSeriesSplit(20), gap = horizon); its out-of-sample
             predictions are split by regime state; for each state the
             unchanged pooled_metrics and qualify (gate_v1) run on exactly the
             state's rows for all candidates and baselines - "gate_v1 logic
             unchanged; evaluation population is the pre-registered
             regime-conditioned validation population".
  sample     a state with < 100 pooled OOS rows -> INSUFFICIENT_DATA, no test
  family     all eligible regime-conditioned comparisons (at most 198 x 12 =
             2,376); Holm across the complete family, fixed before results
  qualifies  gate_v1 QUALIFIED within the state AND >= 100 rows AND raw DM
             p < 0.05 AND Holm-adjusted p < 0.05
  periods    development 2017-02-01 .. 2024-09-30 (1,928 rows; 1,820 pooled
             OOS rows per run); confirmation holdout 2024-10-01 .. 2026-09-25
             (not pristine) only for qualifiers with --confirm

Leakage controls: regime variables use bars dated <= D only (unit tests:
future AAPL / SPY prices leave every regime variable at D unchanged);
thresholds use training rows only (changing validation values has no
effect); labels use the frozen threshold; regime labels are independent of
targets; no target column is a regime input and no regime variable is a
model feature; folds and gap identical to Phase 10A. Final audit: regime
variables recomputed from raw closes dated <= D match; every one of the 360
fold thresholds equals the median of that fold's training rows and the state
labels reproduce.

Regime diagnostics (development, before any model; nearly identical for
h = 1/3/5): R1 LOW 728 / HIGH 1,092; R2 LOW 618 / HIGH 1,202; R3 POSITIVE
1,312 / NEGATIVE 508; R4 POSITIVE 1,261 / NEGATIVE 559; R5 LOW_POS 557 /
LOW_NEG 61 / HIGH_POS 755 / HIGH_NEG 447 (h = 1). The volatility states are
skewed to HIGH because each threshold is the median of the earlier training
window and volatility rose over the sample (frozen rule, not adjusted).

Development result (18 runs, about 10 min, exit 0, 0 warnings): NEGATIVE.
  comparisons             2,376 (198 base experiments x 12 states)
  insufficient data       198 - the single state R5 LOW_SPY_VOL_NEGATIVE_TREND
                          (61 pooled OOS rows at every horizon)
  Holm family             2,178 eligible comparisons (198 x 11 states)
  raw p < 0.05            0 (smallest 0.0703; only one below 0.10)
  Holm-adjusted p < 0.05  0 (smallest Holm p 1.0000)
  gate_v1 passes          0 (also 0 before Holm)
  fully qualified         0 -> NO DEVELOPMENT QUALIFIERS
  Descriptive only (not a signal, not ranked): 26 of 2,178 comparisons beat
  their reference baseline on the primary metric within the state (22
  regression, 4 classification; 21 at h = 1), all NOT_QUALIFIED with raw
  p >= 0.070 and Holm p = 1.0000 (largest gap MSE -0.739% for
  p10c_A_h1_rf_regressor in R4 POSITIVE_20D_RETURN, p = 0.0703).
  Unconditional reference (not in the family): 198 / 198 NOT_QUALIFIED and
  bit-identical to the Phase 10A development results.
  Confirmation: NOT_APPLICABLE - no development qualifier; the Phase-10C
  confirmation holdout was not read and no holdout file exists.

Conclusion (scoped): for these pre-registered AAPL regime definitions,
features, horizons and fixed models, regime conditioning revealed no
statistically reliable out-of-sample performance under gate_v1 with Holm
correction. This is not a statement about predictive signal in general.

Final audit (read-only, 18 / 18): artifacts present and hash-verified;
inputs and matrix unchanged; family 2,178 / insufficient 198 / single
insufficient state confirmed; Holm recomputed; gate_v1 re-applied from the
artifacts reproduces all 2,178 statuses; no raw or Holm p < 0.05; holdout
not read; regime variables and fold thresholds independently recomputed;
unconditional reference == Phase 10A; no warnings.

Statistical methodology and limitations:
  1. DM regime-subset limitation: within regime-conditioned subsets,
     observations are not necessarily consecutive in calendar/session time.
     The existing Newey-West correction therefore treats the pooled
     regime-filtered sequence as consecutive observations. This approximation
     is retained to preserve gate_v1 comparability with Phases 9-10B.
  2. One model is trained per fold on all regimes; separate per-regime models
     were not part of Phase 10C.
  3. One fixed state (R5 LOW_SPY_VOL_NEGATIVE_TREND) had too few rows to test.
  4. Median thresholds come from earlier training windows, so the volatility
     states are unbalanced in the out-of-sample period.
  5. The development period has now been used by Phase 9, 10A, 10B and 10C;
     Holm is applied within Phase 10C only.
  6. Inherited: single symbol; fixed parameters only; cumulative sentiment
     features; keyword event labels; FinBERT model-level determinism not
     verified (section 11.5).
  7. DEVELOPMENT RERUN DETERMINISM NOT VERIFIED for the full Phase 10C run
     (frame, matrix and development-target construction verified identical at
     freeze; the unconditional reference reproduced Phase 10A bit for bit).

15.6 Phase 11.0 cross-stock generalization - DESIGN FROZEN (no data, no experiments)

Question: is the absence of a qualified predictive signal specific to AAPL,
or does the FindMarketDriver methodology fail to generalize across equities?

  code       training/phase11_design.py (design definition, affected-row and
             eligibility semantics; no data access)
  registry   data/results/research/phase11/design_registry.json (write-once)
             design sha256 266daaddba0e8cb18163b4ec9db7eadabe2d5ab574bf410e043a7d28d3247650

Frozen decisions:
  universe   historical DJIA membership on 2016-12-31 (source: Wikipedia,
             "Historical components of the Dow Jones Industrial Average",
             section "March 19, 2015", in force until 2017-09-01; used only to
             establish membership, not corporate-action mechanics). 29
             securities: AAPL AXP BA CAT CSCO CVX DIS GE GS HD IBM INTC JNJ JPM
             KO MCD MMM MRK MSFT NKE PFE PG RTX TRV UNH V VZ WMT XOM.
             Later-removed members are kept (historical, not current, membership).
             DD EXCLUDED: the 2016 member (E.I. du Pont de Nemours) has no single
             continuous listed security series over the research interval; no
             successor substitution, no synthetic series.
             UTX -> RTX: legal/security continuation (UTC was the legal survivor
             of the Raytheon merger completed 2020-04-03), subject to Phase 11.1
             verification of the Carrier/Otis adjustment.
             No security may be removed for predictive performance or low news
             coverage.
  corporate  provider-adjusted history; every material event (pre-registered
  actions    for RTX/UTX, GE, MMM, MRK, IBM, PFE, JNJ, plus any further material
             action in the provider records) is verified in Phase 11.1; properly
             adjusted -> rows kept; not adjusted -> rows whose feature lookback
             or target spans the event are excluded and counted: for first
             reflecting session E, rows D in [E-h, E+49] (session indices; 50-
             session maximum lookback); no price repair, no splicing, no whole-
             stock exclusion without a separately frozen rule. Event dates and
             ratios other than the 2020-04-03 merger completion are recorded as
             REQUIRES_PHASE_11_1_VERIFICATION (none invented).
  sample     >= 1,800 labelled development rows per stock AFTER all exclusions;
             missing sessions never filled
  intervals  research frame 2017-02-01..2026-09-28; development
             2017-02-01..2024-09-30; confirmation 2024-10-01..2026-09-25
  target     Phase 10A excess return vs SPY (each stock's own adjusted close);
             horizons 1, 3, 5
  features   Phase 10A sets A-F (20/35/43/58/29/64), aapl_minus_spy_* renamed
             stock_minus_spy_* (same formula); rules_v1 events unchanged (its
             Apple-specific product keywords are a recorded limitation)
  models     the 11 Phase 9 fixed models, seed 42, no tuning
  panel      PRIMARY: all eligible stocks pooled; TimeSeriesSplit(20) over
             unique development dates with gap = h (all stocks of a date in the
             same fold); one model per fold; no ticker identity feature;
             baselines Zero, pooled Mean, per-stock Mean / Always UP, pooled and
             per-stock Base Rate (reference = strongest by pooled OOS loss);
             DM on the per-date cross-sectional mean loss with the existing
             one-sided test (Newey-West h - 1 lags); gate_v1 logic unchanged
  per-stock  SECONDARY: existing per-symbol harness for each stock
  counts     per-stock 29 x 6 x 3 x 11 = 5,742; panel 198; AAPL-excluded panel
             sensitivity 198 (descriptive only)
  families   Holm within the primary panel family (198) and within the
             secondary per-stock family (5,742); never merged after results
  qualifies  gate_v1 + raw p < 0.05 + Holm p < 0.05 within the family;
             confirmation only for qualifiers with authorization, else
             NOT_APPLICABLE; a generalization claim requires a confirmed panel
             qualifier
  staging    11A: A, E (prices only); 11B: B, C, D, F (multi-symbol news) -
             runs regardless of 11A
  AAPL       included; per-stock A/E compared with Phase 10A as a
             reproducibility check only (wording amended - see below)

Amendment 01 (data/results/research/phase11/amendment_01_reproducibility.json,
training/phase11_reproducibility.py): the Phase 11.0 registry
(design_registry.json, design sha256 266daadd...) remains immutable and was
not edited. The amendment replaces ONLY the AAPL / Phase 10A reproducibility-
check wording ("checked for equivalence ... where data and design are
mathematically identical"), because the canonical Phase 11.1 price batch
differs from the Phase 10 snapshots by about 1e-6 relative (provider
re-adjustment at retrieval), so bit-identical equivalence is impossible.
  input      Phase 11 uses the canonical Phase 11.1 price batch
             (data/processed/phase11/market/, verification report
             price_verification.json)
  prices     AAPL, SPY and QQQ vs the Phase 10 snapshots on every common date:
             |relative close difference| <= 1e-5 and |absolute daily-return
             difference| <= 1e-5
  results    the 66 AAPL A/E comparisons (2 feature sets x 3 horizons x 11
             models) vs the matching Phase 10A experiments: identical gate_v1
             status, primary metric (MSE / log loss) within 0.5% relative, raw
             DM p-value within 0.02
  outcome    REPRODUCED_WITHIN_TOLERANCE or REPRODUCIBILITY_CHECK_FAILED (every
             difference reported); a failure does not modify the Phase 11
             methodology or results
All other Phase 11 design decisions are unchanged.

Phase 11.1 status (complete; data and integrity verification only): price
acquisition and session validation are complete (one yfinance batch, 29
securities + SPY + QQQ, 2016-01-04 .. 2026-09-29). No missing sessions were
found. All 29 securities passed the >= 1,800 development-row requirement (1,927 /
1,925 / 1,923 labelled development rows for h = 1/3/5). No corporate-action
rows required exclusion: every documented event was determined properly
adjusted (or no adjustment required) under the frozen Phase 11.1 rule, including
RTX/UTX continuity across the 2020-04-03 merger and the Carrier/Otis adjustment.
The canonical Phase 11.1 price batch (data/processed/phase11/market/) is now the
input for Phase 11A. Detailed observations are recorded in
data/results/research/phase11/price_verification.json; the AAPL / Phase 10A
reproducibility amendment is recorded separately in
amendment_01_reproducibility.json. Phase 11A has NOT started.

16. EXPERIMENT GATE

Every new feature group must pass the same evaluation harness.

Experiment matrix:

Experiment A
Technical only

Experiment B
News sentiment only

Experiment C
Technical + sentiment

Experiment D
Technical + sentiment + events

Each experiment receives:

dataset version
feature version
model version
metrics

Decision:

Does the new feature set improve out-of-sample performance
relative to the established baseline?

If not:

Do not add it to production.

17. PROBABILITY CALIBRATION

Raw classifier probabilities should not automatically be displayed as trustworthy probabilities.

Pipeline:

Classifier
    |
    v
Raw probability
    |
    v
Calibration
    |
    v
Calibrated probability

Evaluate:

Brier score

Log loss

Reliability/calibration curve

Possible methods:

Platt scaling
Isotonic regression

Choose using time-aware validation.

18. FINAL ML OUTPUT

The production prediction object should eventually look like:

{
  "ticker": "AAPL",
  "as_of": "2026-09-29",
  "prediction": "UP",
  "probability": 0.64,
  "expected_return": 0.008,
  "model_version": "v1.0",
  "feature_version": "v1.0"
}

This is an ML output, not financial advice.

19. SHAP EXPLAINABILITY

After a validated model exists:

Prediction
    |
    v
SHAP
    |
    v
Feature contributions
    |
    v
Driver generator

Example:

Prediction: UP

Top model-attributed factors:

1. Positive sentiment      +0.08
2. RSI recovery            +0.04
3. Earnings news           +0.03
4. Short-term return       -0.02

Important:

These are model attributions.

They must not be presented as causal explanations.

20. MARKET DRIVER GENERATOR

Convert model attribution into readable text.

Example:

The model's UP prediction is primarily associated with:

- Positive recent financial-news sentiment
- Improving short-term momentum
- Reduced recent volatility

Negative contribution:
- Recent short-term price reversal

The UI should label this:

Why the model predicts UP

rather than:

Why AAPL will rise

21. FLASK BACKEND

Backend responsibility:

React
  |
  | HTTP
  v
Flask
  |
  +-- Stock Service
  +-- News Service
  +-- ML Service
  +-- Explainability Service
  +-- Database

Suggested API:

GET /api/health

GET /api/stocks/<ticker>

GET /api/stocks/<ticker>/history

GET /api/news/<ticker>

GET /api/predictions/<ticker>

GET /api/drivers/<ticker>

GET /api/evaluation/<ticker>

GET /api/models

22. API RESPONSE DESIGN

Prediction endpoint:

{
  "ticker": "AAPL",
  "as_of": "2026-09-29",
  "direction": "UP",
  "probability": 0.64,
  "expected_return": 0.008,
  "model_version": "v1.0",
  "feature_version": "v1.0"
}

Drivers endpoint:

{
  "ticker": "AAPL",
  "prediction": "UP",
  "drivers": [
    {
      "feature": "mean_sentiment",
      "contribution": 0.08,
      "direction": "positive"
    },
    {
      "feature": "RSI",
      "contribution": 0.04,
      "direction": "positive"
    }
  ]
}

23. DATABASE

Recommended later storage:

PostgreSQL

Core tables:

stocks
stock_prices
news_articles
news_sentiment
news_events
features
model_versions
predictions
prediction_outcomes
driver_attributions

Prediction table should store:

ticker
as_of_timestamp
bar_date
model_version
feature_version
prediction
probability
expected_return

Outcome table:

prediction_id
actual_return
actual_direction
evaluated_at

This allows:

What did the model predict yesterday?
What actually happened?

24. REDIS

Redis is optional initially.

Use later for:

API caching

Latest prediction caching

Rate limiting

Frequently requested stock data

Do not introduce Redis before the core ML pipeline works.

25. FRONTEND

React dashboard structure:

Dashboard
│
├── Search Bar
│
├── Stock Overview
│   ├── Ticker
│   ├── Current Price
│   └── Daily Change
│
├── Prediction Card
│   ├── UP / DOWN
│   ├── Probability
│   └── Model Version
│
├── Candlestick Chart
│
├── Technical Indicators
│
├── News Timeline
│   ├── Headline
│   ├── Sentiment
│   └── Event
│
├── Model Drivers
│   ├── SHAP chart
│   └── Explanation
│
└── Historical Evaluation
    ├── Accuracy
    ├── AUC
    ├── Brier Score
    └── Recent predictions

26. FRONTEND USER FLOW

User opens application
        |
        v
Searches AAPL
        |
        v
Backend fetches/loads stock data
        |
        v
Latest completed market data
        |
        v
Feature pipeline
        |
        v
Production model
        |
        v
Prediction
        |
        v
Probability calibration
        |
        v
SHAP
        |
        v
Driver summary
        |
        v
React dashboard

27. LIVE PREDICTION PIPELINE

Only after the offline model is validated:

Scheduled Job
     |
     v
Check market session
     |
     v
Fetch completed daily bar
     |
     v
Fetch latest eligible news
     |
     v
Generate features
     |
     v
Load model version
     |
     v
Prediction
     |
     v
Probability calibration
     |
     v
SHAP
     |
     v
Store prediction

Next trading day:

Prediction
    |
    v
Fetch actual result
    |
    v
Score prediction
    |
    v
Store outcome

28. MODEL MONITORING

Track:

Accuracy
Balanced Accuracy
AUC
Brier Score
Log Loss

Prediction distribution
Probability calibration
Feature distribution
Data freshness
Missing values
API failures

Monitor drift:

Training feature distribution
vs
Live feature distribution

29. TESTING STRATEGY

Unit tests

Test:

Feature calculations

Target creation

Trading-day assignment

News deduplication

Sentiment aggregation

Prediction schema

Probability calibration

Integration tests

Test:

data -> features -> model -> prediction

API tests

Test:

GET /api/health
GET /api/stocks/AAPL
GET /api/predictions/AAPL
GET /api/drivers/AAPL

ML tests

Test:

No future leakage

Chronological ordering

Feature schema

Baseline calculation

Fold isolation

Target alignment

30. SECURITY

Secrets must live in:

.env

Never commit:

.env
API keys
database passwords
tokens

Provide:

.env.example

Example:

ALPHA_VANTAGE_API_KEY=
DATABASE_URL=
NEWS_API_KEY=

31. DEPENDENCY MANAGEMENT

Current requirements.txt is incomplete.

The final environment should explicitly record packages actually used.

At minimum, depending on the implemented pipeline:

numpy
pandas
scikit-learn
scipy
joblib
yfinance
xgboost
lightgbm
python-dotenv
flask
flask-cors
transformers
torch
shap

Do not install every package immediately.

Add dependencies when their corresponding project phase begins.

Python version should also be explicitly documented.

32. DOCKER ARCHITECTURE

Later target:

docker-compose
│
├── backend
├── frontend
├── postgres
└── redis

Optional later:

worker
scheduler

The ML model can initially run inside the backend process.

Separate ML inference service only if scale requires it.

33. CI/CD

GitHub Actions later:

Push
 |
 v
Lint
 |
 v
Unit Tests
 |
 v
ML Sanity Tests
 |
 v
Build Backend
 |
 v
Build Frontend
 |
 v
Docker Build
 |
 v
Deploy

Do not deploy until:

Evaluation harness passes

Production model is validated

API tests pass

Live prediction pipeline works

34. IMPLEMENTATION PHASES

PHASE 1 — ML FOUNDATION

Status: CURRENT

Tasks:

1. Clean feature set
2. Build evaluation_harness.py
3. Add regression baselines
4. Add classification baselines
5. Add Logistic Regression
6. 20-fold walk-forward evaluation
7. Save official baseline

Deliverable:

data/results/evaluation/technical_baseline_YYYYMMDD.json

PHASE 2 — DATA PIPELINE HARDENING

Tasks:

1. Completed-bar validation
2. Minimum-history validation
3. Consistent yfinance schema
4. Remove app.py duplicate/broken path
   (done: app.py is now an explicit placeholder reserved for Flask, Phase 11)
5. Fix Config/config case
   (canonical: lowercase config/, matching every import)
6. Fix requirements.txt
7. Fix live evaluation adjustment issue
8. Raw data snapshots

PHASE 3 — PRODUCT TARGET

Tasks:

1. UP/DOWN target                     (implemented: section 8.1)
2. Logistic Regression baseline       (implemented, + RF / XGBoost / LightGBM)
3. Time-aware evaluation              (central harness, gap = horizon)
4. Probability calibration            (reliability analysis implemented;
                                       fitting a calibrator deferred until a
                                       model qualifies - calibrating a model
                                       with no skill cannot create skill)
5. Brier score                        (harness, since Phase 1)
6. Log loss                           (harness, since Phase 1)

PHASE 4 — HISTORICAL NEWS

Tasks:

1. Verify historical source          (done manually: Alpaca AAPL news back
                                      to 2017-01, pagination, content)
2. Obtain timestamped news           (4A: provider abstraction + Alpaca)
3. Store raw articles                (4A: hash-verified raw snapshots)
4. Deduplicate                       (4A: provider + provider_article_id)
5. Assign trading dates              (4B: section 12.1 - eligibility by
                                      information_available_at, session labels)
6. Map tickers                       (4B: canonical symbols only, no inference;
                                      alias/company-name mapping is later work)

Phase 4A does not add sentiment, events, news features, real-time
ingestion, or any change to the training dataset or models.

PHASE 5 — FINBERT

Tasks:

1. Load pretrained FinBERT
2. Process headlines/articles
3. Generate sentiment probabilities
4. Generate sentiment score
5. Aggregate daily
6. Evaluate sentiment-only model
7. Evaluate technical + sentiment

PHASE 6 — EVENT CLASSIFICATION

Tasks:

1. Define event taxonomy          (taxonomy_v1, section 14.3)
2. Build event classifier         (baseline: keyword rules_v1)
3. Generate event labels          (classifier output, not ground truth)
4. Aggregate daily                (event_features_v1)
5. Evaluate incremental value     (central-harness evaluation done in
                                   Phase 8, experiment C: no qualified
                                   model, section 15.2; human-labelled
                                   validation still NOT done)

PHASE 7 — FINAL FEATURE STORE

Tasks:

1. Technical features             (reused: technical_v2)
2. Sentiment features             (reused: sentiment_features_v1)
3. Event features                 (reused: event_features_v1)
4. Versioned combined dataset     (feature_store_v1, section 15.1)
5. Reproducible feature pipeline  (content hash, write-once)
6. Incremental A/B/C evaluation   (runner done; Phase 7 sample ->
                                   INSUFFICIENT_DATA; run in Phase 8)

PHASE 8 — HISTORICAL NEWS BACKFILL (inserted 2026-10-01; later phases
shift by one in the working roadmap)

Status: COMPLETE (2026-10-01). No model qualified in A, B or C.

Tasks:

1. Resumable chunked ingestion   (section 11.5) - done, 119 chunks
2. Backfill audit                (section 11.5) - done
3. Resumable, revision-pinned FinBERT scoring (section 11.5) - done,
   28,992 articles
4. Rebuild events, daily features, feature store; rerun A/B/C if >= 252
   sessions are covered          (sections 11.5, 15.2) - done, 2,431
                                   evaluable rows; NO QUALIFIED MODEL
5. Determinism                   downstream rebuild byte/bit-identical;
                                   FinBERT model-level determinism NOT
                                   verified (section 11.5)

PHASE 9 — PREDICTIVE SIGNAL RESEARCH (inserted 2026-10-01; AAPL only;
research, no production model / API)

Status: DEVELOPMENT COMPLETE (2026-10-01) - 0 development qualifiers;
        Phase-9 confirmation holdout not applicable (not evaluated)

Tasks:

1. Pre-register the experiment matrix  (section 15.3) - frozen, 198 experiments
2. Sentiment-window features           (sentiment_window_v1) - done
3. Market-context features             (market_context_v1, SPY/QQQ) - done
4. Development evaluation              (<= 2024-09-30, gate_v1) - done,
                                         NO DEVELOPMENT QUALIFIERS
5. Confirmation holdout                (2024-10-01 .. 2026-09-25) - not
                                         applicable; not evaluated
6. Explainability of a qualified model - not applicable (none qualified)

PHASE 10B — VOLATILITY-NORMALIZED EXCESS-RETURN RESEARCH (AAPL only;
research, no production model / API)

Status: DEVELOPMENT COMPLETE (2026-10-01) - 0 development qualifiers;
        Phase-10B confirmation holdout NOT_APPLICABLE (not read)

Tasks:

1. Normalized target + prediction-time volatility (section 15.4) - done
2. Pre-register the experiment matrix  - frozen, 198 experiments
3. Target diagnostics (development only) - done
4. Development evaluation              (<= 2024-09-30, gate_v1) - done,
                                         NO DEVELOPMENT QUALIFIERS
5. Confirmation holdout                - not applicable; not read

PHASE 10C — MARKET-REGIME-CONDITIONED RESEARCH (AAPL only; research, no
production model / API)

Status: DEVELOPMENT COMPLETE (2026-10-01) - NEGATIVE RESULT: 0 qualified
        regime-conditioned comparisons; Phase-10C confirmation holdout
        NOT_APPLICABLE (not read)

Tasks:

1. Regime variables and fold-fitted thresholds (section 15.5) - done
2. Pre-register matrix, regimes, sample rule, Holm family - frozen
3. Regime diagnostics (development only) - done
4. Development evaluation (2,376 comparisons, 2,178 eligible) - done,
   NO DEVELOPMENT QUALIFIERS
5. Final audit                          - done, 18 / 18
6. Confirmation holdout                - not applicable; not read

PHASE 11 — CROSS-STOCK GENERALIZATION RESEARCH (29 historical DJIA members;
research, no production model / API)

Status: 11.0 DESIGN FROZEN (2026-10-01) - section 15.6; no data acquired,
        no experiment run

Tasks:

1. 11.0 design freeze (universe, corporate-action rule, matrix) - done
2. 11.1 price data, corporate-action verification, eligibility   - not started
3. 11.2 panel evaluator + per-stock runner                       - not started
4. 11.3 / 11A development (A, E)                                  - not started
5. 11.4 multi-symbol news, FinBERT, events                       - not started
6. 11.5 / 11B development (B, C, D, F)                            - not started
7. 11.6 confirmation (qualifiers only) and report                 - not started

PHASE 8 (original numbering) — FINAL MODEL

Tasks:

1. Compare candidate models
2. Select only if baseline is beaten
3. Calibrate probability
4. Save artifact
5. Save metadata

PHASE 9 (original numbering) — EXPLAINABILITY

Tasks:

1. SHAP
2. Feature attribution
3. Driver ranking
4. Natural-language driver generation

PHASE 10 — BACKEND

Tasks:

1. Flask
2. API schemas
3. Stock endpoint
4. News endpoint
5. Prediction endpoint
6. Drivers endpoint
7. Evaluation endpoint

PHASE 11 — DATABASE

Tasks:

1. PostgreSQL
2. Predictions
3. News
4. Features
5. Model versions
6. Outcomes

PHASE 12 — FRONTEND

Tasks:

1. React
2. Stock search
3. Chart
4. Prediction card
5. Probability
6. News timeline
7. Sentiment
8. Drivers
9. Historical evaluation

PHASE 13 — LIVE SYSTEM

Tasks:

1. Scheduler
2. Daily prediction
3. Prediction storage
4. Next-day outcome
5. Automatic scoring
6. Monitoring

PHASE 14 — DEPLOYMENT

Tasks:

1. Docker
2. Docker Compose
3. CI/CD
4. Cloud deployment
5. Monitoring

35. IMPLEMENTATION RULE

Always work one phase at a time.

For every phase:

Implement
   ↓
Run
   ↓
Test
   ↓
Inspect output
   ↓
Commit
   ↓
Move to next phase

Never implement five phases simultaneously.

36. CURRENT CHECKPOINT

Current repository truth:

Git:
    Clean / pushed

Stock ingestion:
    Working

Technical features:
    Working but require cleanup

Target:
    Working

Regression:
    Working but production model is invalid

Classification:
    Experimental / not production

News:
    Missing

FinBERT:
    Missing

Events:
    Missing

Feature store:
    Missing

SHAP:
    Missing

Flask:
    Missing

React:
    Missing

Database:
    Missing

Docker:
    Missing

CI/CD:
    Missing

37. IMMEDIATE NEXT TASK

Do NOT start React.

Do NOT start Flask.

Do NOT start SHAP.

Do NOT start FinBERT yet.

First create:

training/evaluation_harness.py

Then clean:

features/feature_engineering.py

Then run:

technical-only baseline

Expected experiment:

Technical Features
        |
        v
20-fold TimeSeriesSplit
        |
        +--> Mean baseline
        +--> Zero baseline
        +--> Logistic Regression
        +--> Ridge
        +--> Random Forest
        +--> XGBoost
        +--> LightGBM
        |
        v
Pooled OOS predictions
        |
        v
Metrics
        |
        v
Official baseline report

Only after this succeeds do we begin historical news.

38. DEFINITION OF DONE

FindMarketDriver is complete when a user can:

1. Search a stock
2. View historical price data
3. View relevant news
4. View sentiment
5. View detected events
6. Receive UP/DOWN prediction
7. See calibrated probability
8. See model-attributed drivers
9. Inspect historical model performance
10. See whether previous predictions were correct

And the system can:

1. Reproduce training
2. Validate new features
3. Prevent temporal leakage
4. Version models
5. Store predictions
6. Score outcomes
7. Monitor data/model drift
8. Deploy reproducibly

39. IMPORTANT PRODUCT LANGUAGE

Use:

"Model prediction"
"Model probability"
"Model-attributed drivers"
"Feature contribution"

Avoid:

"This news caused the stock to rise"
"The stock will rise"
"This guarantees an increase"
"AI knows why the market moved"

The system is a predictive/analytical tool, not a causal inference engine.

40. MASTER IMPLEMENTATION ORDER

                    CURRENT
                       |
                       v
              [1] EVALUATION HARNESS
                       |
                       v
              [2] DATA HARDENING
                       |
                       v
              [3] UP/DOWN TARGET
                       |
                       v
             [4] HISTORICAL NEWS
                       |
                       v
                 [5] FINBERT
                       |
                       v
            [6] EVENT CLASSIFIER
                       |
                       v
            [7] FEATURE STORE
                       |
                       v
             [8] FINAL MODEL
                       |
                       v
           [9] PROBABILITY CALIBRATION
                       |
                       v
                [10] SHAP
                       |
                       v
          [11] MARKET DRIVER OUTPUT
                       |
                       v
               [12] FLASK API
                       |
                       v
              [13] POSTGRESQL
                       |
                       v
              [14] REACT UI
                       |
                       v
            [15] LIVE EVALUATION
                       |
                       v
             [16] DOCKER / CI/CD
                       |
                       v
                  DEPLOYMENT