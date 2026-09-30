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

1. Define event taxonomy
2. Build event classifier
3. Generate event labels
4. Aggregate daily
5. Evaluate incremental value

PHASE 7 — FINAL FEATURE STORE

Tasks:

1. Technical features
2. Sentiment features
3. Event features
4. Versioned combined dataset
5. Reproducible feature pipeline

PHASE 8 — FINAL MODEL

Tasks:

1. Compare candidate models
2. Select only if baseline is beaten
3. Calibrate probability
4. Save artifact
5. Save metadata

PHASE 9 — EXPLAINABILITY

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