FindMarketDriver — Requirements Specification

Version: 1.0
Date: 2026-09-30

1. PURPOSE

FindMarketDriver is an evidence-based stock analysis platform that combines:

Market data

Technical indicators

Financial news

Financial sentiment analysis

Event classification

Machine-learning prediction

Probability calibration

SHAP-based model attribution

Historical prediction evaluation

Interactive visualization

The system is intended to answer:

"Given the information available before the next trading session, what direction does the model predict, how confident is it, and which features contributed most to that prediction?"

The system must distinguish model attribution from causal claims about why a stock moved.

2. PRODUCT SCOPE

2.1 Core functionality

The completed platform shall allow a user to:

Search for a stock ticker.

View historical market data.

View technical indicators.

View relevant historical/current news.

View financial-news sentiment.

View detected financial events.

Receive an UP/DOWN model prediction.

View a calibrated prediction probability.

View expected return where supported by the validated model.

View model-attributed drivers.

View historical prediction performance.

Inspect previous predictions and their realized outcomes.

3. NON-GOALS

The system shall NOT claim to:

Guarantee future stock prices.

Guarantee investment returns.

Establish that a news article caused a stock movement.

Provide causal inference merely from SHAP.

Present an uncalibrated classifier score as a trustworthy probability.

Present an unvalidated model as a production forecast.

The system is an analytical/predictive platform.

4. CURRENT BASELINE REQUIREMENTS

The existing technical-only pipeline must first establish an official benchmark.

The current audit found:

Technical-only models do not beat the mean/always-UP baselines.

The current production Lasso is a constant prediction.

Tree models overfit.

Historical news/NLP is not implemented.

The Flask backend is not implemented.

The React frontend is not implemented.

SHAP is not implemented.

Therefore, no dashboard or production prediction feature shall be built around the current constant model.

5. REQUIREMENT PRIORITY

Use:

P0 = Blocking / critical
P1 = Required
P2 = Important
P3 = Optional / later

6. DATA REQUIREMENTS

REQ-DATA-001 — Stock market data

Priority: P0

The system shall obtain daily market data containing at least:

Date
Open
High
Low
Close
Volume

The current implementation uses yfinance.

REQ-DATA-002 — Completed daily bars

Priority: P0

The prediction pipeline shall use completed daily trading bars.

An unfinished intraday bar shall not be treated as the daily close.

REQ-DATA-003 — Historical range

Priority: P1

The technical training pipeline should support multi-year historical data.

The current benchmark uses approximately ten years of AAPL data.

REQ-DATA-004 — Data validation

Priority: P0

Before feature generation, validate:

Date ordering

Duplicate dates

Missing values

Infinite values

Required columns

Minimum history

Ticker identity

Trading-day validity

REQ-DATA-005 — Reproducible raw data

Priority: P1

Raw source data should eventually be stored/snapshotted so that model training can be reproduced without depending entirely on a changing external data source.

7. TECHNICAL FEATURE REQUIREMENTS

REQ-FEATURE-001 — Technical indicators

Priority: P0

The system shall support technical features including:

Returns

Return_1
Return_2
Return_3
Return_5
Return_10
Log_Return

Trend

Close_to_SMA_7
Close_to_SMA_30
Close_to_EMA_12
Close_to_EMA_26

MACD

MACD_Norm
MACD_Signal_Norm
MACD_Histogram_Norm

Momentum

RSI

Bollinger Bands

BB_Position
BB_Width

Volatility

Volatility

Volume

Volume_Change_1
Volume_Change_2
Volume_Change_5

REQ-FEATURE-002 — Feature cleanup

Priority: P0

The technical feature set shall not include:

Dividends
Stock Splits
Daily_Return

where these are not part of the intended predictive feature definition.

REQ-FEATURE-003 — Feature naming

Priority: P1

Features previously named:

Return_Lag_1
Return_Lag_2
...

shall be renamed because they represent cumulative k-day returns rather than one-day lag values.

Use:

Return_1
Return_2
Return_3
Return_5
Return_10

Similarly:

Volume_Change_1
Volume_Change_2
Volume_Change_5

REQ-FEATURE-004 — No target leakage

Priority: P0

No feature may contain information unavailable at prediction time.

All rolling features must use only information available on or before the prediction timestamp.

8. TARGET REQUIREMENTS

REQ-TARGET-001 — Product target

Priority: P0

The primary product target shall eventually be:

UP / DOWN

Definition:

future_return = Close[t+1] / Close[t] - 1

target = 1 if future_return > 0 else 0

Implemented (Phase 3): training/targets.py, target_version direction_v1.
For horizon h: future_return = Close[t+h] / Close[t] - 1.

REQ-TARGET-003 — Zero-return boundary

Priority: P0

An exactly-zero future return shall be labelled DOWN (0). No dead zone
and no epsilon shall be applied, because excluding small moves would
select rows using the future outcome. Rows without a future close
shall receive no label. Evaluation reports shall state how many rows
were labelled DOWN because their return was exactly zero.

REQ-TARGET-002 — Regression target

Priority: P2

Next-day return regression may remain available for experimentation.

Definition:

Target = Close[t+1] / Close[t] - 1

Regression must not automatically become the product-facing prediction.

9. EVALUATION REQUIREMENTS

REQ-EVAL-001 — Central evaluation harness

Priority: P0

A single evaluation harness shall evaluate every candidate model and feature set.

Required file:

training/evaluation_harness.py

REQ-EVAL-002 — Walk-forward validation

Priority: P0

Primary validation shall be chronological.

Required configuration:

TimeSeriesSplit
n_splits = 20
gap = prediction horizon

REQ-EVAL-003 — Regression baselines

Priority: P0

Regression experiments shall include:

Mean-return baseline
Zero-return baseline

REQ-EVAL-004 — Classification baselines

Priority: P0

Classification experiments shall include:

Always-UP baseline
Base-rate probability baseline

REQ-EVAL-005 — Regression metrics

Priority: P0

Report:

MAE
RMSE
R²

Metrics shall be calculated on pooled out-of-sample predictions.

REQ-EVAL-006 — Classification metrics

Priority: P0

Report:

Accuracy
Balanced Accuracy
ROC-AUC
Brier Score
Log Loss

REQ-EVAL-007 — Fold-level results

Priority: P1

The evaluation system should retain fold-level results so performance variation over time can be inspected.

REQ-EVAL-008 — Pooled out-of-sample results

Priority: P0

Final benchmark metrics must be calculated using predictions generated without training on their corresponding test observations.

REQ-EVAL-009 — Feature-selection isolation

Priority: P0

Any feature selection must occur independently inside each training fold.

Forbidden:

Entire dataset
      ↓
Feature selection
      ↓
Cross-validation

Required:

Training fold
      ↓
Feature selection
      ↓
Model
      ↓
Test fold

REQ-EVAL-010 — Model qualification gate

Priority: P0

A candidate model shall not be considered production-qualified unless it demonstrates improvement over the appropriate baseline under the official evaluation procedure.

If no model qualifies:

No production model

is the correct result.

10. MODEL REQUIREMENTS

REQ-ML-001 — Initial models

Priority: P0

The classification benchmark shall include:

Logistic Regression
Random Forest
XGBoost
LightGBM

Ridge may remain in regression experiments.

REQ-ML-002 — Model versioning

Priority: P1

Every production model must have a version.

Example:

model_version = v1.0

REQ-ML-003 — Model metadata

Priority: P1

Store:

Model name
Model version
Training date
Dataset version
Feature version
Target definition
Training period
Validation configuration
Metrics
Python version
Dependency versions

REQ-ML-004 — No constant-model promotion

Priority: P0

A model that collapses to a constant baseline shall not be presented as a meaningful production prediction.

11. PROBABILITY REQUIREMENTS

REQ-PROB-001 — Probability calibration

Priority: P0

Production classification probabilities shall be calibrated.

Potential methods:

Platt scaling
Isotonic regression

Selection shall be based on time-aware evaluation.

REQ-PROB-002 — Probability metrics

Priority: P0

Evaluate probabilities using:

Brier Score
Log Loss
Calibration / reliability analysis

REQ-PROB-003 — UI labeling

Priority: P0

The frontend shall label the result as:

Model probability

or equivalent.

It shall not imply certainty.

12. NEWS REQUIREMENTS

REQ-NEWS-001 — Historical news source

Priority: P0

Before implementing the final news pipeline, verify availability of multi-year, timestamped financial news for the target tickers.

REQ-NEWS-002 — Raw article schema

Priority: P1

Store at least:

article_id
ticker
headline
text
source
url
published_at
assigned_trading_date

REQ-NEWS-003 — Deduplication

Priority: P1

Duplicate articles must be identified and removed or linked.

REQ-NEWS-004 — Timestamp normalization

Priority: P0

Article timestamps shall be normalized to a documented timezone.

The system shall correctly handle:

Market hours

After-market news

Weekends

Market holidays

REQ-NEWS-005 — Trading-date assignment

Priority: P0

Each article shall be assigned to the trading session during which it becomes eligible as model information.

Default after-close rule:

Before/equal cutoff:
    current trading date

After cutoff:
    next trading session

Weekend/holiday news shall roll forward to the next trading session.

The eligibility time shall be information_available_at (REQ-NEWS-008),
not the raw publication or update time.

REQ-NEWS-006 — Provider abstraction

Priority: P0

News shall be obtained through a NewsProvider interface. Alpaca
(Benzinga) is the primary and canonical historical provider. Alpha
Vantage NEWS_SENTIMENT (reference) and NewsData.io (optional) may be
added later as providers. Providers shall NOT be merged into the
training dataset, and no provider-specific ML features shall exist.

REQ-NEWS-007 — Canonical news schema

Priority: P0

Every provider shall return the canonical schema (ARCHITECTURE.md
11.2): provider, provider_article_id, headline, summary, content,
symbols, source, source_url, created_at, updated_at,
information_available_at, fetched_at, provider_metadata. Timestamps
shall be timezone-aware UTC; naive timestamps shall be rejected. The
ML-facing view shall exclude provider-specific fields.

REQ-NEWS-008 — Information availability

Priority: P0

created_at, updated_at, fetched_at and information_available_at shall
be stored separately. updated_at shall never silently replace
created_at. For historical backfill, information_available_at =
max(created_at, updated_at) (rule historical_backfill_v1).

REQ-NEWS-010 — Prediction-time eligibility

Priority: P0

An article shall inform a prediction only if
information_available_at <= prediction_timestamp. Calendar-date or
session matching shall never grant eligibility. For a direction_v1
row t, prediction_timestamp = completion time of bar t (D_t 16:30
America/New_York); the prediction horizon shall not change it.
Articles with information_available_at > fetched_at shall be rejected.
All timestamps shall be timezone-aware; UTC internally, exchange-local
only for interpretation.

REQ-NEWS-011 — Trading sessions for news

Priority: P0

Trading sessions shall be derived from completed daily bars (no
weekday-only heuristics). Dates outside the known sessions shall raise
an error rather than be guessed. News after the 16:00 close, on
weekends or on holidays shall be assigned to the next known session.
Session assignment is descriptive and shall not decide eligibility.

REQ-NEWS-012 — Symbol safety

Priority: P0

News shall be available only to symbols listed in the article's
canonical `symbols`; no symbol shall be inferred or added.

REQ-NEWS-013 — Historical ingestion runs

Priority: P0

Multi-period history shall be ingested in bounded chunks through the
NewsProvider interface, one verified snapshot per chunk, tied together
by a run manifest written only after every chunk succeeded. Errors
shall propagate; results shall never be silently truncated.

REQ-NEWS-014 — Canonical news dataset

Priority: P0

The dataset shall be built only from verified snapshots and shall:
keep articles with start <= created_at < end; keep articles whose
canonical symbols contain a requested symbol (symbols never altered or
inferred); deduplicate by (provider, provider_article_id) independently
of input order (latest information_available_at, then fetched_at, then
smallest serialized record; never by headline similarity); sort
deterministically; and record its rules, counts, source hashes and
sha256. The same snapshots shall produce byte-identical output.

REQ-NEWS-015 — Interval is not eligibility

Priority: P0

The ingestion interval selects which published articles are stored.
It shall never be used as prediction eligibility, which remains
information_available_at <= prediction_timestamp (REQ-NEWS-010).
information_available_at shall be stored unchanged.

REQ-NEWS-009 — Ingestion provenance and safety

Priority: P0

Historical ingestion shall record provider, request parameters
(excluding credentials), retrieval time, page count, raw/duplicate
counts and a sha256 of the stored snapshot. Credentials shall come
only from ALPACA_API_KEY / ALPACA_API_SECRET and shall never be
logged, printed, stored or included in errors. Pagination shall be
bounded (repeated-token detection and a page cap). Unit tests shall
not require network access or credentials.

13. FINBERT REQUIREMENTS

REQ-NLP-001 — Pretrained model

Priority: P1

Initially use a pretrained financial sentiment model rather than fine-tuning.

REQ-NLP-002 — Sentiment outputs

Priority: P1

For each article generate:

positive_probability
neutral_probability
negative_probability
sentiment_score

Implemented for single articles in Phase 5A (ARCHITECTURE.md 13.1).

REQ-NLP-004 — FinBERT inference contract

Priority: P0

Model ProsusAI/finbert, inference only, loaded once, eval mode, no
gradients, CPU by default (CUDA optional, never required). Input text
follows text_v1: headline + summary, content only as a fallback, never
metadata; empty text is an error. Truncation by tokens at the model
maximum. sentiment_score = positive_probability - negative_probability.
Probabilities shall be validated (range, sum = 1, label = argmax).
Every result shall record inference version, model name, resolved
model revision and a sha256 of the input text. Model weights shall
never be committed; secrets shall never appear in errors or logs.

REQ-NLP-005 — Offline unit tests

Priority: P0

Unit tests shall not download models or need network access; the real
model shall be exercised only by opt-in integration tests.

REQ-NLP-003 — Daily sentiment aggregation

Priority: P1

Aggregate by ticker and trading day:

news_count
mean_sentiment
positive_news_count
negative_news_count
max_positive_sentiment
max_negative_sentiment
recent_news_count

Phase 5B status: implemented as sentiment_features_v1 (ARCHITECTURE.md
14.1) except recent_news_count, which needs a time window (not in v1).

REQ-NLP-006 — Leakage-safe sentiment features (sentiment_features_v1)

Priority: P0

For each (symbol, completed-bar trading date D), features shall use only
articles with symbol in their canonical symbols and
information_available_at <= prediction_timestamp(D) = D 16:30
America/New_York (REQ-NEWS-010, reusing the Phase 4B implementation).
Features: news_count; positive/negative/neutral count and ratio;
mean_sentiment; population sentiment_std; mean and max of each class
probability. With no eligible news every feature is 0 (never NaN).
Semantics are cumulative (no windows in v1). Sentiment results shall be
re-validated and matched to their article by input_text_hash; identical
duplicates collapse, conflicting duplicates raise. Output shall be
deterministic, independent of input order and of the prediction
horizon; prediction timestamps shall remain timezone-aware. No
persistence or feature store in this phase.

REQ-NLP-007 — Historical sentiment dataset

Priority: P0

A canonical news dataset shall be scored once per unique article with
the Phase 5A service and stored as article-level records
(sentiment_records_v1: identity, symbols, information_available_at,
probabilities, label, sentiment_score, input_text_hash and full
provenance), in deterministic order with a sha256 and metadata linking
the source dataset hash. A dataset shall contain exactly one
(model_name, model_revision, inference_version, text_policy); mixing or
missing provenance shall fail. Reloading shall verify both hashes and
re-validate every record against its canonical article (text hash,
identity, availability, symbols). Existing output with different
content shall never be overwritten.

REQ-NLP-008 — Daily sentiment dataset

Priority: P0

The daily dataset shall be produced only by the Phase 5B aggregation
(REQ-NLP-006) on completed-bar sessions from a verified market snapshot,
from re-validated sentiment records, and shall record its sources and
sha256. Sentiment is an NLP signal and shall not be presented as a
causal explanation of price movements.

14. EVENT CLASSIFICATION REQUIREMENTS

REQ-EVENT-001 — Event taxonomy

Priority: P2

Support financial event categories such as:

Earnings
Product
Legal
Management
Macro
Analyst
Regulatory
M&A

The final taxonomy must be documented before production use.

Phase 6: taxonomy_v1 documented in ARCHITECTURE.md 14.3 and defined only
in services/event_taxonomy.py.

REQ-EVENT-002 — Event aggregation

Priority: P2

Aggregate event counts/features by ticker and trading date.

Phase 6: event_features_v1 (ARCHITECTURE.md 14.3).

REQ-EVENT-004 — Event classification contract

Priority: P0

Each unique article shall receive exactly one event_type from the
central taxonomy, an event_confidence in [0, 1] documented as what it
is (rules_v1: share of rule evidence, not a probability), the evidence
used (matched_rules), and an event_impact taken from its FinBERT label -
type and impact are separate. Classification shall be deterministic,
replaceable behind EventClassifier, and recorded with classifier name,
version and taxonomy version.

REQ-EVENT-005 — Leakage-safe event features

Priority: P0

Daily event features for (symbol, D) shall use only articles listing
the symbol with prev(D) < information_available_at <= D 16:30 New York
(Phase 4B rule as the upper bound, calendar-derived lower bounds);
duplicates shall not inflate counts; zero windows are all 0 / NONE; no
dependence on the prediction horizon.

REQ-EVENT-006 — No fabricated ground truth

Priority: P0

Classifier output shall never be reported as accuracy. Supervised
metrics (precision, recall, F1, macro/weighted F1, confusion matrix,
baselines) shall be computed only from human labels, and small samples
shall be reported as descriptive only.

REQ-EVENT-003 — Incremental evaluation

Priority: P0

Event features shall be evaluated through the same central evaluation harness.

They shall not automatically be added to production.

15. FEATURE STORE REQUIREMENTS

REQ-STORE-001 — Versioned features

Priority: P1

Combined feature datasets shall be versioned.

Example:

feature_version = v1.0

REQ-STORE-002 — Combined feature table

Priority: P1

The final daily feature representation should combine:

Technical features
+
Sentiment features
+
Event features

REQ-STORE-003 — Feature reproducibility

Priority: P1

Given the same source data and feature version, the feature pipeline should reproduce the same feature values.

Phase 7: implemented as feature_store_v1 (ARCHITECTURE.md 15.1);
REQ-STORE-001..003 satisfied for the available data.

REQ-STORE-004 — Grain, keys and join

Priority: P0

One row per (symbol, trading_date), the existing prediction grain.
Duplicate or null keys shall be detected before joining and raise;
every key shall be a calendar session; sentiment/event prediction
timestamps shall equal the canonical D 16:30 America/New_York; joins
are 1:1.

REQ-STORE-005 — Missing vs no observations

Priority: P0

Inside a family's coverage its own zero convention applies and every
session must be present (a gap is an error); outside coverage values are
NaN with an explicit coverage flag. No blanket zero imputation.

REQ-STORE-006 — Targets and leakage

Priority: P0

Targets keep the existing definitions (next-day return; direction_v1),
are verified against Close, and never appear in feature families.
Feature values for an earlier date shall not change when later news is
added.

REQ-STORE-007 — Incremental evaluation

Priority: P0

Technical / +sentiment / +events shall be evaluated only through the
central harness on identical rows, targets, splits, candidates and gate.
Below the minimum evaluable sample the result shall be reported as
INSUFFICIENT_DATA without metrics.

16. EXPERIMENT REQUIREMENTS

Every feature experiment shall follow:

Experiment
    ↓
Dataset version
    ↓
Feature version
    ↓
Evaluation harness
    ↓
Baseline comparison
    ↓
Results

Required initial experiments:

Experiment A

Technical only

Experiment B

News sentiment only

Experiment C

Technical + sentiment

Experiment D

Technical + sentiment + events

A new feature group shall be retained only if the evaluation evidence supports its usefulness.

17. EXPLAINABILITY REQUIREMENTS

REQ-XAI-001 — SHAP

Priority: P1

After a validated production model exists, implement SHAP-based feature attribution.

REQ-XAI-002 — Feature contribution

Priority: P1

For every prediction, where supported, provide:

Feature
Contribution
Direction

REQ-XAI-003 — Driver ranking

Priority: P1

Rank the most influential model features for the current prediction.

REQ-XAI-004 — Causal disclaimer

Priority: P0

The system shall explicitly distinguish:

Model attribution

from:

Causal explanation

SHAP shall not be described as proving that a factor caused the market movement.

18. MARKET DRIVER REQUIREMENTS

REQ-DRIVER-001 — Driver summary

Priority: P1

Generate a readable explanation such as:

The model's UP prediction is primarily associated with:

- Positive recent news sentiment
- Improving momentum
- Lower recent volatility

REQ-DRIVER-002 — Attribution wording

Priority: P0

Use language such as:

"contributed to the model prediction"
"model-attributed factor"
"feature contribution"

Avoid:

"caused the stock to rise"
"guarantees the stock will rise"

19. BACKEND REQUIREMENTS

REQ-API-001 — Backend framework

Priority: P1

Use Flask for the initial backend.

REQ-API-002 — Health endpoint

Priority: P1

Implement:

GET /api/health

REQ-API-003 — Stock endpoint

Priority: P1

Implement:

GET /api/stocks/<ticker>

REQ-API-004 — Historical data endpoint

Priority: P1

Implement:

GET /api/stocks/<ticker>/history

REQ-API-005 — News endpoint

Priority: P1

Implement:

GET /api/news/<ticker>

REQ-API-006 — Prediction endpoint

Priority: P1

Implement:

GET /api/predictions/<ticker>

REQ-API-007 — Drivers endpoint

Priority: P1

Implement:

GET /api/drivers/<ticker>

REQ-API-008 — Evaluation endpoint

Priority: P2

Implement:

GET /api/evaluation/<ticker>

20. API PREDICTION RESPONSE

The prediction endpoint should eventually return:

{
  "ticker": "AAPL",
  "as_of": "2026-09-29",
  "direction": "UP",
  "probability": 0.64,
  "expected_return": 0.008,
  "model_version": "v1.0",
  "feature_version": "v1.0"
}

21. API DRIVER RESPONSE

Example:

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

22. DATABASE REQUIREMENTS

REQ-DB-001

Priority: P1

Use PostgreSQL for persistent application storage.

REQ-DB-002 — Required entities

Eventually support:

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

REQ-DB-003 — Prediction storage

Each prediction should store:

ticker
as_of_timestamp
bar_date
model_version
feature_version
prediction
probability
expected_return

REQ-DB-004 — Outcome storage

After the next trading day, store:

prediction_id
actual_return
actual_direction
evaluated_at

23. LIVE EVALUATION REQUIREMENTS

REQ-LIVE-001

Priority: P1

The system shall record live predictions.

REQ-LIVE-002

Priority: P1

Each prediction shall be linked to its later realized outcome.

REQ-LIVE-003

Priority: P1

The system shall calculate historical live metrics only after sufficient observations exist.

A single prediction is not sufficient evidence of model performance.

24. FRONTEND REQUIREMENTS

REQ-UI-001 — Stock search

Priority: P1

Provide a stock ticker search.

REQ-UI-002 — Stock overview

Priority: P1

Display:

Ticker
Current/latest completed price
Daily change

REQ-UI-003 — Candlestick chart

Priority: P1

Display historical price movement.

REQ-UI-004 — Prediction card

Priority: P1

Display:

UP / DOWN
Model probability
Expected return if available
Model version
As-of date

REQ-UI-005 — News timeline

Priority: P1

Display:

Headline
Source
Timestamp
Sentiment
Event

REQ-UI-006 — Driver visualization

Priority: P1

Display model-attributed drivers using a suitable chart.

REQ-UI-007 — Historical evaluation

Priority: P2

Display:

Accuracy
AUC
Brier Score
Recent prediction outcomes

25. STORAGE/CACHING REQUIREMENTS

REQ-CACHE-001

Priority: P2

Redis may be introduced for:

Stock data caching

Latest prediction caching

API rate limiting

Frequently requested resources

Redis is not required for the first ML milestone.

26. SECURITY REQUIREMENTS

REQ-SEC-001

Priority: P0

Secrets shall never be committed to Git.

REQ-SEC-002

Priority: P0

Use:

.env

for local secrets.

Provide:

.env.example

without secret values.

REQ-SEC-003

Priority: P1

API keys must be accessed through environment variables/configuration.

REQ-SEC-004

Priority: P1

Production APIs should implement appropriate:

Input validation

Error handling

CORS policy

Rate limiting

Logging

27. DEPENDENCY REQUIREMENTS

The project must maintain reproducible dependencies.

The environment must explicitly record packages actually used.

Expected packages may include:

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

Packages shall be added when their corresponding phase begins rather than installing the entire future stack immediately.

Python version must be documented.

28. TEST REQUIREMENTS

REQ-TEST-001 — Unit tests

Priority: P1

Test:

Feature calculations
Target construction
News date assignment
News deduplication
Sentiment aggregation
Prediction schema
Probability calibration

REQ-TEST-002 — ML tests

Priority: P0

Test:

No future leakage
Chronological ordering
Target alignment
Fold isolation
Feature schema
Baseline calculation

REQ-TEST-003 — Integration tests

Priority: P1

Test:

Data
 ↓
Features
 ↓
Model
 ↓
Prediction

REQ-TEST-004 — API tests

Priority: P1

Test:

GET /api/health
GET /api/stocks/AAPL
GET /api/stocks/AAPL/history
GET /api/predictions/AAPL
GET /api/drivers/AAPL

29. REPRODUCIBILITY REQUIREMENTS

REQ-REP-001

Priority: P0

A fresh clone shall be able to install the documented dependencies.

REQ-REP-002

Priority: P1

Training metadata must identify:

Dataset
Features
Target
Model
Training period
Validation method
Metrics
Environment

REQ-REP-003

Priority: P1

Model artifacts must not be the only source of model identity.

Metadata must accompany every production model.

30. DEPLOYMENT REQUIREMENTS

REQ-DEP-001

Priority: P2

The project should eventually support Docker.

Target:

Backend
Frontend
PostgreSQL
Redis

REQ-DEP-002

Priority: P2

The project should support automated CI checks.

Pipeline:

Git Push
   ↓
Lint
   ↓
Unit Tests
   ↓
ML Sanity Tests
   ↓
Backend Build
   ↓
Frontend Build
   ↓
Docker Build

31. DOCUMENTATION REQUIREMENTS

The repository shall contain:

README.md
ARCHITECTURE.md
REQUIREMENTS.md
docs/

Recommended documentation:

docs/data-contracts.md
docs/api-contracts.md
docs/ml-pipeline.md
docs/news-pipeline.md
docs/explainability.md
docs/deployment.md

32. GIT REQUIREMENTS

Each completed phase should be committed separately.

Example:

feat: add evaluation harness
fix: clean technical feature schema
feat: add classification baseline
feat: add historical news pipeline
feat: add FinBERT sentiment
feat: add event classification
feat: add probability calibration
feat: add SHAP explanations
feat: add Flask prediction API
feat: add React dashboard

Do not commit:

.env
.venv/
large temporary files
secret keys
unnecessary model artifacts
scratchpad files

33. PHASE CHECKLIST

PHASE 1 — Evaluation Harness

Status: IMPLEMENTED (2026-09-30) — pending review/commit

[x] Clean technical features (technical_v2)
[x] Create evaluation_harness.py
[x] Add TimeSeriesSplit(20)
[x] Add gap=horizon
[x] Add mean baseline
[x] Add zero baseline
[x] Add always-UP baseline
[x] Add Logistic Regression
[x] Add pooled OOS metrics
[x] Add feature-selection isolation
[x] Save dated results
[x] Establish official technical baseline
    (data/results/evaluation/technical_baseline_20260930.json:
     NO QUALIFIED MODEL for regression and classification)

PHASE 2 — Data Pipeline Hardening

Status: COMPLETE (commit e132c2c; 93 unit + 2 integration tests passed,
        deterministic snapshot rebuild verified)
        (policies: ARCHITECTURE.md section 6.2)

[x] Completed-bar validation
[x] Minimum-history validation
[x] Consistent yfinance schema
[x] Fix fallback behavior
[x] Remove/fix stale app.py (placeholder reserved for Flask)
[x] Fix Config/config case (requires the user's git index fix)
[x] Fix requirements.txt
[x] Fix live evaluation adjustment issue
[x] Add raw snapshots

PHASE 3 — Classification

Status: COMPLETE (commit 73d01bb)

[x] UP/DOWN target (training/targets.py, direction_v1, REQ-TARGET-003)
[x] Logistic Regression
[x] Random Forest
[x] XGBoost
[x] LightGBM
[x] Classification evaluation (python -m training.train_classification)
[ ] Probability calibration — reliability analysis implemented; fitting
    Platt/isotonic calibration is deferred until a model qualifies
    (REQ-PROB-001 applies to production probabilities; there is none)
[x] Brier score
[x] Log loss

PHASE 4 — Historical News

Status: 4A COMPLETE (c9184e0); 4B COMPLETE (a16920f); 4C COMPLETE (c896c2b)

[x] Verify source coverage (manual: Alpaca AAPL news back to 2017-01)
[x] Obtain historical articles (4A: NewsProvider + AlpacaNewsProvider)
[x] Store raw articles (4A: hash-verified raw snapshots, local only)
[x] Normalize timestamps (4A: UTC, information_available_at)
[x] Deduplicate (4A: provider + provider_article_id)
[x] Assign trading dates (4B: services/news_alignment.py, REQ-NEWS-010/011)
[ ] Map tickers — partial (4B): canonical symbols only, never inferred
    (REQ-NEWS-012); alias / company-name mapping not implemented
[x] Chunked historical ingestion runs + manifest (4C, REQ-NEWS-013)
[x] Deterministic canonical news dataset (4C, REQ-NEWS-014/015)
[ ] Full 2017-present AAPL backfill (run by the user; data stays local)

PHASE 5 — FinBERT

Status: 5A COMPLETE (9dd32d2); 5B COMPLETE (d470e52);
        5C COMPLETE (846d3de)

[x] Load pretrained model (5A: ProsusAI/finbert, inference only)
[x] Process news — historical pipeline (5C, REQ-NLP-007); full 2017-present
    corpus not yet scored (user-run, data stays local)
[x] Generate sentiment (5A: article-level probabilities + score, REQ-NLP-004)
[x] Aggregate daily (5B: cumulative as-of-prediction aggregation, REQ-NLP-006)
[ ] Create sentiment features — v1 contract implemented in memory (5B);
    not yet joined to the market dataset / feature store; no windows yet
[ ] Evaluate sentiment-only
[ ] Evaluate technical + sentiment

PHASE 6 — Events

Status: COMPLETE (baseline, commit 6c2b7ab)

[x] Define taxonomy (taxonomy_v1)
[x] Build event classifier (baseline: KeywordEventClassifier rules_v1)
[x] Generate event labels (classifier output - NOT ground truth)
[x] Aggregate events (event_features_v1)
[ ] Human-labelled validation sample (template + tooling ready; needs a person)
[ ] Evaluate incremental value (REQ-EVENT-003, central harness)

PHASE 7 — Feature Store

Status: IMPLEMENTED — awaiting user test run, review and commit;
        incremental evaluation INSUFFICIENT_DATA on the current news sample

[x] Version feature schema (feature_store_v1 / feature_store_schema_v1)
[x] Store technical features (reused technical_v2)
[x] Store sentiment features (reused sentiment_features_v1, sentiment__*)
[x] Store event features (reused event_features_v1, event__*)
[x] Build combined feature table ((symbol, trading_date), REQ-STORE-004/005/006)
[x] Add reproducibility metadata (content hash, sources, quality report)
[ ] Incremental A/B/C evaluation with results — runner done (REQ-STORE-007);
    needs >= 252 sessions covered by every family (full news backfill)

PHASE 8 — Production Model

Status: NOT STARTED

[ ] Compare candidates
[ ] Baseline gate
[ ] Select qualified model
[ ] Save model
[ ] Save metadata
[ ] Version model

PHASE 9 — Explainability

Status: NOT STARTED

[ ] Install/configure SHAP
[ ] Generate SHAP values
[ ] Rank feature contributions
[ ] Generate driver summary
[ ] Add attribution wording

PHASE 10 — Backend

Status: NOT STARTED

[ ] Flask app
[ ] Health endpoint
[ ] Stock endpoint
[ ] History endpoint
[ ] News endpoint
[ ] Prediction endpoint
[ ] Drivers endpoint
[ ] Evaluation endpoint
[ ] Error handling
[ ] Validation

PHASE 11 — Database

Status: NOT STARTED

[ ] PostgreSQL
[ ] Schema
[ ] Migrations
[ ] News storage
[ ] Feature storage
[ ] Model versions
[ ] Predictions
[ ] Prediction outcomes
[ ] Driver attributions

PHASE 12 — Frontend

Status: NOT STARTED

[ ] React setup
[ ] Stock search
[ ] Stock overview
[ ] Candlestick chart
[ ] Prediction card
[ ] Probability display
[ ] News timeline
[ ] Sentiment display
[ ] Event display
[ ] Driver visualization
[ ] Historical evaluation

PHASE 13 — Live System

Status: NOT STARTED

[ ] Scheduler
[ ] Completed-bar check
[ ] Daily feature generation
[ ] Prediction generation
[ ] Prediction storage
[ ] Next-day scoring
[ ] Monitoring

PHASE 14 — Deployment

Status: NOT STARTED

[ ] Dockerfile
[ ] Docker Compose
[ ] Environment configuration
[ ] CI tests
[ ] Build pipeline
[ ] Deployment
[ ] Monitoring

34. DEFINITION OF DONE

The project is considered functionally complete when:

[ ] User can search for a stock
[ ] User can view price history
[ ] User can view technical indicators
[ ] User can view relevant news
[ ] User can view sentiment
[ ] User can view events
[ ] User receives UP/DOWN prediction
[ ] Probability is calibrated
[ ] Model-attributed drivers are displayed
[ ] Historical prediction performance is available
[ ] Predictions are stored
[ ] Outcomes are scored
[ ] Model versions are tracked
[ ] Data pipeline is reproducible
[ ] Evaluation prevents temporal leakage
[ ] Backend API is tested
[ ] Frontend is tested
[ ] Deployment is reproducible

35. CURRENT PROJECT STATUS

As of 2026-09-30:

Architecture document       COMPLETE
Requirements document       COMPLETE

Stock ingestion             WORKING (Phase 2 hardening, pending verification)
Technical features          WORKING (technical_v2)
Target creation             WORKING
Regression experiments      WORKING
Classification              EXPERIMENTAL
Evaluation harness          BUILT (Phase 1)
Historical news             MISSING
FinBERT                     MISSING
Event classifier             BASELINE (rules_v1, Phase 6; no labelled evaluation yet)
Feature store               IMPLEMENTED (feature_store_v1; evaluation awaits full news backfill)
Probability calibration    MISSING
SHAP                        MISSING
Market drivers              MISSING
Flask API                   MISSING
PostgreSQL                  MISSING
React frontend              MISSING
Live production system      MISSING
Docker                      MISSING
CI/CD                       MISSING

36. IMMEDIATE NEXT REQUIREMENT

The next implementation task is:

REQ-EVAL-001

Create:

training/evaluation_harness.py

The first successful milestone is:

Clean technical features
        ↓
20-fold walk-forward validation
        ↓
Baselines
        ↓
Candidate models
        ↓
Pooled OOS metrics
        ↓
Official technical baseline

Do not move to FinBERT, SHAP, Flask, or React until this baseline exists.

37. MASTER DEVELOPMENT ORDER

PHASE 1
Evaluation Harness
       ↓
PHASE 2
Data Hardening
       ↓
PHASE 3
UP/DOWN Classification
       ↓
PHASE 4
Historical News
       ↓
PHASE 5
FinBERT
       ↓
PHASE 6
Event Classification
       ↓
PHASE 7
Combined Feature Store
       ↓
PHASE 8
Validated Production Model
       ↓
PHASE 9
Probability Calibration
       ↓
PHASE 10
SHAP + Driver Attribution
       ↓
PHASE 11
Flask API
       ↓
PHASE 12
PostgreSQL
       ↓
PHASE 13
React Frontend
       ↓
PHASE 14
Live Evaluation
       ↓
PHASE 15
Docker + CI/CD
       ↓
DEPLOYMENT

38. CHANGE CONTROL

When a major architectural or ML decision changes:

Update ARCHITECTURE.md.

Update this REQUIREMENTS.md.

Add/update the relevant implementation checklist.

Run the affected tests.

Commit the change.

The architecture and requirements files are the project's source of truth for implementation order and scope.