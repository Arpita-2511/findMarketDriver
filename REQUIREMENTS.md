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

REQ-EVENT-002 — Event aggregation

Priority: P2

Aggregate event counts/features by ticker and trading date.

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

Status: IMPLEMENTED (2026-09-30) — awaiting user test run, review and commit
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

Status: NOT STARTED

[ ] UP/DOWN target
[ ] Logistic Regression
[ ] Random Forest
[ ] XGBoost
[ ] LightGBM
[ ] Classification evaluation
[ ] Probability calibration
[ ] Brier score
[ ] Log loss

PHASE 4 — Historical News

Status: NOT STARTED

[ ] Verify source coverage
[ ] Obtain historical articles
[ ] Store raw articles
[ ] Normalize timestamps
[ ] Deduplicate
[ ] Assign trading dates
[ ] Map tickers

PHASE 5 — FinBERT

Status: NOT STARTED

[ ] Load pretrained model
[ ] Process news
[ ] Generate sentiment
[ ] Aggregate daily
[ ] Create sentiment features
[ ] Evaluate sentiment-only
[ ] Evaluate technical + sentiment

PHASE 6 — Events

Status: NOT STARTED

[ ] Define taxonomy
[ ] Build event classifier
[ ] Generate event labels
[ ] Aggregate events
[ ] Evaluate incremental value

PHASE 7 — Feature Store

Status: NOT STARTED

[ ] Version feature schema
[ ] Store technical features
[ ] Store sentiment features
[ ] Store event features
[ ] Build combined feature table
[ ] Add reproducibility metadata

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
Event classifier             MISSING
Feature store               MISSING
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