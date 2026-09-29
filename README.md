# AI-Based Consumer Behaviour Analytics, Revenue Risk and Consumption Anomaly Prediction for Power Distribution Utilities

This EM630 supervised-learning project will use three years of synthetic monthly electricity billing and consumption data to analyse consumer behaviour, predict next-month consumption, revenue risk and unusual-consumption risk, and present model comparisons in an interactive Streamlit dashboard. Rolling updates will use new actual monthly records to evaluate preserved predictions and forecast the following month.

## Prediction modules
1. Next-month consumption: persistence baseline, Multiple Linear Regression and Random Forest Regressor.
2. Next-month revenue risk: majority-class baseline, Decision Tree Classifier and Random Forest Classifier.
3. Next-month unusual-consumption risk: majority-class baseline, Decision Tree Classifier and Random Forest Classifier. This is not theft or fraud prediction.

## Data and privacy
Public data are synthetic. The 36 monthly pipe-delimited TXT extracts span Apr-2023 through Mar-2026. Source names, including TARRIF, and file contents are preserved. Consumer counts will be calculated from files.

The original Dataset folder is preserved locally and excluded from Git. Private injected-anomaly ground truth and generation configuration must never be used for analysis, modelling, tuning, validation or publication. Only public monthly synthetic TXT extracts are copied into data/raw. Never publish real consumer identities or confidential operational data.

## Folder structure
- Dataset/: original local source, ignored and unchanged.
- data/raw/: public synthetic monthly TXT extracts, tracked.
- data/new/: future monthly actual files, local by default.
- data/processed/: generated local working data, ignored.
- src/: loading, cleaning, EDA, features, splits, training and evaluation.
- models/: local fitted models.
- dashboard/app.py: Streamlit entry point.
- reports/REPORT.md: report starter.
- reports/figures/: report figures.
- reports/summaries/: calculated dashboard outputs.
- reports/internal_validation/: local checks, ignored.

## Planned temporal workflow
Features known through completed month t predict month t+1. Month t payment information is assumed known at prediction time. Train target months end Apr-2025; validation covers May-Oct 2025; final test covers Nov-2025 through Mar-2026. Model selection uses validation, followed by one final-test evaluation. Apr-2026 initially has predictions only, with no actual-based metrics. Preprocessing is fitted on training data only.

## Setup and placeholder run instructions
Run from the project root in PowerShell after installing Python with its launcher:
1. Create the environment: py -m venv .venv
2. Activate: .\.venv\Scripts\Activate.ps1
3. Install: python -m pip install -r requirements.txt
4. Starter dashboard: python -m streamlit run dashboard/app.py

Implemented data-processing commands are listed below. Model commands will be added in later numbered steps; no modelling is implemented yet.

## Implemented data-processing run sequence
With the existing environment active, run from the project root:

1. Step 2 (only when loading or refreshing approved public inputs): `python src/data_loading.py`
2. Step 3: `python src/cleaning.py`

For the existing Step 2 output, run only Step 3. It writes the ignored local `data/processed/consumer_cleaned.csv` and the aggregate `reports/summaries/cleaning_summary.csv`. Detailed inspection files stay under ignored `reports/internal_validation/step3/`. Review flagged values before later work; cleaning does not impute, create targets, split data or train models. Identifiers must be loaded as strings in later scripts. Later numbered stages remain unimplemented.
