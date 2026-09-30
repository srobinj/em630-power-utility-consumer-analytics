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
Features known through completed month t predict month t+1. Payment predictors require a PAYMENT_DATE on or before the last day of month t; later payments are unavailable to predictors. Train target months end Apr-2025; validation covers May-Oct 2025; final test covers Nov-2025 through Mar-2026. Model selection uses validation, followed by one final-test evaluation. Apr-2026 predictions are planned for later stages, with no actual-based metrics. Preprocessing will be fitted on training data only.

## Setup and placeholder run instructions
Run from the project root in PowerShell after installing Python with its launcher:
1. Create the environment: py -m venv .venv
2. Activate: .\.venv\Scripts\Activate.ps1
3. Install: python -m pip install -r requirements.txt
4. Starter dashboard: python -m streamlit run dashboard/app.py

Implemented data-processing and all three model-training commands are listed below.

## Implemented data-processing run sequence
With the existing environment active, run from the project root:

1. Step 2 (only when loading or refreshing approved public inputs): `python src/data_loading.py`
2. Step 3: `python src/cleaning.py`

For the existing Step 2 output, run only Step 3. It writes the ignored local `data/processed/consumer_cleaned.csv` and the aggregate `reports/summaries/cleaning_summary.csv`. Detailed inspection files stay under ignored `reports/internal_validation/step3/`. Review flagged values before later work; cleaning does not impute, create targets, split data or train models. Identifiers must be loaded as strings in later scripts. Implemented Steps 4 and 5 are described below.

## Step 4: exploratory data analysis
Run `python src/eda.py` with the existing environment active. It reads the cleaned CSV without modifying it and writes 15 descriptive figures and aggregate CSV summaries under reports/. These cover monthly consumption and consumer growth, distributions, billing/payment relationships, tariff/solar groups, calendar-month patterns and selected numeric correlations.

Full-period EDA is descriptive and does not select ML features. Two Pearson tests use one available-history average per consumer. The TARRIF versus SOLAR_CONSUMER Chi-square analysis excludes changing/invalid consumer histories and checks expected frequencies before testing; it is skipped for the current sparse table. No replacement tests are added. All assumptions, exclusions and limitations are recorded in statistical_tests.csv, chi_square_sample_checks.csv and eda_findings.csv. EDA creates no targets, lag features, splits or models.

## Step 5: time-safe features and next-month academic targets
Run `python src/features.py` using the existing cleaned CSV. Exact consumer/calendar joins create consumption lags, complete three-month summaries, frozen payment history and arrears history. A missing calendar month stays missing. `TARGET_PERIOD` is exactly t+1; March-2026 rows target April-2026 and use April-2025 consumption for `SAME_MONTH_LAST_YEAR`.

`FEATURE_CUTOFF_DATE` is month-end t. `PAYMENT_RATIO_CURRENT` and `PAYMENT_DELAY_CURRENT` use payments only when `PAYMENT_DATE` is present and on/before this cutoff. Future payments are excluded, not converted to zero. Historical payment ratios stay frozen at their own month-end cutoff. Original payment amounts, dates and due dates are retained for audit only.

`PAYMENT_RATIO_OUTCOME` is label-only: for a positive bill and valid `DUE_DATE`, use the on-time valid payment amount divided by the bill; no payment recorded by the deadline gives a label-only zero. An on-time payment with missing/invalid amount remains undefined. `AT_RISK_CURRENT` is 1 below a 0.50 outcome ratio and otherwise 0 when defined. It is an academic due-date proxy, not an official utility default classification. Both outcome columns are excluded from every predictor list.

The unusual-consumption proxy uses at least three valid strictly earlier observations and their sample standard deviation. A defined absolute z-score above 2.5 gives class 1; insufficient history or zero/invalid prior standard deviation gives no label. This is neither theft nor fraud ground truth. Current proxy labels and prior-statistic audit columns are not predictors. Exact t+1 joins create the three next-month targets; all 2,087 March rows retain unknown April targets.

Outputs: ignored local `data/processed/consumer_features.csv` and public aggregate `feature_dictionary.csv`, `target_availability_summary.csv`, `payment_resolution_summary.csv`, and `feature_validation_summary.csv` under reports/summaries/. Use the explicit candidate lists in features.py, never every numeric column in the master file. Step 5 creates no imputation, feature selection, split, model or April prediction.

Limitations: monthly consumption, bills, arrears, load and categories are treated as the supplied month-end snapshots; historical revision timestamps are unavailable. Due-date labels are retrospective and assume the supplied record adequately describes payment by its deadline; later modelling must consider label maturity/availability. The single payment record does not establish a full instalment ledger. Missing or invalid inputs remain explicit, and extreme values are not automatically removed.

## Step 6: chronological splits and TRAIN-only relevance
Run `python src/splits.py`. Each module keeps its own known-target rows, partitioned strictly by TARGET_PERIOD: TRAIN through April 2025, VALIDATION May–October 2025, FINAL TEST November 2025–March 2026. FUTURE retains all March-2026 feature rows targeting April-2026 with unknown targets. No shuffle is used. The 12 longitudinal partition CSVs remain ignored under data/processed/splits/; consumer_features.csv is unchanged. Public split_summary.csv records exclusions, target distributions and unchanged class imbalance. No resampling, balancing or class weighting is performed.

Only eligible TRAIN rows enter feature relevance. Consumption Pearson tests use one paired predictor/target mean per consumer. Consumption categories and the cyclic TARGET_MONTH receive descriptive group summaries. Classification numeric features receive descriptive monthly Class 0/Class 1 summaries without inferential tests. Classifier Chi-square samples require both the predictor category and target class to stay constant in the consumer's eligible TRAIN history; changing/invalid histories are excluded from that test only. All three current Chi-square tests are skipped because these stable subsets contain just one target class. No majority-class consumer label, category merging or replacement test is introduced.

feature_significance.csv records results and limitations; feature_relevance_contingencies.csv records observed/expected tables. feature_selection_rationale.csv retains all 36 module-specific planned candidates provisionally based on domain relevance, timing and leakage checks, with TRAIN evidence documented. Statistical significance does not select features or establish causal effects. Skewness, missingness, unequal histories, shared conditions and stable-subset selection limit inference. Later model validation must establish predictive usefulness. Step 6 itself performs no model training.

## Step 7: consumption model comparison and April prediction
`python src/train_consumption.py` reads the four existing Step 6 consumption partitions directly. It compares persistence, four prescribed ordinary Multiple Linear Regression feature sets, and four prescribed Random Forest configurations. Numeric median imputation and categorical imputation/one-hot encoding (including TARGET_MONTH) are inside each pipeline and fitted on TRAIN only during comparison. No scaling, extra algorithms or additional parameter search is used.

MLR-4 was selected before FINAL TEST evaluation because it had the lowest VALIDATION RMSE (921.19), with MAE 152.91. The preserved TRAIN-fitted model and persistence alone were evaluated on FINAL TEST: RMSE 934.70 versus 972.04; MAE 130.24 versus 141.15; R² 0.8108 versus 0.7954. Correlated and mathematically related predictors make individual linear coefficients unreliable to interpret; predictors were not automatically removed.

After evaluation, a fresh clone of the entire unchanged MLR-4 pipeline was fitted on all 71,491 labelled TRAIN + VALIDATION + FINAL TEST rows. It generated 2,087 finite April-2026 predictions, covering every FUTURE consumer. April actuals are unavailable, so no April accuracy metric is calculated. Predictions are not clipped; the current April predictions are all nonnegative.

models/consumption_frozen_train_pipeline.joblib preserves the evaluated model; models/consumption_april2026_pipeline.joblib holds the fully refitted prediction model. Comparison, selection, metrics, preprocessing audit and coverage summaries are under reports/summaries/, with two FINAL TEST figures under reports/figures/. Row-level FINAL TEST predictions remain in ignored data/processed/consumption_final_test_predictions.csv. The requested reports/summaries/next_month_consumption_predictions.csv is explicitly ignored because it contains consumer lookup keys. No names or addresses are included.

The script refuses to overwrite existing FINAL TEST results. It has already completed; do not rerun it to retest or retune. Use the saved pipelines and reports. Synthetic data, extreme residuals, correlated predictors and changes across months limit generalization.

## Step 8: academic revenue-risk model and April prediction

`python src/train_revenue_risk.py` reads the four existing Step 6 revenue partitions directly. It compares the TRAIN-majority baseline, four prescribed Decision Trees and four prescribed Random Forests. The strongest unweighted tree and forest are each tested once with `class_weight="balanced"`. Numeric median imputation and TARRIF imputation/one-hot encoding remain inside each pipeline and are fitted on TRAIN only during model comparison.

RF-3-balanced (200 trees, maximum depth 10, balanced class weights and random state 42) was selected using VALIDATION results. Its validation ROC-AUC is 0.6390, Class-0 recall is 0.3939 and Class-0 F1 is 0.2808. The weighting trade-off lowers validation accuracy and Class-1 F1 compared with the majority-like unweighted RF-3, while producing useful Class-0 predictions with nearly the same ROC-AUC. No threshold tuning was performed.

The frozen selected pipeline and TRAIN-majority baseline were evaluated once on FINAL TEST. RF-3-balanced has ROC-AUC 0.6370, Average Precision 0.9214, Class-0 F1 0.2715 and Class-1 F1 0.8570; the constant baseline has ROC-AUC 0.5000, Class-0 F1 0 and Class-1 F1 0.9391. Accuracy is 0.7609 versus 0.8853, illustrating why the imbalanced target is assessed with both-class and ranking metrics rather than accuracy alone.

After FINAL TEST evaluation, a fresh clone of the unchanged selected pipeline was fitted on 70,492 labelled TRAIN + VALIDATION + FINAL TEST rows. It generated 2,087 April-2026 predictions with Class-1 probabilities from 0.2000 to 0.8098 and 100% FUTURE-consumer coverage. April outcomes are unavailable, so no April metric is calculated. These outputs are an academic less-than-50%-paid-by-due-date proxy, not an official utility default or revenue-risk classification.

The saved audits verify Class-1 probability extraction from `classes_`, TRAIN-only preprocessing during selection and full-labelled-history preprocessing only for the April refit. The selected predictor allowlist excludes label-only `PAYMENT_RATIO_OUTCOME`, `AT_RISK_CURRENT` and the target. Row-level prediction outputs and fitted models remain ignored locally. The script protects preserved FINAL TEST outputs from accidental overwrite.

## Step 9: unusual-consumption-risk model and April prediction

`python src/train_anomaly.py` loads the four existing Step 6 anomaly partitions directly. It compares the TRAIN-majority baseline, four prescribed Decision Trees and four prescribed Random Forests. Validation Class-1 F1 selects DT-3 and RF-4 for exactly one balanced-weight experiment each. No resampling, additional model configuration or threshold search is used.

DT-3-balanced (maximum depth 8, balanced class weights and random state 42) is the official validation-selected model. Its validation Class-1 F1 is 0.2148, recall is 0.5238, precision is 0.1351, AP is 0.1609 and ROC-AUC is 0.7012. RF-4-balanced has a marginally higher validation F1 of 0.2306, but much lower anomaly recall of 0.2008 and severe overfitting: its TRAIN-to-validation F1/AP/ROC-AUC gaps are −0.7659/−0.8096/−0.2600, compared with −0.0470/−0.1034/−0.0976 for DT-3-balanced. This TRAIN/VALIDATION-only stability review applies the original multi-metric selection rule. FINAL TEST results did not influence the revision.

The frozen DT-3-balanced model and TRAIN-majority baseline were evaluated on FINAL TEST. DT-3-balanced has Class-1 F1 0.1307, recall 0.3204, precision 0.0821, AP 0.1193 and ROC-AUC 0.6733. The baseline predicts only Class 0, giving Class-1 F1 and recall of zero, AP 0.0298 and ROC-AUC 0.5000. Accuracy is 0.8731 for the selected model and 0.9702 for the baseline; both-class and ranking metrics are required because Class 1 is rare.

After evaluation, a fresh clone of the unchanged DT-3-balanced pipeline was fitted on all 67,324 labelled rows through target March 2026. It generated 2,087 April-2026 predictions with complete consumer coverage, 243 predicted Class-1 cases and probabilities from 0.00 to 1.00. April outcomes are unavailable, so no April accuracy metric is calculated. The previous RF-4-balanced evaluation and deployment artifacts are preserved locally under `data/processed/step9_rf4_balanced_history/`; official row-level outputs and fitted models remain ignored locally.

`ANOMALY_NEXT` is a project-defined statistical proxy for unusual consumption. Predictions are decision-support indicators and do not establish electricity theft, fraud, meter tampering, or wrongdoing. The z=2.5 proxy rule remains unchanged. The predictor allowlist excludes `ANOMALY_CURRENT`, `RULE_Z_CURRENT`, `ANOMALY_NEXT` and unavailable future information.
