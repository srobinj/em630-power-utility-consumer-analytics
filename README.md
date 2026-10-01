# AI-Based Consumer Behaviour Analytics, Revenue Risk and Consumption Anomaly Prediction for Power Distribution Utilities

EM630 supervised-learning project for next-month electricity-consumer analytics. It uses 36 public synthetic monthly extracts from April 2023 through March 2026 and produces April 2026 predictions for 2,087 consumers.

## Final prediction modules

| Module | Models compared | Official selected model | Selection evidence |
| --- | --- | --- | --- |
| Consumption Prediction | Persistence, Multiple Linear Regression, Random Forest Regressor | **MLR-4** | Lowest validation RMSE |
| Revenue Risk | TRAIN-majority baseline, Decision Tree, Random Forest | **RF-3-balanced** | Validation ROC-AUC with both-class performance under imbalance |
| Unusual Consumption Risk | TRAIN-majority baseline, Decision Tree, Random Forest | **DT-3-balanced** | Validation Class-1 F1 supported by anomaly recall and generalization stability |

Revenue Risk is a project-defined academic proxy: Class 1 means less than 50% of the bill was paid by its due date. It is not an official utility default classification.

Unusual Consumption Risk is a project-defined statistical proxy using the existing z = 2.5 rule. It is a decision-support indicator and does not establish theft, fraud, meter tampering or wrongdoing.

## Time-safe evaluation design

Feature-month information available through month *t* predicts exactly month *t+1*. The partitions use `TARGET_PERIOD` and are never shuffled:

- **TRAIN:** earliest eligible target through April 2025
- **VALIDATION:** May 2025 through October 2025
- **FINAL TEST:** November 2025 through March 2026
- **FUTURE:** April 2026, using March 2026 feature rows

Preprocessing is fitted on TRAIN only during model comparison. Model selection uses VALIDATION only. The frozen selected model and its baseline are evaluated once on FINAL TEST. The unchanged selected pipeline is then refitted on all eligible labelled history through March 2026 to generate April 2026 predictions. April actual outcomes are unavailable, so no April performance metric is calculated.

No XGBoost, PCA, Ridge, Lasso, Gradient Boosting, AdaBoost, KNN, neural network or ANOVA is used.

## Final held-out results

- **MLR-4:** FINAL TEST RMSE 934.70, MAE 130.24 and R² 0.8108, compared with persistence RMSE 972.04, MAE 141.15 and R² 0.7954.
- **RF-3-balanced:** FINAL TEST ROC-AUC 0.6370, Class-0 F1 0.2715 and Class-1 F1 0.8570, compared with majority-baseline ROC-AUC 0.5000.
- **DT-3-balanced:** FINAL TEST ROC-AUC 0.6733, Average Precision 0.1193, Class-1 recall 0.3204 and Class-1 F1 0.1307. The majority baseline detects no Class-1 cases.

These held-out results did not change model selection. Classification results must be read with both-class and ranking metrics because the targets are imbalanced.

## April 2026 predictions

The three production pipelines generate one April 2026 prediction per eligible consumer:

- predicted consumption from MLR-4;
- Revenue Risk indicator and Class-1 probability from RF-3-balanced;
- Unusual Consumption Risk indicator and Class-1 probability from DT-3-balanced.

The verified local outputs cover 2,087 unique consumers with no missing or duplicate consumer-period predictions. Consumer-level prediction files remain local and Git-ignored because they contain consumer lookup keys.

## Streamlit dashboard

Run `python -m streamlit run dashboard/app.py`. The dashboard reads existing verified outputs and does not train, refit, tune, optimize thresholds or select models.

It contains exactly four pages:

1. **Overview** — project method, data information, financial-year/month controls, historical KPIs and Plotly trends. It contains no April prediction summary.
2. **All Consumer Predictions** — April predictions, search and risk filters, filtered KPIs and CSV download.
3. **Consumer 360** — safe consumer attributes, historical actual trends and all three April predictions.
4. **Model Comparison & Evaluation** — separate VALIDATION comparisons and held-out FINAL TEST results, confusion matrices, ROC curves and the existing Step 4 statistical table.

The dashboard loads only safe operational fields such as `CONSUMER_NO`, tariff and feeder. It does not display consumer names, addresses, meter numbers or contact information. If ignored local processed files are absent from a public clone, affected pages show a clear message instead of fabricating data.

## Project structure

```text
dashboard/app.py          Final four-page Streamlit dashboard
src/                      Loading, cleaning, EDA, features, splits, training and evaluation
data/raw/                 Public synthetic monthly extracts
data/new/                 Local future monthly actuals; ignored except .gitkeep
data/processed/           Local generated datasets and row-level predictions; ignored
models/                   Local fitted pipelines and checkpoints; ignored
reports/figures/          Public-safe aggregate figures
reports/summaries/        Public-safe aggregate metrics, audits and summaries
reports/internal_validation/  Local detailed checks; ignored
Dataset/                  Original local source folder; ignored
```

## Installation and run sequence

From the project root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The completed scripts are retained for reproducibility:

```powershell
python src/data_loading.py
python src/cleaning.py
python src/eda.py
python src/features.py
python src/splits.py
python src/train_consumption.py
python src/train_revenue_risk.py
python src/train_anomaly.py
python src/evaluation.py
```

Do not rerun model-training scripts merely to launch the dashboard. They protect or overwrite modelling artifacts according to their documented workflow; use the already verified outputs for the final project presentation.

## Data, privacy and publication

The repository uses public synthetic/anonymized project data. The original `Dataset/` source folder, processed row-level data, consumer-level prediction files, fitted models, private anomaly ground truth, generation configuration, virtual environments, caches and internal validation files are excluded from Git.

`CONSUMER_NO` is retained only for grouping, joins, safe lookup and dashboard tracking. Names, addresses, meter numbers and other identifiers are not predictive features. Never substitute real utility data or publish consumer-identifiable operational information.

## Limitations

- Synthetic data support academic demonstration and do not establish operational utility performance.
- Temporal change, class imbalance, extreme values and proxy-label design limit generalization.
- Correlation and model association do not establish causation.
- Consumer-level aggregation used for inferential checks loses some month-to-month information.
- Revenue Risk is an academic payment-behaviour proxy.
- Unusual Consumption Risk is a statistical indicator, not evidence of theft, fraud or wrongdoing.
- April 2026 has predictions only; actual outcomes and April accuracy metrics are unavailable.
