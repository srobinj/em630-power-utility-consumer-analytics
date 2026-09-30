"""Step 10: consolidate verified Step 7-9 evaluation outputs.

This script reads saved CSV/model artifacts only. It does not load fitted models,
retrain, tune, change thresholds, recreate splits or perform model selection.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "reports" / "summaries"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


def read_required(path):
    """Read one explicitly named CSV and fail clearly when it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}")
    return pd.read_csv(path)


def value(row, column):
    """Return an existing metric or NaN when it does not apply."""
    return row[column] if column in row.index else np.nan


def role_text(roles):
    """Combine roles when one configuration is both a family winner and selected."""
    return "; ".join(roles)


def classification_counts(row, prefix):
    """Use saved predicted counts, or sum the saved confusion counts."""
    predicted_zero = f"{prefix}predicted_class_0"
    predicted_one = f"{prefix}predicted_class_1"
    if predicted_zero in row.index and predicted_one in row.index:
        return row[predicted_zero], row[predicted_one]
    needed = [
        f"{prefix}true_0_predicted_0", f"{prefix}true_0_predicted_1",
        f"{prefix}true_1_predicted_0", f"{prefix}true_1_predicted_1",
    ]
    if all(column in row.index for column in needed):
        predicted_0 = row[needed[0]] + row[needed[2]]
        predicted_1 = row[needed[1]] + row[needed[3]]
        return predicted_0, predicted_1
    return np.nan, np.nan


def add_regression_row(rows, source, module, split, roles, selected, baseline, reason):
    """Put one saved consumption result into the common metrics structure."""
    prefix = f"{split}_" if split in ["TRAIN", "VALIDATION"] else ""
    rows.append({
        "module": module,
        "model": source["configuration"],
        "model_family": source.get("algorithm", source["configuration"]),
        "configuration": source["configuration"],
        "model_role": role_text(roles),
        "split": split,
        "selected_model": selected,
        "baseline": baseline,
        "sample_count": value(source, f"{prefix}sample_count"),
        "MAE": value(source, f"{prefix}MAE"),
        "RMSE": value(source, f"{prefix}RMSE"),
        "R2": value(source, f"{prefix}R2"),
        "Accuracy": np.nan,
        "Precision_0": np.nan,
        "Recall_0": np.nan,
        "F1_0": np.nan,
        "Precision_1": np.nan,
        "Recall_1": np.nan,
        "F1_1": np.nan,
        "ROC_AUC": np.nan,
        "AP": np.nan,
        "actual_class_0_count": np.nan,
        "actual_class_1_count": np.nan,
        "predicted_class_0_count": np.nan,
        "predicted_class_1_count": np.nan,
        "VALIDATION_minus_TRAIN_MAE": value(source, "VALIDATION_minus_TRAIN_MAE") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_RMSE": value(source, "VALIDATION_minus_TRAIN_RMSE") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_R2": value(source, "VALIDATION_minus_TRAIN_R2") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_Accuracy": np.nan,
        "VALIDATION_minus_TRAIN_ROC_AUC": np.nan,
        "VALIDATION_minus_TRAIN_AP": np.nan,
        "VALIDATION_minus_TRAIN_F1_0": np.nan,
        "VALIDATION_minus_TRAIN_F1_1": np.nan,
        "selection_reason": reason if selected else "",
        "metric_source": "Saved Step 7 comparison" if split != "FINAL_TEST" else "Saved pre-refit held-out FINAL TEST",
    })


def add_classification_row(rows, source, module, split, roles, selected, baseline, reason):
    """Put one saved classification result into the common metrics structure."""
    prefix = f"{split}_" if split in ["TRAIN", "VALIDATION"] else ""
    predicted_0, predicted_1 = classification_counts(source, prefix)
    rows.append({
        "module": module,
        "model": source["configuration"],
        "model_family": source.get("algorithm", source["configuration"]),
        "configuration": source["configuration"],
        "model_role": role_text(roles),
        "split": split,
        "selected_model": selected,
        "baseline": baseline,
        "sample_count": value(source, f"{prefix}sample_count"),
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Accuracy": value(source, f"{prefix}Accuracy"),
        "Precision_0": value(source, f"{prefix}Precision_0"),
        "Recall_0": value(source, f"{prefix}Recall_0"),
        "F1_0": value(source, f"{prefix}F1_0"),
        "Precision_1": value(source, f"{prefix}Precision_1"),
        "Recall_1": value(source, f"{prefix}Recall_1"),
        "F1_1": value(source, f"{prefix}F1_1"),
        "ROC_AUC": value(source, f"{prefix}ROC_AUC"),
        "AP": value(source, f"{prefix}AP"),
        "actual_class_0_count": value(source, f"{prefix}class_0_count"),
        "actual_class_1_count": value(source, f"{prefix}class_1_count"),
        "predicted_class_0_count": predicted_0,
        "predicted_class_1_count": predicted_1,
        "VALIDATION_minus_TRAIN_MAE": np.nan,
        "VALIDATION_minus_TRAIN_RMSE": np.nan,
        "VALIDATION_minus_TRAIN_R2": np.nan,
        "VALIDATION_minus_TRAIN_Accuracy": value(source, "VALIDATION_minus_TRAIN_Accuracy") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_ROC_AUC": value(source, "VALIDATION_minus_TRAIN_ROC_AUC") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_AP": value(source, "VALIDATION_minus_TRAIN_AP") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_F1_0": value(source, "VALIDATION_minus_TRAIN_F1_0") if split == "VALIDATION" else np.nan,
        "VALIDATION_minus_TRAIN_F1_1": value(source, "VALIDATION_minus_TRAIN_F1_1") if split == "VALIDATION" else np.nan,
        "selection_reason": reason if selected else "",
        "metric_source": "Saved model comparison" if split != "FINAL_TEST" else "Saved pre-refit held-out FINAL TEST",
    })


def verify_final_test(table, selected_name, baseline_name, module):
    """Require exactly the frozen selection and baseline on one held-out sample."""
    expected = {selected_name, baseline_name}
    if len(table) != 2 or set(table["configuration"]) != expected:
        raise ValueError(f"{module} FINAL TEST must contain only {expected}.")
    if not table["partition"].eq("FINAL_TEST").all():
        raise ValueError(f"{module} contains a non-FINAL_TEST result.")
    if table["sample_count"].nunique() != 1:
        raise ValueError(f"{module} selected model and baseline use different FINAL TEST rows.")


def prediction_audit(module, path, prediction_column, label_column=None, probability_column=None, probability_audit=None, official_model=None):
    """Audit one official April file without recalculating predictions."""
    frame = pd.read_csv(path, dtype={"CONSUMER_NO": str})
    key_columns = ["CONSUMER_NO", "TARGET_PERIOD"]
    duplicate_count = int(frame.duplicated(key_columns).sum())
    missing_predictions = int(frame[prediction_column].isna().sum())
    periods = sorted(frame["TARGET_PERIOD"].dropna().astype(str).unique().tolist())
    row = {
        "module": module,
        "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(frame),
        "unique_consumers": frame["CONSUMER_NO"].nunique(),
        "unique_consumer_period_keys": frame[key_columns].drop_duplicates().shape[0],
        "target_periods": "; ".join(periods),
        "expected_target_period": "2026-04",
        "expected_consumers": 2087,
        "duplicate_consumer_period_rows": duplicate_count,
        "missing_predictions": missing_predictions,
        "nonfinite_predictions": 0,
        "unknown_class_labels": np.nan,
        "missing_probabilities": np.nan,
        "probabilities_outside_0_1": np.nan,
        "class_1_probability_audit": "Not applicable",
        "status": "PASS",
    }
    numeric_prediction = pd.to_numeric(frame[prediction_column], errors="coerce")
    row["nonfinite_predictions"] = int((~np.isfinite(numeric_prediction)).sum())
    if label_column is not None and probability_column is not None:
        labels = pd.to_numeric(frame[label_column], errors="coerce")
        probabilities = pd.to_numeric(frame[probability_column], errors="coerce")
        row["unknown_class_labels"] = int((~labels.isin([0, 1])).sum())
        row["missing_probabilities"] = int(probabilities.isna().sum())
        row["probabilities_outside_0_1"] = int(
            ((probabilities < 0) | (probabilities > 1) | ~np.isfinite(probabilities)).sum()
        )
        audit = read_required(probability_audit)
        april_check = audit.loc[
            audit["stage"].eq("APRIL_REFIT") & audit["configuration"].eq(official_model)
        ]
        if len(april_check) != 1:
            raise ValueError(f"{module} lacks one official APRIL_REFIT probability audit row.")
        valid_probability_column = (
            april_check.iloc[0]["classes"] == "[0, 1]"
            and int(april_check.iloc[0]["class_1_column"]) == 1
        )
        row["class_1_probability_audit"] = (
            "PASS: classes_=[0, 1], Class 1 column=1"
            if valid_probability_column else "FAIL"
        )
    checks = [
        len(frame) == 2087,
        frame["CONSUMER_NO"].nunique() == 2087,
        periods == ["2026-04"],
        duplicate_count == 0,
        missing_predictions == 0,
        row["nonfinite_predictions"] == 0,
    ]
    if label_column is not None:
        checks += [
            row["unknown_class_labels"] == 0,
            row["missing_probabilities"] == 0,
            row["probabilities_outside_0_1"] == 0,
            row["class_1_probability_audit"].startswith("PASS"),
        ]
    if not all(checks):
        row["status"] = "FAIL"
    return frame, row


def main():
    # Explicit authoritative input paths. The superseded Step 9 history path is
    # intentionally absent and is never scanned or read by this script.
    consumption_comparison = read_required(SUMMARIES / "consumption_model_comparison.csv")
    consumption_selection = read_required(SUMMARIES / "consumption_model_selection.csv")
    consumption_final = read_required(SUMMARIES / "consumption_final_test_metrics.csv")
    revenue_comparison = read_required(SUMMARIES / "revenue_model_comparison.csv")
    revenue_selection = read_required(SUMMARIES / "revenue_model_selection.csv")
    revenue_final = read_required(SUMMARIES / "revenue_final_test_metrics.csv")
    anomaly_comparison = read_required(SUMMARIES / "anomaly_model_comparison.csv")
    anomaly_selection = read_required(SUMMARIES / "anomaly_model_selection.csv")
    anomaly_final = read_required(SUMMARIES / "anomaly_final_test_metrics.csv")

    # Mandatory Step 9 gate: report an upstream inconsistency rather than fixing it.
    if len(anomaly_selection) != 1 or anomaly_selection.iloc[0]["configuration"] != "DT-3-balanced":
        raise ValueError(
            "STOP: authoritative Step 9 selection is not DT-3-balanced. "
            "Step 9 was not modified."
        )
    expected_anomaly_parameters = ["'max_depth': 8", "'class_weight': 'balanced'", "'random_state': 42"]
    parameter_text = anomaly_selection.iloc[0]["hyperparameters"]
    if not all(item in parameter_text for item in expected_anomaly_parameters):
        raise ValueError("STOP: authoritative Step 9 configuration is inconsistent.")

    selected_names = {
        "Consumption Prediction": consumption_selection.iloc[0]["configuration"],
        "Revenue Risk Prediction": revenue_selection.iloc[0]["configuration"],
        "Unusual Consumption Risk Prediction": anomaly_selection.iloc[0]["configuration"],
    }
    expected_selected = {
        "Consumption Prediction": "MLR-4",
        "Revenue Risk Prediction": "RF-3-balanced",
        "Unusual Consumption Risk Prediction": "DT-3-balanced",
    }
    if selected_names != expected_selected:
        raise ValueError(f"Official selected-model mismatch: {selected_names}")
    for selection, module in [
        (consumption_selection, "Consumption Prediction"),
        (revenue_selection, "Revenue Risk Prediction"),
        (anomaly_selection, "Unusual Consumption Risk Prediction"),
    ]:
        if not bool(selection.iloc[0]["frozen_before_final_test"]):
            raise ValueError(f"{module} was not recorded as frozen before FINAL TEST.")

    verify_final_test(consumption_final, "MLR-4", "Persistence", "Consumption Prediction")
    verify_final_test(revenue_final, "RF-3-balanced", "TRAIN-majority", "Revenue Risk Prediction")
    verify_final_test(anomaly_final, "DT-3-balanced", "TRAIN-majority", "Unusual Consumption Risk Prediction")

    # Family winners are read from saved selection evidence or summarized from
    # the already saved Step 7 validation RMSE table. No models are selected here.
    best_consumption_mlr = "MLR-4"
    best_consumption_rf = "RF-1"
    if consumption_comparison.loc[
        consumption_comparison["algorithm"].eq("Random Forest Regressor"), "VALIDATION_RMSE"
    ].idxmin() != consumption_comparison.index[consumption_comparison["configuration"].eq(best_consumption_rf)][0]:
        raise ValueError("Saved Step 7 family winner is inconsistent with validation RMSE.")
    best_revenue_dt = revenue_selection.iloc[0]["best_unweighted_DT"]
    best_revenue_rf = revenue_selection.iloc[0]["best_unweighted_RF"]
    best_anomaly_dt = anomaly_selection.iloc[0]["best_unweighted_DT"]
    best_anomaly_rf = anomaly_selection.iloc[0]["best_unweighted_RF"]

    metric_rows = []
    consumption_roles = {
        "Persistence": ["baseline"],
        best_consumption_mlr: ["best Multiple Linear Regression", "official selected model"],
        best_consumption_rf: ["best Random Forest"],
    }
    consumption_reason = consumption_selection.iloc[0]["reason"]
    for configuration, roles in consumption_roles.items():
        source = consumption_comparison.loc[
            consumption_comparison["configuration"].eq(configuration)
        ].iloc[0]
        for split in ["TRAIN", "VALIDATION"]:
            add_regression_row(
                metric_rows, source, "Consumption Prediction", split, roles,
                configuration == "MLR-4", configuration == "Persistence", consumption_reason,
            )
    for _, source in consumption_final.iterrows():
        roles = consumption_roles[source["configuration"]]
        add_regression_row(
            metric_rows, source, "Consumption Prediction", "FINAL_TEST", roles,
            source["configuration"] == "MLR-4", source["configuration"] == "Persistence",
            consumption_reason,
        )

    classification_modules = [
        (
            "Revenue Risk Prediction", revenue_comparison, revenue_selection, revenue_final,
            best_revenue_dt, best_revenue_rf, "RF-3-balanced",
        ),
        (
            "Unusual Consumption Risk Prediction", anomaly_comparison, anomaly_selection,
            anomaly_final, best_anomaly_dt, best_anomaly_rf, "DT-3-balanced",
        ),
    ]
    for module, comparison, selection, final_table, best_dt, best_rf, selected_name in classification_modules:
        roles_by_name = {
            "TRAIN-majority": ["baseline"],
            best_dt: ["best unweighted Decision Tree"],
            best_rf: ["best unweighted Random Forest"],
            selected_name: ["official selected model"],
        }
        reason = selection.iloc[0]["reason"]
        for configuration, roles in roles_by_name.items():
            source = comparison.loc[comparison["configuration"].eq(configuration)].iloc[0]
            for split in ["TRAIN", "VALIDATION"]:
                add_classification_row(
                    metric_rows, source, module, split, roles,
                    configuration == selected_name,
                    configuration == "TRAIN-majority",
                    reason,
                )
        for _, source in final_table.iterrows():
            roles = roles_by_name[source["configuration"]]
            add_classification_row(
                metric_rows, source, module, "FINAL_TEST", roles,
                source["configuration"] == selected_name,
                source["configuration"] == "TRAIN-majority",
                reason,
            )

    final_metrics = pd.DataFrame(metric_rows)
    if final_metrics.loc[final_metrics["split"].eq("FINAL_TEST")].groupby("module").size().ne(2).any():
        raise AssertionError("Each module must have exactly two FINAL TEST rows.")
    final_metrics.to_csv(SUMMARIES / "final_metrics.csv", index=False)

    consumption_selected_validation = consumption_comparison.loc[
        consumption_comparison["configuration"].eq("MLR-4")
    ].iloc[0]
    consumption_selected_final = consumption_final.loc[
        consumption_final["configuration"].eq("MLR-4")
    ].iloc[0]
    consumption_baseline_final = consumption_final.loc[
        consumption_final["configuration"].eq("Persistence")
    ].iloc[0]
    revenue_selected_validation = revenue_comparison.loc[
        revenue_comparison["configuration"].eq("RF-3-balanced")
    ].iloc[0]
    revenue_selected_final = revenue_final.loc[
        revenue_final["configuration"].eq("RF-3-balanced")
    ].iloc[0]
    revenue_baseline_final = revenue_final.loc[
        revenue_final["configuration"].eq("TRAIN-majority")
    ].iloc[0]
    anomaly_selected_validation = anomaly_comparison.loc[
        anomaly_comparison["configuration"].eq("DT-3-balanced")
    ].iloc[0]
    anomaly_selected_final = anomaly_final.loc[
        anomaly_final["configuration"].eq("DT-3-balanced")
    ].iloc[0]
    anomaly_baseline_final = anomaly_final.loc[
        anomaly_final["configuration"].eq("TRAIN-majority")
    ].iloc[0]

    comparison_rows = [
        {
            "module": "Consumption Prediction",
            "baseline": "Persistence",
            "best_decision_tree_or_linear": "MLR-4",
            "best_random_forest": "RF-1",
            "official_selected_model": "MLR-4",
            "validation_primary_metric": "RMSE",
            "selected_validation_primary_value": consumption_selected_validation["VALIDATION_RMSE"],
            "selected_validation_secondary_metrics": f"MAE={consumption_selected_validation['VALIDATION_MAE']:.6f}; R2={consumption_selected_validation['VALIDATION_R2']:.6f}",
            "selected_final_test_primary_value": consumption_selected_final["RMSE"],
            "baseline_final_test_primary_value": consumption_baseline_final["RMSE"],
            "final_test_secondary_metrics": f"Selected MAE={consumption_selected_final['MAE']:.6f}, R2={consumption_selected_final['R2']:.6f}; baseline MAE={consumption_baseline_final['MAE']:.6f}, R2={consumption_baseline_final['R2']:.6f}",
            "selection_reason": consumption_reason,
        },
        {
            "module": "Revenue Risk Prediction",
            "baseline": "TRAIN-majority",
            "best_decision_tree_or_linear": best_revenue_dt,
            "best_random_forest": best_revenue_rf,
            "official_selected_model": "RF-3-balanced",
            "validation_primary_metric": "ROC-AUC with Class-1 F1 and both-class support",
            "selected_validation_primary_value": revenue_selected_validation["VALIDATION_ROC_AUC"],
            "selected_validation_secondary_metrics": f"F1_1={revenue_selected_validation['VALIDATION_F1_1']:.6f}; F1_0={revenue_selected_validation['VALIDATION_F1_0']:.6f}; AP={revenue_selected_validation['VALIDATION_AP']:.6f}",
            "selected_final_test_primary_value": revenue_selected_final["ROC_AUC"],
            "baseline_final_test_primary_value": revenue_baseline_final["ROC_AUC"],
            "final_test_secondary_metrics": f"Selected F1_1={revenue_selected_final['F1_1']:.6f}, F1_0={revenue_selected_final['F1_0']:.6f}, AP={revenue_selected_final['AP']:.6f}; baseline F1_1={revenue_baseline_final['F1_1']:.6f}, F1_0={revenue_baseline_final['F1_0']:.6f}",
            "selection_reason": revenue_selection.iloc[0]["reason"],
        },
        {
            "module": "Unusual Consumption Risk Prediction",
            "baseline": "TRAIN-majority",
            "best_decision_tree_or_linear": best_anomaly_dt,
            "best_random_forest": best_anomaly_rf,
            "official_selected_model": "DT-3-balanced",
            "validation_primary_metric": "Class-1 F1 with recall and generalization support",
            "selected_validation_primary_value": anomaly_selected_validation["VALIDATION_F1_1"],
            "selected_validation_secondary_metrics": f"Recall_1={anomaly_selected_validation['VALIDATION_Recall_1']:.6f}; Precision_1={anomaly_selected_validation['VALIDATION_Precision_1']:.6f}; AP={anomaly_selected_validation['VALIDATION_AP']:.6f}; ROC-AUC={anomaly_selected_validation['VALIDATION_ROC_AUC']:.6f}",
            "selected_final_test_primary_value": anomaly_selected_final["F1_1"],
            "baseline_final_test_primary_value": anomaly_baseline_final["F1_1"],
            "final_test_secondary_metrics": f"Selected Recall_1={anomaly_selected_final['Recall_1']:.6f}, Precision_1={anomaly_selected_final['Precision_1']:.6f}, AP={anomaly_selected_final['AP']:.6f}, ROC-AUC={anomaly_selected_final['ROC_AUC']:.6f}; baseline Recall_1=0, ROC-AUC=0.5",
            "selection_reason": anomaly_selection.iloc[0]["reason"],
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(SUMMARIES / "final_comparison.csv", index=False)

    dashboard_rows = [
        {
            "module": "Consumption Prediction",
            "selected_model": "MLR-4",
            "model_configuration": consumption_selection.iloc[0]["hyperparameters"],
            "important_validation_metrics": f"RMSE {consumption_selected_validation['VALIDATION_RMSE']:.2f}; MAE {consumption_selected_validation['VALIDATION_MAE']:.2f}; R2 {consumption_selected_validation['VALIDATION_R2']:.4f}",
            "important_final_test_metrics": f"RMSE {consumption_selected_final['RMSE']:.2f}; MAE {consumption_selected_final['MAE']:.2f}; R2 {consumption_selected_final['R2']:.4f}",
            "baseline_comparison": f"FINAL TEST persistence RMSE {consumption_baseline_final['RMSE']:.2f}; selected improvement {consumption_baseline_final['RMSE'] - consumption_selected_final['RMSE']:.2f}",
            "short_interpretation": "Selected by lowest validation RMSE and improved on persistence on the held-out FINAL TEST.",
            "important_limitation": "Synthetic data, heavy tails, extreme residuals and correlated predictors limit generalization; April is unevaluated.",
        },
        {
            "module": "Revenue Risk Prediction",
            "selected_model": "RF-3-balanced",
            "model_configuration": "Random Forest; 200 trees; max_depth=10; class_weight=balanced; random_state=42",
            "important_validation_metrics": f"ROC-AUC {revenue_selected_validation['VALIDATION_ROC_AUC']:.4f}; F1_1 {revenue_selected_validation['VALIDATION_F1_1']:.4f}; F1_0 {revenue_selected_validation['VALIDATION_F1_0']:.4f}",
            "important_final_test_metrics": f"ROC-AUC {revenue_selected_final['ROC_AUC']:.4f}; AP {revenue_selected_final['AP']:.4f}; F1_1 {revenue_selected_final['F1_1']:.4f}; F1_0 {revenue_selected_final['F1_0']:.4f}",
            "baseline_comparison": f"FINAL TEST baseline ROC-AUC {revenue_baseline_final['ROC_AUC']:.4f} and F1_0 {revenue_baseline_final['F1_0']:.4f}; selected ROC-AUC improvement {revenue_selected_final['ROC_AUC'] - revenue_baseline_final['ROC_AUC']:.4f}",
            "short_interpretation": "Balanced weighting enabled minority Class-0 predictions while retaining useful Class-1 ranking.",
            "important_limitation": "Academic less-than-50%-paid-by-due-date proxy; imbalanced classes and synthetic data; probabilities are not official default probabilities.",
        },
        {
            "module": "Unusual Consumption Risk Prediction",
            "selected_model": "DT-3-balanced",
            "model_configuration": "Decision Tree; max_depth=8; class_weight=balanced; random_state=42",
            "important_validation_metrics": f"F1_1 {anomaly_selected_validation['VALIDATION_F1_1']:.4f}; Recall_1 {anomaly_selected_validation['VALIDATION_Recall_1']:.4f}; AP {anomaly_selected_validation['VALIDATION_AP']:.4f}; ROC-AUC {anomaly_selected_validation['VALIDATION_ROC_AUC']:.4f}",
            "important_final_test_metrics": f"F1_1 {anomaly_selected_final['F1_1']:.4f}; Recall_1 {anomaly_selected_final['Recall_1']:.4f}; Precision_1 {anomaly_selected_final['Precision_1']:.4f}; AP {anomaly_selected_final['AP']:.4f}; ROC-AUC {anomaly_selected_final['ROC_AUC']:.4f}",
            "baseline_comparison": "FINAL TEST majority baseline detected no Class-1 cases and had ROC-AUC 0.5000.",
            "short_interpretation": "Selected from TRAIN/VALIDATION evidence for competitive F1, much stronger anomaly recall and substantially smaller overfit gaps than RF-4-balanced.",
            "important_limitation": "Statistical unusual-consumption proxy only; it does not establish theft, fraud, meter tampering or wrongdoing.",
        },
    ]
    pd.DataFrame(dashboard_rows).to_csv(
        SUMMARIES / "dashboard_model_summary.csv", index=False
    )

    consumption_path = SUMMARIES / "next_month_consumption_predictions.csv"
    revenue_path = SUMMARIES / "next_month_revenue_risk_predictions.csv"
    anomaly_path = SUMMARIES / "next_month_anomaly_predictions.csv"
    consumption_predictions, consumption_audit = prediction_audit(
        "Consumption Prediction", consumption_path, "predicted_consumption"
    )
    revenue_predictions, revenue_audit = prediction_audit(
        "Revenue Risk Prediction", revenue_path, "predicted_risk_class",
        "predicted_risk_class", "predicted_class_1_probability",
        SUMMARIES / "revenue_probability_column_audit.csv", "RF-3-balanced",
    )
    anomaly_predictions, anomaly_audit = prediction_audit(
        "Unusual Consumption Risk Prediction", anomaly_path, "predicted_anomaly_class",
        "predicted_anomaly_class", "predicted_class_1_probability",
        SUMMARIES / "anomaly_probability_column_audit.csv", "DT-3-balanced",
    )
    audit_rows = [consumption_audit, revenue_audit, anomaly_audit]
    if any(row["status"] != "PASS" for row in audit_rows):
        raise ValueError(f"April prediction audit failed: {audit_rows}")
    pd.DataFrame(audit_rows).to_csv(
        SUMMARIES / "april2026_prediction_audit.csv", index=False
    )

    consumption_merge = consumption_predictions.rename(
        columns={"predicted_consumption": "PREDICTED_CONSUMPTION"}
    )
    consumption_merge["HAS_CONSUMPTION_PREDICTION"] = consumption_merge[
        "PREDICTED_CONSUMPTION"
    ].notna()
    revenue_merge = revenue_predictions.rename(columns={
        "predicted_risk_class": "PREDICTED_REVENUE_RISK_CLASS",
        "predicted_class_1_probability": "REVENUE_CLASS_1_PROBABILITY",
    })
    revenue_merge["HAS_REVENUE_PREDICTION"] = revenue_merge[
        "PREDICTED_REVENUE_RISK_CLASS"
    ].notna()
    anomaly_merge = anomaly_predictions.rename(columns={
        "predicted_anomaly_class": "PREDICTED_ANOMALY_CLASS",
        "predicted_class_1_probability": "ANOMALY_CLASS_1_PROBABILITY",
    })
    anomaly_merge["HAS_ANOMALY_PREDICTION"] = anomaly_merge[
        "PREDICTED_ANOMALY_CLASS"
    ].notna()
    keys = ["CONSUMER_NO", "TARGET_PERIOD"]
    consolidated = consumption_merge.merge(
        revenue_merge, on=keys, how="outer", validate="one_to_one"
    ).merge(anomaly_merge, on=keys, how="outer", validate="one_to_one")
    for flag in [
        "HAS_CONSUMPTION_PREDICTION", "HAS_REVENUE_PREDICTION", "HAS_ANOMALY_PREDICTION",
    ]:
        consolidated[flag] = consolidated[flag].fillna(False).astype(bool)
    if consolidated.duplicated(keys).any():
        raise AssertionError("Consolidated April file contains duplicate keys.")
    if not consolidated["TARGET_PERIOD"].eq("2026-04").all():
        raise AssertionError("Consolidated April file contains another target period.")
    if consolidated["CONSUMER_NO"].nunique() != 2087:
        raise AssertionError("Consolidated union does not contain 2,087 consumers.")
    consolidated.to_csv(PROCESSED / "april2026_consolidated_predictions.csv", index=False)

    artifact_paths = [
        ("Consumption Prediction", "Model comparison", SUMMARIES / "consumption_model_comparison.csv", "Required saved comparison"),
        ("Consumption Prediction", "FINAL TEST actual versus predicted", FIGURES / "consumption_final_test_actual_vs_predicted.png", "Required regression evaluation figure"),
        ("Consumption Prediction", "FINAL TEST residual plot", FIGURES / "consumption_final_test_residuals.png", "Required regression evaluation figure"),
        ("Consumption Prediction", "Frozen pre-refit model", MODELS / "consumption_frozen_train_pipeline.joblib", "Confirms held-out evaluator was preserved"),
        ("Revenue Risk Prediction", "Model comparison", SUMMARIES / "revenue_model_comparison.csv", "Required saved comparison"),
        ("Revenue Risk Prediction", "FINAL TEST confusion matrix", FIGURES / "revenue_final_test_confusion_matrix.png", "Required classification figure"),
        ("Revenue Risk Prediction", "FINAL TEST ROC", FIGURES / "revenue_final_test_roc.png", "Required classification figure"),
        ("Revenue Risk Prediction", "Feature importance", FIGURES / "revenue_feature_importance.png", "Required supported-model figure"),
        ("Revenue Risk Prediction", "Frozen pre-refit model", MODELS / "revenue_frozen_train_pipeline.joblib", "Confirms held-out evaluator was preserved"),
        ("Unusual Consumption Risk Prediction", "Model comparison", SUMMARIES / "anomaly_model_comparison.csv", "Current official comparison; RF-4 remains experimental only"),
        ("Unusual Consumption Risk Prediction", "FINAL TEST confusion matrix", FIGURES / "anomaly_final_test_confusion_matrix.png", "Must represent official DT-3-balanced"),
        ("Unusual Consumption Risk Prediction", "FINAL TEST ROC", FIGURES / "anomaly_final_test_roc.png", "Must represent official DT-3-balanced"),
        ("Unusual Consumption Risk Prediction", "Feature importance", FIGURES / "anomaly_feature_importance.png", "Must represent official DT-3-balanced"),
        ("Unusual Consumption Risk Prediction", "Frozen pre-refit model", MODELS / "anomaly_frozen_train_pipeline.joblib", "Must be official DT-3-balanced"),
    ]
    artifact_rows = []
    for module, artifact, path, reason in artifact_paths:
        exists = path.exists()
        artifact_rows.append({
            "module": module,
            "artifact": artifact,
            "expected_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": exists,
            "status_reason": "PASS: present. " + reason if exists else "MISSING: " + reason,
        })
    artifact_audit = pd.DataFrame(artifact_rows)
    artifact_audit.to_csv(SUMMARIES / "evaluation_artifact_audit.csv", index=False)
    if not artifact_audit["exists"].all():
        missing = artifact_audit.loc[~artifact_audit["exists"], "expected_path"].tolist()
        raise FileNotFoundError(f"Expected evaluation artifacts are missing: {missing}")

    print("PASS: consolidated Steps 7-9 without retraining or model selection.")
    print("Official models:", expected_selected)
    print("FINAL TEST rows are saved pre-refit selected-model/baseline pairs only.")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(
        "April union:", len(consolidated), "consumers; module coverage:",
        int(consolidated["HAS_CONSUMPTION_PREDICTION"].sum()),
        int(consolidated["HAS_REVENUE_PREDICTION"].sum()),
        int(consolidated["HAS_ANOMALY_PREDICTION"].sum()),
    )
    print("Artifact audit: all", len(artifact_audit), "explicit artifacts present.")


if __name__ == "__main__":
    main()
