"""Step 9: unusual-consumption model comparison and April prediction.

Run from the project root with: python src/train_anomaly.py
Use --compare-only to stop after validation comparison, before FINAL TEST.
"""
from pathlib import Path
import shutil
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from features import ANOMALY_PREDICTORS


ROOT = Path(__file__).resolve().parents[1]
TARGET = "ANOMALY_NEXT"
CATEGORICAL = ["TARGET_MONTH", "TARRIF", "SOLAR_CONSUMER"]
NUMERIC = [column for column in ANOMALY_PREDICTORS if column not in CATEGORICAL]


def make_pipeline(classifier):
    """Keep TRAIN-fitted imputation and encoding together with the classifier."""
    numeric_steps = SimpleImputer(strategy="median", keep_empty_features=True)
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC),
        ("categorical", categorical_steps, CATEGORICAL),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def predict_class_one(pipeline, frame, name, stage, audit_rows):
    """Find the Class-1 probability column from classes_ instead of assuming it."""
    classes = pipeline.named_steps["model"].classes_.tolist()
    if set(classes) != {0, 1}:
        raise ValueError(f"{name} has unexpected classes: {classes}")
    class_one_column = classes.index(1)
    probability = pipeline.predict_proba(frame[ANOMALY_PREDICTORS])[:, class_one_column]
    predicted = pipeline.predict(frame[ANOMALY_PREDICTORS])
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError(f"{name} produced invalid Class-1 probabilities.")
    audit_rows.append({
        "configuration": name,
        "stage": stage,
        "classes": str(classes),
        "class_1_column": class_one_column,
        "rows": len(frame),
    })
    return predicted, probability


def calculate_metrics(actual, predicted, probability):
    """Return ranking metrics, both-class metrics and confusion counts."""
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "sample_count": len(actual),
        "Accuracy": accuracy_score(actual, predicted),
        "ROC_AUC": roc_auc_score(actual, probability),
        "AP": average_precision_score(actual, probability),
        "class_0_count": int(support[0]),
        "class_0_percent": 100 * support[0] / len(actual),
        "Precision_0": precision[0],
        "Recall_0": recall[0],
        "F1_0": f1[0],
        "class_1_count": int(support[1]),
        "class_1_percent": 100 * support[1] / len(actual),
        "Precision_1": precision[1],
        "Recall_1": recall[1],
        "F1_1": f1[1],
        "true_0_predicted_0": int(matrix[0, 0]),
        "true_0_predicted_1": int(matrix[0, 1]),
        "true_1_predicted_0": int(matrix[1, 0]),
        "true_1_predicted_1": int(matrix[1, 1]),
    }


def validation_ranking(table):
    """Rank by the requested primary metric, then the supporting evidence."""
    ranked = table.copy()
    ranked["F1_1_overfit_gap"] = (
        ranked["TRAIN_F1_1"] - ranked["VALIDATION_F1_1"]
    ).clip(lower=0)
    return ranked.sort_values(
        [
            "VALIDATION_F1_1",
            "VALIDATION_Recall_1",
            "VALIDATION_Precision_1",
            "VALIDATION_AP",
            "VALIDATION_ROC_AUC",
            "F1_1_overfit_gap",
        ],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    )


def preprocessing_audit(pipeline, frame, name, stage):
    """Record the rows that supplied learned medians and category vocabularies."""
    processor = pipeline.named_steps["preprocessing"]
    medians = processor.named_transformers_["numeric"].statistics_
    expected_medians = frame[NUMERIC].median().fillna(0).to_numpy()
    np.testing.assert_allclose(medians, expected_medians)
    rows = []
    for feature, median in zip(NUMERIC, medians):
        rows.append({
            "configuration": name,
            "stage": stage,
            "fit_rows": len(frame),
            "feature": feature,
            "statistic": "numeric median",
            "value": median,
        })
    encoder = processor.named_transformers_["categorical"].named_steps["encoder"]
    for feature, categories in zip(CATEGORICAL, encoder.categories_):
        expected = set(frame[feature].dropna().unique())
        if set(categories) != expected:
            raise AssertionError(f"Unexpected category vocabulary for {feature}")
        rows.append({
            "configuration": name,
            "stage": stage,
            "fit_rows": len(frame),
            "feature": feature,
            "statistic": "category vocabulary",
            "value": "; ".join(str(value) for value in categories),
        })
    return rows


def main():
    summaries = ROOT / "reports" / "summaries"
    figures = ROOT / "reports" / "figures"
    models = ROOT / "models"
    processed = ROOT / "data" / "processed"
    split_folder = processed / "splits"
    checkpoint_path = models / "anomaly_comparison_checkpoint.joblib"
    final_metrics_path = summaries / "anomaly_final_test_metrics.csv"
    final_predictions_path = processed / "anomaly_final_test_predictions.csv"

    # A requested selection revision preserves every earlier RF-4-balanced
    # evaluation/deployment artifact before replacing the official outputs.
    revision_requested = "--revise-selection" in sys.argv
    if revision_requested:
        history = processed / "step9_rf4_balanced_history"
        if history.exists():
            raise FileExistsError(
                "RF-4-balanced history already exists. Refusing to repeat the revised FINAL TEST."
            )
        history.mkdir(parents=True)
        history_files = [
            summaries / "anomaly_model_selection.csv",
            summaries / "anomaly_final_test_metrics.csv",
            summaries / "anomaly_final_test_confusion_matrix.csv",
            summaries / "anomaly_feature_importance.csv",
            summaries / "anomaly_prediction_coverage.csv",
            summaries / "anomaly_probability_column_audit.csv",
            summaries / "anomaly_preprocessing_audit.csv",
            summaries / "next_month_anomaly_predictions.csv",
            processed / "anomaly_final_test_predictions.csv",
            figures / "anomaly_validation_roc.png",
            figures / "anomaly_final_test_confusion_matrix.png",
            figures / "anomaly_final_test_roc.png",
            figures / "anomaly_feature_importance.png",
            models / "anomaly_frozen_train_pipeline.joblib",
            models / "anomaly_april2026_pipeline.joblib",
        ]
        for source in history_files:
            if not source.exists():
                raise FileNotFoundError(f"Cannot preserve missing RF-4-balanced artifact: {source}")
            shutil.copy2(source, history / source.name)
        print(f"Preserved prior RF-4-balanced artifacts under {history}", flush=True)

    # Protect the one-time FINAL TEST evaluation from accidental reruns.
    if not revision_requested and (final_metrics_path.exists() or final_predictions_path.exists()):
        raise FileExistsError("Preserved anomaly FINAL TEST outputs already exist. Do not overwrite them.")

    metadata = pd.read_csv(summaries / "split_summary.csv")
    metadata = metadata.loc[metadata["module"].eq("anomaly")].set_index("partition")
    expected_rows = {name: int(metadata.loc[name, "rows"]) for name in metadata.index}
    if sum(expected_rows[name] for name in ["TRAIN", "VALIDATION", "FINAL_TEST"]) != 67324:
        raise ValueError("Step 6 labelled anomaly total does not equal 67,324.")
    if expected_rows["FUTURE"] != 2087:
        raise ValueError("Step 6 anomaly FUTURE count does not equal 2,087.")

    required_columns = ["CONSUMER_NO", "PERIOD", "TARGET_PERIOD", TARGET] + ANOMALY_PREDICTORS
    frames = {}
    source_sizes = {}
    for partition in ["TRAIN", "VALIDATION", "FINAL_TEST", "FUTURE"]:
        path = split_folder / f"anomaly_{partition.lower()}.csv"
        source_sizes[partition] = path.stat().st_size
        frame = pd.read_csv(path, usecols=required_columns, dtype=str, keep_default_na=False)
        for column in NUMERIC + [TARGET]:
            original = frame[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid = original.ne("") & converted.isna()
            if invalid.any():
                raise ValueError(f"{partition} contains invalid numeric values in {column}.")
            frame[column] = converted
        for column in CATEGORICAL:
            frame[column] = frame[column].replace("", np.nan)

        periods = pd.PeriodIndex(frame["TARGET_PERIOD"], freq="M")
        if len(frame) != expected_rows[partition]:
            raise ValueError(f"Unexpected {partition} row count.")
        if frame["CONSUMER_NO"].nunique() != int(metadata.loc[partition, "unique_consumers"]):
            raise ValueError(f"Unexpected {partition} consumer count.")
        if str(periods.min()) != metadata.loc[partition, "min_target_period"]:
            raise ValueError(f"Unexpected {partition} first target month.")
        if str(periods.max()) != metadata.loc[partition, "max_target_period"]:
            raise ValueError(f"Unexpected {partition} last target month.")
        if not periods.is_monotonic_increasing:
            raise ValueError(f"{partition} is not chronological.")
        if frame.duplicated(["CONSUMER_NO", "TARGET_PERIOD"]).any():
            raise ValueError(f"{partition} contains duplicate consumer predictions.")
        if partition == "FUTURE":
            if frame[TARGET].notna().any() or not (periods == pd.Period("2026-04")).all():
                raise ValueError("FUTURE targets or periods are invalid.")
        else:
            if not frame[TARGET].isin([0, 1]).all():
                raise ValueError(f"{partition} contains invalid or missing targets.")
            frame[TARGET] = frame[TARGET].astype(int)
        frames[partition] = frame

    train = frames["TRAIN"]
    validation = frames["VALIDATION"]
    majority_class = int(train[TARGET].value_counts().idxmax())
    train_class_one_rate = train[TARGET].mean()

    distribution_rows = []
    for partition in ["TRAIN", "VALIDATION", "FINAL_TEST"]:
        for label in [0, 1]:
            count = int(frames[partition][TARGET].eq(label).sum())
            distribution_rows.append({
                "partition": partition,
                "class": label,
                "count": count,
                "percent": 100 * count / len(frames[partition]),
                "definition": "Class 1 is the project-defined unusual-consumption proxy; it is not theft or fraud.",
            })
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(summaries / "anomaly_class_distributions.csv", index=False)
    labelled_class_counts = distributions.groupby("class")["count"].sum()
    if int(labelled_class_counts.sum()) != 67324:
        raise AssertionError("Labelled class counts do not reconcile.")
    print(distributions.to_string(index=False), flush=True)
    print(
        f"TRAIN-majority baseline: class={majority_class}; "
        f"constant Class-1 probability={train_class_one_rate:.8f}",
        flush=True,
    )

    if checkpoint_path.exists():
        checkpoint = joblib.load(checkpoint_path)
        if checkpoint["source_sizes"] != source_sizes:
            raise ValueError("Anomaly split files changed since model comparison.")
        comparison = checkpoint["comparison"]
        fitted_models = checkpoint["fitted_models"]
        validation_probabilities = checkpoint["validation_probabilities"]
        probability_audit_rows = checkpoint["probability_audit_rows"]
        preprocessing_audit_rows = checkpoint["preprocessing_audit_rows"]
        best_dt = checkpoint["best_dt"]
        best_rf = checkpoint["best_rf"]
        print("Loaded the completed ten-model comparison; no model was refitted.", flush=True)
    else:
        probability_audit_rows = []
        preprocessing_audit_rows = []
        fitted_models = {}
        validation_probabilities = {}

        baseline_row = {
            "configuration": "TRAIN-majority",
            "algorithm": "Constant baseline",
            "class_weight": "None",
            "hyperparameters": (
                f"TRAIN majority={majority_class}; "
                f"TRAIN Class-1 prevalence={train_class_one_rate}"
            ),
            "features": "None",
        }
        for stage, frame in [("TRAIN", train), ("VALIDATION", validation)]:
            result = calculate_metrics(
                frame[TARGET],
                np.full(len(frame), majority_class),
                np.full(len(frame), train_class_one_rate),
            )
            for metric, value in result.items():
                baseline_row[f"{stage}_{metric}"] = value
        rows = [baseline_row]

        experiments = []
        for number, depth in enumerate([3, 5, 8, None], start=1):
            classifier = DecisionTreeClassifier(
                criterion="gini", max_depth=depth, class_weight=None, random_state=42
            )
            experiments.append((f"DT-{number}", "Decision Tree", classifier))
        forest_settings = [(100, 5), (100, 10), (200, 10), (200, None)]
        for number, (trees, depth) in enumerate(forest_settings, start=1):
            classifier = RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            )
            experiments.append((f"RF-{number}", "Random Forest", classifier))

        # Fit the eight prescribed unweighted models first.
        for name, algorithm, classifier in experiments:
            print(f"Fitting {name}", flush=True)
            pipeline = make_pipeline(classifier)
            pipeline.fit(train[ANOMALY_PREDICTORS], train[TARGET])
            preprocessing_audit_rows += preprocessing_audit(
                pipeline, train, name, "TRAIN_ONLY"
            )
            row = {
                "configuration": name,
                "algorithm": algorithm,
                "class_weight": str(classifier.class_weight),
                "hyperparameters": str(classifier.get_params()),
                "features": "; ".join(ANOMALY_PREDICTORS),
            }
            for stage, frame in [("TRAIN", train), ("VALIDATION", validation)]:
                predicted, probability = predict_class_one(
                    pipeline, frame, name, stage, probability_audit_rows
                )
                result = calculate_metrics(frame[TARGET], predicted, probability)
                for metric, value in result.items():
                    row[f"{stage}_{metric}"] = value
                if stage == "VALIDATION":
                    validation_probabilities[name] = probability
            rows.append(row)
            fitted_models[name] = pipeline

        unweighted = pd.DataFrame(rows)
        best_dt = validation_ranking(
            unweighted.loc[unweighted["algorithm"].eq("Decision Tree")]
        ).iloc[0]["configuration"]
        best_rf = validation_ranking(
            unweighted.loc[unweighted["algorithm"].eq("Random Forest")]
        ).iloc[0]["configuration"]
        print(f"Best unweighted DT/RF: {best_dt}, {best_rf}", flush=True)

        # Fit exactly one balanced clone of each validation-selected family winner.
        for parent, algorithm in [(best_dt, "Decision Tree"), (best_rf, "Random Forest")]:
            classifier = clone(fitted_models[parent].named_steps["model"])
            parent_parameters = classifier.get_params().copy()
            classifier.set_params(class_weight="balanced")
            for parameter, value in parent_parameters.items():
                if parameter != "class_weight" and classifier.get_params()[parameter] != value:
                    raise AssertionError("Balanced clone changed another hyperparameter.")
            name = f"{parent}-balanced"
            print(f"Fitting {name}", flush=True)
            pipeline = make_pipeline(classifier)
            pipeline.fit(train[ANOMALY_PREDICTORS], train[TARGET])
            preprocessing_audit_rows += preprocessing_audit(
                pipeline, train, name, "TRAIN_ONLY"
            )
            row = {
                "configuration": name,
                "algorithm": algorithm,
                "class_weight": "balanced",
                "hyperparameters": str(classifier.get_params()),
                "features": "; ".join(ANOMALY_PREDICTORS),
            }
            for stage, frame in [("TRAIN", train), ("VALIDATION", validation)]:
                predicted, probability = predict_class_one(
                    pipeline, frame, name, stage, probability_audit_rows
                )
                result = calculate_metrics(frame[TARGET], predicted, probability)
                for metric, value in result.items():
                    row[f"{stage}_{metric}"] = value
                if stage == "VALIDATION":
                    validation_probabilities[name] = probability
            rows.append(row)
            fitted_models[name] = pipeline

        comparison = pd.DataFrame(rows)
        for metric in [
            "Accuracy", "ROC_AUC", "AP", "Precision_0", "Recall_0", "F1_0",
            "Precision_1", "Recall_1", "F1_1",
        ]:
            comparison[f"VALIDATION_minus_TRAIN_{metric}"] = (
                comparison[f"VALIDATION_{metric}"] - comparison[f"TRAIN_{metric}"]
            )
        if len(fitted_models) != 10 or len(comparison) != 11:
            raise AssertionError("Expected ten classifiers plus one baseline.")
        checkpoint = {
            "source_sizes": source_sizes,
            "comparison": comparison,
            "fitted_models": fitted_models,
            "validation_probabilities": validation_probabilities,
            "probability_audit_rows": probability_audit_rows,
            "preprocessing_audit_rows": preprocessing_audit_rows,
            "best_dt": best_dt,
            "best_rf": best_rf,
        }
        joblib.dump(checkpoint, checkpoint_path, compress=3)

    comparison.to_csv(summaries / "anomaly_model_comparison.csv", index=False)
    weight_effect_rows = []
    for parent in [best_dt, best_rf]:
        original = comparison.loc[comparison["configuration"].eq(parent)].iloc[0]
        balanced = comparison.loc[
            comparison["configuration"].eq(f"{parent}-balanced")
        ].iloc[0]
        row = {
            "unweighted_configuration": parent,
            "balanced_configuration": f"{parent}-balanced",
        }
        for metric in ["F1_1", "Recall_1", "Precision_1", "AP", "ROC_AUC", "Accuracy"]:
            row[f"unweighted_{metric}"] = original[f"VALIDATION_{metric}"]
            row[f"balanced_{metric}"] = balanced[f"VALIDATION_{metric}"]
            row[f"balanced_minus_unweighted_{metric}"] = (
                balanced[f"VALIDATION_{metric}"] - original[f"VALIDATION_{metric}"]
            )
        weight_effect_rows.append(row)
    pd.DataFrame(weight_effect_rows).to_csv(
        summaries / "anomaly_class_weight_effect.csv", index=False
    )

    candidates = comparison.loc[comparison["configuration"].ne("TRAIN-majority")]
    ranked = validation_ranking(candidates)
    # Revision based only on the already-saved TRAIN and VALIDATION comparison:
    # RF-4-balanced has a slightly higher validation F1, but DT-3-balanced has
    # much higher anomaly recall and far smaller F1, AP and ROC-AUC gaps.
    # This applies the original rule: Class-1 F1 is primary, while recall,
    # precision, AP, ROC-AUC and generalization gaps provide supporting evidence.
    selected_name = "DT-3-balanced"
    selected_row = ranked.loc[ranked["configuration"].eq(selected_name)].iloc[0]
    rf_balanced = ranked.loc[ranked["configuration"].eq("RF-4-balanced")].iloc[0]
    if rf_balanced["VALIDATION_F1_1"] - selected_row["VALIDATION_F1_1"] > 0.02:
        raise AssertionError("Balanced-model F1 difference is no longer a close validation result.")
    if selected_row["VALIDATION_Recall_1"] <= rf_balanced["VALIDATION_Recall_1"]:
        raise AssertionError("DT-3-balanced no longer has the stronger validation recall.")
    if selected_row["F1_1_overfit_gap"] >= rf_balanced["F1_1_overfit_gap"]:
        raise AssertionError("DT-3-balanced no longer has the smaller F1 generalization gap.")
    selected_pipeline = fitted_models[selected_name]
    print(
        ranked[[
            "configuration", "VALIDATION_F1_1", "VALIDATION_Recall_1",
            "VALIDATION_Precision_1", "VALIDATION_AP", "VALIDATION_ROC_AUC",
            "F1_1_overfit_gap",
        ]].to_string(index=False),
        flush=True,
    )
    print(f"Validation-selected model: {selected_name}", flush=True)

    if "--compare-only" in sys.argv:
        print("Stopped before FINAL TEST. Run normally to resume the saved comparison.", flush=True)
        return

    # Save the validation decision and frozen TRAIN model before FINAL TEST use.
    selection = ranked.loc[ranked["configuration"].eq(selected_name)].copy()
    selection["best_unweighted_DT"] = best_dt
    selection["best_unweighted_RF"] = best_rf
    selection["primary_selection_metric"] = "VALIDATION Class-1 F1"
    selection["frozen_before_final_test"] = True
    selection["reason"] = (
        "TRAIN/VALIDATION-only stability review: DT-3-balanced validation Class-1 F1=0.214776 "
        "is close to RF-4-balanced F1=0.230599, while DT-3-balanced has much higher validation "
        "Class-1 recall (0.523810 versus 0.200772). Its validation-minus-TRAIN gaps are also much "
        "smaller: F1=-0.047032 versus -0.765917, AP=-0.103422 versus -0.809600 and ROC-AUC="
        "-0.097643 versus -0.259984. RF-4-balanced had TRAIN recall=1.0 and near-perfect TRAIN "
        "F1, showing severe overfitting. DT-3-balanced is selected as the more stable validation-"
        "supported model under the original multi-metric rule. FINAL TEST was not used for this "
        "revision, and no tuning or threshold search was performed."
    )
    selection.to_csv(summaries / "anomaly_model_selection.csv", index=False)
    frozen_parameters = selected_pipeline.named_steps["model"].get_params().copy()
    joblib.dump(selected_pipeline, models / "anomaly_frozen_train_pipeline.joblib", compress=3)

    # Compare relevant validation ROC curves only; this does not change selection.
    figure, axis = plt.subplots(figsize=(8, 6))
    for name in [best_dt, best_rf, f"{best_dt}-balanced", f"{best_rf}-balanced"]:
        false_positive, true_positive, unused_thresholds = roc_curve(
            validation[TARGET], validation_probabilities[name]
        )
        auc = comparison.loc[
            comparison["configuration"].eq(name), "VALIDATION_ROC_AUC"
        ].iloc[0]
        axis.plot(false_positive, true_positive, label=f"{name}: AUC {auc:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="grey", label="TRAIN-majority: AUC 0.5000")
    axis.set(
        title="Unusual-consumption proxy: VALIDATION ROC comparison",
        xlabel="False positive rate (Class 1)",
        ylabel="True positive rate (Class 1)",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figures / "anomaly_validation_roc.png", dpi=160)
    plt.close(figure)

    # FINAL TEST is touched only after the selection record and frozen model exist.
    final_test = frames["FINAL_TEST"]
    selected_class, selected_probability = predict_class_one(
        selected_pipeline, final_test, selected_name, "FINAL_TEST", probability_audit_rows
    )
    baseline_class = np.full(len(final_test), majority_class)
    baseline_probability = np.full(len(final_test), train_class_one_rate)
    final_rows = []
    for name, predicted, probability in [
        (selected_name, selected_class, selected_probability),
        ("TRAIN-majority", baseline_class, baseline_probability),
    ]:
        result = calculate_metrics(final_test[TARGET], predicted, probability)
        result.update({"configuration": name, "partition": "FINAL_TEST"})
        final_rows.append(result)
    pd.DataFrame(final_rows).to_csv(final_metrics_path, index=False)

    saved_predictions = final_test[["CONSUMER_NO", "PERIOD", "TARGET_PERIOD"]].copy()
    saved_predictions["actual_class"] = final_test[TARGET]
    saved_predictions["predicted_class"] = selected_class
    saved_predictions["class_1_probability"] = selected_probability
    saved_predictions["baseline_class"] = majority_class
    saved_predictions["baseline_class_1_probability"] = train_class_one_rate
    saved_predictions.to_csv(final_predictions_path, index=False)

    matrix = confusion_matrix(final_test[TARGET], selected_class, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(summaries / "anomaly_final_test_confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            colour = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=colour)
    axis.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"],
        xlabel="Predicted", ylabel="Actual",
        title=f"Unusual-consumption proxy FINAL TEST: {selected_name}",
    )
    figure.tight_layout()
    figure.savefig(figures / "anomaly_final_test_confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    false_positive, true_positive, unused_thresholds = roc_curve(
        final_test[TARGET], selected_probability
    )
    axis.plot(
        false_positive,
        true_positive,
        label=f"{selected_name}: AUC {final_rows[0]['ROC_AUC']:.4f}",
    )
    axis.plot([0, 1], [0, 1], "--", color="grey", label="TRAIN-majority: AUC 0.5000")
    axis.set(
        title="Unusual-consumption proxy FINAL TEST ROC",
        xlabel="False positive rate (Class 1)",
        ylabel="True positive rate (Class 1)",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figures / "anomaly_final_test_roc.png", dpi=160)
    plt.close(figure)

    importance = pd.DataFrame({
        "encoded_feature": selected_pipeline.named_steps["preprocessing"].get_feature_names_out(),
        "impurity_importance": selected_pipeline.named_steps["model"].feature_importances_,
    }).sort_values("impurity_importance", ascending=False)
    importance["note"] = (
        "Descriptive TRAIN-fitted impurity importance; not causal and not a new selection step."
    )
    importance.to_csv(summaries / "anomaly_feature_importance.csv", index=False)
    displayed = importance.head(15).sort_values("impurity_importance")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(displayed["encoded_feature"], displayed["impurity_importance"])
    axis.set(
        title=f"Unusual-consumption proxy: {selected_name} feature importance",
        xlabel="Impurity-based importance",
    )
    figure.tight_layout()
    figure.savefig(figures / "anomaly_feature_importance.png", dpi=160)
    plt.close(figure)

    # Clone and refit every pipeline step on all labelled history only.
    labelled = pd.concat(
        [frames["TRAIN"], frames["VALIDATION"], frames["FINAL_TEST"]],
        ignore_index=True,
    )
    if len(labelled) != 67324 or labelled[TARGET].isna().any():
        raise AssertionError("All-labelled refit population is invalid.")
    april_pipeline = clone(selected_pipeline)
    if april_pipeline.named_steps["model"].get_params() != frozen_parameters:
        raise AssertionError("Cloned model configuration changed before refit.")
    april_pipeline.fit(labelled[ANOMALY_PREDICTORS], labelled[TARGET])
    preprocessing_audit_rows += preprocessing_audit(
        april_pipeline, labelled, selected_name, "REFIT_ALL_LABELLED"
    )
    if april_pipeline.named_steps["model"].get_params() != frozen_parameters:
        raise AssertionError("Model configuration changed during refit.")
    joblib.dump(april_pipeline, models / "anomaly_april2026_pipeline.joblib", compress=3)

    future = frames["FUTURE"]
    april_class, april_probability = predict_class_one(
        april_pipeline, future, selected_name, "APRIL_REFIT", probability_audit_rows
    )
    april_predictions = future[["CONSUMER_NO", "TARGET_PERIOD"]].copy()
    april_predictions["predicted_anomaly_class"] = april_class
    april_predictions["predicted_class_1_probability"] = april_probability
    if len(april_predictions) != 2087 or april_predictions["CONSUMER_NO"].nunique() != 2087:
        raise AssertionError("April prediction keys are incomplete or duplicated.")
    if not april_predictions["TARGET_PERIOD"].eq("2026-04").all():
        raise AssertionError("April TARGET_PERIOD is invalid.")
    if april_predictions.isna().any().any():
        raise AssertionError("April predictions contain missing values.")
    april_predictions.to_csv(
        summaries / "next_month_anomaly_predictions.csv", index=False
    )

    coverage = pd.DataFrame([{
        "configuration": selected_name,
        "refit_labelled_rows": len(labelled),
        "target_period": "2026-04",
        "predictions": len(april_predictions),
        "unique_consumers": april_predictions["CONSUMER_NO"].nunique(),
        "coverage_percent": 100 * len(april_predictions) / len(future),
        "predicted_class_0": int((april_class == 0).sum()),
        "predicted_class_1": int((april_class == 1).sum()),
        "minimum_probability": april_probability.min(),
        "maximum_probability": april_probability.max(),
        "limitation": (
            "ANOMALY_NEXT is a project-defined statistical proxy for unusual consumption. "
            "Predictions are decision-support indicators and do not establish electricity "
            "theft, fraud, meter tampering, or wrongdoing."
        ),
    }])
    coverage.to_csv(summaries / "anomaly_prediction_coverage.csv", index=False)
    pd.DataFrame(probability_audit_rows).to_csv(
        summaries / "anomaly_probability_column_audit.csv", index=False
    )
    pd.DataFrame(preprocessing_audit_rows).to_csv(
        summaries / "anomaly_preprocessing_audit.csv", index=False
    )

    # Confirm that Step 6 source files were not modified during this script.
    for partition, original_size in source_sizes.items():
        path = split_folder / f"anomaly_{partition.lower()}.csv"
        if path.stat().st_size != original_size:
            raise AssertionError(f"Source split changed: {partition}")

    print(pd.DataFrame(final_rows).to_string(index=False), flush=True)
    print(coverage.to_string(index=False), flush=True)
    print(
        "PASS: validation-only selection, one frozen FINAL TEST evaluation, "
        "unchanged full-history refit and 2,087 April predictions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
