"""Step 7: nine consumption comparisons, one final test, then April refit.
Run python src/train_consumption.py. Step 6 partitions are read, never recreated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import CONSUMPTION_PREDICTORS

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'CONSUMPTION_NEXT'
CATEGORIES = ['TARRIF', 'SOLAR_CONSUMER', 'TARGET_MONTH']


def metrics(actual, predicted):
    """All configurations use the same finite target rows and original units."""
    assert np.isfinite(actual).all() and np.isfinite(predicted).all()
    return {'sample_count': len(actual), 'MAE': mean_absolute_error(actual, predicted),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)), 'R2': r2_score(actual, predicted)}


def make_pipeline(features, estimator):
    """Median/category imputation and encoding are fitted only inside fit()."""
    numeric = [column for column in features if column not in CATEGORIES]
    categorical = [column for column in features if column in CATEGORIES]
    transformers = [('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric)]
    if categorical:
        categories = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                               ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transformers.append(('categorical', categories, categorical))
    return Pipeline([('preprocessing', ColumnTransformer(transformers)), ('model', estimator)])


def check_preprocessing(pipeline, fitted_rows, features, name, stage):
    """Confirm numeric medians/category vocabulary match only the intended fit rows."""
    numeric = [column for column in features if column not in CATEGORIES]
    categorical = [column for column in features if column in CATEGORIES]
    learned = pipeline.named_steps['preprocessing']
    medians = fitted_rows[numeric].median().fillna(0)
    np.testing.assert_allclose(learned.named_transformers_['numeric'].statistics_, medians)
    rows = []
    for column, value in zip(numeric, learned.named_transformers_['numeric'].statistics_):
        rows.append({'configuration': name, 'stage': stage, 'fit_rows': len(fitted_rows),
                     'feature': column, 'statistic': 'numeric median', 'value': value})
    if categorical:
        encoder = learned.named_transformers_['categorical'].named_steps['encoder']
        for column, categories in zip(categorical, encoder.categories_):
            assert set(categories) == set(fitted_rows[column].dropna().unique())
            rows.append({'configuration': name, 'stage': stage, 'fit_rows': len(fitted_rows),
                         'feature': column, 'statistic': 'category vocabulary', 'value': '; '.join(categories)})
    return rows


def main():
    summaries = ROOT / 'reports/summaries'
    models = ROOT / 'models'
    local = ROOT / 'data/processed'
    figures = ROOT / 'reports/figures'
    final_metrics_path = summaries / 'consumption_final_test_metrics.csv'
    final_predictions_path = local / 'consumption_final_test_predictions.csv'
    # Protect the one-time final evaluation and its evidence from accidental reruns.
    # An incomplete run after final evaluation must be recovered without retesting.
    if final_metrics_path.exists() or final_predictions_path.exists():
        raise FileExistsError('Preserved FINAL TEST outputs already exist. Stop: do not rerun evaluation or overwrite them.')
    for directory in [summaries, models, local, figures]:
        directory.mkdir(parents=True, exist_ok=True)
    split_summary = pd.read_csv(summaries / 'split_summary.csv')
    expected = split_summary.loc[split_summary.module.eq('consumption')].set_index('partition')
    partitions = {}
    input_copies = {}
    columns = ['CONSUMER_NO', 'PERIOD', 'TARGET_PERIOD', TARGET] + CONSUMPTION_PREDICTORS
    # Read each existing file directly; boundaries are verified against Step 6 metadata.
    # No period masks, shuffling or new partition construction occur here.
    for name in ['TRAIN', 'VALIDATION', 'FINAL_TEST', 'FUTURE']:
        path = local / 'splits' / f'consumption_{name.lower()}.csv'
        input_copies[path] = path.read_bytes()
        frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
        for column in [TARGET] + CONSUMPTION_PREDICTORS:
            if column in ['TARRIF', 'SOLAR_CONSUMER']:
                frame[column] = frame[column].replace('', np.nan)
            else:
                original = frame[column]
                parsed = pd.to_numeric(original, errors='coerce')
                if (original.ne('') & ~np.isfinite(parsed)).any():
                    raise ValueError(f'{name}: invalid numeric {column}')
                frame[column] = parsed
        frame['TARGET_MONTH'] = frame.TARGET_MONTH.astype('Int64').astype(str).replace('<NA>', np.nan)
        periods = pd.PeriodIndex(frame.TARGET_PERIOD, freq='M')
        assert len(frame) == int(expected.loc[name, 'rows'])
        assert frame.CONSUMER_NO.nunique() == int(expected.loc[name, 'unique_consumers'])
        assert str(periods.min()) == expected.loc[name, 'min_target_period']
        assert str(periods.max()) == expected.loc[name, 'max_target_period']
        assert periods.is_monotonic_increasing
        assert frame.CONSUMER_NO.str.strip().ne('').all()
        assert not frame.duplicated(['CONSUMER_NO', 'TARGET_PERIOD']).any()
        if name == 'FUTURE':
            assert frame[TARGET].isna().all()
            assert (periods == pd.Period('2026-04')).all()
            assert (pd.PeriodIndex(frame.PERIOD, freq='M') == pd.Period('2026-03')).all()
        else:
            assert frame[TARGET].notna().all()
        partitions[name] = frame
    train = partitions['TRAIN']
    validation = partitions['VALIDATION']
    print('Loaded verified Step 6 rows:', {name: len(frame) for name, frame in partitions.items()}, flush=True)

    # TRAIN-only collinearity audit. No automatic feature removal.
    numeric = [column for column in CONSUMPTION_PREDICTORS if column not in CATEGORIES]
    correlation = train[numeric].corr()
    related = []
    for i, first in enumerate(numeric):
        for second in numeric[i+1:]:
            r = correlation.loc[first, second]
            if abs(r) >= 0.90:
                related.append({'kind': 'TRAIN pairwise correlation', 'feature_1': first, 'feature_2': second,
                    'correlation': r, 'complete_rows': len(train[[first, second]].dropna()),
                    'note': 'Absolute Pearson correlation >=0.90; coefficient interpretation can be unstable; predictor retained.'})
    for relation in ['RECENT_MEAN_3 = (CURRENT_CONSUMPTION + CONSUMPTION_LAG1 + CONSUMPTION_LAG2)/3 on complete rows.',
                     'CONSUMPTION_TREND_3 = CURRENT_CONSUMPTION - CONSUMPTION_LAG2 on complete rows.',
                     'Independent median imputation can break exact relationships on incomplete rows; strong dependence remains.',
                     'Full one-hot indicators with an intercept also create coefficient redundancy; ordinary least squares handles rank deficiency.']:
        related.append({'kind': 'mathematical relationship', 'feature_1': '', 'feature_2': '',
                        'correlation': np.nan, 'complete_rows': np.nan, 'note': relation})
    complete = train[['CURRENT_CONSUMPTION', 'CONSUMPTION_LAG1', 'CONSUMPTION_LAG2', 'RECENT_MEAN_3', 'CONSUMPTION_TREND_3']].dropna()
    np.testing.assert_allclose(complete.RECENT_MEAN_3, complete[['CURRENT_CONSUMPTION','CONSUMPTION_LAG1','CONSUMPTION_LAG2']].mean(axis=1))
    np.testing.assert_allclose(complete.CONSUMPTION_TREND_3, complete.CURRENT_CONSUMPTION-complete.CONSUMPTION_LAG2)
    pd.DataFrame(related).to_csv(summaries / 'consumption_predictor_relationships.csv', index=False)

    # Exactly four nested MLR sets and the four requested forest configurations.
    mlr1 = ['CURRENT_CONSUMPTION', 'CONSUMPTION_LAG1', 'CONSUMPTION_LAG2', 'CONSUMPTION_LAG3']
    mlr2 = mlr1 + ['RECENT_MEAN_3', 'RECENT_STD_3', 'CONSUMPTION_TREND_3']
    mlr3 = mlr2 + ['SAME_MONTH_LAST_YEAR', 'TARGET_MONTH']
    mlr4 = mlr3 + ['CONTRACT_LOAD', 'TARRIF', 'SOLAR_CONSUMER']
    experiments = []
    for number, features in enumerate([mlr1, mlr2, mlr3, mlr4], start=1):
        experiments.append({'name': f'MLR-{number}', 'algorithm': 'Multiple Linear Regression', 'features': features,
                            'parameters': 'LinearRegression defaults; no scaling', 'estimator': LinearRegression()})
    for number, (trees, depth) in enumerate([(100, 5), (100, 10), (200, 10), (200, None)], start=1):
        experiments.append({'name': f'RF-{number}', 'algorithm': 'Random Forest Regressor', 'features': CONSUMPTION_PREDICTORS.copy(),
            'parameters': f'n_estimators={trees}; max_depth={depth}; random_state=42; n_jobs=-1; other sklearn defaults',
            'estimator': RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=-1)})
    comparison = []
    baseline = {'configuration': 'Persistence', 'algorithm': 'Persistence baseline', 'features': 'CURRENT_CONSUMPTION', 'hyperparameters': 'None; no fitting'}
    for label, frame in [('TRAIN', train), ('VALIDATION', validation)]:
        result = metrics(frame[TARGET], frame.CURRENT_CONSUMPTION)
        for key, value in result.items():
            baseline[label + '_' + key] = value
    comparison.append(baseline)
    fitted = {}
    preprocessing_audit = []
    for experiment in experiments:
        name = experiment['name']
        features = experiment['features']
        print('Fitting', name, experiment['parameters'], flush=True)
        pipeline = make_pipeline(features, experiment['estimator'])
        pipeline.fit(train[features], train[TARGET])
        preprocessing_audit += check_preprocessing(pipeline, train, features, name, 'TRAIN_ONLY')
        row = {'configuration': name, 'algorithm': experiment['algorithm'], 'features': '; '.join(features),
               'hyperparameters': experiment['parameters']}
        for label, frame in [('TRAIN', train), ('VALIDATION', validation)]:
            prediction = pipeline.predict(frame[features])
            result = metrics(frame[TARGET], prediction)
            for key, value in result.items():
                row[label + '_' + key] = value
        comparison.append(row)
        fitted[name] = pipeline
        print(f"{name}: TRAIN RMSE={row['TRAIN_RMSE']:.4f}; VALIDATION RMSE={row['VALIDATION_RMSE']:.4f}", flush=True)
    comparison = pd.DataFrame(comparison)
    for metric in ['MAE', 'RMSE', 'R2']:
        comparison['VALIDATION_minus_TRAIN_' + metric] = comparison['VALIDATION_' + metric] - comparison['TRAIN_' + metric]
    assert len(comparison) == 9
    comparison.to_csv(summaries / 'consumption_model_comparison.csv', index=False)

    # Selection is frozen BEFORE any final-test prediction or metric calculation.
    # Baseline is the benchmark; select among the eight fitted model experiments.
    # Sort by validation RMSE, then MAE for an exact RMSE tie; listed order is a
    # deterministic simplicity tie-break (MLR before RF, smaller sets first).
    ranked = comparison.loc[comparison.configuration.ne('Persistence')].sort_values(['VALIDATION_RMSE', 'VALIDATION_MAE'], kind='stable')
    selected_name = ranked.iloc[0].configuration
    selected_experiment = next(experiment for experiment in experiments if experiment['name'] == selected_name)
    selected_features = selected_experiment['features']
    selected = fitted[selected_name]
    frozen_parameters = selected.named_steps['model'].get_params().copy()
    selection = ranked.head(1).copy()
    selection['reason'] = 'Lowest VALIDATION RMSE among eight fitted experiments. MAE, generalization gaps and complexity are recorded; no FINAL TEST result used. Persistence remains the reference.'
    selection['baseline_validation_RMSE'] = baseline['VALIDATION_RMSE']
    selection['frozen_before_final_test'] = True
    selection.to_csv(summaries / 'consumption_model_selection.csv', index=False)
    joblib.dump(selected, models / 'consumption_frozen_train_pipeline.joblib')
    print('Selected and frozen:', selected_name, flush=True)

    # The selected TRAIN-fitted pipeline and baseline are evaluated once only.
    test = partitions['FINAL_TEST']
    predicted = selected.predict(test[selected_features])
    baseline_prediction = test.CURRENT_CONSUMPTION.to_numpy()
    final_results = []
    for name, values in [(selected_name, predicted), ('Persistence', baseline_prediction)]:
        result = metrics(test[TARGET], values)
        result.update({'configuration': name, 'partition': 'FINAL_TEST'})
        final_results.append(result)
    saved_test = test[['CONSUMER_NO', 'PERIOD', 'TARGET_PERIOD']].copy()
    saved_test['actual_consumption'] = test[TARGET]
    saved_test['selected_prediction'] = predicted
    saved_test['baseline_prediction'] = baseline_prediction
    saved_test['residual_actual_minus_predicted'] = test[TARGET]-predicted
    saved_test.to_csv(final_predictions_path, index=False)
    pd.DataFrame(final_results).to_csv(final_metrics_path, index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(test[TARGET], predicted, s=7, alpha=0.3)
    low = min(test[TARGET].min(), predicted.min())
    high = max(test[TARGET].max(), predicted.max())
    ax.plot([low, high], [low, high], '--', color='black', label='Perfect agreement')
    ax.set(title=f'Consumption FINAL TEST: actual vs predicted ({selected_name})', xlabel='Actual consumption (source units)', ylabel='Predicted consumption (source units)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'consumption_final_test_actual_vs_predicted.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, test[TARGET]-predicted, s=7, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--')
    ax.set(title=f'Consumption FINAL TEST residuals ({selected_name})', xlabel='Predicted consumption (source units)', ylabel='Actual minus predicted (source units)')
    fig.tight_layout()
    fig.savefig(figures / 'consumption_final_test_residuals.png', dpi=160)
    plt.close(fig)

    # Clone discards ALL fitted preprocessing/model state while preserving settings.
    # Refit the whole pipeline, not just its regression model; FUTURE is excluded.
    labelled = pd.concat([train, validation, test], ignore_index=True)
    assert labelled[TARGET].notna().all()
    assert len(labelled) == len(train)+len(validation)+len(test)
    refitted = clone(selected)
    assert refitted.named_steps['model'].get_params() == frozen_parameters
    print('Refitting entire frozen configuration on', len(labelled), 'labelled rows.', flush=True)
    refitted.fit(labelled[selected_features], labelled[TARGET])
    preprocessing_audit += check_preprocessing(refitted, labelled, selected_features, selected_name, 'REFIT_ALL_LABELLED')
    assert refitted.named_steps['model'].get_params() == frozen_parameters
    joblib.dump(refitted, models / 'consumption_april2026_pipeline.joblib')
    future = partitions['FUTURE']
    april = refitted.predict(future[selected_features])
    assert np.isfinite(april).all()
    output = future[['CONSUMER_NO', 'TARGET_PERIOD']].copy()
    output['predicted_consumption'] = april
    assert len(output) == future.CONSUMER_NO.nunique()
    output.to_csv(summaries / 'next_month_consumption_predictions.csv', index=False)
    pd.DataFrame(preprocessing_audit).to_csv(summaries / 'consumption_preprocessing_audit.csv', index=False)
    coverage = {'selected_configuration': selected_name, 'labelled_refit_rows': len(labelled),
        'future_rows': len(future), 'predictions': len(output), 'unique_consumers': output.CONSUMER_NO.nunique(),
        'coverage_percent': 100*len(output)/len(future), 'target_period': '2026-04',
        'finite_predictions': int(np.isfinite(april).sum()), 'negative_predictions': int((april < 0).sum()),
        'minimum_prediction': april.min(), 'maximum_prediction': april.max(),
        'notes': 'No April actuals or accuracy metrics. Predictions are not clipped. Synthetic utility history; extremes and distribution changes limit generalization.'}
    pd.DataFrame([coverage]).to_csv(summaries / 'consumption_prediction_coverage.csv', index=False)
    for path, original in input_copies.items():
        assert path.read_bytes() == original, f'Input split changed: {path.name}'
    print(pd.DataFrame(final_results).to_string(index=False), flush=True)
    print('April prediction coverage:', coverage, flush=True)
    print('PASS: nine comparisons, VALIDATION-only selection, two FINAL TEST evaluations, whole-pipeline refit, no April metrics. No Git operations.', flush=True)


if __name__ == '__main__':
    main()
