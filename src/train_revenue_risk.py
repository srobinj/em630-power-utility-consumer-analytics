"""Step 8: fixed revenue classifier experiments, final test, then April refit.
Run python src/train_revenue_risk.py. Optional --compare-only stops before final test.
A saved comparison checkpoint lets the normal command continue without retraining.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, roc_curve, confusion_matrix)
from features import REVENUE_PREDICTORS

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'AT_RISK_NEXT'
NUMERIC = [column for column in REVENUE_PREDICTORS if column != 'TARRIF']


def pipeline_for(estimator):
    """Fit all learned preprocessing with the classifier inside one pipeline."""
    numeric = SimpleImputer(strategy='median', keep_empty_features=True)
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                           ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessing = ColumnTransformer([('numeric', numeric, NUMERIC), ('categorical', categorical, ['TARRIF'])])
    return Pipeline([('preprocessing', preprocessing), ('model', estimator)])


def predictions(pipeline, frame, name, stage, audit):
    """Find Class 1 explicitly; classifier.predict uses the unchanged sklearn rule."""
    classes = pipeline.named_steps['model'].classes_.tolist()
    assert set(classes) == {0, 1}, f'{name}: expected both original classes, got {classes}'
    class_one_column = classes.index(1)
    probabilities = pipeline.predict_proba(frame[REVENUE_PREDICTORS])[:, class_one_column]
    labels = pipeline.predict(frame[REVENUE_PREDICTORS])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.isin(labels, [0, 1]).all()
    audit.append({'configuration': name, 'stage': stage, 'classes': str(classes), 'class_1_column': class_one_column,
                  'rows': len(frame), 'decision_rule': 'Unchanged sklearn predict: maximum probability; exact tie selects Class 0.'})
    return labels, probabilities


def score(actual, predicted, probability):
    """Both-class metrics prevent majority-class accuracy/F1 from hiding failure."""
    precision, recall, f1, counts = precision_recall_fscore_support(actual, predicted, labels=[0, 1], zero_division=0)
    result = {'sample_count': len(actual), 'Accuracy': accuracy_score(actual, predicted),
              'ROC_AUC': roc_auc_score(actual, probability), 'AP': average_precision_score(actual, probability),
              'predicted_class_0': int((predicted == 0).sum()), 'predicted_class_1': int((predicted == 1).sum())}
    for label in [0, 1]:
        result[f'class_{label}_count'] = int(counts[label])
        result[f'class_{label}_percent'] = 100*counts[label]/len(actual)
        result[f'Precision_{label}'] = precision[label]
        result[f'Recall_{label}'] = recall[label]
        result[f'F1_{label}'] = f1[label]
    return result


def check_preprocessing(pipeline, frame, name, stage):
    """Check that the intended fit population produced medians and categories."""
    preprocessing = pipeline.named_steps['preprocessing']
    values = preprocessing.named_transformers_['numeric'].statistics_
    np.testing.assert_allclose(values, frame[NUMERIC].median().fillna(0))
    rows = []
    for column, value in zip(NUMERIC, values):
        rows.append({'configuration': name, 'stage': stage, 'fit_rows': len(frame), 'feature': column,
                     'statistic': 'numeric median', 'value': value})
    categories = preprocessing.named_transformers_['categorical'].named_steps['encoder'].categories_[0]
    assert set(categories) == set(frame.TARRIF.dropna().unique())
    rows.append({'configuration': name, 'stage': stage, 'fit_rows': len(frame), 'feature': 'TARRIF',
                 'statistic': 'category vocabulary', 'value': '; '.join(categories)})
    return rows


def rank_candidates(table):
    """Validation ranking: discrimination first, Class-1 F1 next, then minority F1.

    Remaining exact ties prefer a smaller overfit gap and listed simpler model.
    Full both-class metrics, AP and baseline differences are saved for review.
    No arbitrary minimum ROC-AUC improvement or threshold tuning is introduced.
    """
    ranked = table.copy()
    ranked['AUC_overfit_gap'] = (ranked.TRAIN_ROC_AUC - ranked.VALIDATION_ROC_AUC).clip(lower=0)
    return ranked.sort_values(['VALIDATION_ROC_AUC', 'VALIDATION_F1_1', 'VALIDATION_F1_0', 'AUC_overfit_gap'],
                              ascending=[False, False, False, True], kind='stable')


def main():
    summaries = ROOT / 'reports/summaries'
    models = ROOT / 'models'
    figures = ROOT / 'reports/figures'
    local = ROOT / 'data/processed'
    checkpoint_path = models / 'revenue_comparison_checkpoint.joblib'
    final_metrics_path = summaries / 'revenue_final_test_metrics.csv'
    final_prediction_path = local / 'revenue_final_test_predictions.csv'
    if final_metrics_path.exists() or final_prediction_path.exists():
        raise FileExistsError('Preserved revenue FINAL TEST outputs already exist. Do not overwrite or retest.')
    expected_rows = {'TRAIN': 48212, 'VALIDATION': 12135, 'FINAL_TEST': 10145, 'FUTURE': 2087}
    metadata = pd.read_csv(summaries / 'split_summary.csv')
    metadata = metadata.loc[metadata.module.eq('revenue')].set_index('partition')
    frames = {}
    input_bytes = {}
    columns = ['CONSUMER_NO', 'PERIOD', 'TARGET_PERIOD', TARGET] + REVENUE_PREDICTORS
    for name in expected_rows:
        path = local / 'splits' / f'revenue_{name.lower()}.csv'
        input_bytes[name] = path.read_bytes()
        frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
        for column in NUMERIC + [TARGET]:
            original = frame[column]
            converted = pd.to_numeric(original, errors='coerce')
            assert not (original.ne('') & ~np.isfinite(converted)).any(), f'Invalid {name} {column}'
            frame[column] = converted
        frame['TARRIF'] = frame.TARRIF.replace('', np.nan)
        periods = pd.PeriodIndex(frame.TARGET_PERIOD, freq='M')
        assert len(frame) == expected_rows[name] == int(metadata.loc[name, 'rows'])
        assert frame.CONSUMER_NO.nunique() == int(metadata.loc[name, 'unique_consumers'])
        assert str(periods.min()) == metadata.loc[name, 'min_target_period']
        assert str(periods.max()) == metadata.loc[name, 'max_target_period']
        assert periods.is_monotonic_increasing
        assert frame.CONSUMER_NO.str.strip().ne('').all()
        assert not frame.duplicated(['CONSUMER_NO','TARGET_PERIOD']).any()
        if name == 'FUTURE':
            assert frame[TARGET].isna().all()
            assert (periods == pd.Period('2026-04')).all()
            assert (pd.PeriodIndex(frame.PERIOD, freq='M') == pd.Period('2026-03')).all()
        else:
            assert frame[TARGET].isin([0, 1]).all()
            frame[TARGET] = frame[TARGET].astype(int)
        frames[name] = frame
    train, validation = frames['TRAIN'], frames['VALIDATION']
    prevalence = train[TARGET].mean()
    majority = int(train[TARGET].value_counts().idxmax())
    distributions = []
    for name in ['TRAIN', 'VALIDATION', 'FINAL_TEST']:
        for label in [0, 1]:
            count = int(frames[name][TARGET].eq(label).sum())
            distributions.append({'partition': name, 'class': label, 'count': count, 'percent': 100*count/len(frames[name]),
                                  'definition': 'Academic proxy: Class 1 means less than 50% paid by due date; Class 0 means at least 50%.'})
    pd.DataFrame(distributions).to_csv(summaries / 'revenue_class_distributions.csv', index=False)
    print(f'TRAIN-only baseline: class={majority}, Class-1 probability={prevalence:.8f}', flush=True)

    # A compare-only run can be resumed without repeating any classifier fit.
    if checkpoint_path.exists():
        checkpoint = joblib.load(checkpoint_path)
        assert checkpoint['input_bytes'] == input_bytes, 'Split files changed since comparison.'
        comparison = checkpoint['comparison']
        fitted = checkpoint['fitted']
        validation_probabilities = checkpoint['validation_probabilities']
        class_audit = checkpoint['class_audit']
        preprocessing_audit = checkpoint['preprocessing_audit']
        best_dt, best_rf = checkpoint['best_dt'], checkpoint['best_rf']
        print('Loaded completed ten-classifier comparison; no experiments refitted.', flush=True)
    else:
        class_audit = []
        preprocessing_audit = []
        fitted = {}
        validation_probabilities = {}
        baseline = {'configuration': 'TRAIN-majority', 'algorithm': 'Constant baseline', 'class_weight': 'None',
                    'hyperparameters': f'TRAIN majority={majority}; TRAIN Class-1 prevalence={prevalence}',
                    'features': 'None'}
        for name, frame in [('TRAIN', train), ('VALIDATION', validation)]:
            result = score(frame[TARGET], np.full(len(frame), majority), np.full(len(frame), prevalence))
            for key, value in result.items():
                baseline[name + '_' + key] = value
        rows = [baseline]
        experiments = []
        for number, depth in enumerate([3, 5, 8, None], start=1):
            experiments.append((f'DT-{number}', 'Decision Tree', DecisionTreeClassifier(max_depth=depth, criterion='gini', class_weight=None, random_state=42)))
        for number, (trees, depth) in enumerate([(100, 5), (100, 10), (200, 10), (200, None)], start=1):
            experiments.append((f'RF-{number}', 'Random Forest', RandomForestClassifier(n_estimators=trees, max_depth=depth, class_weight=None, random_state=42, n_jobs=-1)))
        # First eight unweighted models, then exactly one balanced clone per family.
        for phase in ['unweighted', 'balanced']:
            if phase == 'balanced':
                unweighted = pd.DataFrame(rows)
                best_dt = rank_candidates(unweighted.loc[unweighted.algorithm.eq('Decision Tree')]).iloc[0].configuration
                best_rf = rank_candidates(unweighted.loc[unweighted.algorithm.eq('Random Forest')]).iloc[0].configuration
                experiments = []
                for parent, algorithm in [(best_dt, 'Decision Tree'), (best_rf, 'Random Forest')]:
                    estimator = clone(fitted[parent].named_steps['model'])
                    estimator.set_params(class_weight='balanced')
                    parent_parameters = fitted[parent].named_steps['model'].get_params()
                    for key, value in parent_parameters.items():
                        if key != 'class_weight':
                            assert estimator.get_params()[key] == value
                    experiments.append((parent + '-balanced', algorithm, estimator))
                print('Best unweighted DT/RF:', best_dt, best_rf, flush=True)
            for name, algorithm, estimator in experiments:
                print('Fitting', name, flush=True)
                pipeline = pipeline_for(estimator)
                pipeline.fit(train[REVENUE_PREDICTORS], train[TARGET])
                preprocessing_audit += check_preprocessing(pipeline, train, name, 'TRAIN_ONLY')
                row = {'configuration': name, 'algorithm': algorithm, 'class_weight': str(estimator.class_weight),
                       'hyperparameters': str(estimator.get_params()), 'features': '; '.join(REVENUE_PREDICTORS)}
                for stage, frame in [('TRAIN', train), ('VALIDATION', validation)]:
                    labels, probability = predictions(pipeline, frame, name, stage, class_audit)
                    result = score(frame[TARGET], labels, probability)
                    for key, value in result.items():
                        row[stage + '_' + key] = value
                    if stage == 'VALIDATION':
                        validation_probabilities[name] = probability
                fitted[name] = pipeline
                rows.append(row)
                print(f"{name}: validation AUC={row['VALIDATION_ROC_AUC']:.6f}, F1(1)={row['VALIDATION_F1_1']:.6f}, F1(0)={row['VALIDATION_F1_0']:.6f}", flush=True)
        comparison = pd.DataFrame(rows)
        for metric in ['Accuracy', 'ROC_AUC', 'AP', 'Precision_0', 'Recall_0', 'F1_0', 'Precision_1', 'Recall_1', 'F1_1']:
            comparison['VALIDATION_minus_TRAIN_' + metric] = comparison['VALIDATION_' + metric]-comparison['TRAIN_' + metric]
        assert len(fitted) == 10 and len(comparison) == 11
        checkpoint = {'input_bytes': input_bytes, 'comparison': comparison, 'fitted': fitted,
            'validation_probabilities': validation_probabilities, 'class_audit': class_audit,
            'preprocessing_audit': preprocessing_audit, 'best_dt': best_dt, 'best_rf': best_rf}
        joblib.dump(checkpoint, checkpoint_path, compress=3)
    comparison.to_csv(summaries / 'revenue_model_comparison.csv', index=False)
    weight_effects = []
    for parent in [best_dt, best_rf]:
        original = comparison.loc[comparison.configuration.eq(parent)].iloc[0]
        balanced = comparison.loc[comparison.configuration.eq(parent+'-balanced')].iloc[0]
        effect = {'unweighted_configuration': parent, 'balanced_configuration': parent+'-balanced'}
        for metric in ['ROC_AUC', 'AP', 'Accuracy', 'Precision_0', 'Recall_0', 'F1_0', 'Precision_1', 'Recall_1', 'F1_1']:
            effect['unweighted_' + metric] = original['VALIDATION_' + metric]
            effect['balanced_' + metric] = balanced['VALIDATION_' + metric]
            effect['balanced_minus_unweighted_' + metric] = balanced['VALIDATION_' + metric]-original['VALIDATION_' + metric]
        weight_effects.append(effect)
    pd.DataFrame(weight_effects).to_csv(summaries / 'revenue_class_weight_effect.csv',index=False)
    # Rank evidence is created before FINAL TEST use. Inspect both classes explicitly.
    candidates = comparison.loc[comparison.configuration.ne('TRAIN-majority')]
    ranked = rank_candidates(candidates)
    # Reviewed choice for this fixed comparison checkpoint, using VALIDATION only.
    # RF-3 has the highest AUC but no Class-0 recall. Its balanced version loses
    # 0.002147 AUC, identifies 39.39% of Class 0, and retains Class-1 F1=0.8567.
    # This is an explicit trade-off, NOT a rule that balanced weighting must win.
    # Accuracy and Class-1 F1 decrease, and its AUC overfit gap is slightly larger.
    selected_name = 'RF-3-balanced'
    selected_row = ranked.loc[ranked.configuration.eq(selected_name)].iloc[0]
    selected = fitted[selected_name]
    assert selected_row.VALIDATION_ROC_AUC > 0.5, 'No ranking discrimination above baseline; review validation only.'
    assert selected_row.VALIDATION_Recall_0 > 0 and selected_row.VALIDATION_Recall_1 > 0, 'Selected classifier ignores a class; review validation only.'
    print(ranked[['configuration','VALIDATION_ROC_AUC','VALIDATION_F1_1','VALIDATION_Precision_0','VALIDATION_Recall_0','VALIDATION_F1_0','VALIDATION_AP','AUC_overfit_gap']].to_string(index=False), flush=True)
    print('Validation candidate:', selected_name, flush=True)
    if '--compare-only' in sys.argv:
        print('Stopped after comparison. FINAL TEST not predicted/evaluated. Normal command resumes this checkpoint.', flush=True)
        return

    # Freeze the decision and its validation evidence before touching final predictions.
    selection = ranked.loc[ranked.configuration.eq(selected_name)].copy()
    selection['ROC_AUC_improvement_over_baseline'] = selected_row.VALIDATION_ROC_AUC - 0.5
    selection['reason'] = 'VALIDATION-only reviewed trade-off: RF-3-balanced AUC=0.639027 versus highest unweighted RF-3 AUC=0.641174; Class-0 recall/F1 improve from zero to 0.393876/0.280754. Class-1 F1 decreases from 0.937018 to 0.856705; AP is similar (~0.9201), accuracy decreases and AUC overfit gap is slightly larger. Chosen for meaningful both-class predictions with near-best ranking; no arbitrary minimum AUC margin or threshold tuning. Balanced DT is weaker on AUC and both-class F1.'
    selection['best_unweighted_DT'] = best_dt
    selection['best_unweighted_RF'] = best_rf
    selection['frozen_before_final_test'] = True
    selection.to_csv(summaries / 'revenue_model_selection.csv', index=False)
    frozen_parameters = selected.named_steps['model'].get_params().copy()
    joblib.dump(selected, models / 'revenue_frozen_train_pipeline.joblib', compress=3)
    # Relevant ROC curves: unweighted family winners, balanced variants and baseline.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in [best_dt, best_rf, best_dt+'-balanced', best_rf+'-balanced']:
        fpr, tpr, thresholds = roc_curve(validation[TARGET], validation_probabilities[name])
        auc = comparison.loc[comparison.configuration.eq(name), 'VALIDATION_ROC_AUC'].iloc[0]
        ax.plot(fpr, tpr, label=f'{name}: AUC {auc:.4f}')
    ax.plot([0,1],[0,1],'--',color='grey',label='TRAIN-majority: AUC 0.5000')
    ax.set(title='Revenue proxy: VALIDATION ROC comparison', xlabel='False positive rate (Class 1)', ylabel='True positive rate (Class 1)')
    ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(figures / 'revenue_validation_roc.png', dpi=160); plt.close(fig)

    test = frames['FINAL_TEST']
    labels, probability = predictions(selected, test, selected_name, 'FINAL_TEST', class_audit)
    baseline_labels = np.full(len(test), majority)
    baseline_probability = np.full(len(test), prevalence)
    final_rows = []
    for name, predicted, probabilities in [(selected_name, labels, probability), ('TRAIN-majority', baseline_labels, baseline_probability)]:
        result = score(test[TARGET], predicted, probabilities)
        result.update({'configuration': name, 'partition': 'FINAL_TEST'})
        final_rows.append(result)
    saved = test[['CONSUMER_NO','PERIOD','TARGET_PERIOD']].copy()
    saved['actual_class'] = test[TARGET]
    saved['predicted_class'] = labels
    saved['class_1_probability'] = probability
    saved['baseline_class'] = majority
    saved['baseline_class_1_probability'] = prevalence
    saved.to_csv(final_prediction_path, index=False)
    pd.DataFrame(final_rows).to_csv(final_metrics_path, index=False)
    matrix = confusion_matrix(test[TARGET], labels, labels=[0,1])
    pd.DataFrame(matrix, index=['actual_0','actual_1'], columns=['predicted_0','predicted_1']).to_csv(summaries / 'revenue_final_test_confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Class 0','Class 1'],yticklabels=['Class 0','Class 1'],
           xlabel='Predicted',ylabel='Actual',title=f'Revenue proxy FINAL TEST: {selected_name}')
    fig.tight_layout(); fig.savefig(figures / 'revenue_final_test_confusion_matrix.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,6))
    fpr,tpr,thresholds = roc_curve(test[TARGET], probability)
    ax.plot(fpr,tpr,label=f"{selected_name}: AUC {final_rows[0]['ROC_AUC']:.4f}")
    ax.plot([0,1],[0,1],'--',color='grey',label='TRAIN-majority: AUC 0.5000')
    ax.set(title='Revenue proxy FINAL TEST ROC',xlabel='False positive rate (Class 1)',ylabel='True positive rate (Class 1)')
    ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(figures / 'revenue_final_test_roc.png',dpi=160); plt.close(fig)
    importance = pd.DataFrame({'encoded_feature': selected.named_steps['preprocessing'].get_feature_names_out(),
        'impurity_importance': selected.named_steps['model'].feature_importances_})
    importance['note'] = 'Frozen TRAIN-fitted classifier; impurity importance is descriptive, biased by cardinality/correlation, not a new selection step.'
    importance.to_csv(summaries / 'revenue_feature_importance.csv',index=False)
    fig, ax = plt.subplots(figsize=(9,6))
    displayed = importance.sort_values('impurity_importance').tail(15)
    ax.barh(displayed.encoded_feature,displayed.impurity_importance)
    ax.set(title=f'Revenue proxy: {selected_name} TRAIN-fitted importance',xlabel='Impurity-based importance')
    fig.tight_layout(); fig.savefig(figures / 'revenue_feature_importance.png',dpi=160); plt.close(fig)

    # Entire pipeline clone: no TRAIN-only preprocessing state is reused for refit.
    labelled = pd.concat([train,validation,test],ignore_index=True)
    assert labelled[TARGET].notna().all() and len(labelled)==70492
    refitted = clone(selected)
    assert refitted.named_steps['model'].get_params()==frozen_parameters
    refitted.fit(labelled[REVENUE_PREDICTORS],labelled[TARGET])
    preprocessing_audit += check_preprocessing(refitted,labelled,selected_name,'REFIT_ALL_LABELLED')
    assert refitted.named_steps['model'].get_params()==frozen_parameters
    joblib.dump(refitted,models / 'revenue_april2026_pipeline.joblib',compress=3)
    future = frames['FUTURE']
    april_class,april_probability = predictions(refitted,future,selected_name,'APRIL_REFIT',class_audit)
    output = future[['CONSUMER_NO','TARGET_PERIOD']].copy()
    output['predicted_risk_class'] = april_class
    output['predicted_class_1_probability'] = april_probability
    assert len(output)==output.CONSUMER_NO.nunique()==2087
    output.to_csv(summaries / 'next_month_revenue_risk_predictions.csv',index=False)
    coverage = {'configuration': selected_name, 'refit_labelled_rows': len(labelled), 'target_period': '2026-04',
        'predictions': len(output), 'unique_consumers': output.CONSUMER_NO.nunique(), 'coverage_percent': 100*len(output)/len(future),
        'predicted_class_0': int((april_class==0).sum()),'predicted_class_1': int((april_class==1).sum()),
        'minimum_probability': april_probability.min(),'maximum_probability': april_probability.max(),
        'meaning': 'Academic less-than-50%-paid-by-due-date proxy only; not official default/risk probability. No April outcomes or metrics.'}
    pd.DataFrame([coverage]).to_csv(summaries / 'revenue_prediction_coverage.csv',index=False)
    pd.DataFrame(class_audit).to_csv(summaries / 'revenue_probability_column_audit.csv',index=False)
    pd.DataFrame(preprocessing_audit).to_csv(summaries / 'revenue_preprocessing_audit.csv',index=False)
    for name, original in input_bytes.items():
        assert (local / 'splits' / f'revenue_{name.lower()}.csv').read_bytes()==original
    print(pd.DataFrame(final_rows).to_string(index=False),flush=True)
    print('April coverage:',coverage,flush=True)
    print('PASS: ten classifiers, TRAIN-only baseline/preprocessing, fixed selection, preserved test, full refit. No Git commands.',flush=True)


if __name__ == '__main__':
    main()
