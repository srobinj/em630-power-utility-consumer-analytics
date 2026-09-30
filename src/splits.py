"""Step 6: chronological module splits and TRAIN-only exploratory relevance.
Run python src/splits.py. No models, balancing, imputation or target changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from scipy.stats.contingency import expected_freq
from features import CONSUMPTION_PREDICTORS, REVENUE_PREDICTORS, ANOMALY_PREDICTORS

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'consumption': ('CONSUMPTION_NEXT', CONSUMPTION_PREDICTORS),
           'revenue': ('AT_RISK_NEXT', REVENUE_PREDICTORS),
           'anomaly': ('ANOMALY_NEXT', ANOMALY_PREDICTORS)}
CATEGORICAL = ['TARRIF', 'SOLAR_CONSUMER']
BOUNDARIES = {'TRAIN': ('2023-05', '2025-04'), 'VALIDATION': ('2025-05', '2025-10'),
              'FINAL_TEST': ('2025-11', '2026-03'), 'FUTURE': ('2026-04', '2026-04')}


def relevance(train, module, target, predictors):
    """This function receives ONLY eligible TRAIN rows, never other partitions."""
    assert train.TARGET_PERIOD.max() <= pd.Period('2025-04')
    results = []
    cells = []
    for feature in predictors:
        common = {'module': module, 'feature': feature, 'target': target,
            'train_target_min': str(train.TARGET_PERIOD.min()), 'train_target_max': str(train.TARGET_PERIOD.max()),
            'method': '', 'sample_size': 0, 'statistic': np.nan, 'degrees_of_freedom': np.nan, 'p_value': np.nan,
            'alpha': 0.05, 'group': '', 'mean': np.nan, 'median': np.nan, 'standard_deviation': np.nan,
            'train_rows': len(train), 'train_consumers': train.CONSUMER_NO.nunique(),
            'missing_predictor_rows': int(train[feature].isna().sum()),
            'excluded_consumers': np.nan, 'changing_feature_consumers': np.nan, 'changing_target_consumers': np.nan,
            'invalid_category_consumers': np.nan, 'minimum_expected': np.nan, 'percent_expected_below_5': np.nan,
            'assumption_check_note': '', 'status': '', 'interpretation': ''}
        if feature in CATEGORICAL:
            common['missing_predictor_rows'] = int(train[feature].str.strip().eq('').sum())
        # Calendar month is numeric in storage, but represents a cyclic category.
        # A January-to-December linear Pearson ordering is not imposed on it.
        descriptive_group = feature in CATEGORICAL or feature == 'TARGET_MONTH'
        if module == 'consumption' and descriptive_group:
            grouped = train.copy()
            if feature in CATEGORICAL:
                grouped[feature] = grouped[feature].replace('', '[missing]')
            for category, group in grouped.groupby(feature, dropna=False):
                row = common.copy()
                row.update({'method': 'Descriptive', 'sample_size': len(group), 'group': str(category),
                    'mean': group[target].mean(), 'median': group[target].median(), 'standard_deviation': group[target].std(),
                    'status': 'DESCRIPTIVE', 'assumption_check_note': 'Eligible TRAIN monthly rows; repeated consumers allowed for description, not independent inference. TARGET_MONTH is cyclic, so no linear calendar-month test.',
                    'interpretation': 'Target distribution by predictor category; no inferential significance or feature decision.'})
                results.append(row)
        elif module == 'consumption':
            # Average predictor and target over the SAME complete TRAIN pairs.
            # Each consumer then contributes one observation to Pearson inference.
            paired = train[['CONSUMER_NO', feature, target]].dropna()
            consumers = paired.groupby('CONSUMER_NO')[[feature, target]].mean()
            row = common.copy()
            row.update({'method': 'Pearson (two-sided; consumer paired means)', 'sample_size': len(consumers),
                        'excluded_consumers': train.CONSUMER_NO.nunique()-len(consumers)})
            note = (f'Only complete eligible TRAIN pairs; {len(paired)} paired monthly rows aggregated to one mean per consumer. '
                    'Unequal history lengths and aggregation lose timing information. Independent consumers are assumed; shared conditions may remain. '
                    'Missingness, extremes and skewness can influence Pearson r/p. Unadjusted exploratory p-values, not selection thresholds. Zero p is numerical underflow.')
            if len(consumers) < 3 or consumers[feature].nunique() < 2 or consumers[target].nunique() < 2:
                row.update({'status': 'SKIPPED', 'interpretation': 'Insufficient complete consumers or no variation; Pearson not performed.'})
            else:
                r, p = pearsonr(consumers[feature], consumers[target])
                direction = 'positive'
                if r < 0:
                    direction = 'negative'
                elif r == 0:
                    direction = 'zero'
                note += f' Consumer-mean skewness: predictor={consumers[feature].skew():.3f}, target={consumers[target].skew():.3f}.'
                row.update({'statistic': r, 'degrees_of_freedom': len(consumers)-2, 'p_value': p, 'status': 'PERFORMED',
                    'interpretation': f'{direction.capitalize()} linear association between consumer TRAIN means; not causal and not a feature-selection rule.'})
            row['assumption_check_note'] = note
            results.append(row)
        elif feature not in CATEGORICAL:
            # These summaries intentionally describe longitudinal TRAIN rows.
            # No numeric class-comparison hypothesis test is performed.
            for label in [0, 1]:
                values = train.loc[train[target].eq(label), feature].dropna()
                row = common.copy()
                row.update({'method': 'Descriptive', 'sample_size': len(values), 'group': f'Class {label}',
                    'mean': values.mean(), 'median': values.median(), 'standard_deviation': values.std(), 'status': 'DESCRIPTIVE',
                    'assumption_check_note': 'Nonmissing predictor values in eligible TRAIN monthly rows for this target class. Repeated consumers are not treated as independent for a test. TARGET_MONTH, if present, is cyclic; its numeric summary is descriptive only.',
                    'interpretation': 'Class-specific predictor distribution; no inferential test or automatic feature decision.'})
                results.append(row)
        else:
            # A binary target can also change across months. Never invent a
            # majority-class or ever-risk consumer label for this statistical test.
            # Require BOTH feature and target to be constant in eligible TRAIN.
            history = train.groupby('CONSUMER_NO')[[feature, target]].nunique(dropna=False)
            changing_feature = history[feature].gt(1)
            changing_target = history[target].gt(1)
            if feature == 'SOLAR_CONSUMER':
                valid = train[feature].isin(['Y', 'N'])
            else:
                valid = train[feature].isin(['RGPU', 'NRGP', 'LTMD', 'DTR', 'TMP', 'GLP.SL', 'A2', 'GLP'])
            invalid_ids = train.loc[~valid, 'CONSUMER_NO'].unique()
            stable_ids = history.index[~changing_feature & ~changing_target]
            included = train.CONSUMER_NO.isin(stable_ids) & ~train.CONSUMER_NO.isin(invalid_ids)
            consumers = train.loc[included, ['CONSUMER_NO', feature, target]].drop_duplicates('CONSUMER_NO')
            observed = pd.crosstab(consumers[feature], consumers[target])
            row = common.copy()
            row.update({'method': 'Chi-square (stable consumer categories)', 'sample_size': len(consumers),
                'excluded_consumers': len(history)-len(consumers), 'changing_feature_consumers': int(changing_feature.sum()),
                'changing_target_consumers': int(changing_target.sum()), 'invalid_category_consumers': len(invalid_ids), 'status': 'SKIPPED'})
            note = ('One observation per consumer; feature AND target must be constant within eligible TRAIN history. '
                    'Changing/missing/invalid categories excluded only for this test. No majority or ever-risk label invented. '
                    'This stable-consumer subset may not represent changing consumers. Consumer independence assumed; unadjusted exploratory inference. '
                    'Expected rule: no cells <1 and at most 20% <5; no category merging.')
            if not observed.empty:
                expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)
                row['minimum_expected'] = expected.min().min()
                row['percent_expected_below_5'] = 100 * (expected < 5).to_numpy().sum()/expected.size
                for category in observed.index:
                    for label in observed.columns:
                        cells.append({'module': module, 'feature': feature, 'target': target, 'category': category,
                                      'target_class': int(label), 'observed': int(observed.loc[category, label]),
                                      'expected': expected.loc[category, label], 'sample': 'stable eligible TRAIN consumers'})
            if observed.shape[0] < 2 or observed.shape[1] < 2:
                row['interpretation'] = 'SKIPPED: fewer than two observed categories or target classes in the stable-consumer subset.'
            else:
                row['degrees_of_freedom'] = (observed.shape[0]-1)*(observed.shape[1]-1)
                if row['minimum_expected'] < 1 or row['percent_expected_below_5'] > 20:
                    row['interpretation'] = 'SKIPPED: sparse expected frequencies fail the stated assumptions.'
                else:
                    statistic, p, degrees, expected_array = chi2_contingency(observed, correction=False)
                    interpretation = 'Evidence of association at alpha=0.05 within the stable TRAIN-consumer subset.'
                    if p >= 0.05:
                        interpretation = 'Insufficient evidence of association at alpha=0.05 within the stable TRAIN-consumer subset.'
                    row.update({'statistic': statistic, 'degrees_of_freedom': degrees, 'p_value': p,
                                'status': 'PERFORMED', 'interpretation': interpretation + ' No causal or feature-selection claim.'})
            row['assumption_check_note'] = note
            results.append(row)
    return results, cells


def rationale_rows(module, predictors, significance):
    """Keep candidates provisionally; later validation evaluates predictive usefulness."""
    domains = {
        'CURRENT_CONSUMPTION': 'Current demand level provides recent consumption context.',
        'CONSUMPTION_LAG1': 'One-month history may describe short-term persistence.',
        'CONSUMPTION_LAG2': 'Two-month history may describe recent variation.',
        'CONSUMPTION_LAG3': 'Three-month history may describe recent variation.',
        'RECENT_MEAN_3': 'Recent level over three complete calendar observations.',
        'RECENT_STD_3': 'Recent variability over three complete calendar observations.',
        'CONSUMPTION_TREND_3': 'Recent change relative to two calendar months earlier.',
        'SAME_MONTH_LAST_YEAR': 'Target-month-aligned historical consumption provides seasonal context.',
        'TARGET_MONTH': 'Known target calendar month provides seasonal context; cyclic, not a linear ordering.',
        'CONTRACT_LOAD': 'Contracted load provides demand-capacity context.',
        'TARRIF': 'Recorded tariff category describes the billing/customer segment.',
        'SOLAR_CONSUMER': 'Current solar status describes a potentially different consumption pattern.',
        'PAYMENT_RATIO_CURRENT': 'Payment observed by current cutoff provides recent payment context.',
        'PAYMENT_RATIO_LAG1': 'Prior ratio frozen at its own cutoff describes observable payment history.',
        'PAYMENT_RATIO_MEAN3': 'Complete three-month frozen ratios describe recent payment history.',
        'PAYMENT_DELAY_CURRENT': 'Observed payment timing relative to deadline describes payment behaviour.',
        'ARREARS_CURRENT': 'Recorded month-end outstanding balance provides revenue exposure context.',
        'ARREARS_LAG1': 'Exact previous-month outstanding balance provides historical context.',
        'ARREARS_TREND': 'Balance movement describes recent recorded arrears change.',
        'BILL_AMOUNT': 'Current bill size provides revenue exposure context.',
        'TENURE_MONTHS': 'Connection history length provides customer-history context.'}
    rows = []
    for feature in predictors:
        evidence = []
        for result in significance:
            if result['module'] != module or result['feature'] != feature:
                continue
            if result['status'] == 'PERFORMED':
                evidence.append(f"{result['method']}: n={result['sample_size']}, statistic={result['statistic']:.4g}, p={result['p_value']:.4g}.")
            elif result['status'] == 'SKIPPED':
                evidence.append(result['interpretation'])
            else:
                evidence.append(f"Descriptive {result['group']}: n={result['sample_size']}, mean={result['mean']:.4g}, median={result['median']:.4g}.")
        timing = 'Month t or earlier, as defined in Step 5; monthly snapshot availability assumed.'
        if feature.startswith('PAYMENT_'):
            timing = 'Only payments observed by the applicable feature month-end; historical ratios frozen at their own cutoffs.'
        elif feature == 'SAME_MONTH_LAST_YEAR':
            timing = 't-11 = TARGET_PERIOD minus 12 months; strictly historical.'
        elif feature == 'TARGET_MONTH':
            timing = 'Calendar of t+1 is known at t; contains no future outcome.'
        rows.append({'module': module, 'feature': feature, 'statistical_evidence': ' '.join(evidence),
            'domain_relevance': domains[feature], 'temporal_availability': timing,
            'leakage_assessment': 'Step 5 candidate only; no target, identity or label-only audit field. Relevance uses eligible TRAIN only. Source revision timestamps are unavailable.',
            'keep_drop_decision': 'KEEP - provisional candidate',
            'reason': 'Combine domain relevance and time-safe availability with exploratory evidence. No p-value-only decision. Actual predictive usefulness and redundancy remain for later model validation.'})
    return rows


def main():
    source = ROOT / 'data/processed/consumer_features.csv'
    before = source.read_bytes()
    data = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = ['CONSUMER_NO', 'PERIOD', 'TARGET_PERIOD', 'FEATURE_CUTOFF_DATE']
    numeric = []
    for target, predictors in MODULES.values():
        required += [target] + predictors
        numeric += [target] + [feature for feature in predictors if feature not in CATEGORICAL]
    missing = sorted(set(required)-set(data.columns))
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    for column in ['PERIOD', 'TARGET_PERIOD']:
        # Step 5 CSVs may serialize a monthly Period as YYYY-MM or month-end date.
        # Accept these explicit formats only; never reinterpret arbitrary dates.
        if data[column].str.fullmatch(r'\d{4}-\d{2}').all():
            dates = pd.to_datetime(data[column], format='%Y-%m', errors='raise')
        elif data[column].str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            dates = pd.to_datetime(data[column], format='%Y-%m-%d', errors='raise')
            assert dates.dt.is_month_end.all(), f'{column} contains non-month-end dates.'
        else:
            raise ValueError(f'Unexpected or mixed monthly format in {column}.')
        data[column] = dates.dt.to_period('M')
    for column in sorted(set(numeric)):
        original = data[column]
        converted = pd.to_numeric(original, errors='coerce')
        if (original.str.strip().ne('') & (converted.isna() | ~np.isfinite(converted))).any():
            raise ValueError(f'Nonblank invalid numeric values in {column}; input not changed.')
        data[column] = converted
    assert data.TARGET_PERIOD.eq(data.PERIOD+1).all()
    assert not data.duplicated(['CONSUMER_NO', 'PERIOD']).any()
    assert data.CONSUMER_NO.str.strip().ne('').all()
    assert data.TARGET_PERIOD.min() == pd.Period('2023-05')
    assert data.TARGET_PERIOD.max() == pd.Period('2026-04')
    for module, (target, predictors) in MODULES.items():
        known = data[target].dropna()
        if module == 'consumption':
            assert known.ge(0).all()
        else:
            assert known.isin([0, 1]).all()
    data = data.sort_values(['TARGET_PERIOD', 'CONSUMER_NO']).reset_index(drop=True)
    local = ROOT / 'data/processed/splits'
    local.mkdir(parents=True, exist_ok=True)
    summaries = ROOT / 'reports/summaries'
    split_rows = []
    all_significance = []
    all_cells = []
    all_rationale = []
    for module, (target, predictors) in MODULES.items():
        period_sets = []
        train = None
        for name, (start, end) in BOUNDARIES.items():
            window = data.loc[data.TARGET_PERIOD.between(pd.Period(start), pd.Period(end))]
            if name == 'FUTURE':
                part = window.copy()
                excluded = 0
                assert part.PERIOD.eq(pd.Period('2026-03')).all()
                assert part[target].isna().all()
            else:
                part = window.loc[window[target].notna()].copy()
                excluded = int(window[target].isna().sum())
                assert part[target].notna().all()
            assert not part.empty
            assert part.TARGET_PERIOD.is_monotonic_increasing
            assert part.TARGET_PERIOD.max() == pd.Period(end)
            if name != 'TRAIN':
                assert part.TARGET_PERIOD.min() == pd.Period(start)
            periods = set(part.TARGET_PERIOD.unique())
            for previous in period_sets:
                assert not periods.intersection(previous)
            period_sets.append(periods)
            row = {'module': module, 'target': target, 'partition': name, 'window_start': start, 'window_end': end,
                'rows_before_target_filter': len(window), 'rows': len(part), 'unique_consumers': part.CONSUMER_NO.nunique(),
                'min_target_period': str(part.TARGET_PERIOD.min()), 'max_target_period': str(part.TARGET_PERIOD.max()),
                'excluded_target_unavailable': excluded, 'known_targets': int(part[target].notna().sum()),
                'missing_targets_retained': int(part[target].isna().sum()),
                'count': np.nan, 'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'Q1': np.nan, 'Q3': np.nan, 'max': np.nan,
                'class_0': np.nan, 'class_0_percent': np.nan, 'class_1': np.nan, 'class_1_percent': np.nan}
            if name != 'FUTURE':
                values = part[target]
                if module == 'consumption':
                    row.update({'count': len(values), 'mean': values.mean(), 'median': values.median(), 'std': values.std(),
                        'min': values.min(), 'Q1': values.quantile(.25), 'Q3': values.quantile(.75), 'max': values.max()})
                else:
                    for label in [0, 1]:
                        row[f'class_{label}'] = int(values.eq(label).sum())
                        row[f'class_{label}_percent'] = 100 * values.eq(label).mean()
            split_rows.append(row)
            part.to_csv(local / f'{module}_{name.lower()}.csv', index=False)
            if name == 'TRAIN':
                train = part.copy()
        # Relevance receives no held-out data. Descriptive split summaries above
        # are the only target-distribution inspection of the held-out partitions.
        results, cells = relevance(train, module, target, predictors)
        all_significance.extend(results)
        all_cells.extend(cells)
        all_rationale.extend(rationale_rows(module, predictors, results))
    summary = pd.DataFrame(split_rows)
    significance = pd.DataFrame(all_significance)
    rationale = pd.DataFrame(all_rationale)
    assert significance.train_target_max.eq('2025-04').all()
    assert len(rationale) == sum(len(predictors) for target, predictors in MODULES.values())
    assert rationale.keep_drop_decision.eq('KEEP - provisional candidate').all()
    assert source.read_bytes() == before
    summary.to_csv(summaries / 'split_summary.csv', index=False)
    significance.to_csv(summaries / 'feature_significance.csv', index=False)
    rationale.to_csv(summaries / 'feature_selection_rationale.csv', index=False)
    pd.DataFrame(all_cells).to_csv(summaries / 'feature_relevance_contingencies.csv', index=False)
    print(summary[['module', 'partition', 'rows', 'unique_consumers', 'min_target_period', 'max_target_period', 'excluded_target_unavailable', 'class_0', 'class_1']].to_string(index=False))
    print('Inferential results:')
    print(significance.loc[significance.method.ne('Descriptive'), ['module', 'feature', 'sample_size', 'statistic', 'degrees_of_freedom', 'p_value', 'status', 'excluded_consumers']].to_string(index=False))
    print('PASS: chronological partitions, TRAIN-only relevance, independent-consumer inference, unchanged source.')
    print('All candidates retained provisionally. No models, balancing, weighting, target changes or Git commands.')


if __name__ == '__main__':
    main()
