"""Step 4: descriptive EDA and three planned consumer-level statistical analyses.
Run python src/eda.py. No targets, ML features, splits or models are created.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pearsonr
from scipy.stats.contingency import expected_freq
from cleaning import NUMERIC, DATES

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['CONSUMPTION_UNIT', 'BILL_AMOUNT', 'PAYMENT_AMOUNT', 'CLOSING_ARREARS', 'CONTRACT_LOAD']
LABELS = {'CONSUMPTION_UNIT': 'Consumption (source units)', 'BILL_AMOUNT': 'Bill amount (source currency units)',
          'PAYMENT_AMOUNT': 'Payment amount (source currency units)', 'CLOSING_ARREARS': 'Closing arrears (source currency units)',
          'CONTRACT_LOAD': 'Contract load (source units)'}


def save_figure(fig, filename):
    """Apply consistent spacing, save the chart and release its memory."""
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/figures' / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    # WHAT/WHY: Read strings first so numeric-looking identifiers remain identifiers.
    # HOW: Reuse only Step 3 column lists; importing cleaning does not run it.
    # OUTPUT: Temporary typed data for EDA; the input CSV is never written.
    source = ROOT / 'data/processed/consumer_cleaned.csv'
    before_bytes = source.read_bytes()
    data = pd.read_csv(source, dtype=str, keep_default_na=False)
    summaries = ROOT / 'reports/summaries'
    figures = ROOT / 'reports/figures'
    summaries.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    periods = pd.to_datetime(data['PERIOD'], format='%Y-%m', errors='coerce')
    expected = pd.period_range('2023-04', '2026-03', freq='M').astype(str).tolist()
    print(f'Rows: {len(data)}; columns: {len(data.columns)}; unique consumers: {data.CONSUMER_NO.nunique()}')
    print(f'PERIOD: {data.PERIOD.min()} to {data.PERIOD.max()}; months: {data.PERIOD.nunique()}')
    if data.shape != (73578, 91) or sorted(data.PERIOD.unique()) != expected or periods.isna().any():
        raise ValueError('Unexpected cleaned-data shape or coverage; no automatic correction.')
    if data.CONSUMER_NO.str.strip().eq('').any() or data.duplicated(['CONSUMER_NO', 'PERIOD']).any():
        raise ValueError('Missing consumer IDs or duplicate consumer-months; review input.')
    for column in NUMERIC:
        converted = pd.to_numeric(data[column], errors='coerce')
        if (data[column].str.strip().ne('') & converted.isna()).any():
            raise ValueError(f'Unexpected nonblank numeric conversion failure: {column}')
        if (converted.notna() & ~np.isfinite(converted)).any():
            raise ValueError(f'Nonfinite numeric values need review: {column}')
        data[column] = converted
    for column in DATES:
        converted = pd.to_datetime(data[column], format='%Y-%m-%d', errors='coerce')
        if (data[column].str.strip().ne('') & converted.isna()).any():
            raise ValueError(f'Unexpected nonblank date conversion failure: {column}')
        data[column] = converted
    overview = {'total_rows': len(data), 'total_columns': len(data.columns),
                'unique_consumers': data.CONSUMER_NO.nunique(), 'earliest_month': data.PERIOD.min(),
                'latest_month': data.PERIOD.max(), 'total_months': data.PERIOD.nunique(),
                'numeric_columns': len(data.select_dtypes(include='number').columns),
                'categorical_object_columns': len(data.select_dtypes(include=['object', 'string']).columns),
                'date_columns': len(DATES)}
    pd.DataFrame([overview]).to_csv(summaries / 'eda_overview.csv', index=False)

    # WHAT/HOW: Summarise all observed values, including genuine extremes.
    # OUTPUT: Monthly-row descriptive statistics, not independent-consumer tests.
    stats_rows = []
    for column in FIELDS:
        values = data[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        stats_rows.append({'variable': column, 'count': len(values), 'mean': values.mean(),
                           'median': values.median(), 'standard_deviation': values.std(),
                           'minimum': values.min(), 'Q1': q1, 'Q3': q3, 'maximum': values.max(),
                           'IQR': q3 - q1, 'skewness': values.skew(), 'kurtosis': values.kurt()})
    statistics = pd.DataFrame(stats_rows)
    statistics.to_csv(summaries / 'descriptive_statistics.csv', index=False)
    monthly = data.groupby('PERIOD').agg(unique_consumers=('CONSUMER_NO', 'nunique'),
              total_consumption=('CONSUMPTION_UNIT', 'sum'), average_consumption=('CONSUMPTION_UNIT', 'mean'),
              total_bills=('BILL_AMOUNT', 'sum'), total_payments=('PAYMENT_AMOUNT', 'sum'),
              total_closing_arrears=('CLOSING_ARREARS', 'sum'))
    monthly.to_csv(summaries / 'eda_monthly_summary.csv')
    for column, title, ylabel, filename in [
        ('total_consumption', 'Monthly total electricity consumption', LABELS['CONSUMPTION_UNIT'], 'monthly_total_consumption.png'),
        ('average_consumption', 'Monthly average electricity consumption per observed consumer', LABELS['CONSUMPTION_UNIT'], 'monthly_average_consumption.png'),
        ('unique_consumers', 'Monthly unique consumers', 'Unique consumers', 'monthly_consumer_growth.png')]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly.index, monthly[column], marker='o', markersize=3)
        ax.set(title=title + ' | descriptive', xlabel='Period', ylabel=ylabel)
        ax.set_xticks(range(0, len(monthly), 3), monthly.index[::3], rotation=45)
        ax.grid(alpha=0.25)
        save_figure(fig, filename)

    # Histograms include every value. Log frequency exposes sparse tails without trimming.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data.CONSUMPTION_UNIT.dropna(), bins=60, log=True, color='#287b8e')
    ax.set(title='Consumption distribution | all monthly observations', xlabel=LABELS['CONSUMPTION_UNIT'], ylabel='Monthly observations (log scale)')
    save_figure(fig, 'consumption_distribution.png')
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.boxplot(data.CONSUMPTION_UNIT.dropna(), orientation='horizontal', flierprops={'markersize': 2, 'alpha': 0.3})
    ax.set_xscale('symlog', linthresh=1)
    ax.set(title='Consumption boxplot | extremes retained', xlabel='Consumption (source units; symmetric log scale)', yticks=[])
    save_figure(fig, 'consumption_boxplot.png')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i in range(4):
        # Place the four variables in a two-row, two-column chart.
        ax = axes[i // 2, i % 2]
        column = FIELDS[i + 1]
        ax.hist(data[column].dropna(), bins=60, log=True, color='#287b8e')
        ax.set(title=column.replace('_', ' ').title(), xlabel=LABELS[column], ylabel='Monthly rows (log scale)')
    fig.suptitle('Billing, payment, arrears and load distributions | no extremes removed')
    save_figure(fig, 'financial_and_load_distributions.png')

    category_rows = []
    for column in ['TARRIF', 'SOLAR_CONSUMER', 'METER_STATUS']:
        counts = data[column].replace('', '[missing]').value_counts()
        for category, count in counts.items():
            category_rows.append({'variable': column, 'category': category, 'monthly_rows': count,
                                  'percent': 100 * count / len(data)})
        fig, ax = plt.subplots(figsize=(8, 4))
        displayed_counts = counts.head(15).sort_values()
        displayed_counts.plot.barh(ax=ax, color='#287b8e')
        ax.set(title=f'{column}: monthly category counts | descriptive', xlabel='Monthly observations (not unique consumers)', ylabel='Category')
        save_figure(fig, column.lower() + '_distribution.png')
    pd.DataFrame(category_rows).to_csv(summaries / 'eda_category_counts.csv', index=False)

    # Each hexagon counts monthly observations in that area of the plot.
    # This keeps all complete pairs visible without overlapping 73,578 points.
    # Log colour shows both common values and rare extremes; no rows are removed.
    for x, y, filename in [('CONSUMPTION_UNIT', 'BILL_AMOUNT', 'consumption_vs_bill.png'),
                            ('BILL_AMOUNT', 'PAYMENT_AMOUNT', 'bill_vs_payment.png')]:
        pair = data[[x, y]].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        chart = ax.hexbin(pair[x], pair[y], gridsize=45, bins='log', mincnt=1, cmap='viridis')
        fig.colorbar(chart, ax=ax, label='Monthly observations per bin (log colour scale)')
        ax.set(title=f'{x} vs {y}\nAll complete monthly pairs; descriptive', xlabel=LABELS[x], ylabel=LABELS[y])
        save_figure(fig, filename)

    group_rows = []
    for column in ['TARRIF', 'SOLAR_CONSUMER']:
        grouped = data.groupby(column).CONSUMPTION_UNIT.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        for category, row in grouped.iterrows():
            result = row.to_dict()
            result.update({'group_variable': column, 'category': category, 'percent_of_monthly_rows': 100 * row['count'] / len(data)})
            group_rows.append(result)
        fig, ax = plt.subplots(figsize=(9, 4))
        groups = []
        labels = []
        for category in grouped.index:
            groups.append(data.loc[data[column].eq(category), 'CONSUMPTION_UNIT'].dropna())
            labels.append(category)
        ax.boxplot(groups, tick_labels=labels, flierprops={'markersize': 2, 'alpha': 0.25})
        ax.set_yscale('symlog', linthresh=1)
        ax.set(title=f'Consumption by {column} | monthly observations', xlabel=column,
               ylabel='Consumption (source units; symmetric log scale)')
        save_figure(fig, 'consumption_by_' + column.lower() + '.png')
    pd.DataFrame(group_rows).to_csv(summaries / 'eda_consumption_groups.csv', index=False)
    # Grouping by calendar month is temporary EDA only, not a saved model feature.
    calendar = data.groupby(periods.dt.month).CONSUMPTION_UNIT.agg(['count', 'mean', 'median'])
    calendar.index.name = 'calendar_month'
    calendar.to_csv(summaries / 'eda_calendar_month.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(calendar.index, calendar['mean'], marker='o', label='Mean')
    ax.plot(calendar.index, calendar['median'], marker='s', label='Median')
    ax.set(xticks=range(1, 13), xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
           title='Calendar-month consumption pattern | pooled 2023–2026 observations', xlabel='Calendar month', ylabel=LABELS['CONSUMPTION_UNIT'])
    ax.legend()
    save_figure(fig, 'monthly_consumption_pattern.png')
    correlation = data[FIELDS].corr()  # Descriptive coefficients only, no additional tests.
    fig, ax = plt.subplots(figsize=(8, 6))
    chart = ax.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r')
    labels = ['Consumption', 'Bill amount', 'Payment', 'Closing arrears', 'Contract load']
    ax.set(xticks=range(5), yticks=range(5), xticklabels=labels, yticklabels=labels,
           title='Selected numeric correlations | monthly rows, descriptive only')
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment('right')
    for i in range(5):
        for j in range(5):
            text_color = 'black'
            if abs(correlation.iloc[i,j]) > 0.65:
                text_color = 'white'
            ax.text(j, i, f'{correlation.iloc[i,j]:.2f}', ha='center', va='center', color=text_color)
    fig.colorbar(chart, ax=ax, label='Pearson coefficient (no hypothesis tests)')
    save_figure(fig, 'correlation_heatmap.png')

    # WHAT: Exactly two Pearson tests, each with one average per consumer.
    # WHY: Monthly rows for the same consumer are not independent consumers.
    # OUTPUT: Aggregate results only; temporary consumer averages are not saved.
    alpha = 0.05
    consumer_means = data.groupby('CONSUMER_NO')[['CONSUMPTION_UNIT', 'BILL_AMOUNT', 'PAYMENT_AMOUNT']].mean()
    tests = []
    for x, y in [('CONSUMPTION_UNIT', 'BILL_AMOUNT'), ('BILL_AMOUNT', 'PAYMENT_AMOUNT')]:
        pair = consumer_means[[x, y]].dropna()
        result = {'Question': f'Is {x} linearly associated with {y}?', 'Variables': f'{x}; {y}',
                  'H0': 'Population linear correlation is zero.', 'H1': 'Population linear correlation is nonzero.',
                  'Method': 'Pearson correlation (two-sided)', 'Sample definition': 'One available-history mean per consumer; pairwise complete averages; unequal history lengths retained.',
                  'Sample size': len(pair), 'Statistic': np.nan, 'Degrees of freedom': np.nan, 'p-value': np.nan,
                  'alpha': alpha, 'Direction': '', 'Status': 'skipped'}
        note = 'Consumers, not monthly rows, are observations. Consumer independence is assumed; shared conditions may remain. Severe skewness/extremes can influence r and p; p-values are exploratory. A displayed zero p-value means numerical underflow, not exact zero.'
        if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
            result['Interpretation'] = 'Insufficient complete or variable consumer averages; test skipped.'
        else:
            r, p = pearsonr(pair[x], pair[y])
            if r > 0:
                direction = 'positive'
            elif r < 0:
                direction = 'negative'
            else:
                direction = 'zero'
            if p < alpha:
                evidence = 'Evidence of a nonzero linear association at 5%.'
            else:
                evidence = 'Insufficient evidence of a nonzero linear association at 5%.'
            result.update({'Statistic': r, 'Degrees of freedom': len(pair)-2, 'p-value': p, 'Direction': direction, 'Status': 'performed',
                           'Interpretation': f'{evidence} Direction: {direction}. Positive means values tend to increase together; negative means opposite movement; near zero means weak linear association. Association is not causation.'})
        result['Assumption/check note'] = note
        tests.append(result)

    # Stable, valid categories across every observed month give one category per consumer.
    # Do not silently choose the latest status or merge rare tariffs to obtain a test.
    # These are the non-placeholder tariff codes inspected in the Step 3 data.
    # An unfamiliar code is excluded from this test for review, not erased from EDA.
    valid_tariffs = ['RGPU', 'NRGP', 'LTMD', 'DTR', 'TMP', 'GLP.SL', 'A2', 'GLP']
    category_history = data.groupby('CONSUMER_NO')[['TARRIF', 'SOLAR_CONSUMER']].nunique()
    valid_tariff = data['TARRIF'].isin(valid_tariffs)
    valid_solar = data['SOLAR_CONSUMER'].isin(['Y', 'N'])
    invalid = ~valid_tariff | ~valid_solar
    invalid_consumers = data.loc[invalid, 'CONSUMER_NO'].unique()
    stable_tariff = category_history['TARRIF'] == 1
    stable_solar = category_history['SOLAR_CONSUMER'] == 1
    stable = stable_tariff & stable_solar
    reliable_ids = category_history.index[stable]
    included = data['CONSUMER_NO'].isin(reliable_ids)
    included = included & ~data['CONSUMER_NO'].isin(invalid_consumers)
    category_columns = ['CONSUMER_NO', 'TARRIF', 'SOLAR_CONSUMER']
    consumer_categories = data.loc[included, category_columns]
    # Categories are already checked as constant, so keeping one row is safe.
    consumer_categories = consumer_categories.drop_duplicates('CONSUMER_NO')
    table = pd.crosstab(consumer_categories.TARRIF, consumer_categories.SOLAR_CONSUMER)
    table.to_csv(summaries / 'chi_square_observed.csv')
    excluded = data.CONSUMER_NO.nunique() - len(consumer_categories)
    check = {'total_consumers': data.CONSUMER_NO.nunique(), 'changing_tariff': int(category_history.TARRIF.gt(1).sum()),
             'changing_solar': int(category_history.SOLAR_CONSUMER.gt(1).sum()), 'invalid_category_consumers': len(invalid_consumers),
             'excluded_union': excluded, 'included_consumers': len(consumer_categories)}
    result = {'Question': 'Is solar-consumer status associated with tariff category?', 'Variables': 'TARRIF; SOLAR_CONSUMER',
              'H0': 'TARRIF and SOLAR_CONSUMER are independent.', 'H1': 'TARRIF and SOLAR_CONSUMER are associated.',
              'Method': 'Chi-square independence', 'Sample definition': 'One consumer with valid unchanged categories in every observed month; changing or invalid histories excluded.',
              'Sample size': len(consumer_categories), 'Statistic': np.nan, 'Degrees of freedom': np.nan, 'p-value': np.nan,
              'alpha': alpha, 'Direction': 'not applicable', 'Status': 'skipped'}
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        # SciPy calculates row total * column total / grand total for each cell.
        # This only calculates expected counts; it does not run a statistical test.
        expected_counts = expected_freq(table)
        expected_table = pd.DataFrame(expected_counts, index=table.index, columns=table.columns)
        expected_table.to_csv(summaries / 'chi_square_expected.csv')
        smallest = expected_counts.min()
        cells_below_five = (expected_counts < 5).sum()
        small_percent = 100 * cells_below_five / expected_counts.size
        check.update({'minimum_expected': smallest, 'percent_expected_below_5': small_percent})
        note = f'{excluded} consumers excluded. Expected-count rule: none below 1 and at most 20% below 5. Minimum={smallest:.4f}; below 5={small_percent:.2f}%. Independent consumers assumed; shared conditions may remain.'
        result['Degrees of freedom'] = (table.shape[0]-1)*(table.shape[1]-1)
        assumptions_suitable = smallest >= 1 and small_percent <= 20
        if assumptions_suitable:
            statistic, p, degrees, unused_expected = chi2_contingency(table, correction=False)
            interpretation = 'There is statistical evidence of an association between tariff category and solar-consumer status in this exploratory sample.'
            if p >= alpha:
                interpretation = 'There is insufficient statistical evidence of an association between tariff category and solar-consumer status at the 5% significance level.'
            result.update({'Statistic': statistic, 'p-value': p, 'Status': 'performed', 'Interpretation': interpretation + ' Association does not imply causation.'})
        else:
            result['Interpretation'] = 'Test not performed: sparse expected cells make the Chi-square approximation unsuitable. No category merging or replacement test applied.'
    else:
        note = 'Fewer than two valid categories on an axis.'
        result['Interpretation'] = 'Test skipped: contingency table is not suitable.'
    result['Assumption/check note'] = note
    tests.append(result)
    pd.DataFrame([check]).to_csv(summaries / 'chi_square_sample_checks.csv', index=False)
    pd.DataFrame(tests).to_csv(summaries / 'statistical_tests.csv', index=False)

    findings = [
        {'Finding': 'Consumer base changes over time', 'Evidence': f'{monthly.unique_consumers.iloc[0]} to {monthly.unique_consumers.iloc[-1]} unique monthly consumers.', 'Modelling implication': 'Consider changing exposure when examining temporal behaviour later.'},
        {'Finding': 'Consumption varies by calendar month', 'Evidence': f'Highest pooled monthly mean: month {calendar["mean"].idxmax()}; lowest: month {calendar["mean"].idxmin()}. Growth and tariff mix may contribute.', 'Modelling implication': 'Examine historical patterns during authorised Step 5; no feature selected here.'},
        {'Finding': 'Zero use and zero bills remain', 'Evidence': f'{int(data.CONSUMPTION_UNIT.eq(0).sum())} zero-consumption rows; {int(data.BILL_AMOUNT.le(0).sum())} nonpositive bills.', 'Modelling implication': 'Retain observations; later revenue eligibility must be explicit.'},
        {'Finding': 'Extremes and skewness remain', 'Evidence': '; '.join(f'{row.variable} skew={row.skewness:.2f}' for row in statistics.itertuples()), 'Modelling implication': 'Investigate skewness later; do not automatically remove extremes.'},
        {'Finding': 'Solar status can change', 'Evidence': f'{check["changing_solar"]} consumers change solar status; {excluded} excluded from Chi-square sample.', 'Modelling implication': 'Respect information available at each month in later work.'},
        {'Finding': 'Chi-square assumption check', 'Evidence': result['Interpretation'], 'Modelling implication': 'No feature selection from this exploratory check.'},
    ]
    for test in tests[:2]:
        findings.append({'Finding': test['Variables'], 'Evidence': f'Consumer means: n={test["Sample size"]}; r={test["Statistic"]:.4f}; p={test["p-value"]:.4g}.', 'Modelling implication': 'Descriptive association only; assess target-specific relevance in Step 6 using TRAIN only.'})
    tariff_means = data.groupby('TARRIF').CONSUMPTION_UNIT.mean()
    solar_means = data.groupby('SOLAR_CONSUMER').CONSUMPTION_UNIT.mean()
    findings.extend([
        {'Finding': 'Consumption differs across tariff groups',
         'Evidence': f'Highest monthly mean: {tariff_means.idxmax()} ({tariff_means.max():.2f}); lowest: {tariff_means.idxmin()} ({tariff_means.min():.2f}) source units.',
         'Modelling implication': 'Categories may require encoding later; this is not a feature-selection decision.'},
        {'Finding': 'Solar and non-solar consumption distributions differ',
         'Evidence': '; '.join(f'{category}: monthly mean {value:.2f}' for category, value in solar_means.items()) + '. Tariff mix and changing status can confound this comparison.',
         'Modelling implication': 'Do not interpret group differences as an adoption effect.'},
        {'Finding': 'Arrears have a long upper tail',
         'Evidence': f'Mean {data.CLOSING_ARREARS.mean():.2f}; median {data.CLOSING_ARREARS.median():.2f}; maximum {data.CLOSING_ARREARS.max():.2f} source currency units.',
         'Modelling implication': 'Examine extreme balances during later modelling without automatically deleting them.'},
        {'Finding': 'Payments need not equal bills in the same month',
         'Evidence': f'{int(data.PAYMENT_AMOUNT.gt(data.BILL_AMOUNT).sum())} monthly rows have payments above bills; payment timing and prior balances may contribute.',
         'Modelling implication': 'Later risk eligibility and timing rules need explicit definitions; no target is created here.'},
    ])
    limitations = ['Synthetic/anonymized demonstration data may not generalize to real utilities.',
                  'Monthly rows repeat consumers; descriptive pooled summaries weight longer histories more.',
                  'Association does not imply causation; consumer independence and test assumptions matter.',
                  'Consumer averages lose month-to-month information and use unequal available histories.',
                  'Full-period EDA is descriptive and does not select or reject predictive features.',
                  'Extreme utility values are not automatically errors; skewness can distort Pearson summaries.',
                  'Kurtosis is pandas excess kurtosis (normal reference zero); standard deviation uses sample ddof=1.',
                  'No season variable created. Calendar-month differences may reflect growth and category mix, not only seasonality.']
    for note in limitations:
        findings.append({'Finding': 'Limitation / interpretation note', 'Evidence': note, 'Modelling implication': 'Carry this limitation into later interpretation; no automatic feature decision.'})
    pd.DataFrame(findings).to_csv(summaries / 'eda_findings.csv', index=False)
    if source.read_bytes() != before_bytes:
        raise RuntimeError('Cleaned input changed during EDA; investigate before using outputs.')
    print('PASS: cleaned input byte-identical; no targets, features, splits or models created.')
    print(pd.DataFrame(tests)[['Method', 'Sample size', 'Statistic', 'p-value', 'Status']].to_string(index=False))
    print('Chi-square checks:', check)
    print('Saved aggregate CSV summaries and 15 descriptive figures.')


if __name__ == '__main__':
    main()
