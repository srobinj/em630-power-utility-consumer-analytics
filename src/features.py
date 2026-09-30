"""Step 5: calendar-aware predictors and academic next-month targets.
Run python src/features.py. No splitting, selection, imputation or modelling.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COLUMNS = '''CONSUMER_NO PERIOD MONTH YEAR CONSUMPTION_UNIT BILL_AMOUNT
PAYMENT_AMOUNT PAYMENT_DATE DUE_DATE LAST_PAYDATE CLOSING_ARREARS CONNECTION_DATE
CONTRACT_LOAD TARRIF SOLAR_CONSUMER STATUS BILL_TYPE'''.split()
CONSUMPTION_PREDICTORS = '''CURRENT_CONSUMPTION CONSUMPTION_LAG1 CONSUMPTION_LAG2
CONSUMPTION_LAG3 RECENT_MEAN_3 RECENT_STD_3 CONSUMPTION_TREND_3 SAME_MONTH_LAST_YEAR
TARGET_MONTH CONTRACT_LOAD TARRIF SOLAR_CONSUMER'''.split()
REVENUE_PREDICTORS = '''PAYMENT_RATIO_CURRENT PAYMENT_RATIO_LAG1 PAYMENT_RATIO_MEAN3
PAYMENT_DELAY_CURRENT ARREARS_CURRENT ARREARS_LAG1 ARREARS_TREND CURRENT_CONSUMPTION
BILL_AMOUNT TENURE_MONTHS TARRIF CONTRACT_LOAD'''.split()
ANOMALY_PREDICTORS = CONSUMPTION_PREDICTORS.copy()
TARGETS = ['CONSUMPTION_NEXT', 'AT_RISK_NEXT', 'ANOMALY_NEXT']


def calendar_join(frame, history, months_back, columns):
    """Match t to exactly t-months_back. A negative offset matches a future label.

    Moving history's join key forward aligns its original month to the feature
    row that needs it. No row-position shift or next-available match is used.
    columns maps historical column names to output column names.
    """
    lookup = history[['CONSUMER_NO', 'PERIOD'] + list(columns)].copy()
    lookup['PERIOD'] = lookup['PERIOD'] + months_back
    lookup = lookup.rename(columns=columns)
    return frame.merge(lookup, on=['CONSUMER_NO', 'PERIOD'], how='left', validate='one_to_one', sort=False)


def build_features(raw):
    """Calculate only the prescribed features; keep every supplied consumer-month."""
    data = raw[SOURCE_COLUMNS].copy()
    data['PERIOD'] = pd.to_datetime(data.PERIOD, format='%Y-%m', errors='raise').dt.to_period('M')
    data = data.sort_values(['CONSUMER_NO', 'PERIOD']).reset_index(drop=True)
    if data.duplicated(['CONSUMER_NO', 'PERIOD']).any():
        raise ValueError('Duplicate consumer-months prevent safe joins.')
    # WHAT: Parse working copies. Original payment amount/date/deadline stay intact.
    # WHY: Future or missing payments must never be rewritten as zero predictors.
    consumption = pd.to_numeric(data.CONSUMPTION_UNIT, errors='coerce')
    bill = pd.to_numeric(data.BILL_AMOUNT, errors='coerce')
    amount = pd.to_numeric(data.PAYMENT_AMOUNT, errors='coerce')
    payment_date = pd.to_datetime(data.PAYMENT_DATE, format='%Y-%m-%d', errors='coerce')
    due_date = pd.to_datetime(data.DUE_DATE, format='%Y-%m-%d', errors='coerce')
    connection = pd.to_datetime(data.CONNECTION_DATE, format='%Y-%m-%d', errors='coerce')
    data['BILL_AMOUNT'] = bill.where(np.isfinite(bill))
    load = pd.to_numeric(data.CONTRACT_LOAD, errors='coerce')
    data['CONTRACT_LOAD'] = load.where(np.isfinite(load))
    data['FEATURE_CUTOFF_DATE'] = data.PERIOD.dt.to_timestamp(how='end').dt.normalize()
    data['TARGET_PERIOD'] = data.PERIOD + 1
    data['TARGET_MONTH'] = data.TARGET_PERIOD.dt.month
    data['CURRENT_CONSUMPTION'] = consumption.where(np.isfinite(consumption) & consumption.ge(0))
    arrears = pd.to_numeric(data.CLOSING_ARREARS, errors='coerce')
    data['ARREARS_CURRENT'] = arrears.where(np.isfinite(arrears))

    # WHAT: Payment predictors use only payments recorded by month-end t.
    # HOW: Freeze each month's ratio at its own cutoff before making lag joins.
    # A later payment is never used to reconstruct an earlier month's ratio.
    observed = payment_date.notna() & payment_date.le(data.FEATURE_CUTOFF_DATE)
    amount_valid = amount.notna() & np.isfinite(amount)
    bill_valid = bill.gt(0) & np.isfinite(bill)
    ratio_valid = observed & bill_valid & amount_valid
    data['PAYMENT_OBSERVED_BY_CUTOFF'] = observed
    data['PAYMENT_RATIO_CURRENT'] = np.nan
    data.loc[ratio_valid, 'PAYMENT_RATIO_CURRENT'] = amount[ratio_valid] / bill[ratio_valid]
    data['PAYMENT_DELAY_CURRENT'] = (payment_date - due_date).dt.days.where(observed & due_date.notna())

    # WHAT: Outcomes evaluate payment by the bill's due date, not feature cutoff.
    # WHY: These are retrospective label-only fields and never predictors.
    # Invalid nonblank dates remain unknown, rather than being treated as no payment.
    deadline_valid = bill_valid & due_date.notna()
    by_due = payment_date.notna() & due_date.notna() & payment_date.le(due_date)
    no_payment_by_due = due_date.notna() & (data.PAYMENT_DATE.str.strip().eq('') | payment_date.gt(due_date))
    data['PAYMENT_RATIO_OUTCOME'] = np.nan
    paid_valid = deadline_valid & by_due & amount_valid
    data.loc[paid_valid, 'PAYMENT_RATIO_OUTCOME'] = amount[paid_valid] / bill[paid_valid]
    data.loc[deadline_valid & no_payment_by_due, 'PAYMENT_RATIO_OUTCOME'] = 0.0
    for column in ['PAYMENT_RATIO_CURRENT', 'PAYMENT_RATIO_OUTCOME']:
        data[column] = data[column].where(np.isfinite(data[column]))
    data['AT_RISK_CURRENT'] = np.nan
    known_outcome = data.PAYMENT_RATIO_OUTCOME.notna()
    data.loc[known_outcome, 'AT_RISK_CURRENT'] = (data.loc[known_outcome, 'PAYMENT_RATIO_OUTCOME'] < 0.50).astype(int)

    # Tenure is elapsed calendar months to t, never to t+1. Flag negatives locally.
    tenure = (data.PERIOD.dt.year - connection.dt.year) * 12 + data.PERIOD.dt.month - connection.dt.month
    data['NEGATIVE_TENURE_FLAG'] = tenure.lt(0)
    data['TENURE_MONTHS'] = tenure.where(tenure.ge(0))

    # Exact calendar history. Target-month-last-year is t+1-12 = t-11.
    history = data.copy()
    for lag in [1, 2, 3]:
        data = calendar_join(data, history, lag, {'CURRENT_CONSUMPTION': f'CONSUMPTION_LAG{lag}'})
    data = calendar_join(data, history, 11, {'CURRENT_CONSUMPTION': 'SAME_MONTH_LAST_YEAR'})
    data = calendar_join(data, history, 1, {'PAYMENT_RATIO_CURRENT': 'PAYMENT_RATIO_LAG1', 'ARREARS_CURRENT': 'ARREARS_LAG1'})
    data = calendar_join(data, history, 2, {'PAYMENT_RATIO_CURRENT': 'PAYMENT_RATIO_LAG2_AUDIT'})
    recent = data[['CURRENT_CONSUMPTION', 'CONSUMPTION_LAG1', 'CONSUMPTION_LAG2']]
    data['RECENT_MEAN_3'] = recent.mean(axis=1, skipna=False)
    data['RECENT_STD_3'] = recent.std(axis=1, ddof=1, skipna=False)
    data['CONSUMPTION_TREND_3'] = data.CURRENT_CONSUMPTION - data.CONSUMPTION_LAG2
    payments = data[['PAYMENT_RATIO_CURRENT', 'PAYMENT_RATIO_LAG1', 'PAYMENT_RATIO_LAG2_AUDIT']]
    data['PAYMENT_RATIO_MEAN3'] = payments.mean(axis=1, skipna=False)
    data['ARREARS_TREND'] = data.ARREARS_CURRENT - data.ARREARS_LAG1

    # WHAT: Compare consumption to all valid earlier observations for this consumer.
    # HOW: Calculate statistics BEFORE appending the current value to history.
    # This history need not be consecutive; three-month predictors above must be.
    data['PRIOR_COUNT'] = 0
    data['PRIOR_MEAN'] = np.nan
    data['PRIOR_STD'] = np.nan
    for consumer, group in data.groupby('CONSUMER_NO', sort=False):
        earlier_values = []
        for index, current in group.CURRENT_CONSUMPTION.items():
            data.at[index, 'PRIOR_COUNT'] = len(earlier_values)
            if len(earlier_values) > 0:
                data.at[index, 'PRIOR_MEAN'] = np.mean(earlier_values)
            if len(earlier_values) > 1:
                data.at[index, 'PRIOR_STD'] = np.std(earlier_values, ddof=1)
            if pd.notna(current):
                earlier_values.append(current)
    eligible = data.PRIOR_COUNT.ge(3) & data.PRIOR_STD.gt(0) & np.isfinite(data.PRIOR_STD)
    data['RULE_Z_CURRENT'] = ((data.CURRENT_CONSUMPTION - data.PRIOR_MEAN) / data.PRIOR_STD).where(eligible)
    data['RULE_Z_CURRENT'] = data.RULE_Z_CURRENT.where(np.isfinite(data.RULE_Z_CURRENT))
    data['ANOMALY_CURRENT'] = np.nan
    known_z = data.RULE_Z_CURRENT.notna()
    data.loc[known_z, 'ANOMALY_CURRENT'] = data.loc[known_z, 'RULE_Z_CURRENT'].abs().gt(2.5).astype(int)
    data = calendar_join(data, data, -1, {'CURRENT_CONSUMPTION': 'CONSUMPTION_NEXT',
                          'AT_RISK_CURRENT': 'AT_RISK_NEXT', 'ANOMALY_CURRENT': 'ANOMALY_NEXT'})
    return data


def payment_summary(data):
    """Public counts by month; no consumer numbers or individual payment records."""
    amount = pd.to_numeric(data.PAYMENT_AMOUNT, errors='coerce')
    payment = pd.to_datetime(data.PAYMENT_DATE, format='%Y-%m-%d', errors='coerce')
    due = pd.to_datetime(data.DUE_DATE, format='%Y-%m-%d', errors='coerce')
    valid_amount = amount.notna() & np.isfinite(amount)
    observed = data.PAYMENT_OBSERVED_BY_CUTOFF
    by_due = payment.notna() & due.notna() & payment.le(due)
    no_payment = due.notna() & (data.PAYMENT_DATE.str.strip().eq('') | payment.gt(due))
    # Outcome audit counts use the same population as outcome construction.
    # Keep an explicit unknown-date category so eligible rows always reconcile.
    eligible = data.BILL_AMOUNT.gt(0) & np.isfinite(data.BILL_AMOUNT) & due.notna()
    unknown_payment_date = data.PAYMENT_DATE.str.strip().ne('') & payment.isna()
    counts = pd.DataFrame({'PERIOD': data.PERIOD, 'total_rows': 1,
        'outcome_eligible': eligible,
        'positive_bill': data.BILL_AMOUNT.gt(0), 'observed_by_cutoff': observed, 'not_observed_by_cutoff': ~observed,
        'payment_date_null': data.PAYMENT_DATE.str.strip().eq(''), 'payment_date_after_cutoff': payment.gt(data.FEATURE_CUTOFF_DATE),
        'observed_zero_amount': observed & amount.eq(0), 'observed_positive_amount': observed & amount.gt(0),
        'observed_missing_invalid_amount': observed & ~valid_amount,
        'valid_payment_ratio_current': data.PAYMENT_RATIO_CURRENT.notna(), 'missing_payment_ratio_current': data.PAYMENT_RATIO_CURRENT.isna(),
        'valid_payment_delay_current': data.PAYMENT_DELAY_CURRENT.notna(), 'missing_payment_delay_current': data.PAYMENT_DELAY_CURRENT.isna(),
        'valid_payment_ratio_outcome': data.PAYMENT_RATIO_OUTCOME.notna(), 'missing_payment_ratio_outcome': data.PAYMENT_RATIO_OUTCOME.isna(),
        'by_due_valid_amount': eligible & by_due & valid_amount,
        'by_due_missing_invalid_amount': eligible & by_due & ~valid_amount,
        'no_payment_by_due': eligible & no_payment,
        'eligible_unknown_payment_date': eligible & unknown_payment_date,
        'at_risk_current_0': data.AT_RISK_CURRENT.eq(0),
        'at_risk_current_1': data.AT_RISK_CURRENT.eq(1), 'at_risk_current_undefined': data.AT_RISK_CURRENT.isna(),
        'invalid_nonblank_payment_date': data.PAYMENT_DATE.str.strip().ne('') & payment.isna(),
        'negative_tenure': data.NEGATIVE_TENURE_FLAG})
    monthly = counts.groupby('PERIOD').sum().astype(int).reset_index()
    outcome_categories = ['by_due_valid_amount', 'by_due_missing_invalid_amount',
                          'no_payment_by_due', 'eligible_unknown_payment_date']
    assert monthly[outcome_categories].sum(axis=1).eq(monthly.outcome_eligible).all()
    return monthly


def validate_features(data):
    """Independent dictionary lookups validate every calendar join, not just samples."""
    checks = []
    # A dictionary key is the actual (consumer, month), never a row position.
    lookup = {}
    fields = ['CURRENT_CONSUMPTION', 'PAYMENT_RATIO_CURRENT', 'ARREARS_CURRENT', 'AT_RISK_CURRENT', 'ANOMALY_CURRENT']
    for row in data[['CONSUMER_NO', 'PERIOD'] + fields].itertuples(index=False, name=None):
        lookup[(row[0], row[1])] = dict(zip(fields, row[2:]))
    specifications = [('CONSUMPTION_LAG1', 'CURRENT_CONSUMPTION', -1), ('CONSUMPTION_LAG2', 'CURRENT_CONSUMPTION', -2),
        ('CONSUMPTION_LAG3', 'CURRENT_CONSUMPTION', -3), ('SAME_MONTH_LAST_YEAR', 'CURRENT_CONSUMPTION', -11),
        ('PAYMENT_RATIO_LAG1', 'PAYMENT_RATIO_CURRENT', -1), ('PAYMENT_RATIO_LAG2_AUDIT', 'PAYMENT_RATIO_CURRENT', -2),
        ('ARREARS_LAG1', 'ARREARS_CURRENT', -1), ('CONSUMPTION_NEXT', 'CURRENT_CONSUMPTION', 1),
        ('AT_RISK_NEXT', 'AT_RISK_CURRENT', 1), ('ANOMALY_NEXT', 'ANOMALY_CURRENT', 1)]
    for output, original, offset in specifications:
        expected = []
        for consumer, period in zip(data.CONSUMER_NO, data.PERIOD):
            match = lookup.get((consumer, period + offset))
            if match is None:
                expected.append(np.nan)
            else:
                expected.append(match[original])
        np.testing.assert_allclose(data[output], expected, equal_nan=True)
        checks.append({'Check': output + '_exact_calendar_join', 'Rows_checked': len(data), 'Result': 'passed'})
    # Direct prior-history slices for a reproducible sample, including late joiners.
    sample = data.sample(min(250, len(data)), random_state=630)
    for index, row in sample.iterrows():
        prior = data.loc[data.CONSUMER_NO.eq(row.CONSUMER_NO) & data.PERIOD.lt(row.PERIOD), 'CURRENT_CONSUMPTION'].dropna()
        assert row.PRIOR_COUNT == len(prior)
        np.testing.assert_allclose(row.PRIOR_MEAN, prior.mean(), equal_nan=True)
        np.testing.assert_allclose(row.PRIOR_STD, prior.std(ddof=1), equal_nan=True)
    checks.append({'Check': 'strictly_prior_statistics_sample', 'Rows_checked': len(sample), 'Result': 'passed'})
    recent = data[['CURRENT_CONSUMPTION', 'CONSUMPTION_LAG1', 'CONSUMPTION_LAG2']]
    np.testing.assert_allclose(data.RECENT_MEAN_3, recent.mean(axis=1, skipna=False), equal_nan=True)
    np.testing.assert_allclose(data.RECENT_STD_3, recent.std(axis=1, ddof=1, skipna=False), equal_nan=True)
    np.testing.assert_allclose(data.CONSUMPTION_TREND_3, data.CURRENT_CONSUMPTION-data.CONSUMPTION_LAG2, equal_nan=True)
    payment_history = data[['PAYMENT_RATIO_CURRENT', 'PAYMENT_RATIO_LAG1', 'PAYMENT_RATIO_LAG2_AUDIT']]
    np.testing.assert_allclose(data.PAYMENT_RATIO_MEAN3, payment_history.mean(axis=1, skipna=False), equal_nan=True)
    np.testing.assert_allclose(data.ARREARS_TREND, data.ARREARS_CURRENT-data.ARREARS_LAG1, equal_nan=True)
    invalid_anomaly = data.PRIOR_COUNT.lt(3) | data.PRIOR_STD.isna() | data.PRIOR_STD.le(0)
    assert data.loc[invalid_anomaly, ['RULE_Z_CURRENT', 'ANOMALY_CURRENT']].isna().all().all()
    valid_z = data.RULE_Z_CURRENT.notna()
    assert data.loc[valid_z, 'ANOMALY_CURRENT'].eq(data.loc[valid_z, 'RULE_Z_CURRENT'].abs().gt(2.5)).all()
    payment = pd.to_datetime(data.PAYMENT_DATE, errors='coerce', format='%Y-%m-%d')
    due = pd.to_datetime(data.DUE_DATE, errors='coerce', format='%Y-%m-%d')
    amount = pd.to_numeric(data.PAYMENT_AMOUNT, errors='coerce')
    observed = payment.notna() & payment.le(data.FEATURE_CUTOFF_DATE)
    assert data.PAYMENT_OBSERVED_BY_CUTOFF.eq(observed).all()
    assert data.loc[~observed, ['PAYMENT_RATIO_CURRENT', 'PAYMENT_DELAY_CURRENT']].isna().all().all()
    valid_ratio = observed & data.BILL_AMOUNT.gt(0) & np.isfinite(amount)
    np.testing.assert_allclose(data.PAYMENT_RATIO_CURRENT, (amount/data.BILL_AMOUNT).where(valid_ratio), equal_nan=True)
    np.testing.assert_allclose(data.PAYMENT_DELAY_CURRENT, (payment-due).dt.days.where(observed & due.notna()), equal_nan=True)
    by_due_missing = payment.notna() & payment.le(due) & ~np.isfinite(amount)
    assert data.loc[by_due_missing, ['PAYMENT_RATIO_OUTCOME', 'AT_RISK_CURRENT']].isna().all().all()
    no_payment = (data.PAYMENT_DATE.str.strip().eq('') | payment.gt(due)) & due.notna() & data.BILL_AMOUNT.gt(0)
    assert data.loc[no_payment, 'PAYMENT_RATIO_OUTCOME'].eq(0).all()
    assert data.loc[no_payment, 'AT_RISK_CURRENT'].eq(1).all()
    known_outcome = data.PAYMENT_RATIO_OUTCOME.notna()
    assert data.loc[known_outcome, 'AT_RISK_CURRENT'].eq(data.loc[known_outcome, 'PAYMENT_RATIO_OUTCOME'].lt(0.5)).all()
    assert data.loc[~known_outcome, 'AT_RISK_CURRENT'].isna().all()
    assert data.TENURE_MONTHS.dropna().ge(0).all()
    first = data.groupby('CONSUMER_NO', sort=False).head(1)
    assert first[['CONSUMPTION_LAG1', 'CONSUMPTION_LAG2', 'CONSUMPTION_LAG3', 'SAME_MONTH_LAST_YEAR', 'ARREARS_LAG1', 'PAYMENT_RATIO_LAG1']].isna().all().all()
    assert first.PRIOR_COUNT.eq(0).all()
    forbidden = TARGETS + ['CONSUMER_NO', 'PAYMENT_RATIO_OUTCOME', 'AT_RISK_CURRENT', 'ANOMALY_CURRENT', 'RULE_Z_CURRENT', 'PRIOR_MEAN', 'PRIOR_STD']
    for predictors in [CONSUMPTION_PREDICTORS, REVENUE_PREDICTORS, ANOMALY_PREDICTORS]:
        assert not set(predictors).intersection(forbidden)
    for name in ['complete_three_month_windows', 'payment_cutoff_and_due_date_rules', 'first_observation_history_missing', 'anomaly_eligibility_and_threshold', 'predictor_allowlists']:
        checks.append({'Check': name, 'Rows_checked': len(data), 'Result': 'passed'})
    return checks


def edge_case_checks():
    """Small invented examples test absent months and payment cases missing in real data."""
    rows = []
    for period, consumption in [('2025-01', '10'), ('2025-03', '20'), ('2025-04', '30'), ('2025-05', '40')]:
        row = dict.fromkeys(SOURCE_COLUMNS, '')
        row.update({'CONSUMER_NO': 'example', 'PERIOD': period, 'MONTH': period[-2:], 'YEAR': period[:4],
            'CONSUMPTION_UNIT': consumption, 'BILL_AMOUNT': '100', 'PAYMENT_AMOUNT': '50',
            'PAYMENT_DATE': '2025-06-01', 'DUE_DATE': '2025-06-10', 'CONNECTION_DATE': '2025-01-01',
            'CONTRACT_LOAD': '2', 'CLOSING_ARREARS': '10', 'TARRIF': 'RGPU', 'SOLAR_CONSUMER': 'N'})
        rows.append(row)
    # This March payment arrives in April. April must not revise March's ratio.
    rows[1]['PAYMENT_DATE'] = '2025-04-01'
    rows[1]['DUE_DATE'] = '2025-04-05'
    fixture = pd.DataFrame(rows)
    result = build_features(fixture)
    january, march, april, may = [result.iloc[i] for i in range(4)]
    assert pd.isna(january.CONSUMPTION_NEXT)
    assert pd.isna(march.CONSUMPTION_LAG1) and march.CONSUMPTION_LAG2 == 10
    assert pd.isna(march.CONSUMPTION_LAG3) and pd.isna(april.CONSUMPTION_LAG2)
    assert april.CONSUMPTION_LAG3 == 10 and pd.isna(may.CONSUMPTION_LAG3)
    assert pd.isna(april.PAYMENT_RATIO_LAG1)
    assert pd.isna(march.RECENT_MEAN_3) and pd.isna(april.RECENT_STD_3)
    assert may.PRIOR_MEAN == 20 and may.PRIOR_STD == 10 and may.PRIOR_COUNT == 3
    # Changing a later consumption cannot change earlier predictors or prior stats.
    changed = fixture.copy()
    changed.loc[3, 'CONSUMPTION_UNIT'] = '99999'
    altered = build_features(changed)
    columns = list(dict.fromkeys(CONSUMPTION_PREDICTORS + REVENUE_PREDICTORS))
    pd.testing.assert_frame_equal(result.loc[:2, columns], altered.loc[:2, columns])
    assert altered.iloc[3].PRIOR_MEAN == may.PRIOR_MEAN
    # Alter unobserved payment amounts: only label construction may change.
    changed = fixture.copy()
    changed['PAYMENT_AMOUNT'] = '99999'
    altered = build_features(changed)
    pd.testing.assert_frame_equal(result[columns], altered[columns])
    constant = fixture.copy()
    constant['CONSUMPTION_UNIT'] = '10'
    flat = build_features(constant)
    assert flat.iloc[-1].PRIOR_STD == 0 and pd.isna(flat.iloc[-1].ANOMALY_CURRENT)
    # Every case is an independent toy consumer in March, with a positive bill.
    cases = [('zero', '0', '2025-03-20', '2025-03-25'),
             ('missing', '', '2025-03-20', '2025-03-25'),
             ('invalid', 'bad', '2025-03-20', '2025-03-25'),
             ('absent', '99', '', '2025-03-25'),
             ('late', '99', '2025-04-01', '2025-03-25'),
             ('future_on_time', '99', '2025-04-01', '2025-04-05'),
             ('cutoff_boundary', '50', '2025-03-31', '2025-03-31'),
             ('threshold', '50', '2025-03-20', '2025-03-25'),
             ('invalid_due', '50', '2025-03-20', 'bad')]
    payment_rows = []
    for name, amount, date, due in cases:
        row = rows[1].copy()
        row.update({'CONSUMER_NO': name, 'PAYMENT_AMOUNT': amount, 'PAYMENT_DATE': date, 'DUE_DATE': due})
        payment_rows.append(row)
    result = build_features(pd.DataFrame(payment_rows)).set_index('CONSUMER_NO')
    assert result.loc['zero', 'PAYMENT_RATIO_CURRENT'] == 0 and result.loc['zero', 'AT_RISK_CURRENT'] == 1
    assert result.loc['zero', 'PAYMENT_DELAY_CURRENT'] == -5
    assert result.loc[['missing', 'invalid'], ['PAYMENT_RATIO_OUTCOME', 'AT_RISK_CURRENT']].isna().all().all()
    assert result.loc[['absent', 'late'], 'PAYMENT_RATIO_OUTCOME'].eq(0).all()
    assert result.loc[['absent', 'late'], 'AT_RISK_CURRENT'].eq(1).all()
    assert result.loc['future_on_time', 'PAYMENT_RATIO_OUTCOME'] == 0.99
    assert result.loc[['absent', 'late', 'future_on_time'], ['PAYMENT_RATIO_CURRENT', 'PAYMENT_DELAY_CURRENT']].isna().all().all()
    assert result.loc['threshold', 'AT_RISK_CURRENT'] == 0
    assert result.loc['cutoff_boundary', 'PAYMENT_RATIO_CURRENT'] == 0.5
    assert result.loc['cutoff_boundary', 'AT_RISK_CURRENT'] == 0
    assert pd.isna(result.loc['invalid_due', 'AT_RISK_CURRENT'])
    return {'Check': 'toy_missing_month_payment_and_future_perturbation_cases', 'Rows_checked': len(rows)+len(cases), 'Result': 'passed'}


def feature_dictionary(data):
    """Document every output column and its permitted role; never expose row values."""
    definitions = {
        'CONSUMER_NO': ('Grouping/join key only; never a numeric predictor.', 'identity', 'Required; reject missing IDs.'),
        'PERIOD': ('Feature calendar month t.', 't', 'Required valid YYYY-MM.'),
        'FEATURE_CUTOFF_DATE': ('Last calendar day of feature month.', 'end of t', 'Derived from PERIOD.'),
        'TARGET_PERIOD': ('Exactly PERIOD plus one calendar month.', 't+1', 'Retain even when target unavailable.'),
        'TARGET_MONTH': ('Calendar month number of TARGET_PERIOD, known in advance.', 'calendar of t+1', 'Derived from TARGET_PERIOD.'),
        'CURRENT_CONSUMPTION': ('Consumption recorded at feature month.', 't', 'Invalid/nonfinite/negative quantity remains missing.'),
        'RECENT_MEAN_3': ('Mean of current consumption, lag1 and lag2.', 't, t-1, t-2', 'All three exact calendar values required.'),
        'RECENT_STD_3': ('Sample standard deviation of current consumption, lag1 and lag2; ddof=1.', 't, t-1, t-2', 'All three exact calendar values required.'),
        'CONSUMPTION_TREND_3': ('CURRENT_CONSUMPTION minus CONSUMPTION_LAG2.', 't minus t-2', 'Both endpoints required.'),
        'SAME_MONTH_LAST_YEAR': ('Consumption at TARGET_PERIOD minus 12; March-2026 uses April-2025, not March-2025.', 't-11', 'Exact consumer/calendar match required.'),
        'PAYMENT_OBSERVED_BY_CUTOFF': ('PAYMENT_DATE present and on/before FEATURE_CUTOFF_DATE.', 'end of t', 'False if missing/invalid/future date; audit only.'),
        'PAYMENT_RATIO_CURRENT': ('PAYMENT_AMOUNT/BILL_AMOUNT only if observed by cutoff, bill positive, amount finite.', 'on/before end of t', 'Unavailable payment or invalid bill/amount gives NaN; genuine observed zero stays zero.'),
        'PAYMENT_DELAY_CURRENT': ('PAYMENT_DATE minus DUE_DATE in days, only for payments observed by cutoff.', 'on/before end of t', 'Missing/invalid due date or unobserved payment gives NaN; negative delay retained.'),
        'PAYMENT_RATIO_LAG1': ('Frozen PAYMENT_RATIO_CURRENT from exact previous calendar month.', 't-1, at its own cutoff', 'Missing month/ratio remains NaN; no later-payment reconstruction.'),
        'PAYMENT_RATIO_LAG2_AUDIT': ('Frozen PAYMENT_RATIO_CURRENT from exact two months earlier; helper for mean3.', 't-2, at its own cutoff', 'Missing month/ratio remains NaN; not a candidate predictor.'),
        'PAYMENT_RATIO_MEAN3': ('Mean of frozen current, lag1 and lag2 ratios.', 't, t-1, t-2', 'All three valid ratios required; never fill missing with zero.'),
        'ARREARS_CURRENT': ('CLOSING_ARREARS recorded at feature month.', 't', 'Missing/nonfinite remains NaN.'),
        'ARREARS_LAG1': ('CLOSING_ARREARS from exact previous month.', 't-1', 'Missing month/value remains NaN.'),
        'ARREARS_TREND': ('ARREARS_CURRENT minus ARREARS_LAG1.', 't minus t-1', 'Both values required.'),
        'TENURE_MONTHS': ('Calendar months from CONNECTION_DATE to PERIOD, not TARGET_PERIOD.', 't', 'Missing/invalid connection or negative result gives NaN.'),
        'NEGATIVE_TENURE_FLAG': ('Would calculated calendar tenure be negative?', 't', 'Audit flag; negative tenure is never used.'),
        'PAYMENT_RATIO_OUTCOME': ('LABEL ONLY: positive bill and valid deadline; on-time valid amount/bill; no payment by due date gives zero.', 'bill month due-date outcome', 'On-time missing/invalid amount, invalid bill/deadline or invalid nonblank payment date gives NaN.'),
        'AT_RISK_CURRENT': ('LABEL ONLY: 1 if PAYMENT_RATIO_OUTCOME < 0.50, otherwise 0 when defined.', 'bill month due-date outcome', 'Undefined outcome stays NaN. Academic proxy, not official default probability.'),
        'PRIOR_COUNT': ('Number of valid earlier consumption observations for this consumer.', 'strictly before t', 'Zero with no history; audit only.'),
        'PRIOR_MEAN': ('Mean of all valid earlier consumption observations, excluding current.', 'strictly before t', 'NaN with no earlier observations; audit only.'),
        'PRIOR_STD': ('Sample standard deviation of valid earlier consumption, ddof=1.', 'strictly before t', 'NaN with fewer than two earlier observations; audit only.'),
        'RULE_Z_CURRENT': ('(CURRENT_CONSUMPTION-PRIOR_MEAN)/PRIOR_STD; at least three valid earlier observations.', 't versus strictly earlier history', 'NaN if fewer than three prior values, zero/invalid std or invalid current value; never a predictor.'),
        'ANOMALY_CURRENT': ('LABEL ONLY: 1 if abs(RULE_Z_CURRENT)>2.5, else 0 when z is defined.', 't with strictly earlier history', 'Undefined z stays NaN. Academic unusual-consumption proxy, not theft/fraud.'),
        'CONSUMPTION_NEXT': ('Consumption from exactly TARGET_PERIOD for same consumer.', 't+1', 'No exact next-month value means NaN; target only.'),
        'AT_RISK_NEXT': ('AT_RISK_CURRENT from exactly TARGET_PERIOD for same consumer.', 't+1 due-date outcome', 'Missing next month or undefined proxy remains NaN; target only.'),
        'ANOMALY_NEXT': ('ANOMALY_CURRENT from exactly TARGET_PERIOD for same consumer.', 't+1', 'Missing next month or undefined proxy remains NaN; target only.'),
        'BILL_AMOUNT': ('Recorded bill amount at t, not the next bill.', 't', 'Missing/nonfinite remains NaN; nonpositive excluded only from ratio/proxy eligibility.'),
        'CONTRACT_LOAD': ('Recorded contract load at t.', 't', 'Missing/nonfinite remains NaN.'),
        'TARRIF': ('Recorded source tariff category; spelling preserved.', 't', 'No recoding or imputation.'),
        'SOLAR_CONSUMER': ('Recorded solar status at t, never a future status.', 't', 'No recoding or imputation.'),
    }
    for lag in [1, 2, 3]:
        definitions[f'CONSUMPTION_LAG{lag}'] = (f'Consumption exactly {lag} calendar months earlier.', f't-{lag}', 'Missing exact month/value remains NaN.')
    rows = []
    for column in data:
        modules = []
        if column in CONSUMPTION_PREDICTORS:
            modules.append('Consumption')
        if column in REVENUE_PREDICTORS:
            modules.append('Revenue')
        if column in ANOMALY_PREDICTORS:
            modules.append('Unusual consumption')
        role = 'audit only'
        if modules:
            role = 'candidate predictor'
        elif column in TARGETS:
            role = 'target only'
        elif column in ['PAYMENT_RATIO_OUTCOME', 'AT_RISK_CURRENT', 'ANOMALY_CURRENT', 'RULE_Z_CURRENT', 'PRIOR_MEAN', 'PRIOR_STD', 'PRIOR_COUNT']:
            role = 'label construction/audit only'
        definition, timing, missing = definitions.get(column, ('Original source field retained for audit only; never automatically a predictor.', 'source monthly record; event may be later than t', 'Original value retained; no imputation.'))
        rows.append({'Feature': column, 'Module': '; '.join(modules) or 'audit/target', 'Type': str(data[column].dtype),
            'Definition': definition, 'Time_reference': timing, 'Missing_value_rule': missing, 'Predictor_or_target': role,
            'Notes': 't = PERIOD; calendar joins only. Use explicit module predictor lists, never all numeric columns. No feature selection performed.'})
    return pd.DataFrame(rows)


def main():
    source = ROOT / 'data/processed/consumer_cleaned.csv'
    before = source.read_bytes()
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [column for column in SOURCE_COLUMNS if column not in raw]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    expected = pd.period_range('2023-04', '2026-03', freq='M').astype(str).tolist()
    if raw.shape != (73578, 91) or sorted(raw.PERIOD.unique()) != expected:
        raise ValueError('Unexpected input shape/coverage; review rather than silently correcting.')
    if raw.CONSUMER_NO.str.strip().eq('').any():
        raise ValueError('Missing tracking identifiers.')
    parsed_period = pd.to_datetime(raw.PERIOD, format='%Y-%m')
    if not pd.to_numeric(raw.MONTH, errors='coerce').eq(parsed_period.dt.month).all() or not pd.to_numeric(raw.YEAR, errors='coerce').eq(parsed_period.dt.year).all():
        raise ValueError('MONTH/YEAR disagrees with PERIOD.')
    print(f'Input: {raw.shape}; consumers: {raw.CONSUMER_NO.nunique()}; {raw.PERIOD.min()} through {raw.PERIOD.max()}')
    data = build_features(raw)
    checks = validate_features(data)
    checks.append(edge_case_checks())
    # Report conversion uncertainty without inventing replacement values.
    numeric_fields = ['CONSUMPTION_UNIT', 'BILL_AMOUNT', 'PAYMENT_AMOUNT', 'CLOSING_ARREARS', 'CONTRACT_LOAD']
    date_fields = ['PAYMENT_DATE', 'DUE_DATE', 'LAST_PAYDATE', 'CONNECTION_DATE']
    for column in numeric_fields + date_fields:
        if column in numeric_fields:
            parsed = pd.to_numeric(raw[column], errors='coerce')
            invalid = ~np.isfinite(parsed)
        else:
            parsed = pd.to_datetime(raw[column], format='%Y-%m-%d', errors='coerce')
            invalid = parsed.isna()
        failures = raw[column].str.strip().ne('') & invalid
        checks.append({'Check': column + '_nonblank_invalid_values', 'Rows_checked': len(raw),
                       'Result': 'reported', 'Count': int(failures.sum())})
    first_rows = data.groupby('CONSUMER_NO', sort=False).head(1)
    late_joiners = first_rows.PERIOD.gt(data.PERIOD.min())
    checks.append({'Check': 'late_joiners_with_no_backfilled_history', 'Rows_checked': int(late_joiners.sum()), 'Result': 'passed'})
    # Number of absent internal calendar months in each consumer's observed span.
    missing_months = 0
    for consumer, group in data.groupby('CONSUMER_NO'):
        span = len(pd.period_range(group.PERIOD.min(), group.PERIOD.max(), freq='M'))
        missing_months = missing_months + span - len(group)
    checks.append({'Check': 'actual_missing_internal_calendar_months', 'Rows_checked': len(data), 'Result': 'reported', 'Count': missing_months})
    latest = data.PERIOD.max()
    future = data.loc[data.PERIOD.eq(latest)]
    assert future.TARGET_PERIOD.eq(pd.Period('2026-04')).all() and future.TARGET_MONTH.eq(4).all()
    assert future[TARGETS].isna().all().all()
    assert len(data) == len(raw)
    original = raw.sort_values(['CONSUMER_NO', 'PERIOD']).reset_index(drop=True)
    for column in ['CONSUMER_NO', 'PAYMENT_AMOUNT', 'PAYMENT_DATE', 'DUE_DATE']:
        assert data[column].equals(original[column])
    assert source.read_bytes() == before
    checks.append({'Check': 'April_2026_unknown_targets_and_original_input_unchanged', 'Rows_checked': len(future), 'Result': 'passed'})
    # A March feature row uses April of the previous year, via the all-row t-11 check.
    checks.append({'Check': 'March_2026_seasonal_reference_April_2025', 'Rows_checked': len(future), 'Result': 'passed'})
    availability = []
    for target in TARGETS:
        row = {'Target': target, 'Total_rows': len(data), 'Available': int(data[target].notna().sum()), 'Missing': int(data[target].isna().sum()), 'Class_0': np.nan, 'Class_1': np.nan}
        if target != 'CONSUMPTION_NEXT':
            row['Class_0'] = int(data[target].eq(0).sum())
            row['Class_1'] = int(data[target].eq(1).sum())
        availability.append(row)
    summary = ROOT / 'reports/summaries'
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(availability).to_csv(summary / 'target_availability_summary.csv', index=False)
    payment_audit = payment_summary(data)
    payment_audit.to_csv(summary / 'payment_resolution_summary.csv', index=False)
    feature_dictionary(data).to_csv(summary / 'feature_dictionary.csv', index=False)
    pd.DataFrame(checks).to_csv(summary / 'feature_validation_summary.csv', index=False)
    data.to_csv(ROOT / 'data/processed/consumer_features.csv', index=False, date_format='%Y-%m-%d')
    print(f'Output: {data.shape}; future April rows: {len(future)}; all checks passed.')
    print(pd.DataFrame(availability).to_string(index=False))
    print('Payment audit totals:')
    print(payment_audit.drop(columns='PERIOD').sum().to_string())
    print('Predictor observation rule: PAYMENT_DATE present and <= FEATURE_CUTOFF_DATE (month-end t).')
    print('Why: payments after month-end were not knowable at feature time; LAST_PAYDATE/STATUS/BILL_TYPE are not substituted.')
    print('Outcome rule: positive bill and valid DUE_DATE; on-time valid amount/bill, otherwise no-payment-by-deadline zero. On-time invalid amount stays NaN. Label only.')
    print('No Step 6, splits, models, feature selection, private metadata or Git operations.')


if __name__ == '__main__':
    main()
