"""Step 3: conservative cleaning and quality checks, without ML preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = '''MONTH YEAR SD_AMOUNT CONTRACT_LOAD EDUTY_AMOUNT CONTRACT_DEMAND
THEFT_ARREARS LITIGATION_ARREARS FIXED_CHG FUEL_CHG FUSE_CHG CREDIT_ADJUSTMENT
DEBIT_ADJUSTMENT CLOSING_ARREARS CONSUMPTION_UNIT ASSESSMENT_UNIT DPC_AMOUNT
RELIEF_AMOUNT ENERGY_CHARGE BOARD_CHARGE PROV_BILL_AMOUNT PAYMENT_AMOUNT
BILL_AMOUNT START_METER END_METER AVG_UNIT RE_START_READING RE_END_READING
RE_CONSUMPTION ADJUSTMENT_UNIT BILL_DEMAND ACTUAL_DEMAND SOLAR_RATE
IMP_CONSUMPTION EXP_CONSUMPTION SOLAR_PURCHASE SOLAR_ADJUSTMENT TDS_ADJ
SOLAR_LOAD GENRATION_UNIT FINAL_MF TOU_CHG'''.split()
DATES = '''CONNECTION_DATE PDC_DATE SD_AMOUNT_RECEIPT_DATE BILL_DATE DUE_DATE
PAYMENT_DATE METER_CHANGE_DATE LAST_PAYDATE PREV_BILL_DATE SOLAR_AGREEMENT_DATE'''.split()
CATEGORIES = ['TARRIF', 'METER_STATUS', 'SOLAR_CONSUMER', 'PHASE']
NONNEGATIVE = '''CONSUMPTION_UNIT CONTRACT_LOAD CONTRACT_DEMAND START_METER
END_METER AVG_UNIT RE_START_READING RE_END_READING RE_CONSUMPTION BILL_DEMAND
ACTUAL_DEMAND SOLAR_LOAD IMP_CONSUMPTION EXP_CONSUMPTION GENRATION_UNIT FINAL_MF'''.split()
TOKENS = {'', '?', 'NA', 'N/A', 'NULL', 'NONE', '-', '-1', '999', '9999', 'UNKNOWN',
          'NAN', 'NAT', 'NIL', 'NOT AVAILABLE', 'NOT APPLICABLE', 'MISSING', 'UNDEFINED', '--', 'TBD'}


def main():
    # WHAT: Preserve literal input strings and inspect conventional missing values.
    # WHY: Default NA parsing could erase a legitimate category such as NA.
    # HOW: Use a separate default-parser read for diagnostics, not transformations.
    # OUTPUT: A local cleaned CSV plus a public aggregate audit; no imputation.
    source = ROOT / 'data/processed/consumer_combined.csv'
    output = ROOT / 'data/processed/consumer_cleaned.csv'
    summary = ROOT / 'reports/summaries/cleaning_summary.csv'
    local = ROOT / 'reports/internal_validation/step3'
    local.mkdir(parents=True, exist_ok=True)
    # Check file sizes and modification times without reading original TXT data.
    # This is a simple change check, not a byte-for-byte integrity guarantee.
    source_files = [source]
    for period in pd.period_range('2023-04', '2026-03', freq='M'):
        filename = f'512LDG{period.month}{period.year}.txt'
        first_year = period.year
        if period.month < 4:
            first_year = first_year - 1
        folder = f'FY{first_year}-{str(first_year + 1)[-2:]}'
        source_files.append(ROOT / 'data/raw' / filename)
        source_files.append(ROOT / 'Dataset' / folder / filename)
    before_sizes = []
    before_times = []
    for path in source_files:
        before_sizes.append(path.stat().st_size)
        before_times.append(path.stat().st_mtime_ns)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    standard = pd.read_csv(source, dtype=str)
    frame = raw.copy()
    n = len(raw)
    audit = []

    def record(section, column, finding, count=None, denominator=None, value='', action='retain', reason=''):
        """Append one CSV audit row; calculate a percentage only when possible."""
        percent = None
        if denominator and count is not None:
            percent = 100 * count / denominator
        audit.append({'section': section, 'column': column, 'finding': finding,
                      'count': count, 'denominator': denominator, 'percent': percent,
                      'value': value, 'action': action, 'reason': reason})

    def flag(name, mask, reason, column=''):
        """Count True rows in a quality check; never change or delete those rows."""
        record('logical', column, name, int(mask.fillna(False).sum()), len(frame), reason=reason)

    record('shape', '', 'rows_before', n)
    record('shape', '', 'columns_before', len(raw.columns))
    required = NUMERIC + DATES + CATEGORIES + ['CONSUMER_NO', 'PERIOD']
    missing = []
    for column in required:
        if column not in raw.columns:
            missing.append(column)
    missing.sort()
    record('schema', '', 'missing_required_columns', len(missing), value=str(missing))
    if missing:
        pd.DataFrame(audit).to_csv(summary, index=False)
        raise ValueError(f'Missing required columns: {missing}; no output written.')
    print(f'Input shape: {raw.shape}; unique consumers: {raw.CONSUMER_NO.nunique()}')
    print('Columns:', ', '.join(raw.columns))
    print('Small non-identifying sample:')
    print(raw[['PERIOD', 'TARRIF', 'CONSUMPTION_UNIT', 'BILL_AMOUNT']].head(3).to_string(index=False))
    suspicious = []
    for c in raw:
        record('inspection', c, 'input_dtype', value=str(raw[c].dtype))
        record('missing', c, 'standard_pandas_missing', int(standard[c].isna().sum()), n,
               reason='Default NA parsing is diagnostic only; working strings preserved.')
        record('missing', c, 'literal_blank_or_whitespace', int(raw[c].str.strip().eq('').sum()), n)
        # WHAT/WHY: Inspect all object values without guessing the meaning of codes.
        # HOW/OUTPUT: Publish fixed token labels only; other raw candidates stay local.
        for value, count in raw[c].value_counts(dropna=False).items():
            norm = value.strip().upper()
            punctuation_only = norm != ''
            for character in norm:
                if character not in '?-.*_':
                    punctuation_only = False
            other = punctuation_only or norm.startswith(('UNKNOWN', 'MISSING', 'NOT AVAILABLE'))
            if norm in TOKENS or other:
                label = norm if norm in TOKENS else '[other candidate; local details]'
                record('disguised_missing', c, label or '[blank/whitespace]', int(count), n,
                       reason='Candidate only; category codes and numeric sentinel-like values retained unless explicit typed conversion fails.')
                suspicious.append(dict(column=c, original_value=value, count=count))
    pd.DataFrame(suspicious).to_csv(local / 'placeholder_details.csv', index=False)

    artifact = 'Unnamed: 90'
    record('artifact', artifact, 'exists', value=artifact in raw)
    if artifact in raw:
        blank = raw[artifact].str.strip().eq('')
        null = standard[artifact].isna()
        for name, count in [('total_rows', n), ('null_count_standard_parser', null.sum()),
                            ('non_null_count_standard_parser', (~null).sum()),
                            ('blank_count_literal', blank.sum()), ('non_blank_count_literal', (~blank).sum())]:
            record('artifact', artifact, name, int(count), n)
        values = raw.loc[~blank, artifact].unique().tolist()
        record('artifact', artifact, 'unique_nonblank_values', len(values), value='[]' if not values else '[local details]')
        pd.DataFrame({'original_value': values}).to_csv(local / 'unnamed_values.csv', index=False)
        if blank.all():
            frame = frame.drop(columns=artifact)
            record('action', artifact, 'column_removed', 1, action='drop column',
                   reason='Empty source-format artifact created by the trailing pipe delimiter; contains no analytical information.')
        else:
            record('artifact', artifact, 'retained_for_review', int((~blank).sum()), n, reason='Nonblank literal content is never automatically dropped.')

    # WHAT: Check original full rows before conversions could hide differences.
    # WHY: Conflicting duplicate records must not be resolved arbitrarily.
    # HOW: Drop only identical full-record repeats, keeping the first occurrence.
    # OUTPUT: All conflicting variants retained and duplicate details saved locally.
    keys = ['CONSUMER_NO', 'PERIOD']
    duplicates = raw.duplicated(keys, keep=False)
    exact = raw.duplicated(keep='first')
    distinct = raw.loc[~exact]
    conflict_keys = distinct.loc[distinct.duplicated(keys, keep=False), keys].drop_duplicates()
    # A left join marks every original row whose key has conflicting versions.
    # conflict_keys has one row per key, so this join cannot multiply input rows.
    conflict_keys['has_conflict'] = True
    conflict_check = raw[keys].merge(conflict_keys, on=keys, how='left', sort=False)
    conflicts = conflict_check['has_conflict'].eq(True)
    record('duplicates', '', 'duplicate_key_rows_all_members', int(duplicates.sum()), n)
    record('duplicates', '', 'exact_duplicate_rows_all_members', int(raw.duplicated(keep=False).sum()), n)
    record('duplicates', '', 'conflicting_duplicate_rows_all_members', int(conflicts.sum()), n)
    if duplicates.any():
        raw.loc[duplicates].to_csv(local / 'duplicate_rows.csv', index=False)
    frame = frame.loc[~exact].copy()
    record('action', '', 'exact_duplicate_rows_removed', int(exact.sum()), n, action='remove identical repeats only',
           reason='Keep first identical full original record; retain distinct conflicting records.')

    # WHAT/WHY: Convert explicit physical/financial fields, never numeric-looking IDs.
    # HOW: Coerce with pandas and save each failed original value locally.
    # OUTPUT: Typed numbers/dates; missing values remain missing, without imputation.
    failures = []
    for c in NUMERIC + DATES:
        original = frame[c]
        nonblank = original.str.strip().ne('')
        if c in NUMERIC:
            converted = pd.to_numeric(original, errors='coerce')
            conversion_action = 'to_numeric'
            conversion_reason = 'Explicit quantity/amount schema; preserve codes as strings.'
        else:
            converted = pd.to_datetime(original, format='%Y-%m-%d', errors='coerce')
            conversion_action = 'to_datetime'
            conversion_reason = 'Observed YYYY-MM-DD format; blank dates become NaT.'
        failed = nonblank & converted.isna()
        record('conversion', c, 'values_before', len(original))
        record('conversion', c, 'nonblank_before', int(nonblank.sum()), len(original))
        record('conversion', c, 'blank_to_missing', int((~nonblank).sum()), len(original))
        record('conversion', c, 'nonblank_conversion_failures', int(failed.sum()), len(original),
               reason='Original failed values saved locally; records retained.')
        record('action', c, 'type_conversion', action=conversion_action, reason=conversion_reason)
        for idx in frame.index[failed]:
            failures.append(dict(source_csv_row=int(idx) + 2, column=c, original_value=original.loc[idx],
                                 CONSUMER_NO=frame.at[idx, 'CONSUMER_NO'], PERIOD=frame.at[idx, 'PERIOD']))
        frame[c] = converted
    pd.DataFrame(failures, columns=['source_csv_row', 'column', 'original_value', 'CONSUMER_NO', 'PERIOD']).to_csv(local / 'conversion_failures.csv', index=False)

    # WHAT/WHY: Logical findings are flags, not reasons to delete billing history.
    # HOW: Check physical domains and chronology; monetary signs need context.
    # OUTPUT: Counts/percentages only, with no targets or engineered features.
    period_date = pd.to_datetime(frame.PERIOD.where(frame.PERIOD.str.fullmatch(r'\d{4}-\d{2}')), format='%Y-%m', errors='coerce')
    periods = period_date.dt.to_period('M')
    expected = pd.period_range('2023-04', '2026-03', freq='M').astype(str).tolist()
    coverage_ok = sorted(frame.PERIOD.unique()) == expected
    record('coverage', 'PERIOD', 'expected_history_exact_match', value=coverage_ok)
    record('coverage', 'PERIOD', 'minimum', value=str(periods.min()))
    record('coverage', 'PERIOD', 'maximum', value=str(periods.max()))
    flag('invalid_or_missing_PERIOD', periods.isna(), 'Retained for review.', 'PERIOD')
    flag('missing_consumer_identifier', frame.CONSUMER_NO.str.strip().eq(''), 'Identifier remains a string.', 'CONSUMER_NO')
    record('tracking', 'CONSUMER_NO', 'unique_consumers', frame.CONSUMER_NO.nunique())
    for c, lower, upper in [('MONTH', 1, 12), ('YEAR', 2023, 2026)]:
        flag('invalid_' + c, frame[c].isna() | ~frame[c].between(lower, upper) | frame[c].mod(1).ne(0), 'Expected history domain.', c)
        part = period_date.dt.month if c == 'MONTH' else period_date.dt.year
        flag(c + '_PERIOD_mismatch', frame[c].notna() & part.notna() & frame[c].ne(part), 'Must agree with PERIOD.', c)
    before = periods < frame.CONNECTION_DATE.dt.to_period('M')
    flag('record_before_connection_month', before, 'Monthly comparison avoids false positives within connection month.')
    flag('negative_tenure_months', before, 'Same chronology condition; no tenure feature created.')
    for c in NUMERIC:
        flag('nonfinite_numeric', frame[c].notna() & ~np.isfinite(frame[c]), 'Nonfinite measurement requires review.', c)
        if c in NONNEGATIVE:
            flag('negative_physical_quantity', frame[c].lt(0), 'Physically suspect; retained for review.', c)
        elif c not in ['MONTH', 'YEAR']:
            flag('negative_amount_or_adjustment_review', frame[c].lt(0),
                 'Accounting signs are undocumented; credits, reversals and signed adjustments may be valid. Retain.', c)
    for c, name in [('BILL_AMOUNT', 'nonpositive_bill'), ('CONSUMPTION_UNIT', 'zero_consumption'), ('CONTRACT_LOAD', 'zero_contract_load')]:
        flag(name, frame[c].le(0) if c == 'BILL_AMOUNT' else frame[c].eq(0),
             'Retain record; zero use can be genuine; nonpositive bills require later revenue eligibility rules.', c)
    for earlier, later in [('BILL_DATE', 'DUE_DATE'), ('PREV_BILL_DATE', 'BILL_DATE'), ('CONNECTION_DATE', 'BILL_DATE'),
                           ('CONNECTION_DATE', 'PDC_DATE'), ('CONNECTION_DATE', 'METER_CHANGE_DATE'), ('BILL_DATE', 'PAYMENT_DATE')]:
        flag(later + '_before_' + earlier, frame[later] < frame[earlier],
             'Chronology review only; pre-bill payments can be legitimate advances.')

    # WHAT/WHY: Separate zero use from bill eligibility rather than deleting either.
    # HOW/OUTPUT: Aggregate billing context and local rows for physical/date review.
    contexts = {
        'zero_consumption_with_positive_bill': frame.CONSUMPTION_UNIT.eq(0) & frame.BILL_AMOUNT.gt(0),
        'zero_consumption_with_nonpositive_bill': frame.CONSUMPTION_UNIT.eq(0) & frame.BILL_AMOUNT.le(0),
        'nonpositive_bill_with_positive_consumption': frame.BILL_AMOUNT.le(0) & frame.CONSUMPTION_UNIT.gt(0),
        'nonpositive_bill_with_payment': frame.BILL_AMOUNT.le(0) & frame.PAYMENT_AMOUNT.gt(0),
        'negative_bill': frame.BILL_AMOUNT.lt(0),
        'zero_bill': frame.BILL_AMOUNT.eq(0),
    }
    for name, mask in contexts.items():
        flag(name, mask, 'Context only: fixed charges or earlier balances can coexist with zero use/bills. No eligibility target created.')
    review = frame[NONNEGATIVE].lt(0).any(axis=1) | (frame.BILL_DATE < frame.CONNECTION_DATE)
    raw.loc[frame.index[review]].to_csv(local / 'physical_and_connection_date_review.csv', index=False)
    record('disguised_missing', 'IND_TYPE', 'asterisk_semantics', value='*',
           reason='Asterisk is a suspicious categorical code with undocumented meaning; preserved, not converted to missing.')

    for c in ['CONSUMPTION_UNIT', 'BILL_AMOUNT', 'PAYMENT_AMOUNT', 'CLOSING_ARREARS', 'CONTRACT_LOAD']:
        finite = frame.loc[np.isfinite(frame[c]), c]
        for name, value in finite.describe().items():
            record('descriptive', c, name, value=value)
        q1, q3 = finite.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for name, value in [('IQR', iqr), ('IQR_lower', low), ('IQR_upper', high)]:
            record('outlier', c, name, value=value)
        record('outlier', c, 'potential_outliers', int(((finite < low) | (finite > high)).sum()), len(finite),
               reason='1.5 IQR rule on finite observations; extremes retained. Zero IQR is a degenerate fence, not proof of error.')
    for c in CATEGORIES:
        values = frame[c]
        record('categorical', c, 'unique_categories', values.nunique())
        record('categorical', c, 'blank_missing', int(values.str.strip().eq('').sum()), len(values))
        for value, count in values.value_counts(dropna=False).items():
            record('categorical', c, 'category_frequency', int(count), len(values), value=value)
        category_pairs = pd.DataFrame({'raw': values, 'normal': values.str.strip().str.upper()})
        category_pairs = category_pairs.drop_duplicates()
        variants = category_pairs.groupby('normal').size()
        record('categorical', c, 'normalization_collision_groups', int(variants.gt(1).sum()), reason='Diagnostic case/whitespace check only; no categories recoded.')
    record('boundary', '', 'identifier_policy', value='CONSUMER_NO for grouping/joins/tracking only. Names, addresses, meter numbers and all identifiers excluded from future predictive features.')
    record('duplicates', '', 'remaining_duplicate_key_rows', int(frame.duplicated(keys, keep=False).sum()), len(frame))
    record('action', '', 'blanket_placeholder_replacements', 0, reason='Only explicit numeric/date coercions, exact repeat removal and empty artifact removal allowed.')
    for c in frame:
        record('inspection', c, 'output_dtype', value=str(frame[c].dtype))
        record('missing_after', c, 'typed_missing', int(frame[c].isna().sum()), len(frame))
    record('shape', '', 'rows_after', len(frame))
    record('shape', '', 'columns_after', len(frame.columns))
    after_sizes = []
    after_times = []
    for path in source_files:
        after_sizes.append(path.stat().st_size)
        after_times.append(path.stat().st_mtime_ns)
    unchanged = before_sizes == after_sizes and before_times == after_times
    record('verification', '', 'public_sources_and_combined_unchanged', value=unchanged,
           reason='File sizes and modification times unchanged: 36 public originals, 36 raw copies and combined CSV. Metadata check only, not byte verification. Private files never opened.')
    blockers = not coverage_ok or conflicts.any() or frame.CONSUMER_NO.str.strip().eq('').any()
    record('verification', '', 'status', value='manual_review_required' if blockers or failures else 'completed')
    pd.DataFrame(audit).to_csv(summary, index=False)
    if not unchanged:
        raise RuntimeError('Input file size or modification time changed; no cleaned output written.')
    frame.to_csv(output, index=False, date_format='%Y-%m-%d')
    print(f'Output shape: {frame.shape}; coverage: {periods.min()} through {periods.max()}')
    print(f'Duplicate key rows: {int(duplicates.sum())}; exact repeats removed: {int(exact.sum())}; conflicting rows: {int(conflicts.sum())}')
    print(f'Conversion failures: {len(failures)}; source sizes and modification times unchanged: {unchanged}')
    print('Saved cleaned CSV and aggregate cleaning_summary.csv; detailed audits are local/ignored.')
    if blockers or failures:
        raise ValueError('Outputs saved with unresolved review findings; see audit before continuing.')


if __name__ == '__main__':
    main()
