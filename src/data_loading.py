"""Validate and combine public monthly extracts. Run from the project root."""

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


# WHAT: Describe the original schema, including its trailing empty field.
# WHY: Comparing only column counts would miss renamed or reordered columns.
# HOW: pandas names the empty 91st source header 'Unnamed: 90'; retain it.
# OUTPUT: An exact ordered schema; TARRIF and other source names stay unchanged.
SOURCE_HEADER = (
    "MONTH|YEAR|CIRCLE_CODE|CIRCLE_NAME|DIVISION_CODE|DIVISION_NAME|SUBDIV_CODE|"
    "SUBDIV_NAME|LOCATION_CODE|CONSUMER_NO|CONSUMER_NAME|ADDRESS1|ADDRESS2|"
    "CENCUS_CODE|TRANS_LOCATION|SD_AMOUNT|CONNECTION_DATE|POLE_CODE|PDC_DATE|"
    "IND_TYPE|SEASONAL_IND|SD_AMOUNT_RECEIPT_DATE|SD_AMOUNT_RECEIPT_NO|CONTRACT_LOAD|"
    "TARRIF|CYCLE_NO|METER_READER_NO|BOOK_NO|ROUTE_CODE|METER_NO|METER_STATUS|"
    "EDUTY_CODE|EDUTY_AMOUNT|CONTRACT_DEMAND|FEEDER_NO|PHASE|METER_RENT_TYPE|"
    "THEFT_ARREARS|LITIGATION_ARREARS|FIXED_CHG|FUEL_CHG|FUSE_CHG|CREDIT_ADJUSTMENT|"
    "DEBIT_ADJUSTMENT|CLOSING_ARREARS|CONSUMPTION_UNIT|ASSESSMENT_UNIT|DPC_AMOUNT|"
    "RELIEF_AMOUNT|ENERGY_CHARGE|BOARD_CHARGE|PROV_BILL_AMOUNT|PAYMENT_AMOUNT|"
    "BILL_AMOUNT|START_METER|END_METER|AVG_UNIT|RE_START_READING|RE_END_READING|"
    "RE_CONSUMPTION|BILL_DATE|DUE_DATE|PAYMENT_DATE|FEEDER_CODE|FEEDER_NAME|"
    "VILLAGE_NAME|TARRIF_SHORT|METER_CHANGE_DATE|ADJUSTMENT_UNIT|OLD_TARIFF_IND|"
    "BILL_DEMAND|BILL_TYPE|STATUS|LAST_PAYDATE|PREV_BILL_DATE|ACTUAL_DEMAND|"
    "SOLAR_CONSUMER|SOLAR_AGREEMENT_DATE|SOLAR_RATE|IMP_CONSUMPTION|EXP_CONSUMPTION|"
    "SOLAR_PURCHASE|SOLAR_ADJUSTMENT|AMR_INDICATOR|TDS_ADJ|SOLAR_LOAD|SOLAR_CONN_TYPE|"
    "GENRATION_UNIT|FINAL_MF|TOU_CHG|"
)
EXPECTED_COLUMNS = SOURCE_HEADER.split("|")[:-1] + ["Unnamed: 90"]
EXPECTED_PERIODS = pd.period_range("2023-04", "2026-03", freq="M")


def discover_months(folder):
    """Discover TXT files without opening private metadata or other files."""
    records = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        match = re.fullmatch(r"512LDG(1[0-2]|[1-9])(\d{4})\.txt", path.name, re.I)
        if match is None:
            raise ValueError(f"Unexpected TXT filename: {path}. Expected 512LDG42023.txt style.")
        month, year = int(match[1]), int(match[2])
        period = pd.Period(year=year, month=month, freq="M")
        records.append((period, path))
    return sorted(records)


def check_periods(records, expected, label):
    """Report all missing, extra and duplicated periods before stopping."""
    periods = [period for period, _ in records]
    counts = pd.Series(periods, dtype="period[M]").value_counts()
    duplicates = sorted(str(period) for period, count in counts.items() if count > 1)
    missing = sorted(str(period) for period in set(expected) - set(periods))
    extra = sorted(str(period) for period in set(periods) - set(expected))
    print(f"{label}: {len(records)} files; missing={missing}; extra={extra}; duplicate periods={duplicates}")
    if missing or extra or duplicates:
        raise ValueError(f"Fix the {label} file coverage above before loading.")


def read_month(path, period):
    """Check physical fields and read one monthly extract without cleaning it."""
    # WHAT: Validate every record before pandas can pad short rows with missing cells.
    # WHY: A malformed delimiter or incomplete row must not be silently accepted.
    # HOW: csv checks field widths, then pandas loads all source cells as strings.
    # OUTPUT: A validated monthly table, preserving IDs and source placeholders.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="|", strict=True)
        header = next(reader, [])
        if header != SOURCE_HEADER.split("|"):
            expected = SOURCE_HEADER.split("|")
            print(f"{path.name}: missing columns={sorted(set(expected) - set(header))}")
            print(f"{path.name}: extra columns={sorted(set(header) - set(expected))}")
            raise ValueError(f"{path.name}: wrong schema, column order or delimiter; expected 91 pipe-delimited columns.")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 91:
                raise ValueError(f"{path.name}, record {line_number}: found {len(row)} fields; expected 91.")
    frame = pd.read_csv(path, delimiter="|", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"{path.name}: pandas column order does not match the source schema.")
    if frame.empty:
        raise ValueError(f"{path.name}: no consumer rows were found.")
    months = pd.to_numeric(frame["MONTH"], errors="coerce")
    years = pd.to_numeric(frame["YEAR"], errors="coerce")
    wrong_period = (months != period.month) | (years != period.year)
    if wrong_period.any():
        raise ValueError(f"{path.name}: {int(wrong_period.sum())} rows have missing/invalid MONTH or YEAR, or disagree with {period}.")
    if frame["CONSUMER_NO"].str.strip().eq("").any():
        raise ValueError(f"{path.name}: blank CONSUMER_NO values prevent reliable longitudinal checks.")
    if frame["Unnamed: 90"].ne("").any():
        raise ValueError(f"{path.name}: the trailing unnamed source field unexpectedly contains values.")
    return frame


def main():
    # WHAT: Check source coverage and public copies before reading modelling inputs.
    # WHY: Private metadata must never enter modelling, and no month may be lost.
    # HOW: Use relative pathlib paths, filename allowlists and byte hashes.
    # OUTPUT: Verified Apr-2023--Mar-2026 raw inputs plus optional later actuals.
    source = Path("Dataset")
    raw = Path("data/raw")
    new = Path("data/new")
    if not source.is_dir() or not raw.is_dir():
        raise ValueError("Run from the project root; Dataset/ and data/raw/ must exist.")
    year_folders = sorted(path for path in source.iterdir() if path.is_dir())
    if len(year_folders) != 3:
        raise ValueError(f"Expected three source year folders; found {len(year_folders)}.")
    source_records = []
    for folder in year_folders:
        records = discover_months(folder)
        print(f"Source folder {folder.name}: {len(records)} monthly TXT extracts")
        source_records.extend(records)
    check_periods(source_records, EXPECTED_PERIODS, "Original source")
    unexpected = [str(path) for path in raw.rglob("*") if path.is_file() and path.suffix.lower() != ".txt"]
    if unexpected:
        raise ValueError(f"data/raw must contain only public monthly TXT files. Unexpected files: {unexpected}")
    raw_records = discover_months(raw)
    check_periods(raw_records, EXPECTED_PERIODS, "Public raw copies")
    sources = dict(source_records)
    for period, path in raw_records:
        original_hash = hashlib.sha256(sources[period].read_bytes()).hexdigest()
        copy_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if original_hash != copy_hash:
            raise ValueError(f"{path}: differs from the original synthetic source. Restore the approved copy.")
    print("All public raw copies match their original monthly extracts.")
    records = sorted(raw_records + discover_months(new))
    all_expected = pd.period_range(EXPECTED_PERIODS[0], max(period for period, _ in records), freq="M")
    check_periods(records, all_expected, "Combined actual-month coverage")

    # WHAT: Read files chronologically and compute per-file counts.
    # WHY: Counts must reflect input data, including consumer growth.
    # HOW: Validate each frame, then add a monthly Period column and concatenate.
    # OUTPUT: File summaries and a consumer/month-sorted combined table.
    frames = []
    file_summaries = []
    for period, path in records:
        frame = read_month(path, period)
        info = {"filename": path.name, "period": str(period), "row_count": len(frame),
                "column_count": len(frame.columns), "unique_consumers": frame["CONSUMER_NO"].nunique()}
        file_summaries.append(info)
        print(f"{path.name}: period={period}, rows={len(frame)}, columns={len(frame.columns)}, unique consumers={info['unique_consumers']}")
        frame["PERIOD"] = period
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["CONSUMER_NO", "PERIOD"]).reset_index(drop=True)

    # WHAT: Check repeated consumer/month records and connection chronology.
    # WHY: Duplicates and pre-connection readings would invalidate later histories.
    # HOW: Count affected records without dropping them; inspect dates in temporary series.
    # OUTPUT: Clear validation counts; invalid chronology blocks output publication.
    duplicates = combined.duplicated(["CONSUMER_NO", "PERIOD"], keep=False)
    duplicate_count = int(duplicates.sum())
    print(f"Duplicate consumer-month rows (all members): {duplicate_count}")
    if duplicate_count:
        print(combined.loc[duplicates].groupby("PERIOD").size().to_string())
        raise ValueError("Duplicate consumer-month records found. No rows were dropped and no new combined output was saved.")
    connection = pd.to_datetime(combined["CONNECTION_DATE"], errors="coerce").dt.to_period("M")
    unknown_dates = int(connection.isna().sum())
    before_connection = connection.notna() & (combined["PERIOD"] < connection)
    # Use each consumer's earliest valid declared connection across their records.
    first_connection = connection.groupby(combined["CONSUMER_NO"]).transform("min")
    first_appearance = combined.groupby("CONSUMER_NO")["PERIOD"].transform("min")
    invalid_first_appearance = first_connection.notna() & (first_appearance < first_connection)
    connection_versions = connection.groupby(combined["CONSUMER_NO"]).nunique()
    inconsistent_connections = int(connection_versions.gt(1).sum())
    late_joiners = int(combined.loc[first_appearance > combined["PERIOD"].min(), "CONSUMER_NO"].nunique())
    print(f"Late joiners after first dataset month: {late_joiners}")
    print(f"Unverifiable connection-date rows: {unknown_dates}; inconsistent connection-date consumers: {inconsistent_connections}")
    print(f"Rows before their declared connection month: {int(before_connection.sum())}")
    print("First appearance is computed from observed records; no earlier history is fabricated.")
    if before_connection.any() or invalid_first_appearance.any() or inconsistent_connections:
        raise ValueError("Connection chronology is inconsistent. Review source records; no automatic changes were made.")
    if unknown_dates:
        print("WARNING: Missing/invalid connection dates could not be verified; source values are retained.")

    # WHAT: Save local working data and compact non-identifying summaries.
    # WHY: Later stages need a combined input and dynamically computed actual dates.
    # HOW: Aggregate counts, derive the next month, and save only after validation.
    # OUTPUT: One local CSV, a monthly counts CSV and a loading summary JSON.
    monthly = combined.groupby("PERIOD").agg(row_count=("CONSUMER_NO", "size"), unique_consumers=("CONSUMER_NO", "nunique")).reset_index()
    latest = combined["PERIOD"].max()
    following = latest + 1
    print("\nMonthly consumer-count progression:")
    print(monthly.to_string(index=False))
    print(f"Latest Actual Month: {latest}\nNext Prediction Month: {following}")
    summary = {
        "status": "passed_with_warnings" if unknown_dates else "passed",
        "file_count": len(records), "source_column_count": 91,
        "schema_note": "90 named source fields plus the empty trailing field, retained as Unnamed: 90. PERIOD is added as column 92.",
        "total_rows": len(combined), "unique_consumers": combined["CONSUMER_NO"].nunique(),
        "first_actual_month": str(combined["PERIOD"].min()), "latest_actual_month": str(latest),
        "next_prediction_month": str(following), "duplicate_consumer_month_rows": duplicate_count,
        "late_joiner_consumers": late_joiners, "unverifiable_connection_date_rows": unknown_dates,
        "inconsistent_connection_date_consumers": inconsistent_connections,
        "rows_before_connection": int(before_connection.sum()),
        "first_appearance_note": "Observed minimum month, not proof of the real connection date; earlier history is not fabricated.",
        "files": file_summaries,
    }
    processed = Path("data/processed")
    summaries = Path("reports/summaries")
    processed.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed / "consumer_combined.csv", index=False)
    monthly.to_csv(summaries / "monthly_loading_counts.csv", index=False)
    with (summaries / "loading_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Saved data/processed/consumer_combined.csv (local, ignored by Git).")
    print("Saved reports/summaries/loading_summary.json and monthly_loading_counts.csv.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, csv.Error, pd.errors.ParserError) as error:
        print(f"\nLOADING ERROR: {error}")
        print("Fix the reported input problem and rerun. Existing outputs, if any, may be stale; do not use them.")
        raise SystemExit(1)
