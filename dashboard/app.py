"""Four-page EM630 dashboard built only from verified project outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "reports" / "summaries"
PAGES = [
    "Overview",
    "All Consumer Predictions",
    "Consumer 360",
    "Model Comparison & Evaluation",
]
OFFICIAL_MODELS = {
    "Consumption Prediction": "MLR-4",
    "Revenue Risk": "RF-3-balanced",
    "Unusual Consumption Risk": "DT-3-balanced",
}
MONTH_ORDER = [
    "All Months", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "January", "February", "March",
]
MONTH_NUMBERS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
    "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12,
}

st.set_page_config(
    page_title="EM630 Consumer Analytics",
    page_icon=":material/electric_bolt:",
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=40)
def load_csv(relative_path, columns=None):
    """Load a known CSV. Missing or unreadable files return None for safe UI handling."""
    path = ROOT / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(
            path,
            usecols=list(columns) if columns else None,
            dtype={"CONSUMER_NO": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def load_historical_data():
    """Load only the safe fields required by the dashboard."""
    columns = [
        "CONSUMER_NO", "PERIOD", "MONTH", "YEAR", "TARRIF", "FEEDER_NO",
        "FEEDER_CODE", "FEEDER_NAME", "CONTRACT_LOAD", "SOLAR_CONSUMER",
        "CONSUMPTION_UNIT", "BILL_AMOUNT", "PAYMENT_AMOUNT", "CLOSING_ARREARS",
    ]
    data = load_csv("data/public/dashboard_history.csv", columns)
    if data is None:
        return None
    data["PERIOD"] = pd.to_datetime(data["PERIOD"], errors="coerce")
    numeric = [
        "MONTH", "YEAR", "CONTRACT_LOAD", "CONSUMPTION_UNIT", "BILL_AMOUNT",
        "PAYMENT_AMOUNT", "CLOSING_ARREARS",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["FINANCIAL_YEAR"] = np.where(
        data["PERIOD"].dt.month >= 4,
        data["PERIOD"].dt.year.astype("Int64").astype(str)
        + "-"
        + (data["PERIOD"].dt.year + 1).astype("Int64").astype(str),
        (data["PERIOD"].dt.year - 1).astype("Int64").astype(str)
        + "-"
        + data["PERIOD"].dt.year.astype("Int64").astype(str),
    )
    return data


def format_indian(value, decimals=0):
    """Format values using Indian digit grouping."""
    if value is None or pd.isna(value):
        return "Not available"
    number = float(value)
    sign = "-" if number < 0 else ""
    fixed = f"{abs(number):.{decimals}f}"
    whole, dot, fraction = fixed.partition(".")
    if len(whole) > 3:
        end = whole[-3:]
        beginning = whole[:-3]
        pairs = []
        while beginning:
            pairs.insert(0, beginning[-2:])
            beginning = beginning[:-2]
        whole = ",".join(pairs + [end])
    return sign + whole + (dot + fraction if decimals else "")


def money(value):
    return "₹" + format_indian(value, 0) if value is not None and pd.notna(value) else "Not available"


def file_error(component, path):
    st.error(f"{component} is unavailable because the required file is missing or unreadable: `{path}`")


def safe_text(value):
    return "Not available" if value is None or pd.isna(value) or str(value).strip() == "" else str(value)


def value_card(label, value, note=None):
    """Display an important value as wrapping text instead of a truncating metric widget."""
    st.caption(label)
    st.markdown(f"### {safe_text(value)}")
    if note:
        st.caption(note)


def show_plot(figure):
    figure.update_layout(margin=dict(l=20, r=20, t=65, b=25), legend_title_text="")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def calculate_roc(actual, probability):
    """Calculate ROC points from saved labels and probabilities without model code."""
    values = pd.DataFrame({"actual": actual, "probability": probability}).dropna()
    values = values.sort_values("probability", ascending=False)
    positives = values["actual"].eq(1).sum()
    negatives = values["actual"].eq(0).sum()
    true_positive = values["actual"].eq(1).cumsum()
    false_positive = values["actual"].eq(0).cumsum()
    tpr = np.r_[0, true_positive / positives, 1]
    fpr = np.r_[0, false_positive / negatives, 1]
    area = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, area


def load_april_predictions():
    """Load and losslessly merge the three official April prediction files."""
    paths = {
        "Consumption Prediction": "reports/summaries/next_month_consumption_predictions.csv",
        "Revenue Risk": "reports/summaries/next_month_revenue_risk_predictions.csv",
        "Unusual Consumption Risk": "reports/summaries/next_month_anomaly_predictions.csv",
    }
    frames = {name: load_csv(path) for name, path in paths.items()}
    missing = [f"{name}: {paths[name]}" for name, frame in frames.items() if frame is None]
    if missing:
        return None, missing

    consumption = frames["Consumption Prediction"].copy()
    revenue = frames["Revenue Risk"].copy()
    anomaly = frames["Unusual Consumption Risk"].copy()
    for frame in [consumption, revenue, anomaly]:
        frame["CONSUMER_NO"] = frame["CONSUMER_NO"].astype("string")
        frame["TARGET_PERIOD"] = pd.to_datetime(frame["TARGET_PERIOD"], errors="coerce")
        if frame.duplicated("CONSUMER_NO").any():
            return None, ["A required April prediction file contains duplicate consumer keys."]

    consumption = consumption.rename(columns={"predicted_consumption": "APRIL_PREDICTED_CONSUMPTION"})
    revenue = revenue.rename(columns={
        "predicted_risk_class": "REVENUE_RISK_CLASS",
        "predicted_class_1_probability": "REVENUE_RISK_PROBABILITY",
    })
    anomaly = anomaly.rename(columns={
        "predicted_anomaly_class": "ANOMALY_CLASS",
        "predicted_class_1_probability": "ANOMALY_PROBABILITY",
    })
    master = consumption[["CONSUMER_NO", "TARGET_PERIOD", "APRIL_PREDICTED_CONSUMPTION"]]
    master = master.merge(
        revenue[["CONSUMER_NO", "TARGET_PERIOD", "REVENUE_RISK_CLASS", "REVENUE_RISK_PROBABILITY"]],
        on="CONSUMER_NO", how="outer", suffixes=("_CONSUMPTION", "_REVENUE"), validate="one_to_one",
    )
    master = master.merge(
        anomaly[["CONSUMER_NO", "TARGET_PERIOD", "ANOMALY_CLASS", "ANOMALY_PROBABILITY"]],
        on="CONSUMER_NO", how="outer", validate="one_to_one",
    )
    period_columns = [column for column in master.columns if column.startswith("TARGET_PERIOD")]
    master["TARGET_PERIOD"] = master[period_columns].bfill(axis=1).iloc[:, 0]
    master = master.drop(columns=[column for column in period_columns if column != "TARGET_PERIOD"])
    master["HAS_CONSUMPTION_PREDICTION"] = master["APRIL_PREDICTED_CONSUMPTION"].notna()
    master["HAS_REVENUE_PREDICTION"] = master[["REVENUE_RISK_CLASS", "REVENUE_RISK_PROBABILITY"]].notna().all(axis=1)
    master["HAS_ANOMALY_PREDICTION"] = master[["ANOMALY_CLASS", "ANOMALY_PROBABILITY"]].notna().all(axis=1)
    master["REVENUE_RISK"] = master["REVENUE_RISK_CLASS"].map({1: "Risk", 0: "No Risk"})
    master["UNUSUAL_CONSUMPTION"] = master["ANOMALY_CLASS"].map({1: "Unusual", 0: "Normal"})
    return master.sort_values("CONSUMER_NO").reset_index(drop=True), []


def latest_safe_attributes(historical):
    if historical is None or historical.empty:
        return None
    latest_period = historical["PERIOD"].max()
    latest = historical.loc[historical["PERIOD"].eq(latest_period)].copy()
    columns = [
        "CONSUMER_NO", "TARRIF", "FEEDER_NO", "FEEDER_CODE", "FEEDER_NAME",
        "CONTRACT_LOAD", "SOLAR_CONSUMER",
    ]
    return latest[columns].drop_duplicates("CONSUMER_NO")


def predictions_with_attributes():
    predictions, errors = load_april_predictions()
    if predictions is None:
        return None, errors
    attributes = latest_safe_attributes(load_historical_data())
    if attributes is not None:
        predictions = predictions.merge(attributes, on="CONSUMER_NO", how="left", validate="one_to_one")
    return predictions, errors


def page_overview():
    st.title("AI-Based Consumer Behaviour Analytics, Revenue Risk and Consumption Anomaly Prediction for Power Distribution Utilities")
    st.subheader("Project summary")
    st.markdown(
        "This project develops a supervised machine-learning consumer analytics system for a Power Distribution Utility "
        "using historical billing and consumption data. It analyses consumer behaviour and provides three next-month "
        "modules: consumption prediction, revenue-risk prediction and unusual-consumption risk prediction. The dashboard "
        "brings together historical analytics, April 2026 predictions, Consumer 360 views and held-out model evaluation."
    )

    st.subheader("Project objectives")
    objective_columns = st.columns(3)
    objectives = [
        ("Consumption prediction", "Predict next-month electricity consumption using historical consumer information.", "MLR-4"),
        ("Revenue risk", "Identify next-month revenue-risk behaviour using billing, payment and arrears information.", "RF-3-balanced"),
        ("Unusual consumption risk", "Identify next-month unusual consumption relative to historical patterns.", "DT-3-balanced"),
    ]
    for column, (heading, text, model) in zip(objective_columns, objectives):
        with column.container(border=True):
            st.markdown(f"**{heading}**")
            st.markdown(text)
            st.caption(f"Selected model: {model}")

    with st.container(border=True):
        st.markdown("**Compact project methodology**")
        st.markdown(
            "36 monthly files → data loading → cleaning → EDA and statistics → feature engineering → chronological split "
            "→ multiple models → validation selection → held-out final test → future prediction → dashboard"
        )

    split_columns = st.columns(4)
    split_details = [
        ("TRAIN", "Through Apr-2025"), ("VALIDATION", "May-2025 to Oct-2025"),
        ("FINAL TEST", "Nov-2025 to Mar-2026"), ("FUTURE", "Apr-2026"),
    ]
    for column, (name, period) in zip(split_columns, split_details):
        with column.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(period)
    st.caption("Validation data were used for model comparison and selection. The held-out final test was used only for final model evaluation.")

    historical = load_historical_data()
    if historical is None:
        file_error("Historical analytics", "data/public/dashboard_history.csv")
        return

    with st.container(border=True):
        st.markdown("**Data information and privacy**")
        information = [
            ("Historical data", "Apr-2023 to Mar-2026"), ("Monthly source files", "36"),
            ("Latest actual month", "Mar-2026"), ("Future period", "Apr-2026"),
            ("Consumer population", format_indian(historical["CONSUMER_NO"].nunique())),
        ]
        for start, size in [(0, 3), (3, 2)]:
            info_columns = st.columns(size)
            for column, (label, value) in zip(info_columns, information[start:start + size]):
                with column:
                    value_card(label, value)
        st.caption(
            "The project uses an anonymized/synthetic dataset representing the structure and analytical characteristics "
            "of a utility billing dataset. Confidential and consumer-identifiable operational data are not published."
        )

    st.subheader("Historical actual analysis")
    financial_years = sorted(historical["FINANCIAL_YEAR"].dropna().unique())
    default_fy = financial_years.index("2025-2026") if "2025-2026" in financial_years else len(financial_years) - 1
    filter_columns = st.columns(2)
    financial_year = filter_columns[0].selectbox("Financial Year", financial_years, index=default_fy, key="overview_fy")
    selected_month = filter_columns[1].selectbox("Month", MONTH_ORDER, key="overview_month")
    filtered = historical.loc[historical["FINANCIAL_YEAR"].eq(financial_year)].copy()
    if selected_month != "All Months":
        month_number = MONTH_NUMBERS[selected_month]
        filtered = filtered.loc[filtered["PERIOD"].dt.month.eq(month_number)]
        selected_period = filtered["PERIOD"].max()
        view_text = selected_period.strftime("%B %Y") if pd.notna(selected_period) else selected_month
    else:
        view_text = "All Months"
    st.markdown(f"**Viewing: FY {financial_year} | {view_text}**")
    if filtered.empty:
        st.info("No historical rows match the selected period.")
        return

    total_consumption = filtered["CONSUMPTION_UNIT"].sum(min_count=1)
    consumers = filtered["CONSUMER_NO"].nunique()
    end_period = filtered["PERIOD"].max()
    end_arrears = filtered.loc[filtered["PERIOD"].eq(end_period), "CLOSING_ARREARS"].sum(min_count=1)
    kpis = [
        ("Total consumers", format_indian(consumers)),
        ("Total consumption", format_indian(total_consumption, 1) + " units"),
        ("Total bill amount", money(filtered["BILL_AMOUNT"].sum(min_count=1))),
        ("Total payment amount", money(filtered["PAYMENT_AMOUNT"].sum(min_count=1))),
        ("Closing arrears — period end", money(end_arrears)),
        ("Average consumption per consumer", format_indian(total_consumption / consumers, 1) + " units" if consumers else "Not available"),
    ]
    for start in [0, 2, 4]:
        columns = st.columns(2)
        for column, (label, value) in zip(columns, kpis[start:start + 2]):
            with column.container(border=True):
                value_card(label, value)

    monthly = filtered.groupby("PERIOD", as_index=False).agg(
        TOTAL_CONSUMPTION=("CONSUMPTION_UNIT", "sum"),
        BILL_AMOUNT=("BILL_AMOUNT", "sum"), PAYMENT_AMOUNT=("PAYMENT_AMOUNT", "sum"),
        CLOSING_ARREARS=("CLOSING_ARREARS", "sum"), CONSUMERS=("CONSUMER_NO", "nunique"),
    )
    tariff = filtered.groupby("TARRIF", dropna=False, as_index=False).agg(
        TOTAL_CONSUMPTION=("CONSUMPTION_UNIT", "sum"),
        AVERAGE_CONSUMPTION=("CONSUMPTION_UNIT", "mean"),
        BILL_AMOUNT=("BILL_AMOUNT", "sum"), PAYMENT_AMOUNT=("PAYMENT_AMOUNT", "sum"),
        CLOSING_ARREARS=("CLOSING_ARREARS", "sum"), CONSUMERS=("CONSUMER_NO", "nunique"),
    )
    tariff["TARRIF"] = tariff["TARRIF"].fillna("Missing")
    if selected_month == "All Months":
        show_plot(px.line(monthly, x="PERIOD", y="TOTAL_CONSUMPTION", markers=True, title="Monthly electricity consumption", labels={"PERIOD": "Actual month", "TOTAL_CONSUMPTION": "Consumption units"}))
        financial = monthly.melt("PERIOD", ["BILL_AMOUNT", "PAYMENT_AMOUNT"], var_name="Series", value_name="Amount")
        show_plot(px.line(financial, x="PERIOD", y="Amount", color="Series", markers=True, title="Bill amount vs payment amount"))
        show_plot(px.line(monthly, x="PERIOD", y="CLOSING_ARREARS", markers=True, title="Closing arrears by month", labels={"PERIOD": "Actual month", "CLOSING_ARREARS": "Closing arrears"}))
    else:
        show_plot(px.bar(tariff, x="TARRIF", y="TOTAL_CONSUMPTION", title=f"Consumption by tariff — {view_text}", labels={"TARRIF": "Tariff", "TOTAL_CONSUMPTION": "Consumption units"}))
        financial = tariff.melt("TARRIF", ["BILL_AMOUNT", "PAYMENT_AMOUNT"], var_name="Series", value_name="Amount")
        show_plot(px.bar(financial, x="TARRIF", y="Amount", color="Series", barmode="group", title=f"Bill vs payment by tariff — {view_text}"))
        show_plot(px.bar(tariff, x="TARRIF", y="CLOSING_ARREARS", title=f"Closing arrears by tariff — {view_text}", labels={"TARRIF": "Tariff", "CLOSING_ARREARS": "Closing arrears"}))
    show_plot(px.bar(tariff, x="TARRIF", y="AVERAGE_CONSUMPTION", title="Average consumption by tariff", labels={"TARRIF": "Tariff", "AVERAGE_CONSUMPTION": "Average consumption units"}))
    if selected_month == "All Months":
        show_plot(px.line(monthly, x="PERIOD", y="CONSUMERS", markers=True, title="Monthly consumer population", labels={"PERIOD": "Actual month", "CONSUMERS": "Unique consumers"}))
    else:
        show_plot(px.bar(tariff, x="TARRIF", y="CONSUMERS", title=f"Consumer count by tariff — {view_text}", labels={"TARRIF": "Tariff", "CONSUMERS": "Unique consumers"}))


def page_all_predictions():
    st.title("April 2026 — All Consumer Predictions")
    with st.container(border=True):
        st.markdown("**Consumption Prediction — MLR-4  |  Revenue Risk — RF-3-balanced  |  Unusual Consumption — DT-3-balanced**")
    st.caption("April-2026 predictions were generated using the final models selected from validation performance. April actual outcomes are unavailable, so no April accuracy is reported.")
    data, errors = predictions_with_attributes()
    if data is None:
        for error in errors:
            st.error(error)
        return
    invalid_periods = data["TARGET_PERIOD"].dropna().dt.to_period("M").ne(pd.Period("2026-04")).sum()
    if invalid_periods:
        st.error("The required prediction files contain a target period other than April 2026.")
        return

    control_columns = st.columns([2, 1, 1, 1])
    search = control_columns[0].text_input("Consumer Search", placeholder="Enter consumer number", key="all_predictions_search").strip()
    revenue_filter = control_columns[1].selectbox("Revenue Risk", ["All", "Risk", "No Risk"], key="revenue_filter")
    anomaly_filter = control_columns[2].selectbox("Unusual Consumption", ["All", "Unusual", "Normal"], key="anomaly_filter")
    tariffs = sorted(data["TARRIF"].dropna().astype(str).unique()) if "TARRIF" in data else []
    tariff_filter = control_columns[3].selectbox("Tariff", ["All"] + tariffs, key="tariff_filter")

    filtered = data.copy()
    if search:
        filtered = filtered.loc[filtered["CONSUMER_NO"].str.contains(search, case=False, regex=False, na=False)]
    if revenue_filter != "All":
        filtered = filtered.loc[filtered["REVENUE_RISK"].eq(revenue_filter)]
    if anomaly_filter != "All":
        filtered = filtered.loc[filtered["UNUSUAL_CONSUMPTION"].eq(anomaly_filter)]
    if tariff_filter != "All":
        filtered = filtered.loc[filtered["TARRIF"].astype(str).eq(tariff_filter)]
    if filtered.empty:
        st.info("No consumers match the selected filters.")
        return

    consumption_total = filtered["APRIL_PREDICTED_CONSUMPTION"].sum(min_count=1)
    valid_revenue = filtered["REVENUE_RISK_CLASS"].dropna()
    valid_anomaly = filtered["ANOMALY_CLASS"].dropna()
    revenue_count = int(valid_revenue.eq(1).sum())
    anomaly_count = int(valid_anomaly.eq(1).sum())
    columns = st.columns(2)
    with columns[0].container(border=True):
        value_card("Displayed consumers", format_indian(filtered["CONSUMER_NO"].nunique()))
    with columns[1].container(border=True):
        value_card("Predicted April consumption", format_indian(consumption_total, 1) + " units")
    columns = st.columns(2)
    with columns[0].container(border=True):
        value_card("Revenue Risk", format_indian(revenue_count), f"{100 * revenue_count / len(valid_revenue):.1f}% of valid predictions" if len(valid_revenue) else "Prediction unavailable")
    with columns[1].container(border=True):
        value_card("Unusual Consumption", format_indian(anomaly_count), f"{100 * anomaly_count / len(valid_anomaly):.1f}% of valid predictions" if len(valid_anomaly) else "Prediction unavailable")
    st.markdown(f"**Showing {format_indian(len(filtered))} of {format_indian(len(data))} consumers**")

    display_columns = [
        "CONSUMER_NO", "APRIL_PREDICTED_CONSUMPTION", "REVENUE_RISK",
        "REVENUE_RISK_PROBABILITY", "UNUSUAL_CONSUMPTION", "ANOMALY_PROBABILITY",
    ]
    display = filtered[display_columns].rename(columns={
        "CONSUMER_NO": "Consumer No.", "APRIL_PREDICTED_CONSUMPTION": "April Predicted Consumption",
        "REVENUE_RISK": "Revenue Risk", "REVENUE_RISK_PROBABILITY": "Revenue Risk Probability",
        "UNUSUAL_CONSUMPTION": "Unusual Consumption", "ANOMALY_PROBABILITY": "Unusual Probability",
    })
    st.dataframe(
        display, hide_index=True, width="stretch",
        column_config={
            "April Predicted Consumption": st.column_config.NumberColumn(format="%.1f"),
            "Revenue Risk Probability": st.column_config.NumberColumn(format="%.3f"),
            "Unusual Probability": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.download_button(
        "Download current filtered table",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="april2026_filtered_consumer_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def page_consumer_360():
    st.title("Consumer 360")
    predictions, errors = predictions_with_attributes()
    if predictions is None:
        for error in errors:
            st.error(error)
        return
    historical = load_historical_data()
    if historical is None:
        file_error("Consumer history", "data/public/dashboard_history.csv")
        return
    consumers = predictions["CONSUMER_NO"].dropna().sort_values().tolist()
    selected_consumer = st.selectbox("Select Consumer Number", consumers, key="consumer_360_selector")
    row = predictions.loc[predictions["CONSUMER_NO"].eq(selected_consumer)].iloc[0]
    history = historical.loc[historical["CONSUMER_NO"].eq(selected_consumer)].sort_values("PERIOD").copy()
    latest = history.iloc[-1] if not history.empty else None

    with st.container(border=True):
        st.markdown("**Safe consumer information**")
        values = [
            ("Consumer Number", selected_consumer),
            ("Tariff", safe_text(row.get("TARRIF"))),
            ("Feeder", safe_text(row.get("FEEDER_NAME", row.get("FEEDER_NO")))),
            ("Contract Load", safe_text(row.get("CONTRACT_LOAD"))),
            ("Solar Status", safe_text(row.get("SOLAR_CONSUMER"))),
            ("Latest Actual Month", latest["PERIOD"].strftime("%B %Y") if latest is not None and pd.notna(latest["PERIOD"]) else "Not available"),
        ]
        for start in [0, 2, 4]:
            info_columns = st.columns(2)
            for column, (label, value) in zip(info_columns, values[start:start + 2]):
                with column:
                    value_card(label, value)

    march_rows = history.loc[history["PERIOD"].eq(pd.Timestamp("2026-03-01"))]
    march_consumption = march_rows.iloc[0]["CONSUMPTION_UNIT"] if not march_rows.empty else np.nan
    april_consumption = row.get("APRIL_PREDICTED_CONSUMPTION")
    change = april_consumption - march_consumption if pd.notna(april_consumption) and pd.notna(march_consumption) else np.nan
    pct_change = 100 * change / march_consumption if pd.notna(change) and march_consumption != 0 else np.nan
    card_columns = st.columns(3)
    with card_columns[0].container(border=True):
        st.markdown("**April 2026 predicted consumption**")
        value_card("Prediction", format_indian(april_consumption, 1) + " units" if pd.notna(april_consumption) else "Prediction unavailable")
        st.caption("Model: MLR-4")
        st.markdown(f"March actual: **{format_indian(march_consumption, 1)} units**")
        if pd.notna(change):
            st.caption(f"Change from March: {change:+,.1f} units" + (f" ({pct_change:+.1f}%)" if pd.notna(pct_change) else ""))
    with card_columns[1].container(border=True):
        st.markdown("**April 2026 Revenue Risk Indicator**")
        status = row.get("REVENUE_RISK") if pd.notna(row.get("REVENUE_RISK")) else "Prediction unavailable"
        value_card("Status", status)
        st.caption("Model: RF-3-balanced")
        probability = row.get("REVENUE_RISK_PROBABILITY")
        st.markdown(f"Class-1 risk probability: **{probability:.3f}**" if pd.notna(probability) else "Class-1 risk probability: Prediction unavailable")
    with card_columns[2].container(border=True):
        st.markdown("**April 2026 Unusual Consumption Risk Indicator**")
        status = row.get("UNUSUAL_CONSUMPTION") if pd.notna(row.get("UNUSUAL_CONSUMPTION")) else "Prediction unavailable"
        value_card("Status", status)
        st.caption("Model: DT-3-balanced")
        probability = row.get("ANOMALY_PROBABILITY")
        st.markdown(f"Class-1 unusual probability: **{probability:.3f}**" if pd.notna(probability) else "Class-1 unusual probability: Prediction unavailable")

    st.subheader("Historical Actual and April 2026 Prediction")
    if history.empty:
        st.info("Historical actual data are not available for this consumer.")
        return
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history["PERIOD"], y=history["CONSUMPTION_UNIT"], mode="lines+markers", name="Historical Actual"))
    if pd.notna(april_consumption):
        bridge_x = [history.iloc[-1]["PERIOD"], pd.Timestamp("2026-04-01")]
        bridge_y = [history.iloc[-1]["CONSUMPTION_UNIT"], april_consumption]
        figure.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", line=dict(dash="dash"), marker=dict(size=10), name="April 2026 Prediction"))
    figure.update_layout(title="Consumption history and separate April prediction", xaxis_title="Month", yaxis_title="Consumption units")
    show_plot(figure)

    actual = history["CONSUMPTION_UNIT"].dropna()
    recent = actual.tail(3)
    recent_trend = recent.iloc[-1] - recent.iloc[0] if len(recent) >= 2 else np.nan
    stats = [
        ("Latest", actual.iloc[-1] if len(actual) else np.nan), ("Mean", actual.mean()),
        ("Median", actual.median()), ("Minimum", actual.min()), ("Maximum", actual.max()),
        ("Recent 3-month average", recent.mean()), ("Recent trend", recent_trend),
    ]
    for start, size in [(0, 3), (3, 2), (5, 2)]:
        stat_columns = st.columns(size)
        for column, (label, value) in zip(stat_columns, stats[start:start + size]):
            with column.container(border=True):
                value_card(label, format_indian(value, 1) + " units" if pd.notna(value) else "Not available")

    finance = history.melt(id_vars="PERIOD", value_vars=["BILL_AMOUNT", "PAYMENT_AMOUNT"], var_name="Series", value_name="Amount")
    show_plot(px.line(finance, x="PERIOD", y="Amount", color="Series", markers=True, title="Historical bill and payment trend", labels={"PERIOD": "Actual month"}))
    show_plot(px.line(history, x="PERIOD", y="CLOSING_ARREARS", markers=True, title="Historical closing arrears", labels={"PERIOD": "Actual month", "CLOSING_ARREARS": "Closing arrears"}))

    with st.container(border=True):
        st.markdown("**Unusual-consumption context**")
        context = st.columns(3)
        with context[0]:
            value_card("Historical minimum", format_indian(actual.min(), 1) + " units")
        with context[1]:
            value_card("Historical maximum", format_indian(actual.max(), 1) + " units")
        with context[2]:
            value_card("Historical mean", format_indian(actual.mean(), 1) + " units")
        context = st.columns(3)
        with context[0]:
            value_card("Recent 3-month average", format_indian(recent.mean(), 1) + " units")
        with context[1]:
            value_card("April prediction", format_indian(april_consumption, 1) + " units" if pd.notna(april_consumption) else "Prediction unavailable")
        with context[2]:
            value_card("Indicator", safe_text(row.get("UNUSUAL_CONSUMPTION")), f"Probability {row.get('ANOMALY_PROBABILITY'):.3f}" if pd.notna(row.get("ANOMALY_PROBABILITY")) else None)
        st.caption("This project-defined statistical proxy is a decision-support indicator. It does not establish theft, fraud, meter tampering or wrongdoing.")


def prepare_validation_rows(comparison, configurations, classification=False):
    rows = comparison.loc[comparison["configuration"].isin(configurations)].copy()
    order = {name: number for number, name in enumerate(configurations)}
    rows["_ORDER"] = rows["configuration"].map(order)
    rows = rows.sort_values("_ORDER").drop(columns="_ORDER")
    if classification:
        columns = [
            "configuration", "VALIDATION_Accuracy", "VALIDATION_Precision_0", "VALIDATION_Recall_0",
            "VALIDATION_F1_0", "VALIDATION_Precision_1", "VALIDATION_Recall_1", "VALIDATION_F1_1",
            "VALIDATION_ROC_AUC", "VALIDATION_AP", "VALIDATION_minus_TRAIN_ROC_AUC",
            "VALIDATION_minus_TRAIN_F1_1", "VALIDATION_minus_TRAIN_AP",
        ]
        return rows[[column for column in columns if column in rows]]
    return rows


def show_consumption_evaluation():
    comparison = load_csv("reports/summaries/consumption_model_comparison.csv")
    final_metrics = load_csv("reports/summaries/consumption_final_test_metrics.csv")
    predictions = load_csv("data/processed/consumption_final_test_predictions.csv")
    if comparison is None or final_metrics is None:
        file_error("Consumption evaluation", "reports/summaries/consumption_model_comparison.csv or consumption_final_test_metrics.csv")
        return
    rf_rows = comparison.loc[comparison["algorithm"].astype(str).str.contains("Random Forest", case=False, na=False)]
    best_rf = rf_rows.loc[rf_rows["VALIDATION_RMSE"].idxmin(), "configuration"]
    configurations = ["Persistence", "MLR-4", best_rf]
    validation = prepare_validation_rows(comparison, configurations)
    status = {"Persistence": "Persistence Baseline", "MLR-4": "Official selected model", best_rf: "Best Random Forest by validation RMSE"}
    validation["Status"] = validation["configuration"].map(status)
    table = validation[["configuration", "VALIDATION_MAE", "VALIDATION_RMSE", "VALIDATION_R2", "Status"]].rename(columns={
        "configuration": "Model", "VALIDATION_MAE": "MAE", "VALIDATION_RMSE": "RMSE", "VALIDATION_R2": "R²",
    })
    st.subheader("VALIDATION comparison")
    st.dataframe(table, hide_index=True, width="stretch")
    for metric, direction in [("MAE", "lower is better"), ("RMSE", "lower is better"), ("R²", "higher is better")]:
        show_plot(px.bar(table, x="Model", y=metric, color="Status", title=f"VALIDATION {metric} — {direction}"))
    st.caption("MLR-4 remains the frozen official model. The Random Forest representative is selected only by the saved validation RMSE for comparison.")

    st.subheader("Held-out FINAL TEST — MLR-4 vs Persistence Baseline")
    final_table = final_metrics[["configuration", "sample_count", "MAE", "RMSE", "R2"]].rename(columns={"configuration": "Model", "sample_count": "Sample Count", "R2": "R²"})
    st.dataframe(final_table, hide_index=True, width="stretch")
    if predictions is None:
        st.info("Detailed evaluation output unavailable.")
        return
    scatter = px.scatter(predictions, x="actual_consumption", y="selected_prediction", opacity=0.45, title="FINAL TEST actual vs predicted — MLR-4", labels={"actual_consumption": "Actual consumption", "selected_prediction": "Predicted consumption"})
    low = min(predictions["actual_consumption"].min(), predictions["selected_prediction"].min())
    high = max(predictions["actual_consumption"].max(), predictions["selected_prediction"].max())
    scatter.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line=dict(dash="dash"), name="Perfect prediction"))
    show_plot(scatter)
    residual = px.scatter(predictions, x="selected_prediction", y="residual_actual_minus_predicted", opacity=0.45, title="FINAL TEST residual plot — MLR-4", labels={"selected_prediction": "Predicted consumption", "residual_actual_minus_predicted": "Actual minus predicted"})
    residual.add_hline(y=0, line_dash="dash")
    show_plot(residual)


def classification_settings(module):
    if module == "Revenue Risk":
        return {
            "comparison": "reports/summaries/revenue_model_comparison.csv",
            "selection": "reports/summaries/revenue_model_selection.csv",
            "final": "reports/summaries/revenue_final_test_metrics.csv",
            "predictions": "data/processed/revenue_final_test_predictions.csv",
            "selected": "RF-3-balanced", "representative_dt": "DT-3", "representative_rf": "RF-3",
            "title": "Revenue Risk", "class_0": "No Risk", "class_1": "Risk",
            "disclaimer": "Revenue Risk is a project-defined academic proxy based on payment behaviour. It is not an official utility default or revenue-risk classification.",
        }
    return {
        "comparison": "reports/summaries/anomaly_model_comparison.csv",
        "selection": "reports/summaries/anomaly_model_selection.csv",
        "final": "reports/summaries/anomaly_final_test_metrics.csv",
        "predictions": "data/processed/anomaly_final_test_predictions.csv",
        "selected": "DT-3-balanced", "representative_dt": "DT-3-balanced", "representative_rf": "RF-4-balanced",
        "title": "Unusual Consumption Risk", "class_0": "Normal", "class_1": "Unusual",
        "disclaimer": "ANOMALY_NEXT is a project-defined statistical proxy for unusual consumption. Predictions are decision-support indicators and do not establish electricity theft, fraud, meter tampering, or wrongdoing.",
    }


def show_classification_evaluation(module):
    settings = classification_settings(module)
    comparison = load_csv(settings["comparison"])
    selection = load_csv(settings["selection"])
    final_metrics = load_csv(settings["final"])
    predictions = load_csv(settings["predictions"])
    if comparison is None or selection is None or final_metrics is None:
        file_error(f"{module} evaluation", settings["comparison"])
        return
    if str(selection.iloc[0]["configuration"]) != settings["selected"]:
        st.error(f"Official model inconsistency: expected {settings['selected']}; saved selection identifies {selection.iloc[0]['configuration']}.")
        return
    configurations = ["TRAIN-majority", settings["representative_dt"], settings["representative_rf"], settings["selected"]]
    configurations = list(dict.fromkeys(configurations))
    validation = prepare_validation_rows(comparison, configurations, classification=True)
    labels = {
        "TRAIN-majority": "Baseline",
        settings["representative_dt"]: settings["representative_dt"],
        settings["representative_rf"]: settings["representative_rf"],
        settings["selected"]: settings["selected"] + " (Selected)",
    }
    validation["Model"] = validation["configuration"].map(labels)
    if module == "Revenue Risk":
        table_columns = [
            "Model", "VALIDATION_ROC_AUC", "VALIDATION_AP", "VALIDATION_F1_1",
            "VALIDATION_Recall_0", "VALIDATION_F1_0",
            "VALIDATION_minus_TRAIN_ROC_AUC", "VALIDATION_minus_TRAIN_F1_1",
        ]
    else:
        table_columns = [
            "Model", "VALIDATION_ROC_AUC", "VALIDATION_AP", "VALIDATION_Precision_1",
            "VALIDATION_Recall_1", "VALIDATION_F1_1", "VALIDATION_minus_TRAIN_ROC_AUC",
            "VALIDATION_minus_TRAIN_AP", "VALIDATION_minus_TRAIN_F1_1",
        ]
    st.subheader("VALIDATION comparison")
    validation_table = validation[[column for column in table_columns if column in validation]].rename(columns={
        "VALIDATION_ROC_AUC": "ROC-AUC", "VALIDATION_AP": "AP",
        "VALIDATION_Precision_1": "Class 1 Precision", "VALIDATION_Recall_1": "Class 1 Recall",
        "VALIDATION_F1_1": "Class 1 F1", "VALIDATION_Recall_0": "Class 0 Recall",
        "VALIDATION_F1_0": "Class 0 F1", "VALIDATION_minus_TRAIN_ROC_AUC": "AUC Gap",
        "VALIDATION_minus_TRAIN_AP": "AP Gap", "VALIDATION_minus_TRAIN_F1_1": "Class 1 F1 Gap",
    })
    st.dataframe(validation_table, hide_index=True, width="stretch")
    if module == "Revenue Risk":
        st.caption("Primary evidence: validation ROC-AUC. Class-1 F1, Class-0 recall/F1 and TRAIN-to-VALIDATION gaps provide supporting evidence. Accuracy is not used alone.")
    else:
        st.caption("Primary evidence: validation Class-1 F1. Class-1 recall/precision, AP, ROC-AUC and TRAIN-to-VALIDATION gaps provide supporting evidence. Accuracy is not used alone.")
    roc_chart = px.bar(validation, x="Model", y="VALIDATION_ROC_AUC", title="VALIDATION ROC-AUC comparison", labels={"VALIDATION_ROC_AUC": "ROC-AUC"})
    roc_chart.update_yaxes(range=[0, 1])
    show_plot(roc_chart)
    for metric in ["Precision", "Recall", "F1"]:
        chart = validation[["Model", f"VALIDATION_{metric}_0", f"VALIDATION_{metric}_1"]].melt("Model", var_name="Class", value_name=metric)
        chart["Class"] = chart["Class"].map({f"VALIDATION_{metric}_0": f"Class 0 — {settings['class_0']}", f"VALIDATION_{metric}_1": f"Class 1 — {settings['class_1']}"})
        figure = px.bar(chart, x="Model", y=metric, color="Class", barmode="group", title=f"VALIDATION {metric.lower()} by class")
        figure.update_yaxes(range=[0, 1])
        show_plot(figure)
    support = validation[["Model", "VALIDATION_Accuracy", "VALIDATION_AP"]].melt("Model", var_name="Metric", value_name="Value")
    support["Metric"] = support["Metric"].map({"VALIDATION_Accuracy": "Accuracy", "VALIDATION_AP": "Average Precision"})
    support_chart = px.bar(support, x="Model", y="Value", color="Metric", barmode="group", title="VALIDATION accuracy and Average Precision")
    support_chart.update_yaxes(range=[0, 1])
    show_plot(support_chart)

    st.markdown("**Model-selection reasoning**")
    st.markdown(str(selection.iloc[0].get("reason", "Saved validation evidence supports the frozen selected model.")))
    if module == "Unusual Consumption Risk":
        st.caption("DT-3-balanced was selected from TRAIN/VALIDATION evidence because its competitive validation performance, much stronger anomaly recall and smaller generalization gaps were more stable than the severely overfit RF-4-balanced candidate.")

    st.subheader(f"Held-out FINAL TEST — {settings['selected']} vs Majority Baseline")
    final_columns = [
        "configuration", "sample_count", "Accuracy", "Precision_0", "Recall_0", "F1_0",
        "Precision_1", "Recall_1", "F1_1", "ROC_AUC", "AP",
    ]
    final_table = final_metrics[[column for column in final_columns if column in final_metrics]].rename(columns={
        "configuration": "Model", "sample_count": "Sample Count",
        "Precision_0": f"Precision — {settings['class_0']}", "Recall_0": f"Recall — {settings['class_0']}",
        "F1_0": f"F1 — {settings['class_0']}", "Precision_1": f"Precision — {settings['class_1']}",
        "Recall_1": f"Recall — {settings['class_1']}", "F1_1": f"F1 — {settings['class_1']}",
        "ROC_AUC": "ROC-AUC", "AP": "Average Precision",
    })
    st.dataframe(final_table, hide_index=True, width="stretch")
    if predictions is None:
        st.info("Detailed evaluation output unavailable.")
    else:
        matrix = pd.crosstab(predictions["actual_class"], predictions["predicted_class"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        matrix.index = [settings["class_0"], settings["class_1"]]
        matrix.columns = [settings["class_0"], settings["class_1"]]
        confusion = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", title=f"FINAL TEST confusion matrix — {settings['selected']}", labels={"x": "Predicted", "y": "Actual", "color": "Count"})
        show_plot(confusion)
        fpr, tpr, roc_area = calculate_roc(predictions["actual_class"], predictions["class_1_probability"])
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{settings['selected']} (AUC {roc_area:.4f})"))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="No-discrimination reference"))
        roc.update_layout(title=f"FINAL TEST ROC — {settings['selected']}", xaxis_title="False positive rate", yaxis_title="True positive rate")
        show_plot(roc)
    st.info(settings["disclaimer"])


def show_statistics():
    st.divider()
    st.subheader("Statistical analysis")
    tests = load_csv("reports/summaries/statistical_tests.csv")
    if tests is None:
        st.info("Detailed evaluation output unavailable.")
    else:
        pearson_1 = tests.loc[tests["Variables"].eq("CONSUMPTION_UNIT; BILL_AMOUNT")].iloc[0]
        pearson_2 = tests.loc[tests["Variables"].eq("BILL_AMOUNT; PAYMENT_AMOUNT")].iloc[0]
        table = pd.DataFrame([
            {
                "Statistical Test": "Pearson Correlation", "Variables": "Consumption Unit vs Bill Amount",
                "Statistic / Result": f"r = {pearson_1['Statistic']:.6f}", "P-Value / Status": "0.0 (numerical underflow)",
                "Interpretation": "Very strong positive linear association. Association does not imply causation.",
            },
            {
                "Statistical Test": "Pearson Correlation", "Variables": "Bill Amount vs Payment Amount",
                "Statistic / Result": f"r = {pearson_2['Statistic']:.6f}", "P-Value / Status": "0.0 (numerical underflow)",
                "Interpretation": "Very strong positive linear association. Association does not imply causation.",
            },
            {
                "Statistical Test": "Chi-Square Test", "Variables": "Tariff vs Solar Consumer",
                "Statistic / Result": "Not Performed / Skipped", "P-Value / Status": "Assumption Not Satisfied",
                "Interpretation": "Expected cell frequencies were too sparse for a reliable Chi-square test.",
            },
        ])
        st.dataframe(table, hide_index=True, width="stretch")
    descriptive = load_csv("reports/summaries/descriptive_statistics.csv")
    if descriptive is not None:
        wanted = ["CONSUMPTION_UNIT", "BILL_AMOUNT", "PAYMENT_AMOUNT", "CLOSING_ARREARS"]
        descriptive = descriptive.loc[descriptive["variable"].isin(wanted), [
            "variable", "mean", "median", "standard_deviation", "minimum", "maximum", "skewness", "kurtosis",
        ]]
        descriptive = descriptive.rename(columns={
            "variable": "Variable", "mean": "Mean", "median": "Median",
            "standard_deviation": "Standard Deviation", "minimum": "Minimum",
            "maximum": "Maximum", "skewness": "Skewness", "kurtosis": "Kurtosis",
        })
        st.markdown("**Descriptive statistics from Step 4**")
        st.dataframe(descriptive, hide_index=True, width="stretch")


def page_model_evaluation():
    st.title("Model Comparison & Evaluation")
    st.caption("Presentation flow: VALIDATION comparison and selection → frozen-model held-out FINAL TEST → existing Step 4 statistical analysis. April 2026 is future prediction only.")
    module = st.selectbox(
        "Select module",
        ["Consumption Prediction", "Revenue Risk", "Unusual Consumption Risk"],
        key="model_evaluation_selector",
    )
    with st.container(border=True):
        st.markdown(f"**Official selected model: {OFFICIAL_MODELS[module]}**")
        st.caption("Model selection is frozen. The dashboard does not train, tune, threshold-optimize or reselect models.")
    if module == "Consumption Prediction":
        show_consumption_evaluation()
    else:
        show_classification_evaluation(module)
    show_statistics()


st.sidebar.title("EM630 Consumer Analytics")
page = st.sidebar.radio("Page", PAGES, key="main_navigation")
st.sidebar.caption("Historical actuals: Apr-2023 to Mar-2026")
st.sidebar.caption("Future prediction: Apr-2026")

if page == "Overview":
    page_overview()
elif page == "All Consumer Predictions":
    page_all_predictions()
elif page == "Consumer 360":
    page_consumer_360()
else:
    page_model_evaluation()
