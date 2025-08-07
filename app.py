import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly")

# 1) DATA UPLOAD
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Allow user to pick a worksheet
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Worksheet (tab) to use", xls.sheet_names)
    df = xls.parse(sheet)

    st.success(f"Loaded sheet '{sheet}' with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2) MAP COLUMNS
    cols = df.columns.tolist()
    st.markdown("### 1️⃣ Map your columns")
    date_col = st.selectbox("Date column", cols, index=0)
    customer_col = st.selectbox("Customer column", cols, index=1)
    sku_col = st.selectbox("SKU column", cols, index=2)
    value_col = st.selectbox("Sales amount / Quantity column", cols, index=3)

    data = df[[date_col, customer_col, sku_col, value_col]].copy()
    data.columns = ["date", "customer", "sku", "value"]
    data["date"] = pd.to_datetime(data["date"])

    # 3) FILTERS & AGGREGATION LEVEL
    st.markdown("### 2️⃣ Select customers / SKUs")
    customers = st.multiselect("Customer(s)", sorted(data["customer"].unique()), default=list(data["customer"].unique())[:1])
    subset = data[data["customer"].isin(customers)]

    skus = st.multiselect("SKU(s)", sorted(subset["sku"].unique()), default=list(subset["sku"].unique())[:1])
    filtered = subset[subset["sku"].isin(skus)]

    agg_series = (
        filtered.groupby("date")["value"].sum().sort_index()
    )

    st.line_chart(agg_series, height=250)

    # 4) SIDEBAR SETTINGS
    st.sidebar.header("⚙️ Forecast settings")
    horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, value=12)
    model_type = st.sidebar.selectbox("Model", ["Moving Average", "Exponential Smoothing"])
    if model_type == "Moving Average":
        ma_window = st.sidebar.slider("MA window (months)", 2, 24, value=3)
    else:
        alpha = st.sidebar.slider("Smoothing alpha", 0.01, 1.0, value=0.3)

    st.sidebar.markdown("---")
    st.sidebar.header("📅 Events / Lifts (monthly)")
    with st.sidebar.form(key="event_form"):
        event_date = st.date_input("Event month")
        lift_pct = st.number_input("Lift % vs baseline", value=10.0, step=1.0)
        submitted = st.form_submit_button("Add / update event")

    if "events" not in st.session_state:
        st.session_state["events"] = {}

    if submitted:
        event_month = (pd.to_datetime(event_date) + pd.offsets.MonthEnd(0)).normalize()
        st.session_state["events"][event_month] = lift_pct / 100.0

    if st.session_state["events"]:
        st.sidebar.write(
            {d.strftime("%Y-%m"): f"+{int(r*100)}%" for d, r in st.session_state["events"].items()}
        )

    # 5) FORECAST BUTTON
    if st.button("🚀 Run forecast"):
        # Aggregate to monthly frequency
        ts = agg_series.resample("M").sum().fillna(0)

        # MODEL FIT & FORECAST
        if model_type == "Moving Average":
            last_ma = ts.rolling(window=ma_window).mean().iloc[-1]
            future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
            forecast = pd.Series(last_ma, index=future_idx)
        else:
            model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(horizon)

        # APPLY EVENTS
        for d, r in st.session_state["events"].items():
            if d in forecast.index:
                forecast.loc[d] = forecast.loc[d] * (1 + r)

        st.subheader("🔮 Monthly Forecast")
        st.line_chart(forecast, height=250)

        result = forecast.reset_index().rename(columns={"index": "date", 0: "forecast"})

        @st.cache_data
        def _to_csv(df):
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download forecast as CSV",
            data=_to_csv(result),
            file_name="forecast.csv",
            mime="text/csv",
        )
else:
    st.info("👆 Upload an Excel file to begin.")

# 6) FOOTER
st.markdown("---\nMade with ❤️ & Streamlit. | Monthly forecasts with optional customer/SKU aggregation.")
