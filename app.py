import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# Prophet import (works with the community fork `prophet`)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None  # Handle missing dependency gracefully

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
    model_type = st.sidebar.selectbox(
        "Model",
        [
            "Moving Average",
            "Exponential Smoothing (no season)",
            "Holt‑Winters Seasonal",
            "Prophet",
        ],
    )

    # Model‑specific parameters
    if model_type == "Moving Average":
        ma_window = st.sidebar.slider("MA window (months)", 2, 24, value=3)
    elif model_type == "Exponential Smoothing (no season)":
            model = ExponentialSmoothing(
                ts,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(horizon)

        elif model_type == "Holt‑Winters Seasonal":
            model = ExponentialSmoothing(
                ts,
                trend=trend_type,
                seasonal=seasonality,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit()
            forecast = fit.forecast(horizon)

        elif model_type == "Prophet":
            if Prophet is None:
                st.error("Prophet library missing. Add `prophet` to requirements.txt and redeploy.")
                st.stop()
            df_prophet = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
            m = Prophet(yearly_seasonality=True, monthly_seasonality=False, weekly_seasonality=False)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon, freq="M")
            forecast_df = m.predict(future)
            forecast = forecast_df.set_index("ds")["yhat"].iloc[-horizon:]

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
