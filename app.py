import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet import (community package name is `prophet`)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None  # Streamlit will warn if missing

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly")

# 1) DATA UPLOAD
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Allow user to pick a worksheet
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Worksheet (tab) to use", xls.sheet_names)
    df = xls.parse(sheet)

    st.success(
        f"Loaded sheet '{sheet}' with {df.shape[0]} rows and {df.shape[1]} columns."
    )

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
    customers = st.multiselect(
        "Customer(s)",
        sorted(data["customer"].unique()),
        default=list(data["customer"].unique())[:1],
    )
    subset = data[data["customer"].isin(customers)]

    skus = st.multiselect(
        "SKU(s)",
        sorted(subset["sku"].unique()),
        default=list(subset["sku"].unique())[:1],
    )
    filtered = subset[subset["sku"].isin(skus)]

    # Raw history aggregated monthly for display
    monthly_hist = (
        filtered.groupby("date")["value"].sum().sort_index().resample("M").sum()
    )
    hist_display = monthly_hist.copy()
    hist_display.index = hist_display.index.to_period("M").astype(str)
    st.line_chart(hist_display, height=250)

    # 4) SIDEBAR SETTINGS
    st.sidebar.header("⚙️ Forecast settings")
    horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, value=12)

    model_type = st.sidebar.selectbox(
        "Model",
        [
            "Moving Average",
            "Exponential Smoothing (no season)",
            "Holt-Winters Seasonal",
            "Prophet",
        ],
    )

    # Model‑specific parameters
    if model_type == "Moving Average":
        ma_window = st.sidebar.slider("MA window (months)", 2, 24, value=3)
    elif model_type == "Exponential Smoothing (no season)":
        alpha = st.sidebar.slider("Smoothing alpha", 0.01, 1.0, value=0.3)
    elif model_type == "Holt-Winters Seasonal":
        seasonal_periods = st.sidebar.slider("Season length (months)", 3, 24, value=12)
        seasonality = st.sidebar.selectbox("Seasonality type", ["add", "mul"], index=0)
        trend_type = st.sidebar.selectbox("Trend type", ["add", "mul", None], index=0)
    else:
        ma_window = alpha = seasonal_periods = seasonality = trend_type = None

    if model_type == "Prophet" and Prophet is None:
        st.sidebar.error(
            "Prophet not installed. Add `prophet` to requirements.txt and redeploy."
        )

    # 5) EVENTS / LIFTS
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Events / Lifts (monthly)")
    with st.sidebar.form(key="event_form"):
        event_date = st.date_input("Event month")
        lift_pct = st.number_input("Lift % vs baseline", value=10.0, step=1.0)
        submitted = st.form_submit_button("Add / update event")

    if "events" not in st.session_state:
        st.session_state["events"] = {}

    if submitted:
        event_month = (
            pd.to_datetime(event_date) + pd.offsets.MonthEnd(0)
        ).normalize()
        st.session_state["events"][event_month] = lift_pct / 100.0

    if st.session_state["events"]:
        st.sidebar.write(
            {
                d.strftime("%Y-%m"): f"+{int(r*100)}%"
                for d, r in st.session_state["events"].items()
            }
        )

    # 6) RUN FORECAST
    if st.button("🚀 Run forecast"):
        # Monthly series for modelling
        ts = (
            filtered.groupby("date")["value"].sum().sort_index().resample("M").sum().fillna(0)
        )

        if model_type == "Moving Average":
            last_ma = ts.rolling(window=ma_window).mean().iloc[-1]
            future_idx = pd.date_range(
                ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M"
            )
            forecast = pd.Series(last_ma, index=future_idx)

        elif model_type == "Exponential Smoothing (no season)":
            model = ExponentialSmoothing(
                ts, trend=None, seasonal=None, initialization_method="estimated"
            )
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(horizon)

        elif model_type == "Holt-Winters Seasonal":
            if len(ts) < 2 * seasonal_periods:
                st.error(
                    f"Holt‑Winters needs ≥2×season length (≥{2*seasonal_periods} months) "
                    f"but only {len(ts)} available."
                )
                st.stop()
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
                st.error(
                    "Prophet library missing. Add `prophet` to requirements.txt and redeploy."
                )
                st.stop()
            df_prophet = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
            m = Prophet(
                yearly_seasonality=True,
                monthly_seasonality=False,
                weekly_seasonality=False,
            )
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon, freq="M")
            forecast_df = m.predict(future)
            forecast = forecast_df.set_index("ds")["yhat"].iloc[-horizon:]

        # Apply event lifts
        for d, r in st.session_state["events"].items():
            if d in forecast.index:
                forecast.loc[d] *= 1 + r

        # Display forecast with month labels
        forecast_display = forecast.copy()
        forecast_display.index = forecast_display.index.to_period("M").astype(str)

        st.subheader("🔮 Monthly Forecast")
        st.line_chart(forecast_display, height=250)

        # CSV download keeps original datetime index
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

# 7) FOOTER
st.markdown(
    "---\nMade with ❤️ & Streamlit. | Monthly forecasts with optional customer/SKU aggregation."
)
