import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

st.set_page_config(page_title="Sales Forecast", layout="wide")

st.title("📈 Sales Forecasting Tool")

# 1) DATA UPLOAD
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from the file.")

    # 2) MAP COLUMNS
    cols = df.columns.tolist()
    st.markdown("### 1️⃣ Map your columns")
    date_col = st.selectbox("Date column", cols, index=0)
    customer_col = st.selectbox("Customer column", cols, index=1)
    sku_col = st.selectbox("SKU column", cols, index=2)
    value_col = st.selectbox("Sales amount / Quantity column", cols, index=3)

    data = df[[date_col, customer_col, sku_col, value_col]].copy()
    data.columns = ["date", "customer", "sku", "value"]
    data["date"] = pd.to_datetime(data["date"])  # ensure datetime dtype

    # 3) FILTERS
    st.markdown("### 2️⃣ Choose customer & SKU to model")
    customer = st.selectbox("Customer", sorted(data["customer"].unique()))
    sku = st.selectbox(
        "SKU", sorted(data.loc[data["customer"] == customer, "sku"].unique())
    )

    filtered = (
        data[(data["customer"] == customer) & (data["sku"] == sku)]
        .sort_values("date")
        .set_index("date")
    )

    st.line_chart(filtered["value"], height=250)

    # 4) SIDEBAR SETTINGS
    st.sidebar.header("⚙️ Forecast settings")
    horizon = st.sidebar.number_input("Forecast horizon (weeks)", 1, 104, value=26)
    model_type = st.sidebar.selectbox("Model", ["Moving Average", "Exponential Smoothing"])
    if model_type == "Moving Average":
        ma_window = st.sidebar.slider("MA window (weeks)", 2, 52, value=4)
    else:
        alpha = st.sidebar.slider("Smoothing alpha", 0.01, 1.0, value=0.3)

    st.sidebar.markdown("---")
    st.sidebar.header("📅 Events / Lifts")
    with st.sidebar.form("events"):
        event_date = st.date_input("Event date")
        lift_pct = st.number_input("Lift % vs baseline", value=10.0, step=1.0)
        submitted = st.form_submit_button("Add / update event")
    if "events" not in st.session_state:
        st.session_state["events"] = {}
    if submitted:
        st.session_state["events"][pd.to_datetime(event_date)] = lift_pct / 100.0

    if st.session_state["events"]:
        st.sidebar.write(
            {
                d.strftime("%Y-%m-%d"): f"+{int(r*100)}%"
                for d, r in st.session_state["events"].items()
            }
        )

    # 5) FORECAST BUTTON
    if st.button("🚀 Run forecast"):
        ts = filtered["value"].asfreq("W").fillna(0)

        # MODEL FIT & FORECAST
        if model_type == "Moving Average":
            last_ma = ts.rolling(window=ma_window).mean().iloc[-1]
            future_idx = pd.date_range(ts.index[-1] + timedelta(weeks=1), periods=horizon, freq="W")
            forecast = pd.Series(last_ma, index=future_idx)
        else:
            model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(horizon)

        # APPLY EVENTS
        for d, r in st.session_state["events"].items():
            if d in forecast.index:
                forecast.loc[d] = forecast.loc[d] * (1 + r)

        st.subheader("🔮 Forecast")
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
st.markdown(
    "---\nMade with ❤️ & Streamlit.  |  Adjust the model, add events, and tweak lift ratios; results update instantly."
)
