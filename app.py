import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly")

# 1️⃣ Upload
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # --- Load sheet ---
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
    raw = xls.parse(sheet)
    st.success(f"Loaded '{sheet}' → {raw.shape[0]}×{raw.shape[1]}")

    cols = raw.columns.tolist()

    # 2️⃣ Layout selection
    layout = st.radio(
        "How are your dates stored?",
        ["Rows – there is a date column", "Columns – each month is a separate column"],
        horizontal=True,
    )

    if layout == "Rows – there is a date column":
        st.markdown("### Map columns")
        date_col = st.selectbox("Date column", cols, 0)
        customer_col = st.selectbox("Customer column", cols, 1)
        sku_col = st.selectbox("SKU column", cols, 2)
        value_col = st.selectbox("Sales value column", cols, 3)
        data = raw[[date_col, customer_col, sku_col, value_col]].copy()
        data.columns = ["date", "customer", "sku", "value"]
    else:
        st.markdown("### Identify identifier columns (everything else are month columns)")
        id_cols = st.multiselect("Identifier columns", cols, default=[cols[0]])
        month_cols = [c for c in cols if c not in id_cols]
        month_cols = st.multiselect("Month columns", month_cols, default=month_cols)
        long = raw[id_cols + month_cols].melt(id_vars=id_cols, var_name="date", value_name="value")
        long["date"] = pd.to_datetime(long["date"], errors="coerce", infer_datetime_format=True)
        if long["date"].isna().any():
            st.error("Some month headers couldn’t be parsed. Rename columns like '2024-01'.")
            st.stop()
        st.markdown("### Map identifier columns")
        customer_col = st.selectbox("Customer column", id_cols, 0)
        sku_col = st.selectbox("SKU column", id_cols, 1 if len(id_cols) > 1 else 0)
        data = long[["date", customer_col, sku_col, "value"]].copy()
        data.columns = ["date", "customer", "sku", "value"]

    # Standardise types
    data["date"] = pd.to_datetime(data["date"])
    data["customer"] = data["customer"].astype(str)
    data["sku"] = data["sku"].astype(str)

    # 3️⃣ Filters
    st.markdown("### Choose customers / SKUs to forecast")
    cust_opts = sorted(data["customer"].unique())
    customers = st.multiselect("Customer(s)", cust_opts, default=cust_opts[:1])
    subset = data[data["customer"].isin(customers)] if customers else data.copy()

    sku_opts = sorted(subset["sku"].unique())
    skus = st.multiselect("SKU(s)", sku_opts, default=sku_opts[:1])
    filtered = subset[subset["sku"].isin(skus)] if skus else subset.copy()

    if filtered.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # 4️⃣ History chart
    monthly_hist = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum()
    hist_df = monthly_hist.reset_index()
    hist_df["date_str"] = hist_df["date"].dt.to_period("M").astype(str)
    st.altair_chart(
        alt.Chart(hist_df)
        .mark_line(point=True)
        .encode(x="date_str:N", y="value:Q", tooltip=["date_str", "value"])
        .properties(height=250),
        use_container_width=True,
    )

    # 5️⃣ Forecast settings
    st.sidebar.header("⚙️ Forecast settings")
    horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, 12)
    model_type = st.sidebar.selectbox(
        "Model",
        ["Moving Average", "Exponential Smoothing (no season)", "Holt-Winters Seasonal", "Prophet"],
    )

    ma_window = alpha = season_len = seasonality = trend = None
    if model_type == "Moving Average":
        ma_window = st.sidebar.slider("MA window", 2, 24, 3)
    elif model_type == "Exponential Smoothing (no season)":
        alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3)
    elif model_type == "Holt-Winters Seasonal":
        season_len = st.sidebar.slider("Season length", 3, 24, 12)
        seasonality = st.sidebar.selectbox("Seasonality", ["add", "mul"], 0)
        trend = st.sidebar.selectbox("Trend", ["add", "mul", None], 0)

    if model_type == "Prophet" and Prophet is None:
        st.sidebar.error("Prophet not installed. Add `prophet` to requirements.txt and redeploy.")

    # 6️⃣ Events
    st.sidebar.markdown("---")
    st.sidebar.subheader("Events / lifts")
    with st.sidebar.form("event_form"):
        e_date = st.date_input("Event month")
        e_lift = st.number_input("Lift %", value=10.0)
        add_e = st.form_submit_button("Add/update")
    if "events" not in st.session_state:
        st.session_state["events"] = {}
    if add_e:
        mth = (pd.to_datetime(e_date) + pd.offsets.MonthEnd(0)).normalize()
        st.session_state["events"][mth] = e_lift / 100
    if st.session_state["events"]:
        st.sidebar.write({d.strftime("%Y-%m"): f"+{int(r*100)}%" for d, r in st.session_state["events"].items()})

    # 7️⃣ Run forecast
    if st.button("🚀 Forecast"):
        ts = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum().fillna(0)

        # --- Fit model ---
        if model_type == "Moving Average":
            last_ma = ts.rolling(ma_window).mean().iloc[-1]
            idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
            forecast = pd.Series(last_ma, idx)
        elif model_type == "Exponential Smoothing (no season)":
            model = ExponentialSmoothing(ts, trend=None, seasonal=None, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, optimized=False)
            forecast = fit.forecast(horizon)
        elif model_type == "Holt-Winters Seasonal":
            if len(ts) < 2 * season_len:
                st.error("Need at least two seasons of data.")
                st.stop()
            model = ExponentialSmoothing(ts, trend=trend, seasonal=seasonality, seasonal_periods=season_len, initialization_method="estimated")
            forecast = model.fit().forecast(horizon)
        else:  # Prophet
            if Prophet is None:
                st.error("Prophet not installed")
                st.stop()
            dfp = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            m.fit(dfp)
            future = m.make_future_dataframe(horizon, freq="M")
            forecast = m.predict(future).set_index("ds")["yhat"].iloc[-horizon:]

        # --- Apply lifts ---
        for d, r in st.session_state["events"].items():
            if d in forecast.index:
                forecast.loc[d] *= 1 + r

        # --- Post‑process: clip negatives, round ---
        forecast = forecast.clip(lower=0).round(0)
        forecast.name = "forecast"
        forecast.index.name = "date"

        # ---- DISPLAY FORECAST ----
        disp_df = forecast.reset_index()
        disp_df["date_str"] = disp
