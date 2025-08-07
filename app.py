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

# ------------------------------------------------------------------
# 1  Upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.info("👆 Upload an Excel file to begin.")
    st.stop()

# ------------------------------------------------------------------
# 2  Load sheet
# ------------------------------------------------------------------
xls = pd.ExcelFile(uploaded_file)
sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
raw = xls.parse(sheet)
st.success(f"Loaded '{sheet}' → {raw.shape[0]} rows × {raw.shape[1]} cols")
cols = raw.columns.tolist()

# ------------------------------------------------------------------
# 3  Layout selection
# ------------------------------------------------------------------
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

# Ensure correct types
data["date"] = pd.to_datetime(data["date"])
data["customer"] = data["customer"].astype(str)
data["sku"] = data["sku"].astype(str)

# ------------------------------------------------------------------
# 4  Filters
# ------------------------------------------------------------------
st.markdown("### Choose customers / SKUs to forecast")
customer_opts = sorted(data["customer"].unique())
selected_customers = st.multiselect("Customer(s)", customer_opts, default=customer_opts[:1])
subset = data[data["customer"].isin(selected_customers)] if selected_customers else data.copy()

sku_opts = sorted(subset["sku"].unique())
selected_skus = st.multiselect("SKU(s)", sku_opts, default=sku_opts[:1])
filtered = subset[subset["sku"].isin(selected_skus)] if selected_skus else subset.copy()

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ------------------------------------------------------------------
# 5  History chart
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 6  Forecast settings
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Forecast settings")
horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, 12)
model_type = st.sidebar.selectbox(
    "Model",
    ["Moving Average", "Exponential Smoothing (no season)", "Holt-Winters Seasonal", "Prophet"],
)
ma_window = st.sidebar.slider("MA window", 2, 24, 3) if model_type == "Moving Average" else None
alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3) if model_type == "Exponential Smoothing (no season)" else None
if model_type == "Holt-Winters Seasonal":
    season_len = st.sidebar.slider("Season length", 3, 24, 12)
    seasonality = st.sidebar.selectbox("Seasonality", ["add", "mul"], 0)
    st.sidebar.caption(
        """**add** – seasonal fluctuations have a constant magnitude (e.g., +500).  
**mul** – seasonal fluctuations scale with the level (e.g., +15 %)."""
    )
    trend = st.sidebar.selectbox("Trend", ["add", "mul", None], 0)
    st.sidebar.caption(
        """**add** – linear trend (constant increment per period).  
**mul** – exponential trend (% growth/decay).  
**None** – no trend component."""
    )
else:
    season_len = seasonality = trend = None
    season_len = seasonality = trend = None

if model_type == "Prophet" and Prophet is None:
    st.sidebar.error("Prophet not installed. Add `prophet` to requirements.txt and redeploy.")

# ------------------------------------------------------------------
# 7  Events
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Events / lifts")
with st.sidebar.form("event_form"):
    e_date = st.date_input("Event month")
    e_name = st.text_input("Event name", placeholder="Promotion / Holiday ...")
    e_lift = st.number_input("Lift %", value=10.0)
    add_event = st.form_submit_button("Add / update")

if "events" not in st.session_state:
    st.session_state["events"] = {}

if add_event:
    mth = (pd.to_datetime(e_date) + pd.offsets.MonthEnd(0)).normalize()
    st.session_state["events"][mth] = {"name": e_name or mth.strftime("Event %Y-%m"), "lift": e_lift / 100}

# Display event list
if st.session_state["events"]:
    evt_df = pd.DataFrame([
        {"Month": d.strftime("%Y-%m"), "Event": v["name"], "Lift %": int(v["lift"]*100)}
        for d, v in sorted(st.session_state["events"].items())
    ])
    st.sidebar.table(evt_df)
# ------------------------------------------------------------------
# 8  Run forecast
# ------------------------------------------------------------------  Run forecast
# ------------------------------------------------------------------
if st.button("🚀 Forecast"):
    ts = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum().fillna(0)

    # Fit model
    if model_type == "Moving Average":
        last_ma = ts.rolling(ma_window).mean().iloc[-1]
        future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
        forecast = pd.Series(last_ma, index=future_idx)
    elif model_type == "Exponential Smoothing (no season)":
        model = ExponentialSmoothing(ts, trend=None, seasonal=None, initialization_method="estimated")
        forecast = model.fit(smoothing_level=alpha, optimized=False).forecast(horizon)
    elif model_type == "Holt-Winters Seasonal":
        if len(ts) < 2 * season_len:
            st.error("Need at least two seasons of data.")
            st.stop()
        model = ExponentialSmoothing(ts, trend=trend, seasonal=seasonality, seasonal_periods=season_len, initialization_method="estimated")
        forecast = model.fit().forecast(horizon)
    else:
        if Prophet is None:
            st.error("Prophet not installed")
            st.stop()
        dfp = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(dfp)
        future = m.make_future_dataframe(horizon, freq="M")
        forecast = m.predict(future).set_index("ds")["yhat"].iloc[-horizon:]

    # Apply lifts
    for d, r in st.session_state["events"].items():
        if d in forecast.index:
            forecast.loc[d] *= 1 + r

    forecast = forecast.clip(lower=0).round(0)
    forecast.name = "forecast"
    forecast.index.name = "date"

    # Display forecast chart
    disp_df = forecast.reset_index()
    disp_df["date_str"] = disp_df["date"].dt.to_period("M").astype(str)
    fc_chart = (
        alt.Chart(disp_df)
        .mark_line(point=True, color="#ff7f0e")
        .encode(
            x=alt.X("date_str:N", title="Month"),
            y=alt.Y("forecast:Q", title="Forecast"),
            tooltip=["date_str", "forecast"],
        )
        + alt.Chart(disp_df)
        .mark_text(dy=-12, color="#ff7f0e")
        .encode(x="date_str:N", y="forecast:Q", text="forecast:Q")
    ).properties(height=300)

    st.subheader("🔮 Monthly Forecast")
    st.altair_chart(fc_chart, use_container_width=True)

    # ------------------------------------------------------------------
    # Yearly recap table
    # ------------------------------------------------------------------
    combined = pd.concat([ts.rename("actual"), forecast]).sort_index()
    combined_df = combined.to_frame(name="value")
    combined_df["year"] = combined_df.index.year

    annual = combined_df.groupby("year")["value"].sum().round(0)
    recap = annual.to_frame(name="value").reset_index()
    recap["YOY_abs"] = recap["value"].diff().fillna(0).round(0)
    recap["YOY_pct"] = recap["value"].pct_change().mul(100).round(1).fillna(0)

    latest_month = ts.index.max().month
    ytd = (
        combined_df[combined_df.index.month <= latest_month]
        .groupby("year")["value"]
        .sum()
        .round(0)
    )
    recap = recap.merge(ytd.to_frame(name="YTD"), on="year", how="left")
    recap["YTD_vs_prev_abs"] = recap["YTD"].diff().fillna(0).round(0)
    recap["YTD_vs_prev_pct"] = recap["YTD"].pct_change().mul(100).round(1).fillna(0)

    st.subheader("📊 Yearly Recap")
    st.dataframe(
        recap.style.format(
            {
                "value": "{:,.0f}",
                "YOY_abs": "{:+,.0f}",
                "YOY_pct": "{:+.1f}%",
                "YTD": "{:,.0f}",
                "YTD_vs_prev_abs": "{:+,.0f}",
                "YTD_vs_prev_pct": "{:+.1f}%",
            }
        )
    )

    # ------------------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------------------
    csv_data = disp_df[["date", "forecast"]].to_csv(index=False).encode()
    st.download_button("Download forecast CSV", csv_data, "forecast.csv", "text/csv")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    """---
Made with ❤️ & Streamlit | Monthly forecasts, yearly recap, event lifts, non‑negative clipping."""
)
