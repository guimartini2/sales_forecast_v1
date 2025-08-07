import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional Prophet support (add “prophet” to requirements.txt)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly")

# 1 — Upload ------------------------------------------------------------
upl = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])
if upl is None:
    st.info("👆 Upload an Excel file to begin.")
    st.stop()

# 2 — Load sheet --------------------------------------------------------
xls = pd.ExcelFile(upl)
sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
raw = xls.parse(sheet)
st.success(f"Loaded '{sheet}' → {raw.shape[0]} rows × {raw.shape[1]} cols")
cols = raw.columns.tolist()

# 3 — Layout ------------------------------------------------------------
layout = st.radio(
    "How are your dates stored?",
    ["Rows – there is a date column", "Columns – each month is a separate column"],
    horizontal=True,
)

if layout == "Rows – there is a date column":
    st.markdown("### Map columns")
    date_col = st.selectbox("Date column", cols, 0)
    cust_col = st.selectbox("Customer column", cols, 1)
    sku_col  = st.selectbox("SKU column", cols, 2)
    val_col  = st.selectbox("Sales value column", cols, 3)
    data = raw[[date_col, cust_col, sku_col, val_col]].copy()
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
    cust_col = st.selectbox("Customer column", id_cols, 0)
    sku_col  = st.selectbox("SKU column", id_cols, 1 if len(id_cols) > 1 else 0)
    data = long[["date", cust_col, sku_col, "value"]].copy()
    data.columns = ["date", "customer", "sku", "value"]

# Coerce types
data["date"]     = pd.to_datetime(data["date"])
data["customer"] = data["customer"].astype(str)
data["sku"]      = data["sku"].astype(str)

# 4 — Filters -----------------------------------------------------------
st.markdown("### Choose customers / SKUs to forecast")
cust_opts = sorted(data["customer"].unique())
sel_cust  = st.multiselect("Customer(s)", cust_opts, default=cust_opts[:1])
subset    = data[data["customer"].isin(sel_cust)] if sel_cust else data

sku_opts  = sorted(subset["sku"].unique())
sel_sku   = st.multiselect("SKU(s)", sku_opts, default=sku_opts[:1])
filtered  = subset[subset["sku"].isin(sel_sku)] if sel_sku else subset

if filtered.empty:
    st.warning("No data for selected filters.")
    st.stop()

# 5 — History chart -----------------------------------------------------
monthly_hist = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum()
hist_df = monthly_hist.reset_index()
hist_df["date_str"] = hist_df["date"].dt.to_period("M").astype(str)

st.altair_chart(
    alt.Chart(hist_df).mark_line(point=True)
    .encode(x="date_str:N", y="value:Q", tooltip=["date_str", "value"])
    .properties(height=250),
    use_container_width=True,
)

# 6 — Forecast settings -------------------------------------------------
st.sidebar.header("⚙️ Forecast settings")
horizon   = st.sidebar.number_input("Forecast horizon (months)", 1, 36, 12)
model     = st.sidebar.selectbox("Model",
            ["Moving Average", "Exponential Smoothing (no season)", "Holt-Winters Seasonal", "Prophet"])

ma_window = st.sidebar.slider("MA window", 2, 24, 3) \
            if model == "Moving Average" else None
alpha     = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3) \
            if model == "Exponential Smoothing (no season)" else None

if model == "Holt-Winters Seasonal":
    season_len  = st.sidebar.slider("Season length", 3, 24, 12)
    seasonality = st.sidebar.selectbox("Seasonality", ["add", "mul"], 0)
    st.sidebar.caption("**add** constant swing (±500) **mul** % swing (±15 %).")
    trend       = st.sidebar.selectbox("Trend", ["add", "mul", None], 0)
    st.sidebar.caption("**add** linear  **mul** exponential  **None** no trend.")
else:
    season_len = seasonality = trend = None

if model == "Prophet" and Prophet is None:
    st.sidebar.error("Prophet not installed. Add `prophet` to requirements.txt and redeploy.")

# 7 — Events ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Events / lifts")
with st.sidebar.form("event_form"):
    e_date = st.date_input("Event month")
    e_lift = st.number_input("Lift %", value=10.0)
    if st.form_submit_button("Add / update"):
        st.session_state.setdefault("events", {})[
            (pd.to_datetime(e_date) + pd.offsets.MonthEnd(0)).normalize()
        ] = e_lift / 100

events = st.session_state.get("events", {})
if events:
    st.sidebar.write({d.strftime("%Y-%m"): f"+{int(r*100)}%" for d, r in events.items()})

# 8 — Run forecast ------------------------------------------------------
if st.button("🚀 Forecast"):
    ts = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum().fillna(0)

    if model == "Moving Average":
        fc_base = ts.rolling(ma_window).mean().iloc[-1]
        future_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
        forecast = pd.Series(fc_base, index=future_idx)

    elif model == "Exponential Smoothing (no season)":
        forecast = ExponentialSmoothing(ts, trend=None, seasonal=None, initialization_method="estimated") \
                   .fit(smoothing_level=alpha, optimized=False).forecast(horizon)

    elif model == "Holt-Winters Seasonal":
        if len(ts) < 2 * season_len:
            st.error("Need at least two seasons of data."); st.stop()
        forecast = ExponentialSmoothing(
            ts, trend=trend, seasonal=seasonality, seasonal_periods=season_len,
            initialization_method="estimated"
        ).fit().forecast(horizon)

    else:  # Prophet
        dfp = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(dfp)
        future = m.make_future_dataframe(horizon, freq="M")
        forecast = m.predict(future).set_index("ds")["yhat"].iloc[-horizon:]

    # Apply lifts
    for d, r in events.items():
        if d in forecast.index:
            forecast.loc[d] *= 1 + r

    forecast = forecast.clip(lower=0).round(0)
    forecast.name = "forecast"
    forecast.index.name = "date"

    # --- Chart ---------------------------------------------------------
    disp = forecast.reset_index()
    disp["date_str"] = disp["date"].dt.to_period("M").astype(str)

    st.subheader("🔮 Monthly Forecast")
    st.altair_chart(
        alt.Chart(disp).mark_line(point=True, color="#ff7f0e")
        .encode(x="date_str:N", y="forecast:Q", tooltip=["date_str", "forecast"])
        + alt.Chart(disp).mark_text(dy=-12, color="#ff7f0e")
        .encode(x="date_str:N", y="forecast:Q", text="forecast:Q"),
        use_container_width=True,
    )

    # --- Yearly recap --------------------------------------------------
    combined = pd.concat([ts.rename("actual"), forecast]).sort_index().to_frame("value")
    combined["year"] = combined.index.year

    annual = combined.groupby("year")["value"].sum().round(0)
    recap  = annual.to_frame("value").reset_index()
    recap["YOY_abs"] = recap["value"].diff().fillna(0).round(0)
    recap["YOY_pct"] = recap["value"].pct_change().mul(100).round(1).fillna(0)

    latest_month = ts.index.max().month
    ytd = (
        combined[combined.index.month <= latest_month]
        .groupby("year")["value"].sum().round(0)
    )
    recap = recap.merge(ytd.to_frame("YTD"), on="year", how="left")
    recap["YTD_vs_prev_abs"] = recap["YTD"].diff().fillna(0).round(0)
    recap["YTD_vs_prev_pct"] = recap["YTD"].pct_change().mul(100).round(1).fillna(0)

    st.subheader("📊 Yearly Recap")
    st.dataframe(
        recap.style.format(
            {"value": "{:,.0f}",
             "YOY_abs": "{:+,.0f}",
             "YOY_pct": "{:+.1f}%",
             "YTD": "{:,.0f}",
             "YTD_vs_prev_abs": "{:+,.0f}",
             "YTD_vs_prev_pct": "{:+.1f}%"}
        )
    )

    # --- Download ------------------------------------------------------
    st.download_button("Download forecast CSV",
                       disp[["date", "forecast"]].to_csv(index=False).encode(),
                       "forecast.csv",
                       "text/csv")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    "---\nMade with ❤️ & Streamlit | Monthly forecasts, yearly recap, event lifts, "
    "non-negative clipping."
)
