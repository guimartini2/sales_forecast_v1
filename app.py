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

# 1️⃣ Upload ------------------------------------------------------------
file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])
if file is None:
    st.info("👆 Upload an Excel file to begin.")
    st.stop()

# 2️⃣ Load sheet --------------------------------------------------------
xls = pd.ExcelFile(file)
sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
raw = xls.parse(sheet)
st.success(f"Loaded '{sheet}' → {raw.shape[0]} rows × {raw.shape[1]} cols")
cols = raw.columns.tolist()

# 3️⃣ Layout selection --------------------------------------------------
layout = st.radio(
    "How are your dates stored?",
    ["Rows – there is a date column", "Columns – months across columns"],
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
    st.markdown("### Identify identifier columns")
    id_cols = st.multiselect("Identifier columns", cols, default=[cols[0]])
    month_cols = [c for c in cols if c not in id_cols]
    month_cols = st.multiselect("Month columns", month_cols, default=month_cols)
    long = raw[id_cols + month_cols].melt(id_vars=id_cols, var_name="date", value_name="value")
    long["date"] = pd.to_datetime(long["date"], errors="coerce", infer_datetime_format=True)
    if long["date"].isna().any():
        st.error("Some month headers couldn’t be parsed. Rename like '2024-01'.")
        st.stop()
    st.markdown("### Map identifier columns")
    cust_col = st.selectbox("Customer column", id_cols, 0)
    sku_col  = st.selectbox("SKU column", id_cols, 1 if len(id_cols) > 1 else 0)
    data = long[["date", cust_col, sku_col, "value"]].copy()
    data.columns = ["date", "customer", "sku", "value"]

# Coerce types
data["date"] = pd.to_datetime(data["date"])
data["customer"] = data["customer"].astype(str)
data["sku"] = data["sku"].astype(str)

# 4️⃣ Filters -----------------------------------------------------------
st.markdown("### Choose customers / SKUs to forecast")
sel_cust = st.multiselect("Customer(s)", sorted(data["customer"].unique()), default=None)
subset = data[data["customer"].isin(sel_cust)] if sel_cust else data
sel_sku = st.multiselect("SKU(s)", sorted(subset["sku"].unique()), default=None)
filtered = subset[subset["sku"].isin(sel_sku)] if sel_sku else subset

if filtered.empty:
    st.warning("No data after filtering.")
    st.stop()

# 5️⃣ History chart -----------------------------------------------------
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

# 6️⃣ Forecast settings -------------------------------------------------
st.sidebar.header("⚙️ Forecast settings")
horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, 12)
model = st.sidebar.selectbox("Model", [
    "Moving Average", "Exponential Smoothing (no season)", "Holt‑Winters Seasonal", "Prophet"
])

ma_window = st.sidebar.slider("MA window", 2, 24, 3) if model == "Moving Average" else None
alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3) if model == "Exponential Smoothing (no season)" else None
if model == "Holt‑Winters Seasonal":
    season_len = st.sidebar.slider("Season length", 3, 24, 12)
    seasonality = st.sidebar.selectbox("Seasonality", ["add", "mul"], 0)
    st.sidebar.caption("**add** – constant swing; **mul** – % swing.")
    trend = st.sidebar.selectbox("Trend", ["add", "mul", None], 0)
    st.sidebar.caption("**add** linear; **mul** exponential; **None** flat.")
else:
    season_len = seasonality = trend = None

if model == "Prophet" and Prophet is None:
    st.sidebar.error("Prophet not installed; add `prophet` lib in requirements.txt.")

# 7️⃣ Events ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Events / Lifts")
with st.sidebar.form("event_form"):
    e_date = st.date_input("Event month")
    e_name = st.text_input("Event name", placeholder="Promotion / Holiday …")
    e_lift = st.number_input("Lift %", value=10.0)
    add_evt = st.form_submit_button("Add / update")

if "events" not in st.session_state:
    st.session_state["events"] = {}

# Normalise older float events
for k, v in list(st.session_state["events"].items()):
    if isinstance(v, (int, float)):
        st.session_state["events"][k] = {"name": k.strftime("Event %Y-%m"), "lift": float(v)}

events = st.session_state["events"]

if add_evt:
    month_key = (pd.to_datetime(e_date) + pd.offsets.MonthEnd(0)).normalize()
    events[month_key] = {"name": e_name or month_key.strftime("Event %Y-%m"), "lift": e_lift / 100}

if events:
    evt_df = pd.DataFrame([
        {"Month": d.strftime("%Y-%m"), "Event": v["name"], "Lift %": int(v["lift"] * 100)}
        for d, v in sorted(events.items())
    ])
    st.sidebar.table(evt_df)

# 8️⃣ Run forecast ------------------------------------------------------
if st.button("🚀 Forecast"):
    ts = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum().fillna(0)

    if model == "Moving Average":
        base = ts.rolling(ma_window).mean().iloc[-1]
        idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
        forecast = pd.Series(base, index=idx)
    elif model == "Exponential Smoothing (no season)":
        forecast = ExponentialSmoothing(ts, trend=None, seasonal=None, initialization_method="estimated") \
                   .fit(smoothing_level=alpha, optimized=False).forecast(horizon)
    elif model == "Holt‑Winters Seasonal":
        if len(ts) < 2 * season_len:
            st.error("Need at least two seasons of data."); st.stop()
        forecast = ExponentialSmoothing(
            ts, trend=trend, seasonal=seasonality, seasonal_periods=season_len,
            initialization_method="estimated"
        ).fit().forecast(horizon)
    else:
        if Prophet is None:
            st.error("Prophet not installed"); st.stop()
        dfp = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period
