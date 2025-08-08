import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly (with valuation)")

# ------------------------------------------------------------------
# 1  Upload sales data
# ------------------------------------------------------------------
sales_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"], key="sales")
if sales_file is None:
    st.info("👆 Upload sales file to begin.")
    st.stop()

# Optional price list --------------------------------------------------
price_file = st.file_uploader("Upload price list (SKU → price) Excel / CSV", type=["xlsx", "csv"], key="prices")
price_map = {}
if price_file is not None:
    st.markdown("### Price‑list options")
    # Choose header row & sheet/CSV
    if price_file.name.lower().endswith(".csv"):
        header_row = st.number_input("Header row (1‑based)", 1, 50, 1, key="pl_header")
        price_raw = pd.read_csv(price_file, header=header_row - 1)
    else:
        xls_price = pd.ExcelFile(price_file)
        price_sheet = st.selectbox("Price sheet (tab)", xls_price.sheet_names, key="pl_sheet")
        header_row = st.number_input("Header row (1‑based)", 1, 50, 1, key="pl_header_xls")
        price_raw = xls_price.parse(price_sheet, header=header_row - 1)

    with st.expander("Map price‑list columns", expanded=False):
        price_cols = price_raw.columns.tolist()
        sku_price_col   = st.selectbox("SKU column", price_cols, 0, key="sku_price_col")
        value_price_col = st.selectbox("Price column", price_cols, 1, key="value_price_col")

    # Build price map
    price_df = price_raw[[sku_price_col, value_price_col]].copy()
    price_df.columns = ["sku", "price"]
    price_df["sku"] = price_df["sku"].astype(str)
    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")
    price_map = dict(price_df.values)

    st.success(f"Loaded {len(price_map)} SKU prices from price list.")

# ------------------------------------------------------------------
# 2  Load sales sheet
# ------------------------------------------------------------------
xls = pd.ExcelFile(sales_file)
sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
raw = xls.parse(sheet)
st.success(f"Sales: {raw.shape[0]} rows × {raw.shape[1]} cols")
cols = raw.columns.tolist()

# ------------------------------------------------------------------
# 3  Layout selection
# ------------------------------------------------------------------
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
    qty_col  = st.selectbox("Sales qty column", cols, 3)
    data = raw[[date_col, cust_col, sku_col, qty_col]].copy()
    data.columns = ["date", "customer", "sku", "qty"]
else:
    st.markdown("### Identify identifier columns")
    id_cols = st.multiselect("Identifier columns", cols, default=[cols[0]])
    month_cols = [c for c in cols if c not in id_cols]
    month_cols = st.multiselect("Month columns", month_cols, default=month_cols)
    long = raw[id_cols + month_cols].melt(id_vars=id_cols, var_name="date", value_name="qty")
    long["date"] = pd.to_datetime(long["date"], errors="coerce", infer_datetime_format=True)
    if long["date"].isna().any():
        st.error("Some month headers couldn’t be parsed."); st.stop()
    st.markdown("### Map identifier columns")
    cust_col = st.selectbox("Customer column", id_cols, 0)
    sku_col  = st.selectbox("SKU column", id_cols, 1 if len(id_cols) > 1 else 0)
    data = long[["date", cust_col, sku_col, "qty"]].copy()
    data.columns = ["date", "customer", "sku", "qty"]

# Coerce types
data["date"] = pd.to_datetime(data["date"])
data["customer"] = data["customer"].astype(str)
data["sku"] = data["sku"].astype(str)
# ensure qty numeric
data["qty"] = pd.to_numeric(data["qty"], errors="coerce").fillna(0)

# Attach price if available
if price_map:
    data["price"] = pd.to_numeric(data["sku"].map(price_map), errors="coerce").fillna(0.0)
    missing = (data["price"] == 0).sum()
    if missing:
        st.warning(f"{missing} rows missing price; revenue uses 0 for those.")
    data["rev"] = data["qty"] * data["price"]
else:
    data["price"] = 0.0
    data["rev"] = 0.0

# ------------------------------------------------------------------
# 4  Filters
# ------------------------------------------------------------------
st.markdown("### Choose customers / SKUs to forecast")
sel_cust = st.multiselect("Customer(s)", sorted(data["customer"].unique()))
subset = data[data["customer"].isin(sel_cust)] if sel_cust else data
sel_sku = st.multiselect("SKU(s)", sorted(subset["sku"].unique()))
filtered = subset[subset["sku"].isin(sel_sku)] if sel_sku else subset

if filtered.empty:
    st.warning("No data after filtering."); st.stop()

# ------------------------------------------------------------------
# 5  History chart (qty & revenue)
# ------------------------------------------------------------------
qty_hist = filtered.groupby("date")["qty"].sum().sort_index().resample("M").sum()
hist_df = qty_hist.reset_index()
hist_df["date_str"] = hist_df["date"].dt.to_period("M").astype(str)

qty_chart = (
    alt.Chart(hist_df)
    .mark_line(point=True)
    .encode(x="date_str:N", y="qty:Q", tooltip=["date_str", "qty"])
    .properties(height=250)
)
st.altair_chart(qty_chart, use_container_width=True)

# If prices provided, show revenue chart too
if price_map:
    rev_hist = filtered.groupby("date")["rev"].sum().sort_index().resample("M").sum()
    rev_df = rev_hist.reset_index()
    rev_df["date_str"] = rev_df["date"].dt.to_period("M").astype(str)
    st.altair_chart(
        alt.Chart(rev_df)
        .mark_line(point=True, color="#2ca02c")
        .encode(x="date_str:N", y="rev:Q", tooltip=["date_str", "rev"])
        .properties(height=250),
        use_container_width=True,
    )

# ------------------------------------------------------------------
# 6  Forecast settings
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Forecast settings")
horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 36, 12)
model = st.sidebar.selectbox("Model", [
    "Moving Average", "Exponential Smoothing (no season)", "Holt-Winters Seasonal", "Prophet"
])

ma_window = st.sidebar.slider("MA window", 2, 24, 3) if model == "Moving Average" else None
alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.3) if model == "Exponential Smoothing (no season)" else None
if model == "Holt-Winters Seasonal":
    season_len = st.sidebar.slider("Season length", 3, 24, 12)
    seasonality = st.sidebar.selectbox("Seasonality", ["add", "mul"], 0)
    trend = st.sidebar.selectbox("Trend", ["add", "mul", None], 0)
else:
    season_len = seasonality = trend = None

# 7  Events block kept as‑is (omitted here for brevity)
# ... (reuse existing events code) ...

# ------------------------------------------------------------------
# 8  Run forecast (qty) and revenue if price provided
# ------------------------------------------------------------------
if st.button("🚀 Forecast"):
    ts_qty = filtered.groupby("date")["qty"].sum().sort_index().resample("M").sum().fillna(0)

    # Fit model on qty series (same as before: MA / ETS / HW / Prophet)
    # ... (reuse existing logic but use ts_qty) ...
    # Result: forecast_qty Series

    # Valuation
    if price_map:
        if sel_sku:
            avg_price = filtered[filtered["sku"].isin(sel_sku)]["price"].mean()
        else:
            avg_price = filtered["price"].mean()
        forecast_rev = forecast_qty * avg_price
    # Display charts & tables similar to before for qty and revenue
    # ...

st.markdown("---\nMade with ❤️ & Streamlit – now supports price valuation ☑️")
