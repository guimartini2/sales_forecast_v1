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
    # clean price strings (remove commas, currency symbols)
    price_df["price"] = (
        price_df["price"].astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )
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
    data["price"] = pd.to_numeric(data["sku"].map(price_map), errors="coerce")
    valid_mask = data["price"].notna() & (data["price"] > 0)
    missing = (~valid_mask).sum()
    if missing:
        st.warning(f"{missing} rows without a valid price were excluded from valuation.")
    data = data[valid_mask].copy()
    data["rev"] = data["qty"] * data["price"]
else:
    st.warning("No price list supplied – revenue metrics disabled.")
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
# 5  Unit price summary
# ------------------------------------------------------------------
if price_map:
    unit_price = filtered["price"].mean()
    st.markdown(f"**Unit price used for valuation:** {unit_price:,.2f}")

# ------------------------------------------------------------------
# 6  History chart (qty & revenue)
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
# ------------------------------------------------------------------
# 8️⃣ Run forecast (qty) and revenue if price provided
# ------------------------------------------------------------------
if st.button("🚀 Forecast"):
    ts_qty = filtered.groupby("date")["qty"].sum().sort_index().resample("M").sum().fillna(0)

    # ----------------- fit model on qty -----------------
    if model == "Moving Average":
        base = ts_qty.rolling(ma_window).mean().iloc[-1]
        idx = pd.date_range(ts_qty.index[-1] + pd.offsets.MonthEnd(1), periods=horizon, freq="M")
        forecast_qty = pd.Series(base, index=idx)

    elif model == "Exponential Smoothing (no season)":
        forecast_qty = ExponentialSmoothing(
            ts_qty, trend=None, seasonal=None, initialization_method="estimated"
        ).fit(smoothing_level=alpha, optimized=False).forecast(horizon)

    elif model == "Holt-Winters Seasonal":
        if len(ts_qty) < 2 * season_len:
            st.error("Need at least two seasons of data."); st.stop()
        forecast_qty = ExponentialSmoothing(
            ts_qty,
            trend=trend,
            seasonal=seasonality,
            seasonal_periods=season_len,
            initialization_method="estimated",
        ).fit().forecast(horizon)

    else:  # Prophet
        if Prophet is None:
            st.error("Prophet not installed"); st.stop()
        dfp = ts_qty.reset_index().rename(columns={"date": "ds", "qty": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.fit(dfp)
        future = m.make_future_dataframe(horizon, freq="M")
        forecast_qty = m.predict(future).set_index("ds")["yhat"].iloc[-horizon:]

    forecast_qty = forecast_qty.clip(lower=0).round(0)

    # ----------------- revenue valuation -----------------
    if price_map and not forecast_qty.empty:
        if sel_sku:
            sel_prices = filtered[filtered["sku"].isin(sel_sku)]["price"]
        else:
            sel_prices = filtered["price"]
        avg_price = sel_prices.replace(0, pd.NA).dropna().mean()
        if pd.isna(avg_price):
            st.warning("No valid price to value forecast; using 0.")
            avg_price = 0.0
        forecast_rev = (forecast_qty * avg_price).round(0)
    else:
        forecast_rev = None

    # ----------------- display -----------------
    fc_df = forecast_qty.reset_index()
    fc_df.columns = ["date", "forecast_qty"]
    fc_df["date_str"] = fc_df["date"].dt.to_period("M").astype(str)

    qty_chart_fc = (
        alt.Chart(fc_df)
        .mark_line(point=True, color="#ff7f0e")
        .encode(x="date_str:N", y="forecast_qty:Q", tooltip=["date_str", "forecast_qty"])
        .properties(height=250)
    )
    st.subheader("🔮 Forecast – Quantity")
    st.altair_chart(qty_chart_fc, use_container_width=True)

    if forecast_rev is not None:
        rev_df = forecast_rev.reset_index()
        rev_df.columns = ["date", "forecast_rev"]
        rev_df["date_str"] = rev_df["date"].dt.to_period("M").astype(str)
        rev_chart = (
            alt.Chart(rev_df)
            .mark_line(point=True, color="#2ca02c")
            .encode(x="date_str:N", y="forecast_rev:Q", tooltip=["date_str", "forecast_rev"])
            .properties(height=250)
        )
        st.subheader("💰 Forecast – Revenue")
        st.altair_chart(rev_chart, use_container_width=True)

    # ----------------- annual table -----------------
    annual_qty = forecast_qty.groupby(forecast_qty.index.year).sum().rename("Units")
    if forecast_rev is not None:
        annual_rev = forecast_rev.groupby(forecast_rev.index.year).sum().rename("Revenue")
        annual_tbl = pd.concat([annual_qty, annual_rev], axis=1).round(0)
    else:
        annual_tbl = annual_qty.to_frame().round(0)

    st.subheader("📅 Annual Forecast Summary")
    fmt = {"Units": "{:,.0f}"}
    if "Revenue" in annual_tbl.columns:
        fmt["Revenue"] = "{:,.0f}$"
    st.dataframe(annual_tbl.style.format(fmt))

    # ----------------- download -----------------
    export_cols = ["date", "forecast_qty"]
    if forecast_rev is not None:
        fc_df = fc_df.merge(rev_df, on="date")
        export_cols.append("forecast_rev")
    st.download_button(
        "Download forecast CSV",
        fc_df[export_cols].to_csv(index=False).encode(),
        file_name="forecast.csv",
        mime="text/csv",
    )(
        "Download forecast CSV",
        fc_df[export_cols].to_csv(index=False).encode(),
        "forecast.csv",
        "text/csv",
    ).encode(),
        "forecast.csv",
        "text/csv",
    )

# ------------------------------------------------------------------
st.markdown("""---
Made with ❤️ & Streamlit – valuation ready ✅""")
