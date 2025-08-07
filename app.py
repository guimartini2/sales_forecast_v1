import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet is optional (add `prophet` to requirements.txt if desired)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Sales Forecasting Tool – Monthly")

# 1️⃣ Upload -------------------------------------------------------------
uploaded_file = st.file_uploader("Upload sales history Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # --- Load sheet ---
    xls = pd.ExcelFile(uploaded_file)
    sheet = st.selectbox("Worksheet (tab)", xls.sheet_names)
    raw = xls.parse(sheet)
    st.success(f"Loaded '{sheet}' → {raw.shape[0]}×{raw.shape[1]}")

    cols = raw.columns.tolist()

    # 2️⃣ Layout selection ------------------------------------------------
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

    # 3️⃣ Filters ---------------------------------------------------------
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

    # 4️⃣ History chart ---------------------------------------------------
    monthly_hist = filtered.groupby("date")["value"].sum().sort_index().resample("M").sum()
    hist_df = monthly_hist.reset_index()
    hist_df["date_str"] = hist_df["date"].dt.to_period("M").astype(str)
            st.subheader("🔮 Monthly Forecast")
        st.altair_chart(chart, use_container_width=True)

        # ---- Yearly Recap --------------------------------------------
        combined = pd.concat([ts.rename("actual"), forecast]).sort_index()
        combined_df = combined.to_frame(name="value")
        combined_df["year"] = combined_df.index.year
        annual = combined_df.groupby("year")["value"].sum().round(0)
        recap = annual.to_frame(name="value").reset_index()
        recap["YOY_abs"] = recap["value"].diff().fillna(0).round(0)
        recap["YOY_pct"] = recap["value"].pct_change().mul(100).round(1).fillna(0)
        latest_month = ts.index.max().month
        ytd = combined_df[combined_df.index.month <= latest_month].groupby("year")["value"].sum().round(0)
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

        # ---- Download CSV -------------------------------------------
        csv = disp_df[["date", "forecast"]].to_csv(index=False).encode()
        st.download_button("Download forecast CSV", csv, "forecast.csv", "text/csv")

# Footer ---------------------------------------------------------------
st.markdown("---
Made with ❤️ & Streamlit | Monthly forecasts, recap table, non‑negative clipping, events & more.")
