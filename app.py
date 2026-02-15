import io
import json
import re
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


DATA_DIR = Path("data")
CONFIG_PATH = Path("etl_config.json")


def parse_maturity_to_months(maturity: str) -> float:
    if maturity is None:
        return np.nan
    m = str(maturity).strip().upper()
    if not m:
        return np.nan

    m1 = re.fullmatch(r"(\d+)\s*([MY])", m)
    m2 = re.fullmatch(r"([MY])\s*(\d+)", m)

    if m1:
        value, unit = int(m1.group(1)), m1.group(2)
    elif m2:
        unit, value = m2.group(1), int(m2.group(2))
    else:
        return np.nan

    return float(value * 12 if unit == "Y" else value)


def setup_page():
    st.set_page_config(page_title="Market Data Visualization", layout="wide")
    st.title("Data Visualization")
    st.caption(
        "Line charts, regime overlays, term structure, heatmaps, and yield curves "
        "from local ETL data."
    )


@st.cache_data(show_spinner=False)
def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


st.write("CWD:", Path.cwd())
st.write("CONFIG exists:", CONFIG_PATH.exists(), "->", str(CONFIG_PATH.resolve()))
st.write("DATA_DIR exists:", DATA_DIR.exists(), "->", str(DATA_DIR.resolve()))
st.write("processed exists:", (DATA_DIR/"processed").exists())

if (DATA_DIR/"processed").exists():
    st.write("files in data/processed:",
             sorted([p.name for p in (DATA_DIR/"processed").glob("*")])[:200])

cfg = load_config(CONFIG_PATH)
ids = [s.get("id") for s in cfg.get("series", [])]
st.write("series ids in config (first 50):", ids[:50])

missing = []
for sid in ids:
    p = DATA_DIR/"processed"/f"{sid}.parquet"
    if not p.exists():
        missing.append(str(p))
st.write("missing parquet paths (first 50):", missing[:50])

@st.cache_data(show_spinner=False)
def load_data(config_path: Path, data_dir: Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    frames = []

    for series in cfg.get("series", []):
        sid = series["id"]
        path = data_dir / "processed" / f"{sid}.parquet"
        if not path.exists():
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        if "return" not in df.columns or df["return"].isna().all():
            df["return"] = np.log(df["price"] / df["price"].shift(1))

        df["series_id"] = sid
        df["series_name"] = series.get("name", sid)
        df["asset_class"] = series.get("asset_class", "unknown")
        df["ticker"] = series.get("ticker", "")
        df["curve_group"] = series.get("curve_group")
        df["maturity"] = series.get("maturity")
        df["maturity_months"] = parse_maturity_to_months(series.get("maturity"))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["series_id", "date"]).reset_index(drop=True)
    out["cum_perf"] = (
        out["return"].fillna(0.0).groupby(out["series_id"]).cumsum()
        * 100.0
    )
    out["daily_return_pct"] = out["return"] * 100.0
    return out


def build_regimes(df: pd.DataFrame, benchmark_series_id: str) -> pd.DataFrame:
    bench = df[df["series_id"] == benchmark_series_id][["date", "return"]].dropna().copy()
    bench = bench.sort_values("date")
    if bench.empty:
        return pd.DataFrame(columns=["date", "regime"])

    bench["trend_63d"] = bench["return"].rolling(63).sum()
    q1, q2 = bench["trend_63d"].quantile(0.33), bench["trend_63d"].quantile(0.66)
    if pd.isna(q1) or pd.isna(q2):
        bench["regime"] = "Neutral"
    else:
        bench["regime"] = np.select(
            [bench["trend_63d"] <= q1, bench["trend_63d"] >= q2],
            ["Bear", "Bull"],
            default="Neutral",
        )
    return bench[["date", "regime"]]


def build_regime_intervals(regimes: pd.DataFrame) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame(columns=["start", "end", "regime"])

    regimes = regimes.sort_values("date").drop_duplicates("date")
    intervals = []
    start = regimes.iloc[0]["date"]
    current = regimes.iloc[0]["regime"]

    for i in range(1, len(regimes)):
        row = regimes.iloc[i]
        if row["regime"] != current:
            intervals.append({"start": start, "end": row["date"], "regime": current})
            start = row["date"]
            current = row["regime"]

    intervals.append(
        {
            "start": start,
            "end": regimes.iloc[-1]["date"] + pd.Timedelta(days=1),
            "regime": current,
        }
    )
    return pd.DataFrame(intervals)


def select_snapshot_per_series(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    snapshot = (
        df[df["date"] <= date]
        .sort_values(["series_id", "date"])
        .groupby("series_id", as_index=False)
        .tail(1)
    )
    return snapshot


def prepare_curve_values(snapshot: pd.DataFrame, curve_mode: str) -> tuple[pd.DataFrame, str]:
    s = snapshot.copy()
    unit = "native units"

    if curve_mode == "Price":
        s["value"] = s["price"]
        unit = "price"
    elif curve_mode == "Yield proxy (%)":
        s["value"] = np.where(s["asset_class"] == "fixed_income", 100.0 - s["price"], np.nan)
        unit = "%"
    elif curve_mode == "Spread vs shortest maturity":
        base_idx = s["maturity_months"].idxmin() if s["maturity_months"].notna().any() else s.index[0]
        base_value = s.loc[base_idx, "price"]
        s["value"] = s["price"] - base_value
        unit = "spread"
    else:
        s["value"] = s["price"]

    s = s.dropna(subset=["value"]).sort_values(["maturity_months", "maturity", "series_name"])
    return s, unit


def infer_region(curve_group: str, series_id: str, name: str) -> str | None:
    text = " ".join([str(curve_group or ""), str(series_id or ""), str(name or "")]).upper()
    if "US" in text or "TREAS" in text:
        return "US"
    if "DE" in text or "GERM" in text or "BUND" in text:
        return "Germany"
    return None


def compute_metric(returns: pd.Series, metric_name: str) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan

    if metric_name == "Annualized Return (%)":
        return (np.exp(r.sum() * 252.0 / len(r)) - 1.0) * 100.0

    if metric_name == "Annualized Volatility (%)":
        return r.std(ddof=0) * np.sqrt(252.0) * 100.0

    if metric_name == "Sharpe Ratio":
        vol = r.std(ddof=0)
        return np.nan if vol == 0 else (r.mean() / vol) * np.sqrt(252.0)

    if metric_name == "Max Drawdown (%)":
        wealth = np.exp(r.cumsum())
        drawdown = wealth / wealth.cummax() - 1.0
        return drawdown.min() * 100.0

    if metric_name == "Hit Ratio (%)":
        return (r > 0).mean() * 100.0

    return np.nan


def build_performance_export_png(
    df: pd.DataFrame,
    metric_col: str,
    y_title: str,
    intervals: pd.DataFrame | None,
) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)

    regime_colors = {"Bear": "#fca5a5", "Neutral": "#fde68a", "Bull": "#86efac"}
    if intervals is not None and not intervals.empty:
        for _, row in intervals.iterrows():
            color = regime_colors.get(row["regime"], "#e5e7eb")
            ax.axvspan(row["start"], row["end"], color=color, alpha=0.15)

    for series_name, grp in df.groupby("series_name"):
        ax.plot(grp["date"], grp[metric_col], linewidth=1.8, label=series_name)

    ax.set_title("Performance Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def section_line_charts(df: pd.DataFrame):
    st.subheader("Line Charts")
    st.caption("Performance time series with optional regime overlay.")

    available = (
        df[["series_id", "series_name"]]
        .drop_duplicates()
        .sort_values("series_name")
        .to_records(index=False)
        .tolist()
    )
    name_to_id = {name: sid for sid, name in available}
    default_series = [name for _, name in available[:4]]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected_names = st.multiselect(
            "Series",
            options=[x[1] for x in available],
            default=default_series,
        )
    with c2:
        metric_label = st.selectbox(
            "Metric",
            options=[
                "Cumulative log return (%)",
                "Price",
                "Daily log return (%)",
            ],
            index=0,
        )
    with c3:
        overlay_regimes = st.checkbox("Overlay regimes", value=True)

    if not selected_names:
        st.warning("Select at least one series.")
        return

    d1, d2 = st.columns(2)
    with d1:
        date_start = st.date_input(
            "Start date",
            value=df["date"].min().date(),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="line_date_start",
        )
    with d2:
        date_end = st.date_input(
            "End date",
            value=df["date"].max().date(),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="line_date_end",
        )

    df = df[(df["date"] >= pd.Timestamp(date_start)) & (df["date"] <= pd.Timestamp(date_end))].copy()
    df["cum_perf"] = (
        df["return"].fillna(0.0).groupby(df["series_id"]).cumsum() * 100.0
    )

    metric_map = {
        "Cumulative log return (%)": ("cum_perf", "Cumulative log return (%)"),
        "Price": ("price", "Price (native units)"),
        "Daily log return (%)": ("daily_return_pct", "Daily log return (%)"),
    }
    metric_col, y_title = metric_map[metric_label]

    selected_ids = [name_to_id[n] for n in selected_names]
    chart_df = (
        df[df["series_id"].isin(selected_ids)][["date", "series_name", metric_col]]
        .dropna()
        .copy()
    )
    chart_df = chart_df.rename(columns={metric_col: "value"})

    interval_df = None
    if overlay_regimes:
        benchmark = st.selectbox(
            "Regime benchmark",
            options=selected_names,
            index=0,
            help="Regime is computed from the 63-day trend of this series.",
        )
        regimes = build_regimes(df, name_to_id[benchmark])
        interval_df = build_regime_intervals(regimes)

    line = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(labelAngle=-45, format="%b %Y")),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("series_name:N", title="Series"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series_name:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title, format=".4f"),
            ],
        )
        .properties(
            height=420,
            title={
                "text": "Performance Time Series",
                "subtitle": "Lines per series with tooltips, axes and units",
            },
        )
    )

    if interval_df is not None and not interval_df.empty:
        regime_colors = alt.Scale(
            domain=["Bear", "Neutral", "Bull"],
            range=["#ef4444", "#f59e0b", "#22c55e"],
        )
        bands = alt.Chart(interval_df).mark_rect(opacity=0.12).encode(
            x="start:T",
            x2="end:T",
            color=alt.Color("regime:N", title="Regime", scale=regime_colors),
            tooltip=[
                alt.Tooltip("regime:N", title="Regime"),
                alt.Tooltip("start:T", title="Start"),
                alt.Tooltip("end:T", title="End"),
            ],
        )
        chart = alt.layer(bands, line).resolve_scale(color="independent")
    else:
        chart = line

    st.altair_chart(chart, width="stretch")

    png_data = build_performance_export_png(
        chart_df.rename(columns={"value": metric_col}),
        metric_col=metric_col,
        y_title=y_title,
        intervals=interval_df,
    )
    st.download_button(
        "Export PNG (line chart)",
        data=png_data,
        file_name="line_chart.png",
        mime="image/png",
    )


def section_term_structure(df: pd.DataFrame):
    st.subheader("Term Structure")
    st.caption(
        "Curve by maturity (x=maturity, y=price/yield/spread) at selected date."
    )

    curves = df[df["curve_group"].notna() & df["maturity"].notna()].copy()
    if curves.empty:
        st.info("No curve/maturity data found in configuration.")
        return

    counts = (
        curves.groupby("curve_group")["maturity"]
        .nunique()
        .sort_values(ascending=False)
        .rename("maturity_count")
        .reset_index()
    )
    eligible = counts[counts["maturity_count"] >= 2]["curve_group"].tolist()

    if not eligible:
        st.info(
            "Term structure not plottable: need at least 2 maturities "
            "in the same `curve_group`."
        )
        st.dataframe(counts, width="stretch")
        return

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        curve_group = st.selectbox("Curve group", options=eligible)
    with c2:
        curve_mode = st.selectbox(
            "Y Variable",
            options=["Price", "Yield proxy (%)", "Spread vs shortest maturity"],
        )
    with c3:
        date_pick = st.date_input(
            "Selected date",
            value=curves["date"].max().date(),
            min_value=curves["date"].min().date(),
            max_value=curves["date"].max().date(),
        )

    group_df = curves[curves["curve_group"] == curve_group].copy()
    snap = select_snapshot_per_series(group_df, pd.Timestamp(date_pick))
    snap, unit = prepare_curve_values(snap, curve_mode)

    if snap.empty:
        st.warning("No curve points available for this date/variable.")
        return

    snap["maturity_label"] = snap["maturity"].astype(str)
    snap["snapshot_date"] = snap["date"]

    chart = (
        alt.Chart(snap)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("maturity_label:O", title="Maturity"),
            y=alt.Y("value:Q", title=f"{curve_mode} ({unit})"),
            tooltip=[
                alt.Tooltip("series_name:N", title="Series"),
                alt.Tooltip("snapshot_date:T", title="Date"),
                alt.Tooltip("maturity_label:N", title="Maturity"),
                alt.Tooltip("value:Q", title=f"{curve_mode}", format=".4f"),
            ],
        )
        .properties(
            height=360,
            title={
                "text": f"Term Structure: {curve_group}",
                "subtitle": "Snapshot at selected date",
            },
        )
    )
    st.altair_chart(chart, width="stretch")


def section_heatmap(df: pd.DataFrame):
    st.subheader("Heatmap: Asset Classes x Regimes")
    st.caption("Performance metrics by asset class and regime.")

    available = (
        df[["series_id", "series_name"]]
        .drop_duplicates()
        .sort_values("series_name")
        .to_records(index=False)
        .tolist()
    )
    name_to_id = {name: sid for sid, name in available}

    c1, c2 = st.columns([1, 1.2])
    with c1:
        benchmark = st.selectbox(
            "Regime benchmark (heatmap)",
            options=[x[1] for x in available],
            index=0,
            key="heatmap_benchmark",
        )
    with c2:
        metric = st.selectbox(
            "Heatmap metric",
            options=[
                "Annualized Return (%)",
                "Annualized Volatility (%)",
                "Sharpe Ratio",
                "Max Drawdown (%)",
                "Hit Ratio (%)",
            ],
            index=0,
        )

    regimes = build_regimes(df, name_to_id[benchmark])
    if regimes.empty:
        st.warning("Unable to compute regimes for the heatmap.")
        return

    merged = df.merge(regimes, on="date", how="inner")
    if merged.empty:
        st.warning("No common dates between series and regimes.")
        return

    rows = []
    grouped = merged.groupby(["asset_class", "series_id", "regime"])
    for (asset_class, _, regime), g in grouped:
        value = compute_metric(g["return"], metric)
        rows.append({"asset_class": asset_class, "regime": regime, "value": value})
    metric_df = pd.DataFrame(rows).dropna()
    if metric_df.empty:
        st.warning("Not enough data to compute the heatmap.")
        return

    heatmap_df = (
        metric_df.groupby(["asset_class", "regime"], as_index=False)["value"]
        .mean()
        .copy()
    )
    heatmap_df["regime"] = pd.Categorical(
        heatmap_df["regime"], categories=["Bear", "Neutral", "Bull"], ordered=True
    )
    heatmap_df = heatmap_df.sort_values(["asset_class", "regime"])

    if metric in {"Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)"}:
        vmax = np.nanmax(np.abs(heatmap_df["value"].values))
        color = alt.Color(
            "value:Q",
            title=metric,
            scale=alt.Scale(scheme="redyellowgreen", domain=[-vmax, vmax]),
        )
    else:
        color = alt.Color("value:Q", title=metric, scale=alt.Scale(scheme="teals"))

    chart = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("regime:N", title="Regime"),
            y=alt.Y("asset_class:N", title="Asset Class"),
            color=color,
            tooltip=[
                alt.Tooltip("asset_class:N", title="Asset Class"),
                alt.Tooltip("regime:N", title="Regime"),
                alt.Tooltip("value:Q", title=metric, format=".4f"),
            ],
        )
        .properties(
            height=320,
            title={
                "text": f"Heatmap: {metric}",
                "subtitle": "Aggregated by asset class and regime",
            },
        )
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Consistent color scale across the heatmap; units match the selected metric.")


def section_yield_curves(df: pd.DataFrame):
    st.subheader("Yield Curves: US vs Germany")
    st.caption("Yield curves on matching maturities where possible, mapped to months.")

    fixed = df[df["asset_class"] == "fixed_income"].copy()
    if fixed.empty:
        st.info("No fixed_income series available.")
        return

    fixed = fixed[fixed["maturity"].notna()].copy()
    fixed["region"] = fixed.apply(
        lambda r: infer_region(r["curve_group"], r["series_id"], r["series_name"]),
        axis=1,
    )
    fixed = fixed[fixed["region"].isin(["US", "Germany"])]

    if fixed.empty:
        st.info(
            "US/Germany yield curves not available (regions not detected in config)."
        )
        return

    date_pick = st.date_input(
        "Yield curves date",
        value=fixed["date"].max().date(),
        min_value=fixed["date"].min().date(),
        max_value=fixed["date"].max().date(),
        key="yield_date",
    )

    snap = select_snapshot_per_series(fixed, pd.Timestamp(date_pick))
    if snap.empty:
        st.warning("No curve points available at this date.")
        return

    snap["yield_proxy"] = 100.0 - snap["price"]
    snap["maturity_label"] = snap["maturity"].astype(str)

    counts = snap.groupby("region")["series_id"].nunique().to_dict()
    if counts.get("US", 0) < 2 or counts.get("Germany", 0) < 2:
        st.info(
            "Not enough data to plot full US & Germany curves "
            "(need at least 2 maturities per region)."
        )
        st.dataframe(
            snap[["region", "series_name", "maturity_label", "maturity_months", "yield_proxy", "date"]]
            .sort_values(["region", "maturity_months"]),
            width="stretch",
        )
        return

    chart = (
        alt.Chart(snap.sort_values(["region", "maturity_months"]))
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("maturity_months:Q", title="Maturity (months)"),
            y=alt.Y("yield_proxy:Q", title="Yield proxy (%)"),
            color=alt.Color("region:N", title="Region"),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("series_name:N", title="Series"),
                alt.Tooltip("maturity_label:N", title="Maturity"),
                alt.Tooltip("yield_proxy:Q", title="Yield proxy (%)", format=".4f"),
                alt.Tooltip("date:T", title="Date"),
            ],
        )
        .properties(
            height=360,
            title={
                "text": "US vs Germany Yield Curves",
                "subtitle": "Maturity axis mapped to months",
            },
        )
    )
    st.altair_chart(chart, width="stretch")


def main():
    setup_page()

    if not CONFIG_PATH.exists():
        st.error(f"Config not found: {CONFIG_PATH}")
        return

    df = load_data(CONFIG_PATH, DATA_DIR)
    if df.empty:
        st.error("No data loaded from data/processed.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Line Charts",
            "Term Structure",
            "Heatmaps",
            "Yield Curves",
        ]
    )

    with tab1:
        section_line_charts(df)
    with tab2:
        section_term_structure(df)
    with tab3:
        section_heatmap(df)
    with tab4:
        section_yield_curves(df)


if __name__ == "__main__":
    main()
