"""
MIBEL Spread Surveillance — Streamlit dashboard.

Single-page app that lets the user explore the system:
  - headline KPIs (recall, pinball, GW p-value, alerts/mo)
  - spread time series with cascade alerts overlaid
  - distribution of alert levels per month
  - documented event panel + system detection lag
  - model comparison table

Run locally:
    streamlit run app/dashboard.py

Deploy to Streamlit Cloud:
    Connect the GitHub repo, point at app/dashboard.py.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231.parquet"
ALERTS = BASE / "results" / "alerts_registry_v3.csv"
EVENTS = BASE / "data" / "events_panel.csv"
EVENT_VALID = BASE / "data" / "event_validation.csv"
BENCH = BASE / "results" / "benchmark_comparison.csv"
ABLATION = BASE / "results" / "ablation_lag_components.csv"
CROSS_YEAR = BASE / "results" / "cross_year_validation.csv"
PERM_IMP = BASE / "results" / "permutation_importance_v3.txt"
GW_REAL = BASE / "results" / "gw_pinball_real_data.csv"
LQRA = BASE / "results" / "lqra_results.csv"

PALETTE = {
    "VERDE":   "#27AE60",
    "ÁMBAR":   "#F1C40F",
    "NARANJA": "#E67E22",
    "ROJA":    "#C0392B",
}
LABEL_EN = {
    "VERDE":   "Green",
    "ÁMBAR":   "Amber",
    "NARANJA": "Orange",
    "ROJA":    "Red",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_alerts():
    df = pd.read_csv(ALERTS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_events():
    df = pd.read_csv(EVENTS, encoding="utf-8")
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["date_end"] = pd.to_datetime(df["date_end"])
    return df


@st.cache_data(show_spinner=False)
def load_event_valid():
    if EVENT_VALID.exists():
        df = pd.read_csv(EVENT_VALID)
        df["date_start"] = pd.to_datetime(df["date_start"])
        df["date_end"] = pd.to_datetime(df["date_end"])
        return df
    return None


@st.cache_data(show_spinner=False)
def load_csv_safe(path):
    if path.exists():
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MIBEL Spread Surveillance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MIBEL Spread Surveillance")
st.caption(
    "Residual-based anomaly detection on the Spain–France day-ahead "
    "electricity spread. Open data, 2019–2024."
)
st.markdown(
    "**Architecture**: XGBoost fundamentals model + Green/Amber/Orange/Red "
    "alert cascade on residuals. Companion quantile model for tail prediction. "
    "Methodology and results in `docs/Memoria_MIBEL_v2.pdf`; code in "
    "`scripts/`."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_dataset()
alerts = load_alerts()
events = load_events()
event_valid = load_event_valid()
bench = load_csv_safe(BENCH)
cross_year = load_csv_safe(CROSS_YEAR)
gw_real = load_csv_safe(GW_REAL)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", [2022, 2023, 2024], index=2)
alert_levels = st.sidebar.multiselect(
    "Show alert levels",
    ["Green", "Amber", "Orange", "Red"],
    default=["Amber", "Orange", "Red"],
)
show_naive = st.sidebar.checkbox(
    "Overlay naive |spread|>p99 benchmark", value=False
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Repository:** "
    "[github.com/cmvdata/mibel-congestion-monitor]"
    "(https://github.com/cmvdata/mibel-congestion-monitor)"
)
st.sidebar.markdown(
    "**Working paper:** [docs/Memoria_MIBEL_v2.pdf]"
    "(https://github.com/cmvdata/mibel-congestion-monitor/blob/main/"
    "docs/Memoria_MIBEL_v2.pdf)"
)

# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------
st.subheader("Headline results")
k1, k2, k3, k4, k5 = st.columns(5)

# Events in panel + in-window
n_total_events = len(events)
n_in_window = len(events[
    (events["date_end"] >= "2022-01-01") &
    (events["date_start"] <= "2024-12-31")
])
k1.metric("Events documented", n_total_events,
          help="Pre-specified panel of historical ES-FR congestion events")
k1.caption(f"{n_in_window} fall in alert window 2022-2024")

# Recall + lag from event_validation
if event_valid is not None:
    iw = event_valid[~event_valid["out_of_alert_window"]]
    recall_sys = iw["system_detected"].mean() if len(iw) else np.nan
    naive_recall = iw["naive_detected"].mean() if len(iw) else np.nan
    median_lag = iw.loc[iw["system_detected"], "system_lag_hours"].median()
    k2.metric("Recall @ event (system)",
              f"{recall_sys:.3f}",
              help="Fraction of in-window events detected by the cascade")
    k2.caption(f"naive régime p99: {naive_recall:.3f}")
    k3.metric("Median detection lag", f"{median_lag:.0f} h",
              help="Time between event start and first system alert")
else:
    k2.metric("Recall @ event", "n/a")
    k3.metric("Median lag", "n/a")

# GW pinball
if gw_real is not None and len(gw_real) > 0:
    ar1_row = gw_real[gw_real["benchmark"] == "AR(1)"]
    if len(ar1_row):
        pv = ar1_row["gw_pinball_pvalue"].iloc[0]
        k4.metric("GW pinball q=0.95 vs AR(1)",
                  f"p = {pv:.4f}",
                  help="Model A v3 mean-error against AR(1). "
                       "Mean-error model does not beat AR(1) on this "
                       "tail metric; v4-quantile companion does.")
else:
    k4.metric("GW pinball v3 vs AR(1)", "n/a")

# v4-quantile pinball headline (hardcoded from results)
k5.metric("v4-quantile pinball q=0.95", "0.232",
          help="vs AR(1) = 0.330; GW p<10⁻³ in favor of v4-quantile "
               "on test 2024")
k5.caption("v3-mean: 0.454")

st.divider()

# ---------------------------------------------------------------------------
# Time series: spread + alerts
# ---------------------------------------------------------------------------
st.subheader(f"Spread and cascade alerts — {year}")

sub = df[df["timestamp"].dt.year == year].copy()
sub_alerts = alerts[alerts["timestamp"].dt.year == year].copy()

# Plot spread baseline
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sub["timestamp"], y=sub["spread_da"],
    mode="lines", name="Spread DA",
    line=dict(color="#34495E", width=1),
    opacity=0.5,
))

# Overlay alerts by level (mapping ES->EN)
level_es_to_en = {"VERDE": "Green", "ÁMBAR": "Amber",
                  "NARANJA": "Orange", "ROJA": "Red"}
for lvl_es, lvl_en in level_es_to_en.items():
    if lvl_en not in alert_levels:
        continue
    mask = sub_alerts["alert_level"] == lvl_es
    if mask.sum() == 0:
        continue
    al = sub_alerts[mask]
    # Get spread at those timestamps
    al_with_spread = al.merge(sub[["timestamp", "spread_da"]], on="timestamp", how="left")
    fig.add_trace(go.Scatter(
        x=al_with_spread["timestamp"],
        y=al_with_spread["spread_da"],
        mode="markers", name=f"{lvl_en} ({len(al):,})",
        marker=dict(color=PALETTE[lvl_es],
                    size=4 if lvl_en == "Green" else 7,
                    opacity=0.6 if lvl_en == "Green" else 0.85),
    ))

# Overlay documented events as shaded regions
events_year = events[
    (events["date_start"].dt.year <= year) &
    (events["date_end"].dt.year >= year)
]
for _, ev in events_year.iterrows():
    start = max(ev["date_start"], pd.Timestamp(f"{year}-01-01"))
    end = min(ev["date_end"], pd.Timestamp(f"{year}-12-31"))
    if start <= end:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="purple", opacity=0.08,
            line_width=0,
        )

# Naive overlay if requested
if show_naive and "regime" in sub.columns:
    # régime p99 over the year (look-ahead, just for visualization)
    p99 = sub.groupby("regime")["spread_da"].transform(
        lambda s: s.abs().quantile(0.99))
    naive_flag = sub["spread_da"].abs() > p99
    fig.add_trace(go.Scatter(
        x=sub.loc[naive_flag, "timestamp"],
        y=sub.loc[naive_flag, "spread_da"],
        mode="markers", name=f"naive |spread|>p99 régime ({int(naive_flag.sum())})",
        marker=dict(color="black", size=4, symbol="x", opacity=0.5),
    ))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Spread DA (EUR/MWh)",
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Purple shaded bands: documented panel events. Markers: cascade alerts. "
    "Optional black ×: naive |spread|>p99 régime-specific benchmark "
    "(look-ahead calibration, illustrative only)."
)

st.divider()

# ---------------------------------------------------------------------------
# Two-column section: alert distribution + per-magnitude recall
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Monthly alert volume")
    high = sub_alerts[sub_alerts["alert_level"].isin(["NARANJA", "ROJA"])].copy()
    if not high.empty:
        high["month"] = high["timestamp"].dt.to_period("M").astype(str)
        monthly = high.groupby(["month", "alert_level"]).size().reset_index(name="hours")
        monthly["level_en"] = monthly["alert_level"].map(level_es_to_en)
        fig2 = px.bar(
            monthly, x="month", y="hours", color="level_en",
            color_discrete_map={"Orange": "#E67E22", "Red": "#C0392B"},
            labels={"hours": "Hours of Orange/Red alert", "month": ""},
        )
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No Orange/Red alerts in {year}")

with col_right:
    st.subheader("Detection by event magnitude")
    if event_valid is not None:
        iw = event_valid[~event_valid["out_of_alert_window"]]
        if len(iw):
            by_mag = iw.groupby("magnitude").agg(
                n=("event_id", "count"),
                detected=("system_detected", "sum"),
            ).reset_index()
            by_mag["recall"] = by_mag["detected"] / by_mag["n"]
            order = ["MUY_ALTA", "ALTA", "MEDIA", "BAJA"]
            by_mag["magnitude"] = pd.Categorical(by_mag["magnitude"],
                                                   categories=order, ordered=True)
            by_mag = by_mag.sort_values("magnitude")
            fig3 = px.bar(
                by_mag, x="magnitude", y="recall",
                hover_data={"n": True, "detected": True},
                labels={"recall": "Recall @ event (in-window)", "magnitude": ""},
                color="recall", color_continuous_scale="Viridis", range_color=[0, 1],
            )
            fig3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                                showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Documented events table (filterable)
# ---------------------------------------------------------------------------
st.subheader("Documented panel events")
if event_valid is not None:
    show_cols = ["event_id", "date_start", "date_end", "event_type",
                 "magnitude", "direction", "system_detected",
                 "system_n_orange_or_higher", "system_tier_max",
                 "system_lag_hours", "naive_detected"]
    show_cols = [c for c in show_cols if c in event_valid.columns]
    only_inwindow = st.checkbox("Show only events inside alert window (2022-2024)",
                                value=True)
    e = event_valid.copy()
    if only_inwindow and "out_of_alert_window" in e.columns:
        e = e[~e["out_of_alert_window"]]
    st.dataframe(e[show_cols], use_container_width=True, hide_index=True)
else:
    st.dataframe(events, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Model comparison: benchmark + cross-year + GW pinball
# ---------------------------------------------------------------------------
st.subheader("Model comparison")
tab_b, tab_cy, tab_gw = st.tabs([
    "Point-prediction benchmarks (test 2024)",
    "Cross-year robustness",
    "Giacomini–White pinball q=0.95",
])

with tab_b:
    if bench is not None:
        st.dataframe(bench, use_container_width=True, hide_index=True)
        st.caption(
            "Diebold-Mariano against Model A v3 (positive DM stat ⇒ v3 better). "
            "RMSE 95% CI from circular block-bootstrap (1000 reps, 24h blocks)."
        )
    else:
        st.info("benchmark_comparison.csv not found")

with tab_cy:
    if cross_year is not None:
        cy = cross_year.copy()
        cols = ["fold", "rmse_v3_mean", "rmse_v4_quant", "rmse_AR1",
                "pinball_v3_mean", "pinball_v4_quant", "pinball_AR1",
                "GW_v4_vs_AR1_pvalue"]
        cols = [c for c in cols if c in cy.columns]
        st.dataframe(cy[cols], use_container_width=True, hide_index=True)
        st.caption(
            "v4-quantile beats AR(1) on pinball q=0.95 in 2 of 3 folds; "
            "fails on 2022 (Ukraine + FR nuclear crisis + Iberian Exception "
            "onset concurrent). Tail-prediction advantage is robust under "
            "stable regimes, not under first-order structural shocks."
        )
    else:
        st.info("cross_year_validation.csv not found")

with tab_gw:
    if gw_real is not None:
        st.dataframe(gw_real, use_container_width=True, hide_index=True)
        st.caption(
            "v3 (mean-error) does not statistically beat AR(1) on pinball at "
            "q=0.95. The v4-quantile companion model (same FEATURES_A_CLEAN, "
            "only the objective changes to reg:quantileerror with α=0.95) does, "
            "with GW p<10⁻³ on test 2024."
        )
    else:
        st.info("gw_pinball_real_data.csv not found")

st.divider()

# ---------------------------------------------------------------------------
# Footer / methodology summary
# ---------------------------------------------------------------------------
st.subheader("Methodology in one minute")
st.markdown("""
- **Data**: OMIE prices, JAO NTC, Yahoo Finance fundamentals (TTF, CO₂).
  Hourly, 2019-01-01 to 2024-12-31 (52,530 observations).
- **Two-model architecture sharing the same 26-feature set
  (`FEATURES_A_CLEAN`)**:
  - **v3-mean** — XGBoost MSE. Residuals feed the operational cascade.
  - **v4-quantile** — same XGBoost, objective `reg:quantileerror`,
    α=0.95. Provides formal econometric evidence on tail behaviour
    (GW p<10⁻³ vs AR(1) on test 2024).
- **Detection**: rolling 168 h z-score with mandatory shift(1) +
  Isolation Forest + bilateral CUSUM. Régime-specific p90/p95/p99
  thresholds calibrated train-only.
- **Cascade**: Green / Amber / Orange / Red.
- **Validation**: pre-specified panel of 40 documented events
  (2019-2024). Recall @ event = 0.833; median detection lag = 180 h.
- **Operational calibration**: Pareto frontier exposes a 4.3 alerts/mo
  configuration with the same recall and 43% higher precision @ 72h.
- **Regulatory positioning**: REMIT II Article 15 compliant (4-week
  notification window from awareness; cascade provides the
  human-screening filter expected by ACER Guidance v6.1).
""")
