# MIBEL Spread Surveillance — Dashboard

Single-page Streamlit app that lets you explore the system interactively.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`.

## What the dashboard shows

- **Headline KPIs**: documented events, recall, median detection lag, GW pinball p-value, v4-quantile pinball.
- **Spread + alerts time series** for the selected year, with cascade alerts overlaid as markers and documented panel events as shaded bands. Optional naive `|spread|>p99` régime-specific benchmark overlay.
- **Monthly alert volume** (Orange + Red) and **recall by event magnitude** side by side.
- **Documented panel events table** with system detection lag and naive-benchmark comparison.
- **Model comparison tabs**: point-prediction benchmarks (RMSE + DM), cross-year robustness (3 folds), Giacomini–White pinball at q=0.95.

## Deploy to Streamlit Cloud

1. The repo already has the right structure (`app/dashboard.py` + `requirements.txt`).
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. New app → repository `cmvdata/mibel-congestion-monitor`, branch `main`, main file `app/dashboard.py`.
4. Deploy. First boot installs the requirements (~2 min).

## Data dependencies

The dashboard reads these files (relative to repo root):

| File | Provided by |
|---|---|
| `data/processed/mibel_dataset_20190101_20241231.parquet` | `scripts/build_dataset.py` |
| `results/alerts_registry_v3.csv` | `scripts/anomaly_detection_v3.py` |
| `data/events_panel.csv` | Hand-curated panel, committed |
| `data/event_validation.csv` | `scripts/event_panel_validation.py` |
| `results/benchmark_comparison.csv` | `scripts/benchmarks_dm.py` |
| `results/cross_year_validation.csv` | `scripts/cross_year_validation.py` |
| `results/gw_pinball_real_data.csv` | `scripts/apply_gw_to_residuals.py` |

If any are missing the dashboard degrades gracefully and shows a placeholder for that section.
