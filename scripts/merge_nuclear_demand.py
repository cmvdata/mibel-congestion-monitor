"""
merge_nuclear_demand.py
-----------------------
Anade 3 features estructurales al dataset v4 (que ya tiene las 4
renovables): fr_nuclear_avail_mw, es_demand_fc_mw, fr_demand_fc_mw.

Output:
    data/processed/mibel_dataset_20190101_20241231_v5.parquet
    data/processed/nuclear_demand_coverage_report.csv
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw" / "standardized"
MASTER_V4 = PROC_DIR / "mibel_dataset_20190101_20241231_v4.parquet"
OUT = PROC_DIR / "mibel_dataset_20190101_20241231_v5.parquet"

NEW_VARS = [
    ("fr_nuclear_avail_mw_hourly_mw.csv", "fr_nuclear_avail"),
    ("es_demand_fc_mw_hourly_mw.csv",     "es_demand_fc"),
    ("fr_demand_fc_mw_hourly_mw.csv",     "fr_demand_fc"),
]


def load_csv(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_local = ts_utc.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    out = pd.DataFrame({
        "timestamp": ts_local,
        column_name: df["value_mw"].astype(float),
    })
    out = out.drop_duplicates(subset="timestamp", keep="first")
    return out


def main():
    print("=" * 72)
    print("  Integrando nuclear FR + demand ES/FR al dataset v4 -> v5")
    print("=" * 72)
    master = pd.read_parquet(MASTER_V4)
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    print(f"  Master v4: {len(master):,} filas")
    merged = master.copy()
    coverage_rows = []
    for csv_name, col in NEW_VARS:
        ext = load_csv(RAW_DIR / csv_name, col)
        merged = merged.merge(ext, on="timestamp", how="left")
        n_non_null = int(merged[col].notna().sum())
        coverage = n_non_null / len(merged)
        print(f"  {col:20s}  {n_non_null:>6,}/{len(merged):,} "
              f"({coverage*100:.2f}%)  missing={len(merged) - n_non_null}")
        coverage_rows.append({
            "variable": col,
            "n_non_null": n_non_null,
            "n_total": len(merged),
            "coverage_pct": coverage * 100,
            "n_missing": len(merged) - n_non_null,
            "min": float(merged[col].min()) if n_non_null else np.nan,
            "max": float(merged[col].max()) if n_non_null else np.nan,
            "mean": float(merged[col].mean()) if n_non_null else np.nan,
        })

    print(f"\n  Master v5 shape: {merged.shape} (v4 was {master.shape})")
    new_cols = [c for c in merged.columns if c not in master.columns]
    print(f"  Nuevas columnas: {new_cols}")
    merged.to_parquet(OUT, index=False)
    print(f"  Guardado: {OUT}")
    cov = pd.DataFrame(coverage_rows)
    cov_path = PROC_DIR / "nuclear_demand_coverage_report.csv"
    cov.to_csv(cov_path, index=False)
    print(f"  Coverage: {cov_path}")


if __name__ == "__main__":
    main()
