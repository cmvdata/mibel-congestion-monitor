"""
merge_renewables.py
-------------------
Integra los 4 CSVs de forecast renovable (Manus, mayo 2026) en el
dataset maestro. Los CSVs vienen con timestamp_utc + timestamp_local;
el dataset maestro usa hora local Madrid timezone-naive.

Outputs:
    data/processed/mibel_dataset_20190101_20241231_v4.parquet
    data/processed/renewables_coverage_report.csv

NO sobreescribe el dataset original.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw" / "renewables"
MASTER = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
OUT = PROC_DIR / "mibel_dataset_20190101_20241231_v4.parquet"

RENEWABLES = [
    ("fr_solar_fc_hourly_mw.csv", "fr_solar_fc"),
    ("fr_wind_fc_hourly_mw.csv",  "fr_wind_fc"),
    ("es_solar_fc_hourly_mw.csv", "es_solar_fc"),
    ("es_wind_fc_hourly_mw.csv",  "es_wind_fc"),
]


def load_renewable_csv(csv_path: Path, column_name: str) -> pd.DataFrame:
    """Load a renewable forecast CSV and return DataFrame with
    [timestamp (naive local Madrid), column_name] columns.

    Parse timestamp_utc (uniform tz), convert to Europe/Madrid, drop tz
    to align with the master parquet which is tz-naive local Madrid.
    """
    df = pd.read_csv(csv_path)
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_local = ts_utc.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    out = pd.DataFrame({
        "timestamp": ts_local,
        column_name: df["value_mw"].astype(float),
    })
    # DST fall-back creates one repeated local hour; keep first occurrence
    out = out.drop_duplicates(subset="timestamp", keep="first")
    return out


def main() -> None:
    print("=" * 72)
    print("  Integrando forecasts renovables al dataset maestro")
    print("=" * 72)

    master = pd.read_parquet(MASTER)
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    print(f"  Master: {len(master):,} filas  "
          f"({master['timestamp'].min()} -> {master['timestamp'].max()})")

    merged = master.copy()
    coverage_rows = []
    for csv_name, col in RENEWABLES:
        path = RAW_DIR / csv_name
        ren = load_renewable_csv(path, col)
        print(f"\n  {col}: {len(ren):,} filas con valor en {csv_name}")

        merged = merged.merge(ren, on="timestamp", how="left")
        n_non_null = int(merged[col].notna().sum())
        n_total = len(merged)
        coverage = n_non_null / n_total
        print(f"    en master: {n_non_null:,}/{n_total:,} ({coverage*100:.2f}%) "
              f"con valor; {n_total - n_non_null:,} NaN")
        coverage_rows.append({
            "variable": col,
            "n_master": n_total,
            "n_non_null_after_merge": n_non_null,
            "coverage_pct": coverage * 100,
            "n_missing": n_total - n_non_null,
            "min": float(merged[col].min()) if n_non_null > 0 else np.nan,
            "max": float(merged[col].max()) if n_non_null > 0 else np.nan,
            "mean": float(merged[col].mean()) if n_non_null > 0 else np.nan,
        })

    # Sanity check on new columns
    print(f"\n  Master enriched: {merged.shape}  (orig {master.shape})")
    new_cols = [c for c in merged.columns if c.endswith("_fc")]
    print(f"  Nuevas columnas: {new_cols}")

    merged.to_parquet(OUT, index=False)
    print(f"\n  Guardado: {OUT}")

    cov_df = pd.DataFrame(coverage_rows)
    cov_path = PROC_DIR / "renewables_coverage_report.csv"
    cov_df.to_csv(cov_path, index=False)
    print(f"  Coverage report: {cov_path}")

    print("\n  Resumen cobertura:")
    print(cov_df[["variable", "n_non_null_after_merge", "coverage_pct",
                  "n_missing"]].to_string(index=False))


if __name__ == "__main__":
    main()
