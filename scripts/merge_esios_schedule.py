"""
merge_esios_schedule.py
-----------------------
Combina los indicadores ESIOS 28 (FR->ES) y 32 (ES->FR) en una serie
unificada de flujo programado neto ES->FR.

Convencion: scheduled_net_es_to_fr_mw
  > 0  -> ES exporta a FR
  < 0  -> ES importa de FR

ESIOS reporta:
  ind 28 positivo (X) = importacion desde FR  ->  net_es_to_fr = -X
  ind 32 negativo (Y) = exportacion hacia FR  ->  net_es_to_fr = -Y  ( = |Y| positivo)

Los indicadores son mutuamente exclusivos por hora (cuando uno publica,
el otro queda vacio). Si las dos publican la misma hora, priorizamos
ind 32 porque la exportacion es la condicion de congestion ES->FR.

Tambien anade la version absoluta para que H1' la use:
  scheduled_abs_mw = |scheduled_net_es_to_fr_mw|

Y NTC absoluta (max de las dos NTC direcciones ya en el master):
  utilization_d1 = scheduled_abs_mw / NTC_absoluta

Output: data/processed/mibel_dataset_20190101_20241231_v6.parquet
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "standardized"
PROC = BASE / "data" / "processed"
MASTER_V5 = PROC / "mibel_dataset_20190101_20241231_v5.parquet"
OUT = PROC / "mibel_dataset_20190101_20241231_v6.parquet"


def load_indicator(csv_name: str, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / csv_name)
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)
    ts_local = ts_utc.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    return pd.DataFrame({
        "timestamp": ts_local,
        col_name: df["value_mw"].astype(float),
    }).drop_duplicates("timestamp", keep="first")


def main():
    print("=" * 72)
    print("  Merging ESIOS scheduled exchange indicators 28 + 32 -> v6")
    print("=" * 72)

    master = pd.read_parquet(MASTER_V5)
    master["timestamp"] = pd.to_datetime(master["timestamp"])
    print(f"  Master v5: {len(master):,} rows")

    ind28 = load_indicator(
        "fr_to_es_scheduled_pbf_mw_hourly_mw.csv", "raw_28_fr_to_es")
    ind32 = load_indicator(
        "es_to_fr_scheduled_pbf_mw_hourly_mw.csv", "raw_32_es_to_fr")
    print(f"  ind 28 (FR->ES import): {len(ind28):,} rows")
    print(f"  ind 32 (ES->FR export): {len(ind32):,} rows")

    # Merge to master timestamps
    m = master.merge(ind28, on="timestamp", how="left")
    m = m.merge(ind32, on="timestamp", how="left")

    n_both = int(((m["raw_28_fr_to_es"].notna())
                  & (m["raw_32_es_to_fr"].notna())).sum())
    n_only28 = int(((m["raw_28_fr_to_es"].notna())
                    & (m["raw_32_es_to_fr"].isna())).sum())
    n_only32 = int(((m["raw_28_fr_to_es"].isna())
                    & (m["raw_32_es_to_fr"].notna())).sum())
    n_neither = int(((m["raw_28_fr_to_es"].isna())
                     & (m["raw_32_es_to_fr"].isna())).sum())
    print(f"\n  Coverage on master timestamps ({len(m):,} rows):")
    print(f"    only ind 28 (import):  {n_only28:,}  ({n_only28/len(m)*100:.2f}%)")
    print(f"    only ind 32 (export):  {n_only32:,}  ({n_only32/len(m)*100:.2f}%)")
    print(f"    both:                  {n_both:,}")
    print(f"    neither:               {n_neither:,}  ({n_neither/len(m)*100:.2f}%)")

    # Build net flow ES -> FR
    # If ind 32 has value Y (Y<0): net = -Y  (>0, ES exports)
    # Elif ind 28 has value X (X>0): net = -X  (<0, ES imports)
    # Both: prioritize ind 32 (export case more relevant for congestion)
    net = np.where(
        m["raw_32_es_to_fr"].notna(),
        -m["raw_32_es_to_fr"].astype(float),
        np.where(m["raw_28_fr_to_es"].notna(),
                 -m["raw_28_fr_to_es"].astype(float),
                 np.nan),
    )
    m["scheduled_net_es_to_fr_mw"] = net
    m["scheduled_abs_mw"] = np.abs(net)

    # Utilization vs available NTC (whichever direction is relevant)
    # If net > 0 (ES exports), capacity is ntc_es_fr; if < 0, ntc_fr_es
    ntc_dir = np.where(net > 0, m["ntc_es_fr"], m["ntc_fr_es"])
    util = np.where(ntc_dir > 0, np.abs(net) / ntc_dir, np.nan)
    util = np.where(np.isnan(util), util, np.clip(util, 0, 2))  # cap
    m["utilization_d1"] = util

    # Summary
    print(f"\n  scheduled_net_es_to_fr_mw: "
          f"n={m['scheduled_net_es_to_fr_mw'].notna().sum():,}  "
          f"min={m['scheduled_net_es_to_fr_mw'].min():.1f}  "
          f"max={m['scheduled_net_es_to_fr_mw'].max():.1f}  "
          f"mean={m['scheduled_net_es_to_fr_mw'].mean():.1f}")
    print(f"  utilization_d1:  "
          f"n={m['utilization_d1'].notna().sum():,}  "
          f"mean={m['utilization_d1'].mean():.3f}  "
          f"p95={m['utilization_d1'].quantile(0.95):.3f}  "
          f"p99={m['utilization_d1'].quantile(0.99):.3f}")

    # Drop raw intermediate columns
    m = m.drop(columns=["raw_28_fr_to_es", "raw_32_es_to_fr"])

    m.to_parquet(OUT, index=False)
    print(f"\n  Saved: {OUT}")
    print(f"  Shape: {m.shape}  (v5 was master, plus 3 new cols)")


if __name__ == "__main__":
    main()
