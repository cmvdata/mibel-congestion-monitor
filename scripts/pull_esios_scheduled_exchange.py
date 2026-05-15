"""
pull_esios_scheduled_exchange.py
--------------------------------
Descarga los indicadores ESIOS 28 (FR -> ES programado D-1 PBF) y
32 (ES -> FR programado D-1 PBF) para 2019-2024 en chunks anuales.

Autenticación: header `x-api-key` (ESIOS cambió desde el viejo
"Authorization: Token token=" alrededor de 2024).

Output: data/raw/standardized/
   fr_to_es_scheduled_pbf_mw_hourly_mw.csv (indicator 28)
   es_to_fr_scheduled_pbf_mw_hourly_mw.csv (indicator 32)
"""
from pathlib import Path
import os
import sys
import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "raw" / "standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get(
    "ESIOS_TOKEN",
    "c5448911873cb0f38f62e8e6c378b706422a173086f9b7971cc3ca989a0d6464",
)

HEADERS = {
    "Accept": "application/json; application/vnd.api+json",
    "x-api-key": TOKEN,
}

INDICATORS = [
    (28, "fr_to_es_scheduled_pbf",
     "ESIOS indicator 28: Generacion programada PBF Importacion Francia (FR->ES, D-1)"),
    (32, "es_to_fr_scheduled_pbf",
     "ESIOS indicator 32: Generacion programada PBF Exportacion Francia (ES->FR, D-1)"),
]


def pull_month(indicator_id: int, year: int, month: int) -> pd.DataFrame:
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    # Compute last day of month
    if month == 12:
        end_ymd = f"{year}-12-31T23:59:59Z"
    else:
        # First day of next month minus 1 second is overkill; use 28-31
        # by trial:
        last_day = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
        end_ymd = f"{year}-{month:02d}-{last_day:02d}T23:59:59Z"
    params = {
        "start_date": f"{year}-{month:02d}-01T00:00:00Z",
        "end_date":   end_ymd,
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=180)
    r.raise_for_status()
    js = r.json()
    vals = js.get("indicator", {}).get("values", [])
    return pd.DataFrame(vals) if vals else pd.DataFrame()


def standardize(df_raw: pd.DataFrame, var_name: str,
                source_label: str) -> pd.DataFrame:
    # ESIOS datetime field has mixed tz (CET+01, CEST+02 by DST). Parse
    # with utc=True to normalize; then store both UTC and local Madrid
    # for the standard schema.
    ts_utc = pd.to_datetime(df_raw["datetime_utc"], utc=True)
    ts_local = pd.to_datetime(df_raw["datetime"], utc=True).dt.tz_convert(
        "Europe/Madrid")
    out = pd.DataFrame({
        "timestamp_utc":   ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamp_local": ts_local.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
        "variable":        var_name,
        "value_mw":        df_raw["value"].astype(float),
        "unit":            "MW",
        "granularity":     "hourly_power_forecast",
        "source":          source_label,
        "coverage_ratio_hour": 1.0,
        "quality_flag":    "forecast_d1_pbf",
    })
    return out.sort_values("timestamp_utc").drop_duplicates(
        subset="timestamp_utc", keep="first").reset_index(drop=True)


def main():
    for ind_id, var_name, source_label in INDICATORS:
        print(f"\n=== Pulling {var_name} (indicator {ind_id}) ===")
        frames = []
        total = 0
        for y in range(2019, 2025):
            year_total = 0
            for m in range(1, 13):
                try:
                    df = pull_month(ind_id, y, m)
                    if len(df):
                        frames.append(df)
                        year_total += len(df)
                    time.sleep(0.2)
                except requests.HTTPError as e:
                    print(f"  {y}-{m:02d} FAIL HTTP {e.response.status_code}")
                    continue
            total += year_total
            print(f"  year {y}: {year_total:,} rows")
        print(f"  total raw rows: {total:,}")

        if not frames:
            print(f"  No data returned for indicator {ind_id}")
            continue

        df_raw = pd.concat(frames, ignore_index=True)
        # ESIOS retorna multiples geo_id por timestamp (zona peninsular,
        # SIDIRA, etc.). Filtrar a peninsular si existe; si no, usar
        # geo_id == 8741 que es "España peninsular".
        if "geo_id" in df_raw.columns:
            unique_geos = df_raw["geo_id"].unique()
            print(f"  geo_id values present: {list(unique_geos)[:6]}")
            # Preferir peninsular: geo_id=8741 si existe; si no, todos
            if 8741 in unique_geos:
                df_raw = df_raw[df_raw["geo_id"] == 8741].copy()
                print(f"  filtered to geo_id=8741 (peninsular): {len(df_raw):,} rows")

        out = standardize(df_raw, var_name, source_label)
        out_path = OUT_DIR / f"{var_name}_mw_hourly_mw.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(out):,} rows)")
        print(f"  Range: {out['value_mw'].min():.1f} to "
              f"{out['value_mw'].max():.1f} MW  "
              f"mean={out['value_mw'].mean():.1f}")


if __name__ == "__main__":
    main()
