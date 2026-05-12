"""
build_dataset.py
----------------
Pipeline integrador del MIBEL Congestion Monitor.
Une OMIE + TTF/CO2 + JAO ATCs y genera el dataset final con todos los features.

Uso:
    python src/build_dataset.py --start 2019-01-01 --end 2024-12-31
    python src/build_dataset.py --start 2024-01-01 --end 2024-03-31  # test rápido

Genera:
    data/processed/mibel_dataset_YYYYMMDD_YYYYMMDD.parquet
    data/processed/mibel_dataset_YYYYMMDD_YYYYMMDD.csv
    data/processed/data_quality_report.txt
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Módulos del proyecto
import sys
sys.path.insert(0, str(Path(__file__).parent))
from omie_downloader import download_omie_range
from ttf_downloader import download_ttf, merge_ttf_hourly
from jao_downloader import build_atc_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BUILD] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade variables de calendario: hora, día semana, mes, festivos, régimen."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek          # 0=lunes, 6=domingo
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_night"] = ((ts.dt.hour < 7) | (ts.dt.hour >= 22)).astype(int)
    df["is_peak"] = ((ts.dt.hour >= 9) & (ts.dt.hour <= 21)).astype(int)

    # Régimen de mercado (variable clave para validación por regímenes)
    # 2019-2021: pre-crisis energética
    # 2022-2024: excepción ibérica + crisis
    # 2024+: post-excepción
    conditions = [
        ts < pd.Timestamp("2022-01-01"),
        (ts >= pd.Timestamp("2022-01-01")) & (ts < pd.Timestamp("2024-01-01")),
        ts >= pd.Timestamp("2024-01-01"),
    ]
    choices = ["pre_crisis", "excepcion_iberica", "post_excepcion"]
    df["regime"] = np.select(conditions, choices, default="unknown")

    # Estación
    month = ts.dt.month
    df["season"] = pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=["winter", "spring", "summer", "autumn"],
        right=True
    ).astype(str)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade lags temporales del spread y variables clave."""
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lags del spread DA (variable objetivo)
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"spread_da_lag{lag}h"] = df["spread_da"].shift(lag)

    # Media móvil del spread (tendencia reciente)
    df["spread_da_ma24h"] = df["spread_da"].rolling(24, min_periods=12).mean()
    df["spread_da_ma168h"] = df["spread_da"].rolling(168, min_periods=72).mean()

    # Volatilidad reciente del spread
    df["spread_da_std24h"] = df["spread_da"].rolling(24, min_periods=12).std()

    # Lags del ATC (señal de congestión rezagada)
    if "ntc_es_fr" in df.columns:
        df["ntc_es_fr_lag24h"] = df["ntc_es_fr"].shift(24)
    if "atc_fatigue" in df.columns:
        df["atc_fatigue_lag24h"] = df["atc_fatigue"].shift(24)

    # Lags del TTF (coste marginal rezagado)
    if "ttf_eur_mwh" in df.columns:
        df["ttf_lag24h"] = df["ttf_eur_mwh"].shift(24)
        df["ttf_lag168h"] = df["ttf_eur_mwh"].shift(168)

    return df


def compute_data_quality(df: pd.DataFrame, start: str, end: str) -> dict:
    """Calcula métricas de calidad del dataset."""
    total_hours = (pd.Timestamp(end) - pd.Timestamp(start)).days * 24 + 24
    actual_hours = len(df)
    coverage = actual_hours / total_hours * 100

    quality = {
        "total_hours_expected": total_hours,
        "total_hours_actual": actual_hours,
        "coverage_pct": coverage,
        "missing_spread_da": df["spread_da"].isna().sum(),
        "missing_ntc_es_fr": df["ntc_es_fr"].isna().sum() if "ntc_es_fr" in df.columns else "N/A",
        "missing_ttf": df["ttf_eur_mwh"].isna().sum() if "ttf_eur_mwh" in df.columns else "N/A",
        "spread_da_mean": df["spread_da"].mean(),
        "spread_da_std": df["spread_da"].std(),
        "spread_da_min": df["spread_da"].min(),
        "spread_da_max": df["spread_da"].max(),
        "pct_congestion": (df["spread_da"].abs() > 0.5).mean() * 100,
        "regime_counts": df["regime"].value_counts().to_dict() if "regime" in df.columns else {},
        "atc_source_counts": df["atc_source"].value_counts().to_dict() if "atc_source" in df.columns else {},
    }
    return quality


def build_dataset(start: str, end: str) -> pd.DataFrame:
    """
    Pipeline completo de construcción del dataset.
    """
    logger.info(f"=== MIBEL Congestion Monitor — Build Dataset ===")
    logger.info(f"Período: {start} → {end}")

    raw_omie = DATA_DIR / "raw" / "omie"
    raw_jao = DATA_DIR / "raw" / "jao"
    raw_ttf = DATA_DIR / "raw" / "ttf"
    processed = DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # PASO 1: Descargar precios OMIE (day-ahead + intradiario)
    # ----------------------------------------------------------------
    logger.info("PASO 1/4: Descargando precios OMIE...")
    df_omie = download_omie_range(start, end, raw_omie, include_intraday=True)
    if df_omie.empty:
        raise RuntimeError("No se obtuvieron datos de OMIE. Abortando.")
    logger.info(f"  OMIE: {len(df_omie)} registros horarios")

    # ----------------------------------------------------------------
    # PASO 2: Descargar TTF Gas y CO2
    # ----------------------------------------------------------------
    logger.info("PASO 2/4: Descargando TTF Gas y CO2...")
    df_ttf = download_ttf(start, end, raw_ttf)
    df_omie = merge_ttf_hourly(df_omie, df_ttf)
    logger.info(f"  TTF integrado. Spark Spread medio: {df_omie['spark_spread'].mean():.1f} €/MWh")

    # ----------------------------------------------------------------
    # PASO 3: Descargar ATCs (JAO real + ATC implícito para pre-2022)
    # ----------------------------------------------------------------
    logger.info("PASO 3/4: Construyendo serie de ATCs...")
    df_atc = build_atc_series(start, end, raw_jao, df_prices=df_omie)
    if not df_atc.empty:
        # Normalizar timestamp para merge
        df_omie["timestamp"] = pd.to_datetime(df_omie["timestamp"])
        df_atc["timestamp"] = pd.to_datetime(df_atc["timestamp"])
        # Quitar timezone si existe
        if df_atc["timestamp"].dt.tz is not None:
            df_atc["timestamp"] = df_atc["timestamp"].dt.tz_localize(None)
        df = df_omie.merge(df_atc, on="timestamp", how="left")
    else:
        df = df_omie.copy()
        df["ntc_es_fr"] = np.nan
        df["ntc_fr_es"] = np.nan
        df["atc_implicit"] = np.nan
        df["atc_congestion"] = np.nan
        df["atc_fatigue"] = 0
        df["atc_source"] = "implicit_only"

    # ----------------------------------------------------------------
    # PASO 4: Feature engineering
    # ----------------------------------------------------------------
    logger.info("PASO 4/4: Calculando features...")
    df = add_calendar_features(df)
    df = add_lag_features(df)

    # Ordenar columnas de forma lógica
    core_cols = [
        "timestamp", "regime",
        # Precios y spreads
        "price_es", "price_fr", "spread_da",
        "price_id_es", "price_id_pt", "spread_id_es_pt", "spread_da_id",
        # Gas y CO2
        "ttf_eur_mwh", "co2_eur_t", "gas_cost_mwh", "co2_cost_mwh",
        "spark_spread", "clean_spark_spread",
        # ATCs
        "ntc_es_fr", "ntc_fr_es", "atc_implicit", "atc_congestion",
        "atc_fatigue", "atc_source",
        # Calendario
        "hour", "dow", "month", "year", "is_weekend", "is_night", "is_peak", "season",
    ]
    lag_cols = [c for c in df.columns if "lag" in c or "ma" in c or "std" in c]
    final_cols = [c for c in core_cols if c in df.columns] + lag_cols
    df = df[final_cols].copy()

    # ----------------------------------------------------------------
    # Guardar dataset
    # ----------------------------------------------------------------
    start_str = start.replace("-", "")
    end_str = end.replace("-", "")
    out_parquet = processed / f"mibel_dataset_{start_str}_{end_str}.parquet"
    out_csv = processed / f"mibel_dataset_{start_str}_{end_str}.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    logger.info(f"Dataset guardado: {out_parquet}")
    logger.info(f"Dataset guardado: {out_csv}")

    # ----------------------------------------------------------------
    # Reporte de calidad
    # ----------------------------------------------------------------
    quality = compute_data_quality(df, start, end)
    report_path = processed / "data_quality_report.txt"
    with open(report_path, "w") as f:
        f.write("=== MIBEL Congestion Monitor — Data Quality Report ===\n\n")
        f.write(f"Período: {start} → {end}\n")
        f.write(f"Registros totales: {quality['total_hours_actual']:,} / {quality['total_hours_expected']:,} horas\n")
        f.write(f"Cobertura temporal: {quality['coverage_pct']:.1f}%\n\n")
        f.write(f"Huecos spread DA: {quality['missing_spread_da']}\n")
        f.write(f"Huecos NTC ES-FR: {quality['missing_ntc_es_fr']}\n")
        f.write(f"Huecos TTF: {quality['missing_ttf']}\n\n")
        f.write(f"Spread DA — Media: {quality['spread_da_mean']:.2f} €/MWh\n")
        f.write(f"Spread DA — Std:   {quality['spread_da_std']:.2f} €/MWh\n")
        f.write(f"Spread DA — Min:   {quality['spread_da_min']:.2f} €/MWh\n")
        f.write(f"Spread DA — Max:   {quality['spread_da_max']:.2f} €/MWh\n")
        f.write(f"% horas con congestión (|spread|>0.5): {quality['pct_congestion']:.1f}%\n\n")
        f.write("Distribución por régimen:\n")
        for regime, count in quality['regime_counts'].items():
            f.write(f"  {regime}: {count:,} horas ({count/quality['total_hours_actual']*100:.1f}%)\n")
        f.write("\nFuente ATCs:\n")
        for src, count in quality['atc_source_counts'].items():
            f.write(f"  {src}: {count:,} horas\n")

    logger.info(f"Reporte de calidad: {report_path}")
    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset final: {len(df):,} registros, {len(df.columns)} columnas")
    logger.info(f"Cobertura: {quality['coverage_pct']:.1f}%")
    logger.info(f"Spread DA medio: {quality['spread_da_mean']:.2f} €/MWh")
    logger.info(f"{'='*50}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIBEL — Build Dataset")
    parser.add_argument("--start", default="2019-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Fecha fin (YYYY-MM-DD)")
    args = parser.parse_args()

    df = build_dataset(args.start, args.end)
    print(f"\nDataset listo: {len(df):,} filas × {len(df.columns)} columnas")
    print(df[["timestamp", "price_es", "price_fr", "spread_da",
              "ttf_eur_mwh", "spark_spread", "ntc_es_fr", "regime"]].head(10).to_string())
