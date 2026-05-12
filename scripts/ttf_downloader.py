"""
ttf_downloader.py
-----------------
Descarga precios de TTF Gas Natural y CO2 EUA desde Yahoo Finance (yfinance).
Calcula coste marginal CCGT, Spark Spread y Clean Spark Spread.

Fuentes:
- TTF Gas: ticker "TTF=F" (futuros front-month)
- CO2 EUA: ticker "CO2.L" o "EUETS.PA"
- Resolución: Diaria (se interpola a horaria)
NO requiere token.

Variables calculadas:
- ttf_eur_mwh: precio TTF en €/MWh (conversión desde €/MWh o USD/MMBtu)
- co2_eur_t: precio CO2 en €/tonelada
- gas_cost_mwh: coste del gas para generar 1 MWh en CCGT (eficiencia 50%)
- spark_spread: price_es - gas_cost_mwh (margen bruto generación gas)
- clean_spark_spread: spark_spread - co2_cost_mwh (margen neto incluyendo CO2)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TTF] %(message)s")
logger = logging.getLogger(__name__)

# Eficiencia térmica CCGT típica (50%)
CCGT_EFFICIENCY = 0.50
# Factor de emisión gas natural (t CO2 / MWh térmico)
GAS_EMISSION_FACTOR = 0.202  # t CO2 / MWh_th

# Tickers a probar en orden de preferencia
TTF_TICKERS = ["TTF=F", "NG=F"]  # TTF Europa o Henry Hub como fallback
CO2_TICKERS = ["CO2.L", "EUETS.PA", "CARBON.L"]


def download_ttf(start: str, end: str, output_dir: Path) -> pd.DataFrame:
    """
    Descarga TTF Gas y CO2 EUA, calcula Spark Spread y Clean Spark Spread.
    Devuelve serie diaria con interpolación a horaria.

    Columnas resultado:
    - date: fecha
    - ttf_eur_mwh: precio TTF en €/MWh
    - co2_eur_t: precio CO2 en €/t
    - gas_cost_mwh: coste gas para generar 1 MWh (€/MWh)
    - co2_cost_mwh: coste CO2 para generar 1 MWh (€/MWh)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f"ttf_{start[:4]}_{end[:4]}.parquet"

    if cache_file.exists():
        logger.info("TTF: cargando desde caché.")
        return pd.read_parquet(cache_file)

    logger.info(f"Descargando TTF Gas: {start} → {end}")

    # --- TTF Gas ---
    ttf_data = None
    for ticker in TTF_TICKERS:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not raw.empty:
                # Aplanar MultiIndex si existe
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                ttf_data = raw[["Close"]].copy()
                ttf_data.columns = ["ttf_raw"]
                ttf_data.index = pd.to_datetime(ttf_data.index)
                logger.info(f"TTF descargado con ticker {ticker}: {len(ttf_data)} días")
                break
        except Exception as e:
            logger.warning(f"TTF ticker {ticker} falló: {e}")

    if ttf_data is None or ttf_data.empty:
        raise RuntimeError(
            "No se pudo descargar TTF desde yfinance. "
            "Pipeline aborta para evitar entrenamiento con datos sintéticos. "
            "Revisar conectividad o ticker (TTF=F). "
            "Si yfinance falla persistentemente, considerar fuente alternativa: "
            "Euronext/EEX licenciadas."
        )

    # Convertir a €/MWh
    # TTF=F cotiza en €/MWh directamente en algunos brokers, en otros en USD/MMBtu
    # 1 MMBtu = 0.29307 MWh → factor = 1/0.29307 = 3.412
    # Asumimos que si el precio > 100 es USD/MMBtu, si < 100 es €/MWh
    ttf_median = ttf_data["ttf_raw"].median()
    if ttf_median > 100:
        # Probablemente USD/MMBtu → convertir a €/MWh (asumiendo USD/EUR ≈ 0.92)
        ttf_data["ttf_eur_mwh"] = ttf_data["ttf_raw"] * 0.92 / 3.412
        logger.info(f"TTF convertido desde USD/MMBtu. Mediana: {ttf_data['ttf_eur_mwh'].median():.1f} €/MWh")
    else:
        ttf_data["ttf_eur_mwh"] = ttf_data["ttf_raw"]
        logger.info(f"TTF en €/MWh. Mediana: {ttf_data['ttf_eur_mwh'].median():.1f} €/MWh")

    # --- CO2 EUA ---
    co2_data = None
    for ticker in CO2_TICKERS:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                co2_data = raw[["Close"]].copy()
                co2_data.columns = ["co2_eur_t"]
                co2_data.index = pd.to_datetime(co2_data.index)
                logger.info(f"CO2 descargado con ticker {ticker}: {len(co2_data)} días")
                break
        except Exception as e:
            logger.warning(f"CO2 ticker {ticker} falló: {e}")

    if co2_data is None or co2_data.empty:
        raise RuntimeError(
            "No se pudo descargar CO2 EUA desde yfinance. "
            "Pipeline aborta para evitar entrenamiento con valores aproximados "
            "hardcoded (que falsifican la trazabilidad del CO2 spark spread). "
            "Tickers probados: CO2.L, EUETS.PA, CARBON.L. "
            "Si yfinance falla persistentemente, considerar fuente alternativa: "
            "ICE Endex licenciado."
        )

    # --- Ensamblar dataset diario ---
    df_daily = ttf_data[["ttf_eur_mwh"]].copy()
    df_daily = df_daily.join(co2_data[["co2_eur_t"]], how="outer")

    # Rellenar huecos (fines de semana, festivos)
    df_daily = df_daily.resample("D").interpolate(method="linear")
    df_daily = df_daily.ffill().bfill()

    # Calcular costes de generación
    # Coste gas para generar 1 MWh eléctrico en CCGT
    df_daily["gas_cost_mwh"] = df_daily["ttf_eur_mwh"] / CCGT_EFFICIENCY
    # Coste CO2 para generar 1 MWh eléctrico en CCGT
    df_daily["co2_cost_mwh"] = (df_daily["co2_eur_t"] * GAS_EMISSION_FACTOR) / CCGT_EFFICIENCY

    df_daily = df_daily.reset_index().rename(columns={"index": "date", "Date": "date"})
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.normalize()

    # Filtrar al rango solicitado
    df_daily = df_daily[
        (df_daily["date"] >= pd.Timestamp(start)) &
        (df_daily["date"] <= pd.Timestamp(end))
    ].reset_index(drop=True)

    df_daily.to_parquet(cache_file, index=False)
    logger.info(f"TTF guardado: {len(df_daily)} días | "
                f"TTF mediana: {df_daily['ttf_eur_mwh'].median():.1f} €/MWh | "
                f"CO2 mediana: {df_daily['co2_eur_t'].median():.1f} €/t")
    return df_daily


def merge_ttf_hourly(df_hourly: pd.DataFrame, df_ttf: pd.DataFrame) -> pd.DataFrame:
    """
    Hace join del dataset horario con los datos diarios de TTF/CO2.
    Añade columnas: ttf_eur_mwh, co2_eur_t, gas_cost_mwh, co2_cost_mwh,
                    spark_spread, clean_spark_spread.
    """
    df = df_hourly.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    df_ttf_clean = df_ttf[["date", "ttf_eur_mwh", "co2_eur_t",
                            "gas_cost_mwh", "co2_cost_mwh"]].copy()
    df_ttf_clean["date"] = pd.to_datetime(df_ttf_clean["date"]).dt.normalize()

    df = df.merge(df_ttf_clean, on="date", how="left")

    # Spark Spread: precio spot - coste marginal gas
    if "price_es" in df.columns:
        df["spark_spread"] = df["price_es"] - df["gas_cost_mwh"]
        df["clean_spark_spread"] = df["price_es"] - df["gas_cost_mwh"] - df["co2_cost_mwh"]

    df = df.drop(columns=["date"])
    return df


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        df = download_ttf("2024-01-01", "2024-01-31", Path(tmp))
        print(df.head(5).to_string())
        print(f"\nGas cost mediano: {df['gas_cost_mwh'].median():.1f} €/MWh")
        print(f"CO2 cost mediano: {df['co2_cost_mwh'].median():.1f} €/MWh")
