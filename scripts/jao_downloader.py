"""
jao_downloader.py
-----------------
Descarga datos de NTC (Net Transfer Capacity) España-Francia y España-Portugal
desde la API pública de JAO SWE CCR (Joint Allocation Office, South Western Europe).

NO requiere token ni registro.
Endpoint verificado: https://publicationtool.jao.eu/swe/api/data/Adjustment
Cobertura histórica: septiembre 2022 – presente (24 valores horarios/día)

Para 2019 – agosto 2022: se calcula el ATC implícito desde los precios OMIE.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [JAO] %(message)s")
logger = logging.getLogger(__name__)

JAO_BASE = "https://publicationtool.jao.eu/swe/api/data/Adjustment"
JAO_START = pd.Timestamp("2022-09-01")  # Primer día con datos reales en JAO SWE


def fetch_jao_day(date: pd.Timestamp) -> pd.DataFrame | None:
    """Descarga los NTC horarios de un día desde JAO SWE CCR."""
    from_utc = date.strftime("%Y-%m-%dT00:00:00")
    to_utc = (date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    try:
        r = requests.get(
            JAO_BASE,
            params={"FromUtc": from_utc, "ToUtc": to_utc},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            logger.warning(f"JAO error {r.status_code} para {date.date()}")
            return None

        data = r.json().get("data", [])
        if not data:
            return None

        df = pd.DataFrame(data)
        df["dateTimeUtc"] = pd.to_datetime(df["dateTimeUtc"], utc=True).dt.tz_localize(None)
        df = df.rename(columns={
            "dateTimeUtc": "timestamp",
            "finalNtc_ES_FR": "ntc_es_fr",
            "finalNtc_FR_ES": "ntc_fr_es",
            "finalNtc_ES_PT": "ntc_es_pt",
            "finalNtc_PT_ES": "ntc_pt_es",
            "antc_ES_FR": "antc_es_fr",
            "antc_FR_ES": "antc_fr_es",
        })
        cols = ["timestamp", "ntc_es_fr", "ntc_fr_es", "ntc_es_pt", "ntc_pt_es",
                "antc_es_fr", "antc_fr_es"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        logger.error(f"Error descargando JAO {date.date()}: {e}")
        return None


def download_jao_range(start: str, end: str, output_dir: Path) -> pd.DataFrame:
    """
    Descarga NTC horarios para el rango [start, end] desde JAO SWE CCR.
    Guarda ficheros diarios en output_dir y devuelve el DataFrame completo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Ajustar al rango disponible en JAO
    effective_start = max(start_ts, JAO_START)
    if effective_start > end_ts:
        logger.info("El rango solicitado es anterior a sep-2022. Se usará ATC implícito.")
        return pd.DataFrame()

    logger.info(f"Descargando JAO NTC: {effective_start.date()} → {end_ts.date()}")

    frames = []
    current = effective_start
    total_days = (end_ts - effective_start).days + 1
    done = 0

    while current <= end_ts:
        cache_file = output_dir / f"jao_{current.strftime('%Y%m%d')}.parquet"

        if cache_file.exists():
            df_day = pd.read_parquet(cache_file)
            # Asegurar que el timestamp no tiene timezone
            if df_day["timestamp"].dt.tz is not None:
                df_day["timestamp"] = df_day["timestamp"].dt.tz_localize(None)
        else:
            df_day = fetch_jao_day(current)
            if df_day is not None and not df_day.empty:
                df_day.to_parquet(cache_file, index=False)
            time.sleep(0.3)  # Respetar rate limit JAO (100 req/min)

        if df_day is not None and not df_day.empty:
            frames.append(df_day)

        done += 1
        if done % 30 == 0:
            pct = done / total_days * 100
            logger.info(f"  JAO: {done}/{total_days} días ({pct:.0f}%)")

        current += timedelta(days=1)

    if not frames:
        logger.warning("No se obtuvieron datos de JAO.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"JAO descargado: {len(df)} registros horarios.")
    return df


def compute_implicit_atc(df_prices: pd.DataFrame,
                          price_threshold: float = 0.5) -> pd.DataFrame:
    """
    Calcula el ATC implícito a partir de los precios day-ahead ES y FR de OMIE.

    Lógica económica:
    - Si |precio_ES - precio_FR| > threshold → interconexión saturada → ATC_implícito = 0
    - Si |precio_ES - precio_FR| <= threshold → interconexión no saturada → ATC_implícito = 1 (capacidad disponible)

    Esta es una variable binaria de congestión, no un valor en MW.
    Se usa para el período 2019 – agosto 2022 donde JAO no tiene datos.

    Parámetros
    ----------
    df_prices : DataFrame con columnas 'timestamp', 'price_es', 'price_fr'
    price_threshold : umbral en €/MWh para considerar congestión (default 0.5)
    """
    df = df_prices[["timestamp", "price_es", "price_fr"]].copy()
    df["spread_abs"] = (df["price_es"] - df["price_fr"]).abs()
    df["atc_implicit"] = (df["spread_abs"] <= price_threshold).astype(int)
    df["atc_congestion"] = (df["spread_abs"] > price_threshold).astype(int)

    # ATC_Fatigue: horas consecutivas con congestión (señal de tensión sostenida)
    df["atc_fatigue"] = (
        df["atc_congestion"]
        .groupby((df["atc_congestion"] != df["atc_congestion"].shift()).cumsum())
        .cumsum()
        * df["atc_congestion"]
    )

    return df[["timestamp", "atc_implicit", "atc_congestion", "atc_fatigue"]]


def build_atc_series(start: str, end: str,
                     jao_dir: Path,
                     df_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construye la serie completa de ATCs para el período [start, end]:
    - 2019 – ago 2022: ATC implícito calculado desde precios OMIE
    - sep 2022 – presente: NTC real desde JAO SWE CCR

    Parámetros
    ----------
    start, end : fechas en formato 'YYYY-MM-DD'
    jao_dir : directorio donde guardar/leer los ficheros JAO
    df_prices : DataFrame con precios horarios ES y FR (necesario para período pre-JAO)
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    frames = []

    # --- Bloque 1: ATC implícito (2019 – ago 2022) ---
    pre_jao_end = min(end_ts, JAO_START - timedelta(hours=1))
    if start_ts < JAO_START and df_prices is not None:
        logger.info(f"Calculando ATC implícito: {start_ts.date()} → {pre_jao_end.date()}")
        mask = (df_prices["timestamp"] >= start_ts) & (df_prices["timestamp"] <= pre_jao_end)
        df_pre = compute_implicit_atc(df_prices[mask].copy())
        # Añadir columnas NTC como NaN para consistencia
        df_pre["ntc_es_fr"] = np.nan
        df_pre["ntc_fr_es"] = np.nan
        df_pre["atc_source"] = "implicit"
        frames.append(df_pre)

    # --- Bloque 2: NTC real JAO (sep 2022 – presente) ---
    jao_start = max(start_ts, JAO_START)
    if jao_start <= end_ts:
        df_jao = download_jao_range(str(jao_start.date()), str(end_ts.date()), jao_dir)
        if not df_jao.empty:
            # Calcular ATC_congestion y ATC_fatigue desde NTC real
            # Congestión cuando NTC < capacidad instalada nominal (3700 MW para ES-FR)
            NTC_NOMINAL = 3700  # MW, capacidad instalada interconexión ES-FR
            df_jao["atc_implicit"] = (df_jao["ntc_es_fr"] >= NTC_NOMINAL * 0.95).astype(int)
            df_jao["atc_congestion"] = (df_jao["ntc_es_fr"] < NTC_NOMINAL * 0.90).astype(int)
            df_jao["atc_fatigue"] = (
                df_jao["atc_congestion"]
                .groupby((df_jao["atc_congestion"] != df_jao["atc_congestion"].shift()).cumsum())
                .cumsum()
                * df_jao["atc_congestion"]
            )
            df_jao["atc_source"] = "jao_real"
            frames.append(df_jao)

    if not frames:
        logger.error("No se pudo construir ninguna serie de ATCs.")
        return pd.DataFrame()

    df_full = pd.concat(frames, ignore_index=True)
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)

    n_real = (df_full["atc_source"] == "jao_real").sum()
    n_impl = (df_full["atc_source"] == "implicit").sum()
    logger.info(f"Serie ATC completa: {len(df_full)} registros "
                f"({n_real} JAO real, {n_impl} implícito)")

    return df_full


if __name__ == "__main__":
    # Test rápido: descargar 3 días de JAO
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        df = download_jao_range("2024-01-01", "2024-01-03", Path(tmp))
        if not df.empty:
            print(df.head(6).to_string())
            print(f"\nColumnas: {list(df.columns)}")
            print(f"NTC ES→FR rango: {df['ntc_es_fr'].min():.0f} – {df['ntc_es_fr'].max():.0f} MW")
