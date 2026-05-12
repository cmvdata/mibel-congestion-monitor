"""
omie_downloader.py
------------------
Descarga precios horarios del mercado diario (day-ahead) de España y Francia
desde OMIE (Operador del Mercado Ibérico de Energía).

Fuente: https://www.omie.es/es/file-access-list
Fichero: marginalpdbc_YYYYMMDD.1
Formato: CSV con separador ';', sin cabecera estándar
Resolución: Horaria (24 valores por día)
Cobertura: 2019 – presente
NO requiere token.

También descarga precios del mercado intradiario continuo (XBID):
Fichero: precios_pibcic_YYYYMMDD.1
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OMIE] %(message)s")
logger = logging.getLogger(__name__)

OMIE_DA_URL = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename={filename}"
OMIE_ID_URL = "https://www.omie.es/es/file-download?parents=precios_pibcic&filename={filename}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.omie.es/",
}


def fetch_omie_da_day(date: pd.Timestamp) -> pd.DataFrame | None:
    """
    Descarga los precios day-ahead (marginalpdbc) de España y Francia para un día.
    Devuelve DataFrame con columnas: timestamp, price_es, price_fr
    """
    filename = f"marginalpdbc_{date.strftime('%Y%m%d')}.1"
    url = OMIE_DA_URL.format(filename=filename)

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None

        content = r.content.decode("latin-1")
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        records = []
        for line in lines:
            parts = [p.strip() for p in line.split(";")]
            # Formato real OMIE: YYYY;MM;DD;hora;precio_es;precio_fr;
            # Ejemplo: 2024;01;01;1;63.33;63.33;
            # Primera línea es cabecera: MARGINALPDBC;
            if len(parts) >= 6:
                try:
                    year_s, month_s, day_s = parts[0], parts[1], parts[2]
                    hour = int(parts[3])
                    price_es = float(parts[4].replace(",", "."))
                    price_fr = float(parts[5].replace(",", "."))
                    if (year_s.isdigit() and month_s.isdigit() and
                            day_s.isdigit() and 1 <= hour <= 25):
                        ts = pd.Timestamp(
                            f"{year_s}-{month_s.zfill(2)}-{day_s.zfill(2)}"
                        ) + pd.Timedelta(hours=hour - 1)
                        records.append({
                            "timestamp": ts,
                            "price_es": price_es,
                            "price_fr": price_fr,
                        })
                except (ValueError, IndexError):
                    continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        # Mantener solo 24 horas (eliminar hora 25 de cambio de horario)
        df = df[df["timestamp"].dt.date == date.date()].head(24)
        return df if not df.empty else None

    except Exception as e:
        logger.error(f"Error OMIE DA {date.date()}: {e}")
        return None


def fetch_omie_id_day(date: pd.Timestamp) -> pd.DataFrame | None:
    """
    Descarga el precio medio ponderado del mercado intradiario continuo (XBID)
    para un día. Devuelve DataFrame con columnas: timestamp, price_id_es, price_id_pt.

    NOTA: el mercado intradiario continuo de MIBEL es España-Portugal (NO Francia).
    Histórico: este campo se llamaba erróneamente `price_id_fr` en v1/v2.
    """
    filename = f"precios_pibcic_{date.strftime('%Y%m%d')}.1"
    url = OMIE_ID_URL.format(filename=filename)

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None

        content = r.content.decode("latin-1")
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        records = []
        for line in lines:
            parts = [p.strip() for p in line.split(";")]
            # Formato real OMIE intradiario:
            # YYYY;M;D;Hora;MaxES;MaxPT;MaxMO;MinES;MinPT;MinMO;MedioES;MedioPT;MedioMO;
            # Usamos MedioES (col 10) y MedioPT (col 11) como precio representativo
            if len(parts) >= 12:
                try:
                    year_s, month_s, day_s = parts[0], parts[1], parts[2]
                    hour = int(parts[3])
                    price_id_es = float(parts[10].replace(",", "."))
                    price_id_pt = float(parts[11].replace(",", "."))
                    if (year_s.isdigit() and month_s.isdigit() and
                            day_s.isdigit() and 1 <= hour <= 25):
                        ts = pd.Timestamp(
                            f"{year_s}-{month_s.zfill(2)}-{day_s.zfill(2)}"
                        ) + pd.Timedelta(hours=hour - 1)
                        records.append({
                            "timestamp": ts,
                            "price_id_es": price_id_es,
                            # IMPORTANTE: el intradiario MIBEL es España-Portugal,
                            # NO España-Francia. El nombre histórico price_id_fr era
                            # engañoso. Renombrado a price_id_pt.
                            "price_id_pt": price_id_pt,
                        })
                except (ValueError, IndexError):
                    continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df[df["timestamp"].dt.date == date.date()].head(24)
        return df if not df.empty else None

    except Exception as e:
        logger.error(f"Error OMIE ID {date.date()}: {e}")
        return None


def download_omie_range(start: str, end: str, output_dir: Path,
                        include_intraday: bool = True) -> pd.DataFrame:
    """
    Descarga precios horarios DA e intradiario para el rango [start, end].
    Guarda ficheros diarios en caché y devuelve el DataFrame completo.

    Columnas resultado:
    - timestamp: hora UTC (naive, hora peninsular española)
    - price_es: precio day-ahead España (€/MWh)
    - price_fr: precio day-ahead Francia (€/MWh)
    - spread_da: price_es - price_fr (spread day-ahead)
    - price_id_es: precio intradiario España (€/MWh, puede ser NaN)
    - price_id_pt: precio intradiario Portugal (€/MWh, puede ser NaN)
                   (Histórico: campo erróneamente llamado price_id_fr en v1/v2.)
    - spread_id_es_pt: price_id_es - price_id_pt (spread intradiario ES-PT)
    - spread_da_id: spread_da - spread_id_es_pt (diferencial DA vs ID)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    total_days = (end_ts - start_ts).days + 1

    logger.info(f"Descargando OMIE: {start_ts.date()} → {end_ts.date()} ({total_days} días)")

    frames_da = []
    frames_id = []
    missing_da = []
    missing_id = []
    current = start_ts
    done = 0

    while current <= end_ts:
        date_str = current.strftime("%Y%m%d")
        cache_da = output_dir / f"da_{date_str}.parquet"
        cache_id = output_dir / f"id_{date_str}.parquet"

        # Day-ahead
        if cache_da.exists():
            df_da = pd.read_parquet(cache_da)
        else:
            df_da = fetch_omie_da_day(current)
            if df_da is not None:
                df_da.to_parquet(cache_da, index=False)
            else:
                missing_da.append(current.date())
            time.sleep(0.2)

        if df_da is not None and not df_da.empty:
            frames_da.append(df_da)

        # Intradiario
        if include_intraday:
            if cache_id.exists():
                df_id = pd.read_parquet(cache_id)
            else:
                df_id = fetch_omie_id_day(current)
                if df_id is not None:
                    df_id.to_parquet(cache_id, index=False)
                else:
                    missing_id.append(current.date())
                time.sleep(0.2)

            if df_id is not None and not df_id.empty:
                frames_id.append(df_id)

        done += 1
        if done % 60 == 0:
            pct = done / total_days * 100
            logger.info(f"  OMIE: {done}/{total_days} días ({pct:.0f}%) | "
                        f"Huecos DA: {len(missing_da)}, ID: {len(missing_id)}")

        current += timedelta(days=1)

    if not frames_da:
        logger.error("No se obtuvieron datos de OMIE day-ahead.")
        return pd.DataFrame()

    # Ensamblar day-ahead
    df = pd.concat(frames_da, ignore_index=True)
    df["spread_da"] = df["price_es"] - df["price_fr"]

    # Ensamblar intradiario y hacer merge
    if frames_id:
        df_id_all = pd.concat(frames_id, ignore_index=True)
        df_id_all["spread_id_es_pt"] = df_id_all["price_id_es"] - df_id_all["price_id_pt"]
        df = df.merge(df_id_all, on="timestamp", how="left")
        df["spread_da_id"] = df["spread_da"] - df["spread_id_es_pt"]
    else:
        df["price_id_es"] = np.nan
        df["price_id_pt"] = np.nan
        df["spread_id_es_pt"] = np.nan
        df["spread_da_id"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reportar huecos
    if missing_da:
        logger.warning(f"Huecos DA ({len(missing_da)} días): {missing_da[:5]}{'...' if len(missing_da)>5 else ''}")
    if missing_id:
        logger.info(f"Huecos ID ({len(missing_id)} días): primeros 5: {missing_id[:5]}")

    logger.info(f"OMIE descargado: {len(df)} registros | "
                f"Cobertura DA: {(len(df)/(total_days*24))*100:.1f}%")
    return df


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        df = download_omie_range("2024-01-01", "2024-01-07", Path(tmp))
        if not df.empty:
            print(df.head(6).to_string())
            print(f"\nSpread DA medio: {df['spread_da'].mean():.2f} €/MWh")
            print(f"Spread ID (ES-PT) medio: {df['spread_id_es_pt'].mean():.2f} €/MWh")
