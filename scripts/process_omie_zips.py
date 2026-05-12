"""
process_omie_zips.py
--------------------
Procesa los ZIPs anuales de OMIE (2019-2022) y los convierte
a ficheros parquet diarios en el mismo formato que el downloader online.

Los ZIPs ya están descargados en data/raw/omie_zips/
Este script los extrae, parsea y guarda en data/raw/omie/ como parquet.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OMIE-ZIP] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
ZIPS_DIR = BASE_DIR / "data" / "raw" / "omie_zips"
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "omie"


def parse_omie_content(content: str, source: str = "") -> pd.DataFrame | None:
    """Parsea el contenido de un fichero OMIE day-ahead."""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    records = []
    for line in lines:
        parts = [p.strip() for p in line.split(";")]
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
    return df


def process_zip(zip_path: Path, output_dir: Path) -> int:
    """
    Extrae y parsea todos los ficheros de un ZIP anual de OMIE.
    Guarda un parquet por día en output_dir.
    Devuelve el número de días procesados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    logger.info(f"Procesando {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = [f for f in zf.namelist() if f.endswith('.1') and 'marginalpdbc_' in f]
        file_list.sort()

        for fname in file_list:
            # Extraer fecha del nombre del fichero: marginalpdbc_YYYYMMDD.1
            try:
                date_str = fname.split('_')[1].split('.')[0]  # YYYYMMDD
                cache_file = output_dir / f"da_{date_str}.parquet"

                if cache_file.exists():
                    skipped += 1
                    continue

                content = zf.read(fname).decode('latin-1')
                df = parse_omie_content(content, fname)

                if df is not None and not df.empty:
                    # Mantener solo 24 horas del día correcto
                    target_date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
                    df = df[df["timestamp"].dt.date == target_date.date()].head(24)
                    if not df.empty:
                        df.to_parquet(cache_file, index=False)
                        processed += 1

            except Exception as e:
                logger.warning(f"Error procesando {fname}: {e}")
                continue

    logger.info(f"  {zip_path.name}: {processed} días procesados, {skipped} ya en caché")
    return processed


def process_all_zips(years: list[int] = None) -> None:
    """Procesa todos los ZIPs anuales disponibles."""
    if years is None:
        years = [2019, 2020, 2021, 2022]

    total = 0
    for year in years:
        zip_path = ZIPS_DIR / f"marginalpdbc_{year}.zip"
        if not zip_path.exists():
            logger.warning(f"ZIP no encontrado: {zip_path}")
            continue
        n = process_zip(zip_path, OUTPUT_DIR)
        total += n

    logger.info(f"\nTotal días procesados de ZIPs: {total}")
    logger.info(f"Ficheros parquet en {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('da_*.parquet')))}")


def parse_omie_id_content(content: str) -> pd.DataFrame | None:
    """Parsea el contenido de un fichero OMIE intradiario continuo."""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    records = []
    for line in lines:
        parts = [p.strip() for p in line.split(";")]
        # Formato: YYYY;M;D;Hora;MaxES;MaxPT;MaxMO;MinES;MinPT;MinMO;MedioES;MedioPT;MedioMO;
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
                        # IMPORTANTE: intradiario MIBEL = España-Portugal, no Francia.
                        "price_id_pt": price_id_pt,
                    })
            except (ValueError, IndexError):
                continue
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def process_id_zip(zip_path: Path, output_dir: Path) -> int:
    """Procesa un ZIP anual del intradiario continuo de OMIE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    logger.info(f"Procesando intradiario {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = [f for f in zf.namelist() if f.endswith('.1') and 'precios_pibcic_' in f]
        file_list.sort()
        for fname in file_list:
            try:
                date_str = fname.split('_')[-1].split('.')[0]  # YYYYMMDD
                cache_file = output_dir / f"id_{date_str}.parquet"
                if cache_file.exists():
                    skipped += 1
                    continue
                content = zf.read(fname).decode('latin-1')
                df = parse_omie_id_content(content)
                if df is not None and not df.empty:
                    target_date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
                    df = df[df["timestamp"].dt.date == target_date.date()].head(24)
                    if not df.empty:
                        df.to_parquet(cache_file, index=False)
                        processed += 1
            except Exception as e:
                logger.warning(f"Error procesando {fname}: {e}")
                continue
    logger.info(f"  {zip_path.name}: {processed} días procesados, {skipped} ya en caché")
    return processed


if __name__ == "__main__":
    # Procesar day-ahead
    process_all_zips([2019, 2020, 2021, 2022])
    # Procesar intradiario
    for year in [2019, 2020, 2021, 2022]:
        zip_path = ZIPS_DIR / f"precios_pibcic_{year}.zip"
        if zip_path.exists():
            process_id_zip(zip_path, OUTPUT_DIR)
        else:
            logger.warning(f"ZIP intradiario no encontrado: {zip_path}")
    # Verificar
    da_files = sorted(OUTPUT_DIR.glob("da_*.parquet"))
    id_files = sorted(OUTPUT_DIR.glob("id_*.parquet"))
    print(f"\nDA: {len(da_files)} ficheros ({da_files[0].name} → {da_files[-1].name})")
    print(f"ID: {len(id_files)} ficheros ({id_files[0].name} → {id_files[-1].name})")
