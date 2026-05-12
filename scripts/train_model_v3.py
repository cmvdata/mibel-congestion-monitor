"""
train_model_v3.py
-----------------
Modelo A reentrenado SIN features con leakage y CON régimes corregidos.

Cambios respecto a v2:
1. Elimina atc_implicit, atc_congestion, atc_fatigue, atc_fatigue_lag24h del
   feature set. Estas columnas son transformaciones deterministas del target
   (spread_da) durante 2019 - Sep 2022 (ver scripts/jao_downloader.py:146-158).
2. Conserva ntc_es_fr / ntc_fr_es solo desde 2022-09 (datos JAO observados).
   Para fechas anteriores imputa con la mediana del periodo post-Sep-2022 y
   añade el flag binario ntc_is_observed (1=JAO real, 0=imputado) como feature.
3. Régimes corregidos (RD-Ley 10/2022 entró en vigor el 15-jun-2022):
       pre_crisis        : 2019-01-01 → 2021-12-31
       crisis_y_excepcion: 2022-01-01 → 2023-12-31
       post_excepcion    : 2024-01-01 → 2024-12-31
4. Genera el dataset parquet desde el CSV existente al inicio (los scripts
   originales esperaban un parquet en data/processed/ que no se commiteaba).

Salidas:
    data/processed/mibel_dataset_20190101_20241231.parquet
    models/xgb_model_A_v3.json
    data/processed/residuals_v3.parquet
    results/model_metrics_v3.txt
    results/ablation_table_v3.csv
    results/comparison_v2_vs_v3.csv          — Tabla 1 del reporte
    plots/05_model_performance_v3.png
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score, f1_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
for d in [PROC_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CSV_INPUT = DATA_DIR / "mibel_dataset_20190101_20241231.csv"
PARQUET_OUT = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

# ── Régimes corregidos ──────────────────────────────────────────────────────
REGIME_BOUNDS = [
    ("pre_crisis",         pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31 23:00")),
    ("crisis_y_excepcion", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31 23:00")),
    ("post_excepcion",     pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:00")),
]
JAO_START = pd.Timestamp("2022-09-01")  # Primer día con NTC observado en JAO

# ── Features (sin leakage ATC) ──────────────────────────────────────────────
FEATURES_A_CLEAN = [
    # Lags del spread DA
    "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
    "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
    "spread_da_lag48h", "spread_da_lag168h",
    "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    # Fundamentales energéticos
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    # Capacidad real (solo desde 2022-09; imputada antes)
    "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
    # Calendario
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
]

TARGET = "spread_da"
TAU = 2.0  # Umbral de "hora extrema" para Recall@extremo


# ════════════════════════════════════════════════════════════════════════════
#  Carga, conversión CSV→parquet y feature engineering
# ════════════════════════════════════════════════════════════════════════════

def assign_regime(ts: pd.Series) -> pd.Series:
    out = pd.Series(["unknown"] * len(ts), index=ts.index, dtype=object)
    for name, lo, hi in REGIME_BOUNDS:
        mask = (ts >= lo) & (ts <= hi)
        out.loc[mask] = name
    return out


def load_and_prepare(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """Carga el CSV, regenera el parquet, sobrescribe régimes y construye features
    derivadas. Imputa NTC pre-Sep-2022 con la mediana del periodo observado.
    """
    print(f"  Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Régimes corregidos ──────────────────────────────────────────────────
    df["regime"] = assign_regime(df["timestamp"])

    # ── Imputar TTF/CO2/spark spread (gaps en festivos sin cotización) ─────
    for col in ["ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
                "ttf_lag24h", "ttf_lag168h", "gas_cost_mwh", "co2_cost_mwh"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=72)

    # ── NTC: imputar pre-Sep-2022 con mediana del periodo JAO observado ────
    obs_mask = df["timestamp"] >= JAO_START
    df["ntc_is_observed"] = obs_mask.astype(int)

    # Imputación causal: mediana de los primeros 90 días observados
    # (Sep-Dic 2022) en lugar de la mediana del periodo completo, que
    # arrastraría información de 2024 hacia 2019-2022.
    early_window = (df["timestamp"] >= JAO_START) & (
        df["timestamp"] < JAO_START + pd.Timedelta(days=90)
    )

    for col in ["ntc_es_fr", "ntc_fr_es"]:
        if col not in df.columns:
            df[col] = np.nan
        median_early = df.loc[early_window, col].median()
        if pd.isna(median_early):
            median_early = 0.0
        df.loc[~obs_mask, col] = median_early
        df[col] = df[col].ffill().fillna(median_early)

    # ── Asegurar lags de spread (por si el CSV no los tiene completos) ─────
    if "spread_da_lag1h" not in df.columns:
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"spread_da_lag{lag}h"] = df["spread_da"].shift(lag)
    # shift(1) antes de rolling: la ventana NO incluye t (que es el target).
    # Sin shift, spread_da_ma24h[t] contiene spread_da[t] = leakage estructural.
    # NOTA: forzamos recálculo siempre (sobreescribiendo si vienen del CSV)
    # porque el CSV histórico contiene estas columnas pre-calculadas SIN shift
    # (residuo de v1/v2). Si el guard `if not in df.columns` se mantiene, el
    # fix A2 queda inerte y se sigue usando la versión sin shift del CSV.
    df["spread_da_ma24h"] = (
        df["spread_da"].shift(1).rolling(24, min_periods=12).mean()
    )
    df["spread_da_ma168h"] = (
        df["spread_da"].shift(1).rolling(168, min_periods=72).mean()
    )
    df["spread_da_std24h"] = (
        df["spread_da"].shift(1).rolling(24, min_periods=12).std()
    )

    # ── Forward fill sobre lags y medias (≤24h) ────────────────────────────
    lag_cols = [c for c in df.columns
                if any(s in c for s in ("lag", "_ma", "std", "volatility"))]
    df[lag_cols] = df[lag_cols].ffill(limit=24)

    # ── Drop filas con target o lags clave nulos ───────────────────────────
    df = df.dropna(subset=[TARGET, "spread_da_lag1h", "spread_da_lag24h"]).reset_index(drop=True)

    # ── Guardar parquet (para reuso por scripts downstream) ────────────────
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet generado: {parquet_path} ({len(df):,} filas, {len(df.columns)} cols)")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  Métricas
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label="", tau=TAU):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    y_true_bin = (np.abs(y_true) > tau).astype(int)
    y_pred_bin = (np.abs(y_pred) > tau).astype(int)
    if y_true_bin.sum() > 0:
        recall_extreme = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        f1_extreme = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    else:
        recall_extreme = 0.0
        f1_extreme = 0.0

    extreme_mask = np.abs(y_true) > tau
    mae_extreme = (float(mean_absolute_error(y_true[extreme_mask], y_pred[extreme_mask]))
                   if extreme_mask.sum() > 0 else float("nan"))

    return {
        "label": label,
        "n": int(len(y_true)),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "recall_extreme": round(recall_extreme, 4),
        "f1_extreme": round(f1_extreme, 4),
        "mae_extreme": round(mae_extreme, 4) if not np.isnan(mae_extreme) else None,
        "n_extreme": int(extreme_mask.sum()),
    }


def benchmark_persistence(df_val):
    return df_val["spread_da_lag24h"].fillna(0).values


def benchmark_zero(df_val):
    return np.zeros(len(df_val))


# ════════════════════════════════════════════════════════════════════════════
#  Entrenamiento XGBoost
# ════════════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, X_val=None, y_val=None, seed=42):
    params = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )
    model = xgb.XGBRegressor(**params)
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def expanding_window(df: pd.DataFrame, features: list[str], label: str = "A_v3"):
    """Tres pasos expanding-window con cortes alineados a regímenes corregidos.
    El sistema de alertas se alimenta de los residuos out-of-sample concatenados
    de los tres pasos, deduplicados por timestamp (manteniendo la predicción del
    paso con training más amplio en caso de solape).
    """
    print(f"\n{'='*64}\n  Expanding-window — {label}\n{'='*64}")
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠ Features no disponibles: {missing}")
    print(f"  Features activas: {len(available)}")

    all_results: dict = {}
    all_residuals: list[pd.DataFrame] = []

    # Paso 1: pre_crisis → crisis_y_excepcion
    print("\n  [1/3] pre_crisis → crisis_y_excepcion")
    tr = df["regime"] == "pre_crisis"
    va = df["regime"] == "crisis_y_excepcion"
    Xtr, ytr = df.loc[tr, available].fillna(0), df.loc[tr, TARGET]
    Xva, yva = df.loc[va, available].fillna(0), df.loc[va, TARGET]
    m1 = train_xgb(Xtr, ytr, Xva, yva)
    p1 = m1.predict(Xva)
    met1 = compute_metrics(yva.values, p1, f"{label}: pre→crisis_exc")
    all_results["step1_pre_to_exc"] = {
        "model": met1,
        "M0":  compute_metrics(yva.values, benchmark_persistence(df.loc[va]), "M0"),
        "M0b": compute_metrics(yva.values, benchmark_zero(df.loc[va]), "M0b"),
    }
    all_residuals.append(pd.DataFrame({
        "timestamp": df.loc[va, "timestamp"].values,
        "regime": "crisis_y_excepcion",
        "spread_observed": yva.values,
        "spread_predicted": p1,
        "residual": yva.values - p1,
        "wf_step": 1,
    }))
    print(f"    RMSE {met1['rmse']:.3f} | MAE {met1['mae']:.3f} | R² {met1['r2']:.3f} "
          f"| Recall@ext {met1['recall_extreme']:.3f}")

    # Paso 2: pre + exc → post_excepcion
    print("\n  [2/3] pre+crisis_exc → post_excepcion")
    tr2 = df["regime"].isin(["pre_crisis", "crisis_y_excepcion"])
    va2 = df["regime"] == "post_excepcion"
    Xtr2, ytr2 = df.loc[tr2, available].fillna(0), df.loc[tr2, TARGET]
    Xva2, yva2 = df.loc[va2, available].fillna(0), df.loc[va2, TARGET]
    m2 = train_xgb(Xtr2, ytr2, Xva2, yva2)
    p2 = m2.predict(Xva2)
    met2 = compute_metrics(yva2.values, p2, f"{label}: exc→post")
    all_results["step2_exc_to_post"] = {
        "model": met2,
        "M0":  compute_metrics(yva2.values, benchmark_persistence(df.loc[va2]), "M0"),
        "M0b": compute_metrics(yva2.values, benchmark_zero(df.loc[va2]), "M0b"),
    }
    all_residuals.append(pd.DataFrame({
        "timestamp": df.loc[va2, "timestamp"].values,
        "regime": "post_excepcion",
        "spread_observed": yva2.values,
        "spread_predicted": p2,
        "residual": yva2.values - p2,
        "wf_step": 2,
    }))
    print(f"    RMSE {met2['rmse']:.3f} | MAE {met2['mae']:.3f} | R² {met2['r2']:.3f} "
          f"| Recall@ext {met2['recall_extreme']:.3f}")

    # Paso 3: train 2019-2023 → test 2024 (para benchmarks T5)
    print("\n  [3/3] train 2019-2023 → test 2024")
    tr3 = df["timestamp"] < pd.Timestamp("2024-01-01")
    va3 = df["timestamp"] >= pd.Timestamp("2024-01-01")
    Xtr3, ytr3 = df.loc[tr3, available].fillna(0), df.loc[tr3, TARGET]
    Xva3, yva3 = df.loc[va3, available].fillna(0), df.loc[va3, TARGET]
    m3 = train_xgb(Xtr3, ytr3, Xva3, yva3)
    p3 = m3.predict(Xva3)
    met3 = compute_metrics(yva3.values, p3, f"{label}: train2023→test2024")
    all_results["step3_train2023_test2024"] = {
        "model": met3,
        "M0":  compute_metrics(yva3.values, benchmark_persistence(df.loc[va3]), "M0"),
        "M0b": compute_metrics(yva3.values, benchmark_zero(df.loc[va3]), "M0b"),
    }
    print(f"    RMSE {met3['rmse']:.3f} | MAE {met3['mae']:.3f} | R² {met3['r2']:.3f} "
          f"| Recall@ext {met3['recall_extreme']:.3f}")

    # Concatenar residuos y deduplicar 2024 (manteniendo paso 2, training más amplio
    # post_excepcion expandido). El paso 3 con test 2024 se guarda en otro archivo
    # para análisis de benchmarks.
    df_residuals = pd.concat(all_residuals, ignore_index=True)
    df_residuals = df_residuals.sort_values(["timestamp", "wf_step"]).reset_index(drop=True)
    # Deduplicar manteniendo el menor wf_step (paso 1 o 2; paso 3 no entra aquí)
    df_residuals = df_residuals.drop_duplicates(subset=["timestamp"], keep="first")

    # Guardar predicciones del paso 3 por separado (para T5)
    df_test2024 = pd.DataFrame({
        "timestamp": df.loc[va3, "timestamp"].values,
        "regime": "post_excepcion",
        "spread_observed": yva3.values,
        "spread_predicted_modelA": p3,
    })

    return all_results, df_residuals, df_test2024, m3, available


# ════════════════════════════════════════════════════════════════════════════
#  Tabla 1 (comparación v2 vs v3)
# ════════════════════════════════════════════════════════════════════════════

V2_REPORTED = {
    # Métricas reportadas en el PDF original (Modelo A baseline limpio)
    "step1_pre_to_exc":          {"rmse": 2.863, "mae": 0.784, "r2": 0.635, "recall_extreme": 0.701, "source": "PDF v1"},
    "step3_train2023_test2024":  {"rmse": None,  "mae": None,  "r2": None,  "recall_extreme": None,  "source": "PDF v1 no detalla paso 3 separado"},
}


def build_comparison_v2_vs_v3(results_v3: dict) -> pd.DataFrame:
    rows = []
    for step, ref in V2_REPORTED.items():
        m = results_v3[step]["model"]
        rows.append({
            "step": step,
            "version": "v2_original (con leakage ATC)",
            "rmse": ref["rmse"], "mae": ref["mae"], "r2": ref["r2"],
            "recall_extreme": ref["recall_extreme"],
            "note": ref["source"],
        })
        rows.append({
            "step": step,
            "version": "v3_clean (sin leakage ATC)",
            "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
            "recall_extreme": m["recall_extreme"],
            "note": "Reentrenamiento con FEATURES_A_CLEAN",
        })
    return pd.DataFrame(rows)


def plot_performance(df_residuals: pd.DataFrame, results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("MIBEL Congestion Monitor v3 — Modelo A reentrenado (sin leakage)",
                 fontsize=13, fontweight="bold")
    colors = {"crisis_y_excepcion": "#E74C3C", "post_excepcion": "#3498DB"}
    ax = axes[0]
    for r, g in df_residuals.groupby("regime"):
        ax.scatter(g["timestamp"], g["residual"], alpha=0.3, s=2,
                   color=colors.get(r, "gray"), label=r)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(5, color="orange", lw=1, ls="--", label="±5 €/MWh")
    ax.axhline(-5, color="orange", lw=1, ls="--")
    ax.set_title("Residuos out-of-sample concatenados (pasos 1+2 expanding-window)")
    ax.set_ylabel("Residuo (€/MWh)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    steps = list(results.keys())
    rmse_model = [results[s]["model"]["rmse"] for s in steps]
    rmse_m0 = [results[s]["M0"]["rmse"] for s in steps]
    rmse_m0b = [results[s]["M0b"]["rmse"] for s in steps]
    x = np.arange(len(steps))
    w = 0.27
    ax2.bar(x - w, rmse_model, w, label="Modelo A v3", color="#2ECC71")
    ax2.bar(x,     rmse_m0,    w, label="M0 (persistencia)", color="#95A5A6")
    ax2.bar(x + w, rmse_m0b,   w, label="M0b (cero)", color="#34495E")
    ax2.set_xticks(x)
    ax2.set_xticklabels(steps, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE (€/MWh)")
    ax2.set_title("RMSE por paso del expanding-window")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOTS_DIR / "05_model_performance_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot guardado: {out}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  MIBEL Congestion Monitor — Train v3 (sin leakage ATC)")
    print("=" * 64)

    df = load_and_prepare(CSV_INPUT, PARQUET_OUT)
    print("\n  Distribución por régimen:")
    for r, n in df["regime"].value_counts().sort_index().items():
        print(f"    {r:25s} : {n:>7,} horas")

    results, residuals, test2024, model_final, features_used = expanding_window(
        df, FEATURES_A_CLEAN, label="A_v3"
    )

    # Guardar artefactos
    residuals.to_parquet(PROC_DIR / "residuals_v3.parquet", index=False)
    test2024.to_parquet(PROC_DIR / "predictions_test2024_v3.parquet", index=False)
    model_final.save_model(str(MODELS_DIR / "xgb_model_A_v3.json"))
    pd.Series(features_used).to_csv(MODELS_DIR / "xgb_model_A_v3_features.csv",
                                     header=["feature"], index=False)
    print(f"\n  Residuos guardados: {PROC_DIR / 'residuals_v3.parquet'}")
    print(f"  Predicciones test 2024 (modelo paso 3): {PROC_DIR / 'predictions_test2024_v3.parquet'}")
    print(f"  Modelo final: {MODELS_DIR / 'xgb_model_A_v3.json'}")

    # Tabla de ablation v3
    rows = []
    for step, res in results.items():
        rows.append({"step": step, "spec": "A_v3_clean", **res["model"]})
        rows.append({"step": step, "spec": "M0_persistence", **res["M0"]})
        rows.append({"step": step, "spec": "M0b_zero", **res["M0b"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation_table_v3.csv", index=False)

    # Tabla 1 comparación v2 vs v3
    cmp = build_comparison_v2_vs_v3(results)
    cmp.to_csv(RESULTS_DIR / "comparison_v2_vs_v3.csv", index=False)
    print("\n" + "=" * 64)
    print("  Tabla 1 — Modelo A original (v2) vs reentrenado (v3)")
    print("=" * 64)
    print(cmp.to_string(index=False))

    # Métricas detalladas
    with open(RESULTS_DIR / "model_metrics_v3.txt", "w", encoding="utf-8") as f:
        f.write("MIBEL Congestion Monitor v3 — Métricas modelo A reentrenado\n")
        f.write("=" * 64 + "\n")
        f.write(f"Features ({len(features_used)}): {features_used}\n\n")
        for step, res in results.items():
            f.write(f"\n[{step}]\n")
            for k, v in res.items():
                f.write(f"  {k}: {v}\n")

    plot_performance(residuals, results)

    print("\n" + "=" * 64)
    print("  T1 + T3 COMPLETADAS")
    print("=" * 64)


if __name__ == "__main__":
    main()
