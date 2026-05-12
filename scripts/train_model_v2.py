"""
train_model_v2.py
-----------------
Modelo base XGBoost LIMPIO para el MIBEL Congestion Monitor.

CAMBIOS RESPECTO A v1:
- Eliminadas spread_id y spread_da_id (leakage temporal + algebraico confirmado)
- Variables intradiarias solo como rezagos de D-1 (ex ante)
- Tres especificaciones comparativas:
    Modelo A: baseline limpio (calendario + lags + fundamentales)
    Modelo B: baseline + intradiario rezagado D-1
    Modelo C: baseline + intradiario contemporáneo (solo diagnóstico, NO para producción)
- Benchmarks M0 (persistencia: spread_t = spread_t-24) y M0b (nivel: spread_t = 0)
- Walk-forward por regímenes estricto
- Métricas orientadas a vigilancia: Recall@extremos, F1, MAE condicional

Salidas:
    models/xgb_model_A.json          — modelo A (producción)
    models/xgb_model_B.json          — modelo B (producción)
    data/processed/residuals_v2.parquet
    results/model_metrics_v2.txt
    results/ablation_table.csv        — tabla comparativa de los 3 modelos + benchmarks
    plots/05_model_performance_v2.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, recall_score

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET = DATA_DIR / "processed" / "mibel_dataset_20190101_20241231.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIÓN DE FEATURES (todas ex ante o rezagadas)
# ─────────────────────────────────────────────────────────────────────────────

# MODELO A: baseline limpio — sin ninguna variable intradiaria
FEATURES_A = [
    # Lags del spread day-ahead (información pasada, ex ante)
    "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
    "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
    "spread_da_lag48h", "spread_da_lag168h",
    # Medias móviles y volatilidad (ventanas pasadas)
    "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    # Fundamentales energéticos (diarios, ex ante)
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    # ATCs (publicados antes del cierre del DA)
    "atc_implicit", "atc_congestion", "atc_fatigue", "atc_fatigue_lag24h",
    # Calendario
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
]

# MODELO B: baseline + intradiario rezagado D-1 (ex ante, legítimo)
# Estas variables se construyen con shift(24) sobre el intradiario
FEATURES_B = FEATURES_A + [
    "spread_id_lag24h",      # spread intradiario de ayer a la misma hora
    "spread_id_lag48h",      # spread intradiario de anteayer
    "price_id_es_lag24h",    # precio intradiario ES de ayer
    "price_id_fr_lag24h",    # precio intradiario FR de ayer
    "id_volatility_d1",      # std del intradiario del día anterior
    "id_max_spread_d1",      # máximo spread intradiario del día anterior
]

# MODELO C: con intradiario contemporáneo (SOLO DIAGNÓSTICO — NO PRODUCCIÓN)
FEATURES_C = FEATURES_A + [
    "spread_id",             # spread intradiario misma hora t → LEAKAGE TEMPORAL
    "spread_da_id",          # DA - ID misma hora t → LEAKAGE ALGEBRAICO
]

TARGET = "spread_da"

REGIMES = {
    "pre_crisis":        ("2019-01-01", "2021-12-31"),
    "excepcion_iberica": ("2022-01-01", "2023-12-31"),
    "post_excepcion":    ("2024-01-01", "2024-12-31"),
}

# Umbral para definir episodio de desacoplamiento
TAU = 2.0  # €/MWh


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Carga el dataset, construye variables rezagadas de intradiario y prepara features."""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Imputar huecos TTF/CO2 (días festivos sin cotización) ──────────────
    for col in ["ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
                "ttf_lag24h", "ttf_lag168h", "gas_cost_mwh", "co2_cost_mwh"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=72)

    # ── Imputar NTC con 0 cuando no hay datos ──────────────────────────────
    for col in ["ntc_es_fr", "ntc_fr_es", "ntc_es_fr_lag24h"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Construir variables intradiarias rezagadas (D-1) ───────────────────
    # Estas son las únicas variables intradiarias permitidas en producción
    if "spread_id" in df.columns:
        df["spread_id_lag24h"] = df["spread_id"].shift(24)
        df["spread_id_lag48h"] = df["spread_id"].shift(48)

    if "price_id_es" in df.columns:
        df["price_id_es_lag24h"] = df["price_id_es"].shift(24)

    if "price_id_fr" in df.columns:
        df["price_id_fr_lag24h"] = df["price_id_fr"].shift(24)

    # Volatilidad intradiaria del día anterior (std de las 24h previas)
    if "spread_id" in df.columns:
        df["id_volatility_d1"] = df["spread_id"].shift(1).rolling(24).std()
        df["id_max_spread_d1"] = df["spread_id"].shift(1).rolling(24).max()

    # ── Rellenar lags con forward fill (máx 24h) ──────────────────────────
    lag_cols = [c for c in df.columns if "lag" in c or "ma" in c or "std" in c
                or "volatility" in c or "max_spread" in c]
    df[lag_cols] = df[lag_cols].ffill(limit=24)

    # ── Eliminar filas con target o features clave nulas ──────────────────
    df = df.dropna(subset=[TARGET, "spread_da_lag1h", "spread_da_lag24h"])

    return df


def compute_metrics(y_true, y_pred, label="", tau=TAU):
    """Calcula métricas de regresión y vigilancia."""
    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Métricas de vigilancia: detección de episodios de desacoplamiento
    y_true_bin = (np.abs(y_true) > tau).astype(int)
    y_pred_bin = (np.abs(y_pred) > tau).astype(int)

    if y_true_bin.sum() > 0:
        recall_extreme = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_extreme = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    else:
        recall_extreme = 0.0
        f1_extreme = 0.0

    # MAE condicional en episodios extremos
    extreme_mask = np.abs(y_true) > tau
    mae_extreme = mean_absolute_error(y_true[extreme_mask], y_pred[extreme_mask]) \
        if extreme_mask.sum() > 0 else np.nan

    # % de residuos grandes
    pct_large = np.mean(np.abs(residuals) > 5) * 100

    return {
        "label": label,
        "n": len(y_true),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "recall_extreme": round(recall_extreme, 4),
        "f1_extreme": round(f1_extreme, 4),
        "mae_extreme": round(mae_extreme, 4) if not np.isnan(mae_extreme) else None,
        "pct_large_residuals": round(pct_large, 2),
        "n_extreme": int(extreme_mask.sum()),
    }


def benchmark_m0(df_val):
    """M0: persistencia — spread_t = spread_t-24"""
    return df_val["spread_da_lag24h"].fillna(0).values


def benchmark_m0b(df_val):
    """M0b: nivel estructural — spread_t = 0"""
    return np.zeros(len(df_val))


def train_xgb(X_train, y_train, X_val=None, y_val=None):
    """Entrena XGBoost con early stopping."""
    params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = xgb.XGBRegressor(**params)
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def run_walk_forward(df, features, model_name):
    """Ejecuta walk-forward validation por regímenes para un conjunto de features."""
    print(f"\n{'='*60}")
    print(f"  MODELO {model_name}")
    print(f"{'='*60}")

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠ Features no disponibles (se omiten): {missing}")
    print(f"  Features activas: {len(available)}")

    all_results = {}
    all_residuals = []

    # ── Paso 1: pre_crisis → excepcion_iberica ────────────────────────────
    print(f"\n  [1/3] pre_crisis → excepcion_iberica")
    tr = df["regime"] == "pre_crisis"
    va = df["regime"] == "excepcion_iberica"

    X_tr, y_tr = df[tr][available].fillna(0), df[tr][TARGET]
    X_va, y_va = df[va][available].fillna(0), df[va][TARGET]

    m1 = train_xgb(X_tr, y_tr, X_va, y_va)
    p1 = m1.predict(X_va)
    met1 = compute_metrics(y_va.values, p1, f"{model_name}: pre→exc")

    # Benchmarks
    bm0_1  = compute_metrics(y_va.values, benchmark_m0(df[va]),  f"M0: pre→exc")
    bm0b_1 = compute_metrics(y_va.values, benchmark_m0b(df[va]), f"M0b: pre→exc")

    all_results["pre_crisis→excepcion_iberica"] = {
        "model": met1, "M0": bm0_1, "M0b": bm0b_1
    }
    all_residuals.append(pd.DataFrame({
        "timestamp": df[va]["timestamp"].values,
        "regime": "excepcion_iberica",
        "spread_observed": y_va.values,
        "spread_predicted": p1,
        "residual": y_va.values - p1,
        "model": model_name,
    }))

    print(f"    RMSE {met1['rmse']:.3f} | MAE {met1['mae']:.3f} | R² {met1['r2']:.3f} "
          f"| Recall@ext {met1['recall_extreme']:.3f} | F1@ext {met1['f1_extreme']:.3f}")
    print(f"    M0:  RMSE {bm0_1['rmse']:.3f} | MAE {bm0_1['mae']:.3f} | R² {bm0_1['r2']:.3f}")
    print(f"    M0b: RMSE {bm0b_1['rmse']:.3f} | MAE {bm0b_1['mae']:.3f} | R² {bm0b_1['r2']:.3f}")

    # ── Paso 2: pre_crisis + excepcion_iberica → post_excepcion ──────────
    print(f"\n  [2/3] pre_crisis + excepcion_iberica → post_excepcion")
    tr2 = df["regime"].isin(["pre_crisis", "excepcion_iberica"])
    va2 = df["regime"] == "post_excepcion"

    X_tr2, y_tr2 = df[tr2][available].fillna(0), df[tr2][TARGET]
    X_va2, y_va2 = df[va2][available].fillna(0), df[va2][TARGET]

    m2 = train_xgb(X_tr2, y_tr2, X_va2, y_va2)
    p2 = m2.predict(X_va2)
    met2 = compute_metrics(y_va2.values, p2, f"{model_name}: exc→post")

    bm0_2  = compute_metrics(y_va2.values, benchmark_m0(df[va2]),  f"M0: exc→post")
    bm0b_2 = compute_metrics(y_va2.values, benchmark_m0b(df[va2]), f"M0b: exc→post")

    all_results["excepcion_iberica→post_excepcion"] = {
        "model": met2, "M0": bm0_2, "M0b": bm0b_2
    }
    all_residuals.append(pd.DataFrame({
        "timestamp": df[va2]["timestamp"].values,
        "regime": "post_excepcion",
        "spread_observed": y_va2.values,
        "spread_predicted": p2,
        "residual": y_va2.values - p2,
        "model": model_name,
    }))

    print(f"    RMSE {met2['rmse']:.3f} | MAE {met2['mae']:.3f} | R² {met2['r2']:.3f} "
          f"| Recall@ext {met2['recall_extreme']:.3f} | F1@ext {met2['f1_extreme']:.3f}")
    print(f"    M0:  RMSE {bm0_2['rmse']:.3f} | MAE {bm0_2['mae']:.3f} | R² {bm0_2['r2']:.3f}")
    print(f"    M0b: RMSE {bm0b_2['rmse']:.3f} | MAE {bm0b_2['mae']:.3f} | R² {bm0b_2['r2']:.3f}")

    # ── Paso 3: modelo final (todo 2019-2023 → 2024) ──────────────────────
    print(f"\n  [3/3] Modelo final: 2019-2023 → 2024 (out-of-sample)")
    tr3 = df["timestamp"] < "2024-01-01"
    va3 = df["timestamp"] >= "2024-01-01"

    X_tr3, y_tr3 = df[tr3][available].fillna(0), df[tr3][TARGET]
    X_va3, y_va3 = df[va3][available].fillna(0), df[va3][TARGET]

    m_final = train_xgb(X_tr3, y_tr3, X_va3, y_va3)
    p3 = m_final.predict(X_va3)
    met3 = compute_metrics(y_va3.values, p3, f"{model_name}: final_2024")

    bm0_3  = compute_metrics(y_va3.values, benchmark_m0(df[va3]),  f"M0: final_2024")
    bm0b_3 = compute_metrics(y_va3.values, benchmark_m0b(df[va3]), f"M0b: final_2024")

    all_results["final_2024"] = {
        "model": met3, "M0": bm0_3, "M0b": bm0b_3
    }
    all_residuals.append(pd.DataFrame({
        "timestamp": df[va3]["timestamp"].values,
        "regime": "post_excepcion",
        "spread_observed": y_va3.values,
        "spread_predicted": p3,
        "residual": y_va3.values - p3,
        "model": model_name,
    }))

    print(f"    RMSE {met3['rmse']:.3f} | MAE {met3['mae']:.3f} | R² {met3['r2']:.3f} "
          f"| Recall@ext {met3['recall_extreme']:.3f} | F1@ext {met3['f1_extreme']:.3f}")
    print(f"    M0:  RMSE {bm0_3['rmse']:.3f} | MAE {bm0_3['mae']:.3f} | R² {bm0_3['r2']:.3f}")
    print(f"    M0b: RMSE {bm0b_3['rmse']:.3f} | MAE {bm0b_3['mae']:.3f} | R² {bm0b_3['r2']:.3f}")

    df_residuals = pd.concat(all_residuals, ignore_index=True)
    return all_results, df_residuals, m_final


def build_ablation_table(results_A, results_B, results_C):
    """Construye la tabla comparativa de ablation para los 3 modelos + benchmarks."""
    rows = []
    for step in ["pre_crisis→excepcion_iberica", "excepcion_iberica→post_excepcion", "final_2024"]:
        for spec, res in [("A_clean", results_A), ("B_id_lag", results_B), ("C_id_contemp", results_C)]:
            m = res[step]["model"]
            rows.append({
                "step": step, "spec": spec,
                "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
                "recall_extreme": m["recall_extreme"], "f1_extreme": m["f1_extreme"],
                "mae_extreme": m["mae_extreme"],
            })
        # Benchmarks (solo una vez por step)
        bm0 = results_A[step]["M0"]
        bm0b = results_A[step]["M0b"]
        rows.append({
            "step": step, "spec": "M0_persistence",
            "rmse": bm0["rmse"], "mae": bm0["mae"], "r2": bm0["r2"],
            "recall_extreme": bm0["recall_extreme"], "f1_extreme": bm0["f1_extreme"],
            "mae_extreme": bm0["mae_extreme"],
        })
        rows.append({
            "step": step, "spec": "M0b_zero",
            "rmse": bm0b["rmse"], "mae": bm0b["mae"], "r2": bm0b["r2"],
            "recall_extreme": bm0b["recall_extreme"], "f1_extreme": bm0b["f1_extreme"],
            "mae_extreme": bm0b["mae_extreme"],
        })
    return pd.DataFrame(rows)


def plot_results(df_residuals_A, df_residuals_B, ablation_table):
    """Genera gráficos de rendimiento comparativo."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("MIBEL Congestion Monitor — Comparativa de Modelos (sin leakage)",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Residuos modelo A (baseline limpio) ──────────────────────
    ax1 = axes[0, 0]
    colors = {"excepcion_iberica": "#E74C3C", "post_excepcion": "#3498DB"}
    for regime, grp in df_residuals_A.groupby("regime"):
        ax1.scatter(grp["timestamp"], grp["residual"],
                    alpha=0.3, s=2, color=colors.get(regime, "gray"), label=regime)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(5, color="orange", linewidth=1, linestyle="--", label="±5 €/MWh")
    ax1.axhline(-5, color="orange", linewidth=1, linestyle="--")
    ax1.set_title("Residuos — Modelo A (baseline limpio)")
    ax1.set_ylabel("Residuo (€/MWh)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Residuos modelo B (+ intradiario laggeado) ───────────────
    ax2 = axes[0, 1]
    for regime, grp in df_residuals_B.groupby("regime"):
        ax2.scatter(grp["timestamp"], grp["residual"],
                    alpha=0.3, s=2, color=colors.get(regime, "gray"), label=regime)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(5, color="orange", linewidth=1, linestyle="--")
    ax2.axhline(-5, color="orange", linewidth=1, linestyle="--")
    ax2.set_title("Residuos — Modelo B (+ intradiario D-1)")
    ax2.set_ylabel("Residuo (€/MWh)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Ablation table — RMSE comparativo ────────────────────────
    ax3 = axes[1, 0]
    step_final = ablation_table[ablation_table["step"] == "final_2024"]
    specs = step_final["spec"].tolist()
    rmses = step_final["rmse"].tolist()
    bar_colors = ["#2ECC71" if "A_" in s else "#3498DB" if "B_" in s
                  else "#E74C3C" if "C_" in s else "#95A5A6" for s in specs]
    bars = ax3.barh(specs, rmses, color=bar_colors, alpha=0.8)
    ax3.set_title("RMSE — Validación final 2024 (out-of-sample)")
    ax3.set_xlabel("RMSE (€/MWh)")
    for bar, val in zip(bars, rmses):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")

    # ── Panel 4: Recall@extremos comparativo ─────────────────────────────
    ax4 = axes[1, 1]
    recalls = step_final["recall_extreme"].tolist()
    bars2 = ax4.barh(specs, recalls, color=bar_colors, alpha=0.8)
    ax4.set_title(f"Recall en episodios extremos (|spread| > {TAU} €/MWh)")
    ax4.set_xlabel("Recall")
    for bar, val in zip(bars2, recalls):
        ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out = PLOTS_DIR / "05_model_performance_v2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Gráfico guardado: {out}")


def main():
    print("=" * 60)
    print("  MIBEL Congestion Monitor — Entrenamiento limpio")
    print("  (sin leakage temporal ni algebraico)")
    print("=" * 60)

    print(f"\nCargando dataset: {DATASET}")
    df = load_and_prepare(DATASET)
    print(f"Dataset listo: {len(df):,} filas | {len(df.columns)} columnas")

    # Asignar régimen
    df["regime"] = "pre_crisis"
    df.loc[df["timestamp"] >= "2022-01-01", "regime"] = "excepcion_iberica"
    df.loc[df["timestamp"] >= "2024-01-01", "regime"] = "post_excepcion"

    print(f"\nDistribución por régimen:")
    for r, cnt in df["regime"].value_counts().sort_index().items():
        print(f"  {r}: {cnt:,} horas")

    # ── Entrenar los tres modelos ─────────────────────────────────────────
    results_A, residuals_A, model_A = run_walk_forward(df, FEATURES_A, "A_clean")
    results_B, residuals_B, model_B = run_walk_forward(df, FEATURES_B, "B_id_lag")
    results_C, residuals_C, model_C = run_walk_forward(df, FEATURES_C, "C_id_contemp")

    # ── Tabla de ablation ─────────────────────────────────────────────────
    ablation = build_ablation_table(results_A, results_B, results_C)
    ablation.to_csv(RESULTS_DIR / "ablation_table.csv", index=False)
    print(f"\n{'='*60}")
    print("  TABLA DE ABLATION — Validación final 2024")
    print(f"{'='*60}")
    final = ablation[ablation["step"] == "final_2024"][
        ["spec", "rmse", "mae", "r2", "recall_extreme", "f1_extreme"]
    ].to_string(index=False)
    print(final)

    # ── Guardar residuos del modelo A (producción) ────────────────────────
    residuals_A.to_parquet(DATA_DIR / "processed" / "residuals_v2.parquet", index=False)
    print(f"\n  Residuos guardados: data/processed/residuals_v2.parquet")

    # ── Guardar modelos ───────────────────────────────────────────────────
    model_A.save_model(str(MODELS_DIR / "xgb_model_A.json"))
    model_B.save_model(str(MODELS_DIR / "xgb_model_B.json"))
    print(f"  Modelos guardados: models/xgb_model_A.json, xgb_model_B.json")

    # ── Guardar métricas ──────────────────────────────────────────────────
    with open(RESULTS_DIR / "model_metrics_v2.txt", "w") as f:
        f.write("MIBEL Congestion Monitor — Métricas modelo limpio\n")
        f.write("=" * 60 + "\n\n")
        for step in ["pre_crisis→excepcion_iberica", "excepcion_iberica→post_excepcion", "final_2024"]:
            f.write(f"\n[{step}]\n")
            for spec, res in [("A_clean", results_A), ("B_id_lag", results_B),
                               ("C_id_contemp", results_C)]:
                m = res[step]["model"]
                f.write(f"  {spec}: RMSE={m['rmse']} MAE={m['mae']} R²={m['r2']} "
                        f"Recall@ext={m['recall_extreme']} F1@ext={m['f1_extreme']}\n")
            bm0 = results_A[step]["M0"]
            bm0b = results_A[step]["M0b"]
            f.write(f"  M0 (persist): RMSE={bm0['rmse']} MAE={bm0['mae']} R²={bm0['r2']}\n")
            f.write(f"  M0b (zero):   RMSE={bm0b['rmse']} MAE={bm0b['mae']} R²={bm0b['r2']}\n")

    # ── Gráficos ──────────────────────────────────────────────────────────
    plot_results(residuals_A, residuals_B, ablation)

    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print("  Modelo de producción: A_clean (sin leakage)")
    print("  Residuos para detección de anomalías: residuals_v2.parquet")
    print("  Tabla de ablation: results/ablation_table.csv")


if __name__ == "__main__":
    main()
