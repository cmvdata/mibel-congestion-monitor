"""
shap_analysis_v3.py
-------------------
Análisis SHAP del Modelo A v3 (XGBoost reentrenado SIN features con leakage).

KNOWN ISSUE (2026-05): incompatibilidad entre shap 0.42/0.46/0.48 y xgboost
3.2.0 al cargar modelos guardados. El JSON de xgboost 3.x serializa
`base_score` como `'[3.670286E-1]'` (string que envuelve un array), pero
SHAP intenta `float(...)` directo y falla con ValueError. Probado fix de
shap 0.42 da otro error (UnicodeDecodeError en parser binario).
WORKAROUND: bajar a xgboost<3.0 y re-entrenar todo el pipeline (riesgo
de cambio de métricas <0.01 en R²), o esperar a parche en shap 0.49+.
HOY: este script NO corre. Los plots plots/08_shap_global_v3.png y
plots/09_shap_by_regime_v3.png reflejan el modelo v3 PRE-A2real
(cuando shap todavía funcionaba). Marcar como caveat en memoria.

Cambios respecto a la v1:
- Carga el modelo de models/xgb_model_A_v3.json
- Usa FEATURES_A_CLEAN (sin atc_implicit/atc_congestion/atc_fatigue)
- Usa los regímenes corregidos (3 regímenes)

Salidas:
    plots/08_shap_global_v3.png
    plots/09_shap_by_regime_v3.png
    results/shap_findings_v3.txt
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
MODEL = MODELS_DIR / "xgb_model_A_v3.json"
FEATURES_FILE = MODELS_DIR / "xgb_model_A_v3_features.csv"

REGIMES = ["pre_crisis", "crisis_y_excepcion", "post_excepcion"]
REGIME_COLORS = {
    "pre_crisis":         "#3498DB",
    "crisis_y_excepcion": "#E74C3C",
    "post_excepcion":     "#27AE60",
}

LABELS = {
    "spread_da_lag1h":   "Spread DA lag 1h",
    "spread_da_lag2h":   "Spread DA lag 2h",
    "spread_da_lag3h":   "Spread DA lag 3h",
    "spread_da_lag6h":   "Spread DA lag 6h",
    "spread_da_lag12h":  "Spread DA lag 12h",
    "spread_da_lag24h":  "Spread DA lag 24h",
    "spread_da_lag48h":  "Spread DA lag 48h",
    "spread_da_lag168h": "Spread DA lag 168h",
    "spread_da_ma24h":   "Media móvil 24h",
    "spread_da_ma168h":  "Media móvil 168h",
    "spread_da_std24h":  "Volatilidad 24h",
    "ttf_eur_mwh":       "TTF Gas",
    "co2_eur_t":         "CO2 EUA",
    "spark_spread":      "Spark Spread",
    "clean_spark_spread":"Clean Spark Spread",
    "ttf_lag24h":        "TTF lag 24h",
    "ttf_lag168h":       "TTF lag 168h",
    "ntc_es_fr":         "NTC ES→FR",
    "ntc_fr_es":         "NTC FR→ES",
    "ntc_is_observed":   "NTC observado (flag)",
    "hour": "Hora", "dow": "Día semana", "month": "Mes",
    "is_weekend":"Fin de semana","is_night":"Nocturno","is_peak":"Punta",
}


def load():
    df = pd.read_parquet(DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = pd.read_csv(FEATURES_FILE)["feature"].tolist()
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL))
    return df, feats, model


def stratified_sample(df: pd.DataFrame, n_total: int = 5000, seed: int = 42) -> pd.DataFrame:
    parts = []
    for r in REGIMES:
        sub = df[df["regime"] == r]
        if len(sub) == 0:
            continue
        n = max(200, int(n_total * len(sub) / len(df)))
        n = min(n, len(sub))
        parts.append(sub.sample(n, random_state=seed))
    return pd.concat(parts).sort_values("timestamp").reset_index(drop=True)


def category(f: str) -> str:
    if any(s in f for s in ("lag", "_ma", "std")):
        return "lags"
    if "ttf" in f or "co2" in f or "spark" in f:
        return "fuel"
    if "ntc" in f:
        return "capacity"
    if f in ("hour", "dow", "month", "is_weekend", "is_night", "is_peak"):
        return "calendar"
    return "other"


CAT_COLORS = {"lags": "#1565C0", "fuel": "#FF6F00",
              "capacity": "#E53935", "calendar": "#4CAF50",
              "other": "#9C27B0"}


def plot_global(shap_values, X, feats, df_sample):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Análisis SHAP — Modelo A v3 (sin leakage ATC)\n"
                 "Importancia global de features",
                 fontsize=12, fontweight="bold")
    # Bar plot
    ax = axes[0]
    cols = [CAT_COLORS[category(feats[i])] for i in order]
    labels = [LABELS.get(feats[i], feats[i]) for i in order]
    ax.barh(range(len(order)), mean_abs[order], color=cols, alpha=0.85)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Importancia media |SHAP| (€/MWh)")
    leg = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items()]
    ax.legend(handles=leg, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    # Beeswarm-like for top 12
    ax2 = axes[1]
    top = np.argsort(mean_abs)[-12:][::-1]
    for j, i in enumerate(top):
        col_vals = X.iloc[:, i].values.astype(float)
        if col_vals.max() == col_vals.min():
            norm = np.zeros_like(col_vals)
        else:
            norm = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
        idx = np.random.default_rng(42).choice(
            len(col_vals), size=min(2000, len(col_vals)), replace=False
        )
        ax2.scatter(shap_values[idx, i],
                    np.full(len(idx), len(top) - 1 - j) + np.random.normal(0, 0.12, len(idx)),
                    c=norm[idx], cmap="RdBu_r", alpha=0.4, s=3, vmin=0, vmax=1)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([LABELS.get(feats[i], feats[i]) for i in top][::-1], fontsize=8)
    ax2.axvline(0, color="black", lw=0.6, ls="--", alpha=0.7)
    ax2.set_xlabel("SHAP value (€/MWh)")
    ax2.set_title("Top 12 — distribución SHAP\n(azul=valor bajo, rojo=valor alto)")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out = PLOTS_DIR / "08_shap_global_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")
    return mean_abs


def plot_by_regime(shap_values, df_sample, feats):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Análisis SHAP por régimen — Modelo A v3", fontsize=12, fontweight="bold")
    labels = {"pre_crisis": "Pre-crisis (2019-2021)",
              "crisis_y_excepcion": "Crisis y Excepción\n(2022-2023)",
              "post_excepcion": "Post-excepción (2024)"}
    top_n = 10
    for i, r in enumerate(REGIMES):
        ax = axes[i]
        mask = (df_sample["regime"] == r).values
        if mask.sum() < 10:
            ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes, ha="center")
            continue
        sv = shap_values[mask]
        m = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(m)[-top_n:][::-1]
        names = [LABELS.get(feats[j], feats[j]) for j in top_idx]
        cols = [CAT_COLORS[category(feats[j])] for j in top_idx]
        ax.barh(range(top_n), m[top_idx][::-1], color=cols[::-1], alpha=0.85)
        ax.set_yticks(range(top_n)); ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("|SHAP| medio (€/MWh)")
        ax.set_title(f"{labels[r]}\n(n={int(mask.sum()):,})", color=REGIME_COLORS[r])
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out = PLOTS_DIR / "09_shap_by_regime_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def main():
    print("=" * 64)
    print("  SHAP analysis v3 (Modelo A reentrenado sin leakage)")
    print("=" * 64)
    df, feats, model = load()
    df_sample = stratified_sample(df, n_total=5000)
    print(f"  Muestra estratificada: {len(df_sample):,} filas")
    X = df_sample[feats].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"  SHAP values: {shap_values.shape}")

    mean_abs = plot_global(shap_values, X, feats, df_sample)
    plot_by_regime(shap_values, df_sample, feats)

    # Top features texto
    order = np.argsort(mean_abs)[::-1]
    with open(RESULTS_DIR / "shap_findings_v3.txt", "w", encoding="utf-8") as f:
        f.write("MIBEL Congestion Monitor v3 — SHAP findings\n")
        f.write("=" * 50 + "\n\n")
        f.write("Top 10 features por |SHAP| medio:\n")
        for i in order[:10]:
            f.write(f"  {LABELS.get(feats[i], feats[i]):30s}  {mean_abs[i]:8.4f}\n")
    print(f"\n  Findings: {RESULTS_DIR / 'shap_findings_v3.txt'}")


if __name__ == "__main__":
    main()
