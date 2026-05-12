"""
shap_analysis.py
----------------
Análisis SHAP (SHapley Additive exPlanations) del modelo base XGBoost.
MIBEL Congestion Monitor.

Genera:
- Importancia global de features (beeswarm + bar plot)
- Importancia por régimen (comparativa)
- SHAP values para los top episodios de alerta
- Interpretación económica de los hallazgos

Salidas:
    plots/08_shap_global.png        — importancia global
    plots/09_shap_by_regime.png     — importancia por régimen
    plots/10_shap_alert_episodes.png — SHAP en episodios de alerta
    results/shap_findings.txt       — hallazgos económicos
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
import shap

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

DATASET_PATH = DATA_DIR / "processed" / "mibel_dataset_20190101_20241231.parquet"
ANOMALY_PATH = DATA_DIR / "processed" / "anomaly_scores.parquet"
ALERTS_PATH = RESULTS_DIR / "alerts_registry.csv"
MODEL_PATH = MODELS_DIR / "xgb_model.json"

FEATURES = [
    "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
    "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
    "spread_da_lag48h", "spread_da_lag168h",
    "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    "atc_implicit", "atc_congestion", "atc_fatigue", "atc_fatigue_lag24h",
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
    "spread_id", "spread_da_id",
]

REGIME_COLORS = {
    "pre_crisis":        "#2196F3",
    "excepcion_iberica": "#FF5722",
    "post_excepcion":    "#4CAF50",
}

# Nombres legibles para las features
FEATURE_LABELS = {
    "spread_da_lag1h":    "Spread DA lag 1h",
    "spread_da_lag2h":    "Spread DA lag 2h",
    "spread_da_lag3h":    "Spread DA lag 3h",
    "spread_da_lag6h":    "Spread DA lag 6h",
    "spread_da_lag12h":   "Spread DA lag 12h",
    "spread_da_lag24h":   "Spread DA lag 24h",
    "spread_da_lag48h":   "Spread DA lag 48h",
    "spread_da_lag168h":  "Spread DA lag 168h (1 semana)",
    "spread_da_ma24h":    "Media móvil spread 24h",
    "spread_da_ma168h":   "Media móvil spread 168h",
    "spread_da_std24h":   "Volatilidad spread 24h",
    "ttf_eur_mwh":        "TTF Gas (€/MWh)",
    "co2_eur_t":          "CO2 EUA (€/t)",
    "spark_spread":       "Spark Spread",
    "clean_spark_spread": "Clean Spark Spread",
    "ttf_lag24h":         "TTF Gas lag 24h",
    "ttf_lag168h":        "TTF Gas lag 168h",
    "atc_implicit":       "ATC implícito",
    "atc_congestion":     "Congestión ATC",
    "atc_fatigue":        "ATC Fatigue (horas consec.)",
    "atc_fatigue_lag24h": "ATC Fatigue lag 24h",
    "hour":               "Hora del día",
    "dow":                "Día de la semana",
    "month":              "Mes",
    "is_weekend":         "Fin de semana",
    "is_night":           "Período nocturno",
    "is_peak":            "Período punta",
    "spread_id":          "Spread intradiario",
    "spread_da_id":       "Diferencial DA-Intradiario",
}


def load_data():
    df = pd.read_parquet(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Imputar huecos
    for col in ["ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
                "ttf_lag24h", "ttf_lag168h"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=72)
    for col in ["ntc_es_fr", "ntc_fr_es", "ntc_es_fr_lag24h"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    lag_cols = [c for c in df.columns if "lag" in c or "ma" in c or "std" in c]
    df[lag_cols] = df[lag_cols].ffill(limit=24)
    df = df.dropna(subset=["spread_da", "spread_da_lag1h", "spread_da_lag24h"])

    return df


def compute_shap_values(df: pd.DataFrame, model: xgb.XGBRegressor,
                        sample_size: int = 5000):
    """
    Calcula SHAP values sobre una muestra estratificada por régimen.
    5.000 muestras es estándar en la literatura para análisis SHAP de interpretabilidad.
    """
    available_features = [f for f in FEATURES if f in df.columns]

    # Muestra estratificada: proporcional por régimen
    samples = []
    for regime in df["regime"].unique():
        regime_df = df[df["regime"] == regime]
        n = max(100, int(sample_size * len(regime_df) / len(df)))
        n = min(n, len(regime_df))
        samples.append(regime_df.sample(n, random_state=42))
    df_sample = pd.concat(samples).sort_values("timestamp").reset_index(drop=True)

    X_full = df[available_features].fillna(0)
    X_sample = df_sample[available_features].fillna(0)

    print(f"  Muestra estratificada: {len(df_sample):,} filas (de {len(df):,} totales)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample, available_features, df_sample


def plot_shap_global(shap_values, X, feature_names):
    """Importancia global de features."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Análisis SHAP — Importancia Global de Features\nMIBEL Congestion Monitor",
                 fontsize=13, fontweight="bold")

    # Panel 1: Bar plot de importancia media |SHAP|
    ax1 = axes[0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=True)
    labels = [FEATURE_LABELS.get(f, f) for f in feat_importance.index]

    # Colorear por categoría
    colors = []
    for f in feat_importance.index:
        if "lag" in f or "ma" in f or "std" in f:
            colors.append("#1565C0")  # Azul: lags/memoria
        elif "ttf" in f or "co2" in f or "spark" in f or "gas" in f:
            colors.append("#FF6F00")  # Naranja: gas/CO2
        elif "atc" in f or "ntc" in f:
            colors.append("#E53935")  # Rojo: ATCs
        elif f in ["hour", "dow", "month", "is_weekend", "is_night", "is_peak"]:
            colors.append("#4CAF50")  # Verde: calendario
        else:
            colors.append("#9C27B0")  # Morado: intradiario

    bars = ax1.barh(range(len(labels)), feat_importance.values, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Importancia media |SHAP| (€/MWh)", fontsize=9)
    ax1.set_title("Importancia global de features", fontsize=10)

    # Leyenda de categorías
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="Lags / Memoria"),
        Patch(facecolor="#FF6F00", label="Gas / CO2"),
        Patch(facecolor="#E53935", label="ATCs / Congestión"),
        Patch(facecolor="#4CAF50", label="Calendario"),
        Patch(facecolor="#9C27B0", label="Intradiario"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # Panel 2: Top 15 features con distribución SHAP
    ax2 = axes[1]
    top_features_idx = np.argsort(mean_abs_shap)[-15:][::-1]
    top_names = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in top_features_idx]
    top_shap = shap_values[:, top_features_idx]
    top_vals = X.iloc[:, top_features_idx].values

    # Normalizar valores de features para colorear
    for j, (name, idx) in enumerate(zip(top_names, top_features_idx)):
        shap_col = top_shap[:, j]
        feat_col = top_vals[:, j]
        # Muestra aleatoria
        sample_idx = np.random.choice(len(shap_col), min(2000, len(shap_col)), replace=False)
        feat_norm = (feat_col[sample_idx] - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-8)
        sc = ax2.scatter(shap_col[sample_idx],
                         np.full(len(sample_idx), len(top_names) - 1 - j) + np.random.normal(0, 0.1, len(sample_idx)),
                         c=feat_norm, cmap="RdBu_r", alpha=0.3, s=3, vmin=0, vmax=1)

    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names[::-1], fontsize=8)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.set_xlabel("SHAP value (impacto en spread predicho, €/MWh)", fontsize=9)
    ax2.set_title("Top 15 features: distribución SHAP\n(azul=valor bajo, rojo=valor alto)", fontsize=10)
    plt.colorbar(sc, ax=ax2, label="Valor de la feature (normalizado)")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out = PLOTS_DIR / "08_shap_global.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")

    return feat_importance


def plot_shap_by_regime(df: pd.DataFrame, shap_values, feature_names):
    """Importancia SHAP por régimen."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Importancia SHAP por Régimen de Mercado — MIBEL Congestion Monitor",
                 fontsize=13, fontweight="bold")

    regime_order = ["pre_crisis", "excepcion_iberica", "post_excepcion"]
    regime_labels = ["Pre-crisis\n(2019-2021)", "Excepción Ibérica\n(2022-2023)", "Post-excepción\n(2024)"]

    top_n = 12
    for i, (regime, label) in enumerate(zip(regime_order, regime_labels)):
        ax = axes[i]
        mask = df["regime"] == regime
        shap_regime = shap_values[mask.values]
        mean_abs = np.abs(shap_regime).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]

        top_names = [FEATURE_LABELS.get(feature_names[j], feature_names[j]) for j in top_idx]
        top_vals = mean_abs[top_idx]

        # Colorear por categoría
        colors = []
        for j in top_idx:
            f = feature_names[j]
            if "lag" in f or "ma" in f or "std" in f:
                colors.append("#1565C0")
            elif "ttf" in f or "co2" in f or "spark" in f:
                colors.append("#FF6F00")
            elif "atc" in f or "ntc" in f:
                colors.append("#E53935")
            elif f in ["hour", "dow", "month", "is_weekend", "is_night", "is_peak"]:
                colors.append("#4CAF50")
            else:
                colors.append("#9C27B0")

        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel("|SHAP| medio (€/MWh)", fontsize=8)
        ax.set_title(f"{label}\n(n={mask.sum():,} horas)", fontsize=9,
                     color=REGIME_COLORS[regime])
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out = PLOTS_DIR / "09_shap_by_regime.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def generate_shap_findings(feat_importance: pd.Series, df: pd.DataFrame,
                            shap_values, feature_names: list) -> str:
    """Genera el informe de hallazgos económicos basado en SHAP."""

    # Top features globales
    top5 = feat_importance.sort_values(ascending=False).head(5)

    # Importancia por régimen
    regime_top = {}
    for regime in ["pre_crisis", "excepcion_iberica", "post_excepcion"]:
        mask = df["regime"] == regime
        shap_r = shap_values[mask.values]
        mean_abs = np.abs(shap_r).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-3:][::-1]
        regime_top[regime] = [(feature_names[j], mean_abs[j]) for j in top_idx]

    # Correlación SHAP de ATCs con spread
    atc_features = [f for f in feature_names if "atc" in f]
    atc_shap_mean = {}
    for f in atc_features:
        idx = feature_names.index(f)
        atc_shap_mean[f] = np.abs(shap_values[:, idx]).mean()

    findings = []
    findings.append("=" * 65)
    findings.append("MIBEL Congestion Monitor — Hallazgos SHAP (Interpretabilidad)")
    findings.append("=" * 65)

    findings.append("\n1. VARIABLES MÁS IMPORTANTES (Global)")
    findings.append("-" * 40)
    for feat, val in top5.items():
        label = FEATURE_LABELS.get(feat, feat)
        findings.append(f"   {label}: {val:.4f} €/MWh de impacto medio")

    findings.append("\n2. HALLAZGOS POR RÉGIMEN")
    findings.append("-" * 40)
    for regime, tops in regime_top.items():
        findings.append(f"\n   {regime.upper()}:")
        for feat, val in tops:
            label = FEATURE_LABELS.get(feat, feat)
            findings.append(f"     → {label}: {val:.4f} €/MWh")

    findings.append("\n3. IMPORTANCIA DE LOS ATCs (Congestión de Red)")
    findings.append("-" * 40)
    for feat, val in sorted(atc_shap_mean.items(), key=lambda x: -x[1]):
        label = FEATURE_LABELS.get(feat, feat)
        findings.append(f"   {label}: {val:.4f} €/MWh")

    findings.append("\n4. INTERPRETACIÓN ECONÓMICA")
    findings.append("-" * 40)

    # Determinar si los lags dominan o los fundamentales
    lag_importance = feat_importance[[f for f in feat_importance.index
                                       if "lag" in f or "ma" in f or "std" in f]].sum()
    fundamental_importance = feat_importance[[f for f in feat_importance.index
                                               if "ttf" in f or "co2" in f or "spark" in f
                                               or "atc" in f]].sum()
    total = feat_importance.sum()

    findings.append(f"   Memoria del mercado (lags): {lag_importance/total*100:.1f}% de la importancia total")
    findings.append(f"   Fundamentos (gas/CO2/ATCs): {fundamental_importance/total*100:.1f}% de la importancia total")

    if lag_importance > fundamental_importance:
        findings.append("\n   → El spread ES-FR está dominado por la MEMORIA DEL MERCADO:")
        findings.append("     El spread de las últimas horas es el mejor predictor del spread actual.")
        findings.append("     Esto es consistente con un mercado con alta autocorrelación,")
        findings.append("     donde los episodios de congestión tienden a persistir.")
    else:
        findings.append("\n   → El spread ES-FR está dominado por los FUNDAMENTOS ECONÓMICOS:")
        findings.append("     El coste marginal del gas y las restricciones de red son los")
        findings.append("     principales determinantes del spread entre zonas de precio.")

    findings.append("\n5. IMPLICACIONES PARA LA VIGILANCIA DE MERCADO")
    findings.append("-" * 40)
    findings.append("   Un residuo elevado (spread observado >> spread predicho por fundamentales)")
    findings.append("   señala que el spread no puede explicarse por:")
    findings.append("   - El precio del gas (TTF)")
    findings.append("   - Las restricciones físicas de red (ATCs)")
    findings.append("   - Los patrones históricos de autocorrelación")
    findings.append("   Esto es consistente con comportamiento estratégico o retención de capacidad,")
    findings.append("   lo que activa los módulos de alerta del sistema.")

    report_text = "\n".join(findings)
    out = RESULTS_DIR / "shap_findings.txt"
    with open(out, "w") as f:
        f.write(report_text)
    print(f"\nHallazgos SHAP guardados: {out}")
    print(report_text)
    return report_text


if __name__ == "__main__":
    print("=== MIBEL Congestion Monitor — Análisis SHAP ===\n")

    print("Cargando datos y modelo...")
    df = load_data()
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    print(f"Dataset: {df.shape} | Modelo cargado: {MODEL_PATH.name}")

    print("\nCalculando SHAP values (muestra estratificada 5.000 filas)...")
    shap_values, X, available_features, df_sample = compute_shap_values(df, model, sample_size=5000)
    print(f"SHAP values calculados: {shap_values.shape}")

    print("\nGenerando gráfico de importancia global...")
    feat_importance = plot_shap_global(shap_values, X, available_features)

    print("\nGenerando gráfico por régimen...")
    plot_shap_by_regime(df_sample, shap_values, available_features)

    print("\nGenerando informe de hallazgos económicos...")
    findings = generate_shap_findings(feat_importance, df_sample, shap_values, available_features)

    print(f"\n{'='*60}")
    print("Análisis SHAP completado.")
    print(f"Gráficos: {PLOTS_DIR}")
    print(f"Hallazgos: {RESULTS_DIR / 'shap_findings.txt'}")
