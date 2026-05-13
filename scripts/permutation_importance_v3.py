"""
permutation_importance_v3.py
-----------------------------
Feature importance del Modelo A v3 mediante permutation importance
(sklearn.inspection), evitando completamente la dependencia de SHAP y
su loader de XGBoost (incompatible con xgboost>=3.1 hasta shap 0.49.1
por el cambio en learner_model_param['base_score'] de float a list).

WHY PERMUTATION INSTEAD OF SHAP?
    1. Es model-agnostic: no depende del parser binario/JSON interno
       de XGBoost. Carga del Booster con la API estándar (load_model +
       Booster.predict(DMatrix)). Inmune a futuros breaking changes del
       formato de serialización.
    2. Defendible ante un revisor econométra: la métrica reportada es la
       degradación del RMSE out-of-sample cuando se permuta una columna,
       directamente interpretable como "cuánto contribuye esta feature
       a la capacidad predictiva del modelo".
    3. Reproducible vía n_repeats=30 + random_state=42, con error bars
       (std de las repeticiones) que SHAP no proporciona de forma directa.

DESIGN NOTES:
    - Usamos xgb.Booster.load_model(...) directo en lugar de XGBRegressor.
      Esto sortea cualquier discrepancia de wrapper sklearn vs Booster.
    - sklearn.inspection.permutation_importance requiere un estimator con
      .predict(); envolvemos el Booster en un thin adapter (BoosterAdapter)
      que expone predict(X)->np.ndarray y satisface el duck-typing de
      sklearn.
    - El orden de columnas DEBE coincidir con el feature_names del Booster.
      Lo forzamos leyendo xgb_model_A_v3_features.csv y reindexando X.
    - NaN policy: fillna(0) explícito. XGBoost maneja NaN nativamente vía
      `missing=np.nan` en DMatrix, pero permutation_importance permuta
      sobre los valores tal cual están en el DataFrame; permutar NaN
      contra valores reales introduce ruido espurio. Rellenar con 0 es
      consistente con la práctica de v3 (ver shap_analysis_v3.py:177) y
      con el preprocesado de entrenamiento del Modelo A.

USAGE:
    python -m src.permutation_importance_v3
    # o
    python permutation_importance_v3.py

OUTPUTS:
    plots/permutation_importance_global.png
    plots/permutation_importance_by_regime.png
    plots/permutation_importance_boxplot.png
    results/permutation_top10_correlations.csv
    results/permutation_importance_v3.txt
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
RESIDUALS = PROC_DIR / "residuals_v3.parquet"  # contiene la columna 'regime'
MODEL_PATH = MODELS_DIR / "xgb_model_A_v3.json"
FEATURES_FILE = MODELS_DIR / "xgb_model_A_v3_features.csv"

TARGET_COL = "spread_da"   # target del Modelo A: spread DA ES-FR
REGIME_COL = "regime"
TIMESTAMP_COL = "timestamp"
TEST_START = "2024-01-01"

# Regímenes que tienen overlap con el periodo de test (2024)
REGIMES_IN_TEST = ["crisis_y_excepcion", "post_excepcion"]

# Etiquetas en español (idénticas a shap_analysis_v3.py para consistencia
# entre los plots del paper)
LABELS = {
    "spread_da_lag1h":   "Spread DA lag 1h",
    "spread_da_lag2h":   "Spread DA lag 2h",
    "spread_da_lag3h":   "Spread DA lag 3h",
    "spread_da_lag6h":   "Spread DA lag 6h",
    "spread_da_lag12h":  "Spread DA lag 12h",
    "spread_da_lag24h":  "Spread DA lag 24h",
    "spread_da_lag48h":  "Spread DA lag 48h",
    "spread_da_lag168h": "Spread DA lag 168h",
    "spread_da_ma24h":   "Moving avg 24h",
    "spread_da_ma168h":  "Moving avg 168h",
    "spread_da_std24h":  "Volatility 24h",
    "ttf_eur_mwh":       "TTF Gas",
    "co2_eur_t":         "CO2 EUA",
    "spark_spread":      "Spark Spread",
    "clean_spark_spread": "Clean Spark Spread",
    "ttf_lag24h":        "TTF lag 24h",
    "ttf_lag168h":       "TTF lag 168h",
    "ntc_es_fr":         "NTC ES->FR",
    "ntc_fr_es":         "NTC FR->ES",
    "ntc_is_observed":   "NTC observed (flag)",
    "hour": "Hour", "dow": "Weekday", "month": "Month",
    "is_weekend": "Weekend", "is_night": "Night", "is_peak": "Peak",
}

CAT_COLORS = {
    "lags":     "#1565C0",
    "fuel":     "#FF6F00",
    "capacity": "#E53935",
    "calendar": "#4CAF50",
    "other":    "#9C27B0",
}

REGIME_COLORS = {
    "pre_crisis":         "#3498DB",
    "crisis_y_excepcion": "#E74C3C",
    "post_excepcion":     "#27AE60",
}


def _category(f: str) -> str:
    if any(s in f for s in ("lag", "_ma", "std")):
        return "lags"
    if "ttf" in f or "co2" in f or "spark" in f:
        return "fuel"
    if "ntc" in f:
        return "capacity"
    if f in ("hour", "dow", "month", "is_weekend", "is_night", "is_peak"):
        return "calendar"
    return "other"


# ---------------------------------------------------------------------------
# Booster adapter — duck-typing para sklearn
# ---------------------------------------------------------------------------
class BoosterAdapter(RegressorMixin, BaseEstimator):
    """Thin wrapper para que un xgb.Booster sea válido como `estimator`
    en sklearn.inspection.permutation_importance.

    Heredamos de RegressorMixin + BaseEstimator porque sklearn ≥1.6
    introduce el sistema de `__sklearn_tags__` y exige que cualquier
    estimator que pase por scorers (incluido `permutation_importance`)
    sea identificable como regresor o clasificador. Sin la herencia,
    sklearn 1.6.1+ lanza AttributeError al construir el scorer.

    permutation_importance solo necesita `estimator.predict(X)` operativo;
    fit/get_params/set_params los provee BaseEstimator pero los dejamos
    como no-ops (el Booster ya viene entrenado de disco).
    """

    def __init__(self, booster: xgb.Booster = None,
                 feature_names: Optional[list[str]] = None):
        # IMPORTANTE: BaseEstimator espera que todos los args de __init__
        # sean asignados a self con el mismo nombre (introspection vía
        # get_params/set_params). Por eso usamos defaults None.
        self.booster = booster
        self.feature_names = feature_names

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_ = X[self.feature_names].values
        else:
            X_ = np.asarray(X)
        dmat = xgb.DMatrix(X_, feature_names=self.feature_names)
        return self.booster.predict(dmat)

    def fit(self, X, y=None):  # no-op, el Booster viene pre-entrenado
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # sklearn >=1.6 puede pedir esto antes de invocar predict
        return self.booster is not None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PermImportanceResult:
    """Resultado tipado de un run de permutation_importance.

    `raw` mantiene la matriz (n_features, n_repeats) tal cual la devuelve
    sklearn; necesaria para boxplots y bootstrap CIs posteriores.
    """
    features: list[str]
    importances_mean: np.ndarray   # shape (n_features,)
    importances_std: np.ndarray    # shape (n_features,)
    raw: np.ndarray                # shape (n_features, n_repeats)
    baseline_score: float          # neg_RMSE del modelo sin permutar
    n_samples: int
    label: str = "global"
    metadata: dict = field(default_factory=dict)

    @property
    def rmse_baseline(self) -> float:
        # scoring='neg_root_mean_squared_error' devuelve neg, lo invertimos
        return -self.baseline_score

    def top_n(self, n: int = 10) -> pd.DataFrame:
        order = np.argsort(self.importances_mean)[::-1][:n]
        return pd.DataFrame({
            "feature": [self.features[i] for i in order],
            "importance_mean": self.importances_mean[order],
            "importance_std":  self.importances_std[order],
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "importance_mean": self.importances_mean,
            "importance_std":  self.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga & preparación de datos
# ---------------------------------------------------------------------------
def load_model_and_features() -> tuple[xgb.Booster, list[str]]:
    """Carga el Booster sin pasar por XGBRegressor, evitando el wrapper
    sklearn. El orden de features se lee de xgb_model_A_v3_features.csv
    (NO de booster.feature_names, que puede venir como ['f0','f1',...] si
    el modelo se guardó sin nombres explícitos)."""
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    feats = pd.read_csv(FEATURES_FILE)["feature"].tolist()
    # Sanity check: si el Booster tiene feature_names, deben coincidir
    if booster.feature_names is not None and not all(
        f.startswith("f") and f[1:].isdigit() for f in booster.feature_names
    ):
        bf = list(booster.feature_names)
        if bf != feats:
            raise ValueError(
                f"Feature order mismatch entre Booster ({bf[:5]}...) "
                f"y CSV ({feats[:5]}...). Aborto antes de generar resultados "
                f"que apuntarían a features equivocadas."
            )
    # Forzamos feature_names en el booster para que DMatrix sea consistente
    booster.feature_names = feats
    return booster, feats


def load_test_data(feats: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Filtra el dataset a 2024 (test). Devuelve X, y, regime_series.

    NaN handling: fillna(0). Justificación: el preprocesado de entrenamiento
    del Modelo A v3 ya rellena NaN en lags y rolling stats con 0 (consistencia
    con shap_analysis_v3.py línea ~177). Hacer dropna aquí desalinearía X
    respecto al periodo de test y produciría n!=8783.
    """
    df = pd.read_parquet(DATASET)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df[df[TIMESTAMP_COL] >= pd.Timestamp(TEST_START)].copy()
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Régimen: preferir residuals_v3.parquet si existe; si no, derivar del df
    if RESIDUALS.exists():
        res = pd.read_parquet(RESIDUALS)
        res[TIMESTAMP_COL] = pd.to_datetime(res[TIMESTAMP_COL])
        df = df.merge(
            res[[TIMESTAMP_COL, REGIME_COL]],
            on=TIMESTAMP_COL, how="left", suffixes=("", "_res"),
        )
        if REGIME_COL + "_res" in df.columns:
            df[REGIME_COL] = df[REGIME_COL].fillna(df[REGIME_COL + "_res"])

    if REGIME_COL not in df.columns:
        raise KeyError(
            f"No se encuentra la columna '{REGIME_COL}' ni en el parquet ni "
            f"en residuals_v3.parquet."
        )

    # X en el orden exacto del Booster
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"Features faltantes en el dataset: {missing}")
    X = df[feats].fillna(0.0).astype(np.float32)
    y = df[TARGET_COL].astype(np.float32)
    regime = df[REGIME_COL].astype(str)
    return X, y, regime


# ---------------------------------------------------------------------------
# Permutation importance — runner
# ---------------------------------------------------------------------------
def run_permutation(
    estimator: BoosterAdapter,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "global",
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
) -> PermImportanceResult:
    """Ejecuta permutation_importance y empaqueta el resultado."""
    # Baseline: RMSE sin permutar
    y_pred = estimator.predict(X)
    rmse = float(np.sqrt(np.mean((y.values - y_pred) ** 2)))

    print(f"  [{label}] n={len(X):,}  baseline RMSE={rmse:.4f} €/MWh")
    print(f"  [{label}] permutando {len(estimator.feature_names)} features "
          f"× {n_repeats} repeats...")

    pi = permutation_importance(
        estimator, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="neg_root_mean_squared_error",
    )
    # sklearn devuelve la *degradación* del score (neg_RMSE):
    # importance = baseline_score - permuted_score.
    # Como score = -RMSE, una importance positiva = el RMSE empeora al
    # permutar = la feature contribuye al modelo. Lo dejamos en esa
    # convención (€/MWh, signo positivo = aporta).
    return PermImportanceResult(
        features=list(estimator.feature_names),
        importances_mean=pi.importances_mean,
        importances_std=pi.importances_std,
        raw=pi.importances,
        baseline_score=-rmse,
        n_samples=len(X),
        label=label,
        metadata={"n_repeats": n_repeats, "random_state": random_state},
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_global(res: PermImportanceResult, out: Path) -> None:
    """Bar chart horizontal con error bars (1 std sobre n_repeats)."""
    df = res.to_dataframe()
    order = df.iloc[::-1].reset_index(drop=True)  # menor arriba, mayor abajo
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [CAT_COLORS[_category(f)] for f in order["feature"]]
    labels = [LABELS.get(f, f) for f in order["feature"]]
    ax.barh(
        range(len(order)),
        order["importance_mean"],
        xerr=order["importance_std"],
        color=colors, alpha=0.85,
        error_kw={"ecolor": "#333", "elinewidth": 0.8, "capsize": 2},
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Permutation importance (Δ RMSE, €/MWh)\n"
                  f"n_repeats={res.metadata['n_repeats']}, "
                  f"baseline RMSE={res.rmse_baseline:.3f}")
    ax.set_title("Model A v3 — Permutation Importance (global)\n"
                 f"Test set 2024 (n={res.n_samples:,})",
                 fontsize=11, fontweight="bold")
    leg = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items()]
    ax.legend(handles=leg, fontsize=8, loc="lower right", title="Category")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_by_regime(
    res_global: PermImportanceResult,
    res_by_regime: dict[str, PermImportanceResult],
    out: Path,
    top_n: int = 15,
) -> None:
    """Heatmap regime × feature (top-N por importancia global)."""
    top_feats = res_global.to_dataframe()["feature"].head(top_n).tolist()
    regimes = ["global"] + list(res_by_regime.keys())

    mat = np.zeros((len(top_feats), len(regimes)))
    mat[:, 0] = [
        res_global.importances_mean[res_global.features.index(f)]
        for f in top_feats
    ]
    for j, r in enumerate(res_by_regime.keys(), start=1):
        r_res = res_by_regime[r]
        for i, f in enumerate(top_feats):
            mat[i, j] = r_res.importances_mean[r_res.features.index(f)]

    fig, ax = plt.subplots(figsize=(8, 9))
    vmax = np.percentile(np.abs(mat), 98)  # robustez ante outliers
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(
        ["Global", *[r.replace("_", "\n") for r in res_by_regime.keys()]],
        fontsize=9,
    )
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([LABELS.get(f, f) for f in top_feats], fontsize=9)
    ax.set_title(f"Permutation importance by regime — top {top_n} features\n"
                 "(values in Δ RMSE, €/MWh)",
                 fontsize=11, fontweight="bold")
    # Anotaciones numéricas
    for i in range(len(top_feats)):
        for j in range(len(regimes)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    color="white" if val > vmax * 0.5 else "black",
                    fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Δ RMSE (€/MWh)")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


def plot_boxplot(res: PermImportanceResult, out: Path, top_n: int = 10) -> None:
    """Boxplot de las n_repeats permutaciones para las top-N features.
    Útil para detectar features cuya importancia es alta pero inestable."""
    order = np.argsort(res.importances_mean)[::-1][:top_n]
    data = [res.raw[i] for i in order]
    labels = [LABELS.get(res.features[i], res.features[i]) for i in order]
    colors = [CAT_COLORS[_category(res.features[i])] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(
        data, vert=True, patch_artist=True,
        widths=0.65, showfliers=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Permutation importance (Δ RMSE, €/MWh)")
    ax.set_title(f"Distribution over {res.metadata['n_repeats']} repetitions "
                 f"— top {top_n} features\n"
                 "Narrow boxes => stable importance; wide boxes => "
                 "noisy estimate",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def save_top10_correlations(
    res: PermImportanceResult,
    X: pd.DataFrame,
    out: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Matriz de correlación de las top-N features + TTF gas + CO2 EUA.
    Sirve para diagnosticar multicolinealidad: si dos features muy
    correladas aparecen ambas en el top y comparten señal, su importancia
    permutada se reparte entre ambas (problema clásico de permutation
    importance con features correladas)."""
    top_feats = res.to_dataframe()["feature"].head(top_n).tolist()
    extras = [f for f in ("ttf_eur_mwh", "co2_eur_t") if f not in top_feats]
    cols = top_feats + extras
    corr = X[cols].corr(method="pearson").round(4)
    corr.to_csv(out, index=True)
    print(f"  Correlations CSV: {out}")
    return corr


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def save_text_summary(
    res_global: PermImportanceResult,
    res_by_regime: dict[str, PermImportanceResult],
    out: Path,
) -> None:
    with open(out, "w", encoding="utf-8") as f:
        f.write("MIBEL Congestion Monitor v3 — Permutation importance\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set: {TEST_START} → 2024-12-31  "
                f"(n={res_global.n_samples:,})\n")
        f.write(f"Baseline RMSE: {res_global.rmse_baseline:.4f} €/MWh\n")
        f.write(f"n_repeats={res_global.metadata['n_repeats']}, "
                f"random_state={res_global.metadata['random_state']}\n\n")
        f.write("Top 10 features (global):\n")
        for _, row in res_global.to_dataframe().head(10).iterrows():
            label = LABELS.get(row["feature"], row["feature"])
            f.write(f"  {label:30s}  {row['importance_mean']:8.4f}  "
                    f"± {row['importance_std']:.4f}\n")
        f.write("\n")
        for r, r_res in res_by_regime.items():
            f.write(f"Top 10 features — régimen {r} "
                    f"(n={r_res.n_samples:,}):\n")
            for _, row in r_res.to_dataframe().head(10).iterrows():
                label = LABELS.get(row["feature"], row["feature"])
                f.write(f"  {label:30s}  {row['importance_mean']:8.4f}  "
                        f"± {row['importance_std']:.4f}\n")
            f.write("\n")
    print(f"  Findings: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
) -> None:
    print("=" * 64)
    print("  Permutation importance — Modelo A v3")
    print("=" * 64)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    booster, feats = load_model_and_features()
    print(f"  Modelo cargado: {MODEL_PATH.name}  ({len(feats)} features)")
    X, y, regime = load_test_data(feats)
    print(f"  Test set: {len(X):,} filas desde {TEST_START}")
    print(f"  Régimenes presentes: {sorted(regime.unique().tolist())}\n")

    estimator = BoosterAdapter(booster, feats)

    # 1) Permutation importance global
    res_global = run_permutation(
        estimator, X, y,
        label="global",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # 2) Permutation importance por régimen (solo los que existen en test)
    res_by_regime: dict[str, PermImportanceResult] = {}
    for r in REGIMES_IN_TEST:
        mask = (regime == r).values
        if mask.sum() < 200:
            print(f"  [skip] régimen '{r}' tiene solo {mask.sum()} muestras "
                  "(<200). No estimamos para evitar varianza absurda.")
            continue
        res_by_regime[r] = run_permutation(
            estimator, X[mask], y[mask],
            label=r,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # 3) Plots
    print("\nGenerando plots...")
    plot_global(res_global, PLOTS_DIR / "permutation_importance_global.png")
    if len(res_by_regime) >= 2:
        plot_by_regime(
            res_global, res_by_regime,
            PLOTS_DIR / "permutation_importance_by_regime.png",
        )
    else:
        print(f"  [skip] heatmap por régimen: solo {len(res_by_regime)} "
              "régimen(es) en el test set. El plot global-vs-régimen seria "
              "tautologico.")
        prev = PLOTS_DIR / "permutation_importance_by_regime.png"
        if prev.exists():
            prev.unlink()
    plot_boxplot(res_global, PLOTS_DIR / "permutation_importance_boxplot.png")

    # 4) Correlations + text summary
    print("\nGenerando artefactos auxiliares...")
    save_top10_correlations(
        res_global, X,
        RESULTS_DIR / "permutation_top10_correlations.csv",
    )
    # Solo pasamos por_regime al summary si hay >=2 (mismo criterio que plot)
    summary_by_regime = res_by_regime if len(res_by_regime) >= 2 else {}
    save_text_summary(
        res_global, summary_by_regime,
        RESULTS_DIR / "permutation_importance_v3.txt",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
