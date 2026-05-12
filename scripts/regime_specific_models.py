"""
regime_specific_models.py
-------------------------
T4 — Entrena un XGBoost separado por régimen (3 regímenes corregidos),
con split temporal 80/20 dentro de cada régimen, y compara contra el
modelo unificado expanding-window.

NOTA: Los modelos régime-específicos se reportan SOLO para Tabla 3.
El sistema de alertas operativo sigue usando los residuales del Modelo A
unificado (residuals_v3.parquet) — ver decisión D1 confirmada por el
usuario.

Salidas:
    results/regime_specific_metrics.csv       (Tabla 3)
    results/comparison_unified_vs_regime.csv
    plots/regime_specific_comparison.png
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
PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PARQUET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
TARGET = "spread_da"
TAU = 2.0

# Mismo feature set que Modelo A v3 (sin leakage ATC)
FEATURES = [
    "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
    "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
    "spread_da_lag48h", "spread_da_lag168h",
    "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
]

REGIMES = ["pre_crisis", "crisis_y_excepcion", "post_excepcion"]


def metrics(y_true, y_pred, tau=TAU):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    yt = (np.abs(y_true) > tau).astype(int)
    yp = (np.abs(y_pred) > tau).astype(int)
    rec = float(recall_score(yt, yp, zero_division=0)) if yt.sum() else 0.0
    f1 = float(f1_score(yt, yp, zero_division=0)) if yt.sum() else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "recall_extreme": rec, "f1_extreme": f1,
            "n": int(len(y_true)), "n_extreme": int(yt.sum())}


def split_chronological(df: pd.DataFrame, train_frac: float = 0.8):
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def train_xgb(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def main():
    print("=" * 64)
    print("  T4 — Regime-specific XGBoost models")
    print("=" * 64)

    df = pd.read_parquet(PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    available = [f for f in FEATURES if f in df.columns]
    print(f"  Features activas: {len(available)}")

    rows = []
    rows_unified = []  # comparación unificado vs régimen-específico

    # Modelo unificado (ya entrenado en train_model_v3.py: train 2019-2023, test 2024)
    # Para comparación rigurosa, también lo entrenamos régime-específicamente:
    # mismo split 80/20 por régimen, pero training pooled de todos los regímenes
    # excepto el test slice del régimen evaluado.
    # Para mantenerlo simple y comparable: cargamos predicciones del modelo unificado
    # en el test slice de cada régimen.
    print("\n  [1/2] Modelos régime-específicos (uno por régimen, 80/20 chronological)")
    print("-" * 64)

    test_slices = {}
    for regime in REGIMES:
        sub = df[df["regime"] == regime].copy()
        if len(sub) < 1000:
            print(f"    ⚠ Régimen {regime}: solo {len(sub)} filas, saltando.")
            continue
        tr, va = split_chronological(sub, 0.8)
        Xtr, ytr = tr[available].fillna(0), tr[TARGET]
        Xva, yva = va[available].fillna(0), va[TARGET]
        m = train_xgb(Xtr, ytr, Xva, yva)
        pred = m.predict(Xva)
        mt = metrics(yva.values, pred)
        rows.append({"regime": regime, "spec": "regime_specific", **mt,
                     "train_period": [str(tr["timestamp"].min()), str(tr["timestamp"].max())],
                     "test_period":  [str(va["timestamp"].min()), str(va["timestamp"].max())]})
        # Guardar modelo y test slice (para comparación con unificado)
        m.save_model(str(MODELS_DIR / f"xgb_regime_{regime}.json"))
        test_slices[regime] = (va.copy(), pred)
        print(f"    {regime:25s}: n_train={len(tr):>6,}  n_test={len(va):>6,}  "
              f"RMSE={mt['rmse']:.3f}  R²={mt['r2']:.3f}  Recall@ext={mt['recall_extreme']:.3f}")

    # Modelo unificado entrenado pooled, evaluado sobre cada test slice ────
    print("\n  [2/2] Modelo unificado pooled (train: 80% temporal pooled de TODOS los regímenes)")
    print("-" * 64)

    # A6: Cutoff temporal único. El unified pooled NO debe ver datos futuros
    # de ningún test slice. Antes: entrenaba con 80% de cada régimen incluido
    # post_excepcion (2024), luego se evaluaba sobre pre_crisis test slice
    # (mayo-dic 2021), viendo 2022-2024 durante el entrenamiento → leakage.
    test_parts = []
    earliest_test_start = None
    for regime in REGIMES:
        sub = df[df["regime"] == regime].sort_values("timestamp").copy()
        tr, va = split_chronological(sub, 0.8)
        test_parts.append(va.assign(_regime_label=regime))
        va_start = va["timestamp"].min()
        if earliest_test_start is None or va_start < earliest_test_start:
            earliest_test_start = va_start
    df_test_pooled = pd.concat(test_parts, ignore_index=True)

    # Unified pooled: entrena con TODO lo anterior al test slice más antiguo
    df_train_pooled = df[df["timestamp"] < earliest_test_start].copy()

    print(f"[A6] unified cutoff temporal único: {earliest_test_start}")
    print(f"[A6] tamaño train pooled (corregido): {len(df_train_pooled)}")
    print(f"[A6] tamaño test pooled: {len(df_test_pooled)}")
    Xtr = df_train_pooled[available].fillna(0); ytr = df_train_pooled[TARGET]
    # Para early stopping, usamos un val set tomado del 10% final del train
    n_es = int(len(Xtr) * 0.9)
    m_unified = train_xgb(Xtr.iloc[:n_es], ytr.iloc[:n_es],
                          Xtr.iloc[n_es:], ytr.iloc[n_es:])
    m_unified.save_model(str(MODELS_DIR / "xgb_unified_v3.json"))

    # Métricas pooled
    Xte = df_test_pooled[available].fillna(0); yte = df_test_pooled[TARGET]
    pred_te = m_unified.predict(Xte)
    mt_pool = metrics(yte.values, pred_te)
    rows.append({"regime": "POOLED_ALL", "spec": "unified", **mt_pool,
                 "train_period": [str(df_train_pooled["timestamp"].min()),
                                  str(df_train_pooled["timestamp"].max())],
                 "test_period":  [str(df_test_pooled["timestamp"].min()),
                                  str(df_test_pooled["timestamp"].max())]})
    print(f"    POOLED_ALL              : n_train={len(df_train_pooled):>6,}  "
          f"n_test={len(df_test_pooled):>6,}  RMSE={mt_pool['rmse']:.3f}  "
          f"R²={mt_pool['r2']:.3f}  Recall@ext={mt_pool['recall_extreme']:.3f}")

    # Métricas del unificado descompuestas por régimen (test slices)
    for regime in REGIMES:
        if regime not in test_slices:
            continue
        va, _ = test_slices[regime]
        Xva = va[available].fillna(0)
        pred_unif_regime = m_unified.predict(Xva)
        mt_u = metrics(va[TARGET].values, pred_unif_regime)
        rows.append({"regime": regime, "spec": "unified", **mt_u,
                     "train_period": [str(df_train_pooled["timestamp"].min()),
                                      str(df_train_pooled["timestamp"].max())],
                     "test_period":  [str(va["timestamp"].min()), str(va["timestamp"].max())]})
        # Comparativa lado a lado
        rs_metrics = next(r for r in rows if r["regime"] == regime and r["spec"] == "regime_specific")
        rows_unified.append({
            "regime": regime,
            "rmse_unified": mt_u["rmse"], "rmse_regime": rs_metrics["rmse"],
            "r2_unified": mt_u["r2"],     "r2_regime":   rs_metrics["r2"],
            "mae_unified": mt_u["mae"],   "mae_regime":  rs_metrics["mae"],
            "recall_unified": mt_u["recall_extreme"], "recall_regime": rs_metrics["recall_extreme"],
            "winner_rmse": "regime" if rs_metrics["rmse"] < mt_u["rmse"] else "unified",
            "delta_rmse_pct": (rs_metrics["rmse"] - mt_u["rmse"]) / mt_u["rmse"] * 100,
        })
        print(f"    {regime:25s} unified->regime test: RMSE_unif={mt_u['rmse']:.3f} | "
              f"RMSE_rs={rs_metrics['rmse']:.3f} | "
              f"winner={'rs' if rs_metrics['rmse'] < mt_u['rmse'] else 'unif'}")

    # Persistir
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(RESULTS_DIR / "regime_specific_metrics.csv", index=False)
    df_cmp = pd.DataFrame(rows_unified)
    df_cmp.to_csv(RESULTS_DIR / "comparison_unified_vs_regime.csv", index=False)

    print("\n  Tabla 3 — Modelos régime-específicos vs unificado")
    print("=" * 64)
    print(df_metrics[["regime", "spec", "n", "rmse", "mae", "r2",
                       "recall_extreme", "f1_extreme"]].to_string(index=False))
    print("\n  Comparativa lado a lado (winner por RMSE):")
    print(df_cmp.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("T4 — Modelos régime-específicos vs unificado pooled\n"
                 "(80/20 split temporal por régimen)", fontsize=12, fontweight="bold")

    # RMSE
    regs = df_cmp["regime"].tolist()
    x = np.arange(len(regs)); w = 0.36
    axes[0].bar(x - w/2, df_cmp["rmse_unified"], w, label="Unificado", color="#34495E", alpha=0.85)
    axes[0].bar(x + w/2, df_cmp["rmse_regime"],  w, label="Régime-específico", color="#1ABC9C", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(regs, rotation=15, ha="right")
    axes[0].set_ylabel("RMSE (€/MWh)"); axes[0].set_title("RMSE por régimen")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis="y")

    # Recall
    axes[1].bar(x - w/2, df_cmp["recall_unified"], w, label="Unificado", color="#34495E", alpha=0.85)
    axes[1].bar(x + w/2, df_cmp["recall_regime"],  w, label="Régime-específico", color="#1ABC9C", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(regs, rotation=15, ha="right")
    axes[1].set_ylabel("Recall@extremo"); axes[1].set_title("Recall en horas extremas (|spread|>2)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOTS_DIR / "regime_specific_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot: {out}")
    print("\n  T4 COMPLETADA")


if __name__ == "__main__":
    main()
