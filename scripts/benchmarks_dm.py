"""
benchmarks_dm.py
----------------
T5 — Benchmarks competitivos + Diebold-Mariano + bootstrap CI

Train: 2019-01-01 → 2023-12-31 (≈ 43.800 horas tras drop NaN)
Test:  2024-01-01 → 2024-12-31 (≈ 8.760 horas, post-excepción puro)

Modelos comparados:
    - Modelo A v3 (XGBoost, predicciones del modelo final guardadas)
    - AR(1)        (statsmodels, predicción rolling 1-paso adelante)
    - AR(24)       (statsmodels, predicción rolling 1-paso adelante)
    - LASSO        (LassoCV(cv=5), features escaladas)
    - M0           (persistencia: spread_t = spread_{t-24})
    - M0b          (constante cero)

Tests:
    - Diebold-Mariano (h=1, varianza HAC Newey-West, lag = h-1 = 0
      pero usamos lag automático ⌊4(n/100)^(2/9)⌋ siguiendo Andrews 1991).
    - Bootstrap CI 95% sobre RMSE por block-bootstrap circular,
      block size 24h y sensibilidad con 168h, 1000 réplicas.

Salidas:
    results/benchmark_comparison.csv          (Tabla 2 base)
    results/dm_test_results.csv
    results/bootstrap_rmse_ci.csv
    plots/benchmark_comparison.png
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score, f1_score
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

PARQUET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
PRED_TEST2024 = PROC_DIR / "predictions_test2024_v3.parquet"

TARGET = "spread_da"
TAU = 2.0

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


# ════════════════════════════════════════════════════════════════════════════
#  Métricas
# ════════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, tau=TAU):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    yt = (np.abs(y_true) > tau).astype(int)
    yp = (np.abs(y_pred) > tau).astype(int)
    rec = float(recall_score(yt, yp, zero_division=0)) if yt.sum() else 0.0
    f1 = float(f1_score(yt, yp, zero_division=0)) if yt.sum() else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "recall_extreme": rec, "f1_extreme": f1}


# ════════════════════════════════════════════════════════════════════════════
#  Diebold-Mariano (h=1, HAC Newey-West)
# ════════════════════════════════════════════════════════════════════════════

def newey_west_lrv(d: np.ndarray, lag: int = None) -> float:
    """Long-run variance estimator (Newey-West, Bartlett kernel).
    Si lag is None, usa la regla de Andrews (1991): lag = ⌊4(n/100)^(2/9)⌋.
    """
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2/9)))
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    s = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(d[k:], d[:-k]) / n)
        w = 1 - k / (lag + 1)  # Bartlett
        s += 2 * w * gamma_k
    return max(s, 1e-12)


def diebold_mariano(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray,
                    h: int = 1, loss: str = "se") -> dict:
    """DM test (Diebold & Mariano 1995): H0 forecast accuracies equal.
    pred1 vs pred2; signo positivo del estadístico → pred1 es PEOR (mayor pérdida).
    HAC Newey-West con regla de Andrews para el lag."""
    y_true = np.asarray(y_true, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    if loss == "se":
        e1 = (y_true - pred1) ** 2
        e2 = (y_true - pred2) ** 2
    elif loss == "ae":
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)
    else:
        raise ValueError(loss)
    d = e1 - e2
    n = len(d)
    mean_d = float(d.mean())
    lrv = newey_west_lrv(d)
    se = np.sqrt(lrv / n)
    dm = mean_d / se if se > 0 else np.nan
    p_two = 2 * (1 - sps.norm.cdf(abs(dm)))
    return {"DM_stat": float(dm), "p_value": float(p_two),
            "mean_loss_diff": mean_d, "n": n}


# ════════════════════════════════════════════════════════════════════════════
#  Bootstrap CI sobre RMSE (block-bootstrap circular)
# ════════════════════════════════════════════════════════════════════════════

def block_bootstrap_circular(errors: np.ndarray, block_size: int, n_reps: int = 1000,
                              seed: int = 42) -> np.ndarray:
    """Devuelve un array de RMSEs bootstrappeados por block-bootstrap circular."""
    rng = np.random.default_rng(seed)
    n = len(errors)
    n_blocks = int(np.ceil(n / block_size))
    rmses = np.empty(n_reps, dtype=float)
    # Pre-construir array circular para evitar wraps repetidos
    e2 = np.concatenate([errors ** 2, errors ** 2])  # circular
    for r in range(n_reps):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = [e2[s:s + block_size] for s in starts]
        sample = np.concatenate(blocks)[:n]
        rmses[r] = float(np.sqrt(sample.mean()))
    return rmses


# ════════════════════════════════════════════════════════════════════════════
#  Modelos
# ════════════════════════════════════════════════════════════════════════════

def fit_predict_ar(y_train: np.ndarray, y_test: np.ndarray, lags: int) -> np.ndarray:
    """AR(p) por OLS sobre el training; predicción 1-paso rolling sobre el test
    usando los valores observados (rolling actualizado, sin re-fit).
    """
    model = AutoReg(y_train, lags=lags, old_names=False).fit()
    coef = model.params  # [const, lag1, ..., lag_p]
    p = lags
    full = np.concatenate([y_train, y_test])
    pred = np.empty(len(y_test), dtype=float)
    for i in range(len(y_test)):
        idx = len(y_train) + i
        prev = full[idx - p:idx][::-1]  # [lag1, lag2, ..., lag_p]
        pred[i] = coef[0] + np.dot(coef[1:], prev)
    return pred


def fit_predict_lasso(X_tr, y_tr, X_te) -> np.ndarray:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr.fillna(0))
    X_te_s = sc.transform(X_te.fillna(0))
    m = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=20000).fit(X_tr_s, y_tr)
    return m.predict(X_te_s), float(m.alpha_), int(np.sum(m.coef_ != 0))


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  T5 — Benchmarks + Diebold-Mariano + bootstrap CI")
    print("=" * 64)

    df = pd.read_parquet(PARQUET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_mask = df["timestamp"] < pd.Timestamp("2024-01-01")
    test_mask = df["timestamp"] >= pd.Timestamp("2024-01-01")
    df_tr = df.loc[train_mask].copy()
    df_te = df.loc[test_mask].copy()
    y_tr = df_tr[TARGET].values
    y_te = df_te[TARGET].values
    print(f"  Train: {len(df_tr):,} horas  ({df_tr['timestamp'].min()} → {df_tr['timestamp'].max()})")
    print(f"  Test:  {len(df_te):,} horas  ({df_te['timestamp'].min()} → {df_te['timestamp'].max()})")

    available = [f for f in FEATURES if f in df.columns]

    # ── Modelo A v3 (predicciones ya guardadas) ────────────────────────────
    pred_A = pd.read_parquet(PRED_TEST2024).sort_values("timestamp")
    assert len(pred_A) == len(df_te), f"Predicciones A ({len(pred_A)}) != test ({len(df_te)})"
    pa = pred_A["spread_predicted_modelA"].values
    print(f"\n  Modelo A v3: predicciones cargadas (n={len(pa):,})")

    # ── AR(1) y AR(24) ─────────────────────────────────────────────────────
    print("\n  Ajustando AR(1) y AR(24) sobre y_train...")
    pred_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    pred_ar24 = fit_predict_ar(y_tr, y_te, lags=24)
    print("    AR(1) listo")
    print("    AR(24) listo")

    # ── LASSO ──────────────────────────────────────────────────────────────
    print("\n  Ajustando LASSO con LassoCV(cv=5)...")
    pred_lasso, alpha_lasso, n_nonzero = fit_predict_lasso(
        df_tr[available], y_tr, df_te[available]
    )
    print(f"    alpha óptimo: {alpha_lasso:.6f} | features no-cero: {n_nonzero}/{len(available)}")

    # ── Benchmarks naïf ────────────────────────────────────────────────────
    pred_m0 = df_te["spread_da_lag24h"].fillna(0).values
    pred_m0b = np.zeros(len(y_te))

    PREDS = {
        "Modelo_A_v3": pa,
        "AR(1)":       pred_ar1,
        "AR(24)":      pred_ar24,
        "LASSO":       pred_lasso,
        "M0_persist":  pred_m0,
        "M0b_zero":    pred_m0b,
    }

    # ── Métricas ───────────────────────────────────────────────────────────
    rows = []
    for name, p in PREDS.items():
        m = metrics(y_te, p)
        rows.append({"model": name, **m})
    df_metrics = pd.DataFrame(rows)
    print("\n  Métricas test 2024:")
    print(df_metrics.to_string(index=False))

    # ── Diebold-Mariano vs Modelo A ────────────────────────────────────────
    dm_rows = []
    print("\n  Diebold-Mariano (Modelo_A_v3 vs benchmark, loss=SE, HAC Newey-West):")
    for name, p in PREDS.items():
        if name == "Modelo_A_v3":
            continue
        # H0: igual precisión. Signo + → A peor que benchmark.
        # Convención: diff = (loss_bench - loss_A); DM>0 → A mejor.
        res = diebold_mariano(y_te, p, pa, h=1, loss="se")
        verdict = "A mejor" if res["DM_stat"] > 0 else "A peor"
        sig = "p<0.05 (sig.)" if res["p_value"] < 0.05 else "no sig."
        dm_rows.append({
            "comparison": f"Modelo_A_v3 vs {name}",
            "DM_stat": res["DM_stat"],
            "p_value": res["p_value"],
            "verdict_A_vs_bench": verdict,
            "significance": sig,
            "n": res["n"],
        })
        print(f"    vs {name:14s}: DM={res['DM_stat']:+8.3f}  p={res['p_value']:.4f}  "
              f"→ {verdict} ({sig})")
    df_dm = pd.DataFrame(dm_rows)

    # ── Bootstrap CI 95% sobre RMSE (block-bootstrap circular 24h y 168h) ──
    print("\n  Bootstrap CI 95% (block-bootstrap circular, 1000 reps)...")
    boot_rows = []
    for name, p in PREDS.items():
        err = (y_te - p)
        for bsize in [24, 168]:
            samples = block_bootstrap_circular(err, block_size=bsize, n_reps=1000, seed=42)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            boot_rows.append({
                "model": name, "block_size_h": bsize,
                "rmse_point": float(np.sqrt(np.mean(err ** 2))),
                "rmse_ci_lo": float(lo), "rmse_ci_hi": float(hi),
                "rmse_boot_mean": float(samples.mean()),
                "rmse_boot_std": float(samples.std()),
            })
        print(f"    {name:14s}: 24h CI=[{boot_rows[-2]['rmse_ci_lo']:.3f}, "
              f"{boot_rows[-2]['rmse_ci_hi']:.3f}]  |  "
              f"168h CI=[{boot_rows[-1]['rmse_ci_lo']:.3f}, "
              f"{boot_rows[-1]['rmse_ci_hi']:.3f}]")
    df_boot = pd.DataFrame(boot_rows)

    # ── Tabla 2 final integrada ────────────────────────────────────────────
    df_table2 = df_metrics.merge(
        df_boot[df_boot["block_size_h"] == 24][["model", "rmse_ci_lo", "rmse_ci_hi"]],
        on="model", how="left"
    ).rename(columns={"rmse_ci_lo": "rmse_ci95_lo_bb24",
                       "rmse_ci_hi": "rmse_ci95_hi_bb24"})
    # Adjuntar DM
    dm_lookup = {row["comparison"].split(" vs ")[1]: row for row in dm_rows}
    df_table2["DM_vs_modelA_stat"] = df_table2["model"].map(
        lambda n: dm_lookup.get(n, {}).get("DM_stat", np.nan))
    df_table2["DM_vs_modelA_pvalue"] = df_table2["model"].map(
        lambda n: dm_lookup.get(n, {}).get("p_value", np.nan))
    df_table2["DM_verdict"] = df_table2["model"].map(
        lambda n: dm_lookup.get(n, {}).get("verdict_A_vs_bench", "—"))

    df_table2.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
    df_dm.to_csv(RESULTS_DIR / "dm_test_results.csv", index=False)
    df_boot.to_csv(RESULTS_DIR / "bootstrap_rmse_ci.csv", index=False)
    print("\n  Artefactos:")
    print(f"    {RESULTS_DIR / 'benchmark_comparison.csv'}")
    print(f"    {RESULTS_DIR / 'dm_test_results.csv'}")
    print(f"    {RESULTS_DIR / 'bootstrap_rmse_ci.csv'}")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("T5 — Benchmark comparison (Test 2024)\n"
                 "RMSE point estimate + bootstrap CI 95% (block 24h)",
                 fontsize=12, fontweight="bold")

    df_b24 = df_boot[df_boot["block_size_h"] == 24].set_index("model").reindex(df_table2["model"])
    models = df_b24.index.tolist()
    point = df_b24["rmse_point"].values
    err_lo = point - df_b24["rmse_ci_lo"].values
    err_hi = df_b24["rmse_ci_hi"].values - point

    cols = ["#27AE60" if m == "Modelo_A_v3" else
            ("#34495E" if m.startswith("M0") else "#3498DB") for m in models]
    axes[0].bar(range(len(models)), point, yerr=[err_lo, err_hi],
                color=cols, alpha=0.85, capsize=6)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("RMSE (€/MWh)"); axes[0].set_title("RMSE ± CI95% (bb24h)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # DM vs Modelo A
    df_dm_plot = df_dm.copy()
    df_dm_plot["bench"] = df_dm_plot["comparison"].str.split(" vs ").str[1]
    cols_dm = ["#C0392B" if p < 0.05 else "#7F8C8D" for p in df_dm_plot["p_value"]]
    axes[1].barh(df_dm_plot["bench"], df_dm_plot["DM_stat"], color=cols_dm, alpha=0.85)
    axes[1].axvline(0, color="black", lw=0.7)
    axes[1].axvline(1.96, color="#C0392B", ls="--", lw=0.8, label="±1.96 (p=0.05)")
    axes[1].axvline(-1.96, color="#C0392B", ls="--", lw=0.8)
    axes[1].set_xlabel("DM stat  (signo + → Modelo_A mejor)")
    axes[1].set_title("Diebold-Mariano vs Modelo_A_v3 (loss=SE)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out = PLOTS_DIR / "benchmark_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")
    print("\n  T5 COMPLETADA")


if __name__ == "__main__":
    main()
