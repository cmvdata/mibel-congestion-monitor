"""
arma_correction.py
------------------
Corrige el bias y la autocorrelacion de los residuos del v3-mean
ajustando un AR(p) sobre los residuos del training y aplicando la
correccion al test.

Modelo:
    r_t = mu + rho_1 * r_{t-1} + ... + rho_p * r_{t-p} + epsilon_t

Prediccion corregida en test:
    pred_corrected_t = pred_v3_t + mu + sum_k rho_k * (obs_{t-k} - pred_v3_{t-k})

El residuo en t-1 usa observed_{t-1} y pred_{t-1}, ambos disponibles
al momento de hacer la prediccion de t (mismo auction DA — leakage
operacional como el de spread_da_lag1h, aceptado en el paper).

Compara: v3 baseline vs v3+AR(p) correction. Reporta RMSE, pinball,
DM test, autocorrelacion residual nueva.

Output: results/arma_correction_results.csv
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from gw_pinball_test import dm_test, pinball_loss, squared_loss
from benchmarks_dm import fit_predict_ar

PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

FEATURES_A_CLEAN = [
    "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
    "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
    "spread_da_lag48h", "spread_da_lag168h",
    "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
]
TARGET = "spread_da"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mp(y, p, q=0.95):
    return float(pinball_loss(np.asarray(y), np.asarray(p), q).mean())


def main():
    print("=" * 72)
    print("  ARMA correction on v3-mean residuals")
    print("=" * 72)
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"
    df_tr = df[tr_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy().reset_index(drop=True)
    df_te = df[te_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy().reset_index(drop=True)
    print(f"  Train: {len(df_tr):,}  Test 2024: {len(df_te):,}")

    # Train v3-mean
    params = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    m = xgb.XGBRegressor(**params).fit(
        df_tr[FEATURES_A_CLEAN], df_tr[TARGET], verbose=False)

    # Predict on both train and test (need train predictions to compute
    # training residuals for AR fitting)
    pred_tr = m.predict(df_tr[FEATURES_A_CLEAN])
    pred_te = m.predict(df_te[FEATURES_A_CLEAN])
    y_tr = df_tr[TARGET].values
    y_te = df_te[TARGET].values

    resid_tr = y_tr - pred_tr
    resid_te_raw = y_te - pred_te

    print(f"\n  Residual diagnostics (train):")
    print(f"    mean = {resid_tr.mean():+.4f}")
    print(f"    autocorr lag 1 = {pd.Series(resid_tr).autocorr(1):.4f}")
    print(f"    autocorr lag 2 = {pd.Series(resid_tr).autocorr(2):.4f}")
    print(f"    autocorr lag 3 = {pd.Series(resid_tr).autocorr(3):.4f}")

    # Fit AR(p) for several p, select by AIC
    print("\n  AR(p) selection by AIC on training residuals:")
    aic_table = {}
    fitted = {}
    for p in [1, 2, 3, 6]:
        try:
            ar = AutoReg(resid_tr, lags=p, old_names=False).fit()
            aic_table[p] = ar.aic
            fitted[p] = ar
            print(f"    AR({p}): AIC = {ar.aic:.1f}  params = "
                  f"{[round(float(c), 3) for c in ar.params]}")
        except Exception as e:
            print(f"    AR({p}) failed: {e}")
    best_p = min(aic_table, key=aic_table.get)
    print(f"  -> best AR order by AIC: AR({best_p})")
    ar = fitted[best_p]
    print(f"  AR({best_p}) params: const={float(ar.params[0]):.4f}  "
          f"rho={[round(float(c), 3) for c in ar.params[1:]]}")

    # Apply correction to test predictions rolling 1-step ahead
    # pred_corrected_t = pred_v3_t + const + sum_k rho_k * (obs_{t-k} - pred_{t-k})
    const = float(ar.params[0])
    rhos = [float(c) for c in ar.params[1:]]
    full_y = np.concatenate([y_tr, y_te])
    full_pred = np.concatenate([pred_tr, pred_te])
    full_resid = full_y - full_pred
    n_tr = len(y_tr)

    pred_te_corrected = pred_te.copy()
    for i in range(len(y_te)):
        idx = n_tr + i
        correction = const
        for k, rho in enumerate(rhos, start=1):
            if idx - k >= 0:
                correction += rho * full_resid[idx - k]
        pred_te_corrected[i] = pred_te[i] + correction

    resid_te_corr = y_te - pred_te_corrected

    # Metrics
    print("\n" + "=" * 72)
    print(f"  Comparativa v3-mean vs v3-mean + AR({best_p}) correction")
    print("=" * 72)
    p_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    print(f"  RMSE v3 baseline      = {rmse(y_te, pred_te):.4f}")
    print(f"  RMSE v3 + AR({best_p}) corr  = {rmse(y_te, pred_te_corrected):.4f}")
    print(f"  RMSE AR(1) benchmark  = {rmse(y_te, p_ar1):.4f}")
    print(f"  Pinball q=0.95 v3 baseline      = {mp(y_te, pred_te):.4f}")
    print(f"  Pinball q=0.95 v3 + AR({best_p}) corr = {mp(y_te, pred_te_corrected):.4f}")
    print(f"  Pinball q=0.95 AR(1) benchmark  = {mp(y_te, p_ar1):.4f}")

    # DM v3 vs corrected
    dm = dm_test(squared_loss(y_te, pred_te), squared_loss(y_te, pred_te_corrected))
    print(f"\n  DM squared error: v3 vs v3+AR({best_p}) corrected")
    print(f"    stat = {dm.statistic:+.4f}  p = {dm.p_value:.4f}  "
          f"mean_diff(baseline-corrected) = {dm.mean_loss_diff:+.4f}")
    sign = "in favor of corrected" if dm.mean_loss_diff > 0 else "in favor of baseline"
    print(f"    -> {sign}")

    # DM corrected vs AR(1)
    dm2 = dm_test(squared_loss(y_te, pred_te_corrected), squared_loss(y_te, p_ar1))
    print(f"\n  DM squared error: v3+AR({best_p}) corrected vs AR(1)")
    print(f"    stat = {dm2.statistic:+.4f}  p = {dm2.p_value:.4f}  "
          f"mean_diff(corrected-AR1) = {dm2.mean_loss_diff:+.4f}")

    # Residual diagnostics after correction
    print(f"\n  Residual diagnostics (test, after correction):")
    print(f"    mean baseline    = {resid_te_raw.mean():+.4f}")
    print(f"    mean corrected   = {resid_te_corr.mean():+.4f}")
    print(f"    autocorr lag 1 baseline  = {pd.Series(resid_te_raw).autocorr(1):.4f}")
    print(f"    autocorr lag 1 corrected = {pd.Series(resid_te_corr).autocorr(1):.4f}")
    # T-test residual mean
    t_b, p_b = stats.ttest_1samp(resid_te_raw, 0.0)
    t_c, p_c = stats.ttest_1samp(resid_te_corr, 0.0)
    print(f"    t-test baseline   mean=0: t={t_b:+.2f}  p={p_b:.4f}")
    print(f"    t-test corrected  mean=0: t={t_c:+.2f}  p={p_c:.4f}")

    # Save
    rows = [
        {"model": "v3_baseline",
         "rmse": rmse(y_te, pred_te),
         "pinball_q95": mp(y_te, pred_te),
         "residual_mean": float(resid_te_raw.mean()),
         "residual_autocorr_lag1": float(pd.Series(resid_te_raw).autocorr(1))},
        {"model": f"v3_AR{best_p}_corrected",
         "rmse": rmse(y_te, pred_te_corrected),
         "pinball_q95": mp(y_te, pred_te_corrected),
         "residual_mean": float(resid_te_corr.mean()),
         "residual_autocorr_lag1": float(pd.Series(resid_te_corr).autocorr(1))},
        {"model": "AR1_benchmark",
         "rmse": rmse(y_te, p_ar1),
         "pinball_q95": mp(y_te, p_ar1),
         "residual_mean": np.nan,
         "residual_autocorr_lag1": np.nan},
    ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "arma_correction_results.csv", index=False)
    print(f"\n  CSV: {RESULTS_DIR / 'arma_correction_results.csv'}")


if __name__ == "__main__":
    main()
