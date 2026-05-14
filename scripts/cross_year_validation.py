"""
cross_year_validation.py
------------------------
Robustez temporal: en lugar de un unico test split (2024), evaluamos
el modelo en tres folds distintos rotando el ano de test:

  Fold 1: train 2019-2021, test 2022 (regimen Crisis + Iberian Exception
          inicial; alta volatilidad)
  Fold 2: train 2019-2022, test 2023 (regimen Iberian Exception puro)
  Fold 3: train 2019-2023, test 2024 (regimen post-Exception, default)

Para cada fold: v3-mean (XGB MSE) y v4-quant (XGB quantile q=0.95) +
AR(1) y AR(24) como benchmarks. Reportamos RMSE, pinball q=0.95, GW
pinball vs AR(1).

Output: results/cross_year_validation.csv
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from gw_pinball_test import compare_models_pinball, dm_test, pinball_loss, squared_loss
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
ALPHA = 0.95


def base_params(objective="reg:squarederror"):
    p = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
        objective=objective,
    )
    if objective == "reg:quantileerror":
        p["quantile_alpha"] = ALPHA
    return p


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mp(y, p, q=ALPHA):
    return float(pinball_loss(np.asarray(y), np.asarray(p), q).mean())


def run_fold(df, train_start, train_end, test_year):
    tr_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    te_mask = ((df["timestamp"] >= f"{test_year}-01-01") &
               (df["timestamp"] < f"{test_year + 1}-01-01"))
    df_tr = df[tr_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy()
    df_te = df[te_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy().reset_index(drop=True)
    y_tr = df_tr[TARGET].values
    y_te = df_te[TARGET].values

    # v3-mean
    m3 = xgb.XGBRegressor(**base_params("reg:squarederror")).fit(
        df_tr[FEATURES_A_CLEAN], y_tr, verbose=False)
    p_v3 = m3.predict(df_te[FEATURES_A_CLEAN])
    # v4-quant
    m4 = xgb.XGBRegressor(**base_params("reg:quantileerror")).fit(
        df_tr[FEATURES_A_CLEAN], y_tr, verbose=False)
    p_v4 = m4.predict(df_te[FEATURES_A_CLEAN])
    # Benchmarks
    p_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    p_ar24 = fit_predict_ar(y_tr, y_te, lags=24)

    # GW pinball: v4 vs AR(1) — only meaningful test
    gw = compare_models_pinball(y_te, p_v4, p_ar1, q=ALPHA, conditional=True)
    diff_v4_ar1 = mp(y_te, p_v4) - mp(y_te, p_ar1)
    dm = dm_test(squared_loss(y_te, p_v3), squared_loss(y_te, p_ar1))

    return {
        "fold": f"test_{test_year}",
        "n_train": len(df_tr),
        "n_test": len(df_te),
        "y_te_mean": float(np.mean(y_te)),
        "y_te_std": float(np.std(y_te)),
        "y_te_q95": float(np.quantile(y_te, 0.95)),
        "rmse_v3_mean": rmse(y_te, p_v3),
        "rmse_v4_quant": rmse(y_te, p_v4),
        "rmse_AR1": rmse(y_te, p_ar1),
        "rmse_AR24": rmse(y_te, p_ar24),
        "pinball_v3_mean": mp(y_te, p_v3),
        "pinball_v4_quant": mp(y_te, p_v4),
        "pinball_AR1": mp(y_te, p_ar1),
        "pinball_AR24": mp(y_te, p_ar24),
        "GW_v4_vs_AR1_stat": float(gw.statistic),
        "GW_v4_vs_AR1_pvalue": float(gw.p_value),
        "GW_v4_vs_AR1_pinball_diff": diff_v4_ar1,
        "DM_v3_vs_AR1_stat": float(dm.statistic),
        "DM_v3_vs_AR1_pvalue": float(dm.p_value),
    }


def main():
    print("=" * 78)
    print("  Cross-year validation: 3 rolling-origin folds")
    print("=" * 78)
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    folds = [
        ("2019-01-01", "2022-01-01", 2022),
        ("2019-01-01", "2023-01-01", 2023),
        ("2019-01-01", "2024-01-01", 2024),
    ]

    rows = []
    for tr_s, tr_e, te_y in folds:
        print(f"\n  Fold test_{te_y}: train {tr_s} -> {tr_e}, test {te_y}-01-01 -> {te_y}-12-31")
        r = run_fold(df, tr_s, tr_e, te_y)
        print(f"    n_train = {r['n_train']:,}  n_test = {r['n_test']:,}")
        print(f"    y_test  mean={r['y_te_mean']:.2f}  std={r['y_te_std']:.2f}  "
              f"q95={r['y_te_q95']:.2f}")
        print(f"    RMSE    v3={r['rmse_v3_mean']:.3f}  v4q={r['rmse_v4_quant']:.3f}  "
              f"AR1={r['rmse_AR1']:.3f}  AR24={r['rmse_AR24']:.3f}")
        print(f"    Pinball v3={r['pinball_v3_mean']:.3f}  v4q={r['pinball_v4_quant']:.3f}  "
              f"AR1={r['pinball_AR1']:.3f}  AR24={r['pinball_AR24']:.3f}")
        print(f"    GW v4 vs AR1: stat={r['GW_v4_vs_AR1_stat']:+.2f}  "
              f"p={r['GW_v4_vs_AR1_pvalue']:.4f}  diff={r['GW_v4_vs_AR1_pinball_diff']:+.3f}")
        print(f"    DM v3 vs AR1: stat={r['DM_v3_vs_AR1_stat']:+.2f}  "
              f"p={r['DM_v3_vs_AR1_pvalue']:.4f}")
        rows.append(r)

    df_out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "cross_year_validation.csv"
    df_out.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("  Summary across folds")
    print("=" * 78)
    cols = ["fold", "rmse_v3_mean", "rmse_v4_quant", "rmse_AR1",
            "pinball_v3_mean", "pinball_v4_quant", "pinball_AR1",
            "GW_v4_vs_AR1_pvalue"]
    print(df_out[cols].to_string(index=False))
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
