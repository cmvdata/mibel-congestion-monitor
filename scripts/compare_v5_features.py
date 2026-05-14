"""
compare_v5_features.py
----------------------
Compara baseline (FEATURES_A_CLEAN, 26) contra extensiones progresivas:
  BASE         FEATURES_A_CLEAN                              (26)
  BASE+REN     FEATURES_A_CLEAN + 4 renovables               (30)
  BASE+ND      FEATURES_A_CLEAN + 3 nuclear/demand           (29)
  BASE+ALL     FEATURES_A_CLEAN + 4 ren + 3 nuclear/demand   (33)

Para cada feature set entrena v3-mean y v4-quantile-cleanONLY, evalua
RMSE, pinball q=0.95, DM/GW vs AR(1) y vs BASE.

Output: results/v5_features_comparison.csv
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
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231_v5.parquet"

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
RENEW = ["fr_solar_fc", "fr_wind_fc", "es_solar_fc", "es_wind_fc"]
NUC_DEM = ["fr_nuclear_avail", "es_demand_fc", "fr_demand_fc"]
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


def train_predict(df_tr, df_te, feats, objective):
    m = xgb.XGBRegressor(**base_params(objective)).fit(
        df_tr[feats], df_tr[TARGET], verbose=False)
    return m.predict(df_te[feats]), m


def main():
    print("=" * 78)
    print("  Compare BASE / +REN / +ND / +ALL feature sets")
    print("=" * 78)
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    sets = {
        "BASE":     FEATURES_A_CLEAN,
        "BASE+REN": FEATURES_A_CLEAN + RENEW,
        "BASE+ND":  FEATURES_A_CLEAN + NUC_DEM,
        "BASE+ALL": FEATURES_A_CLEAN + RENEW + NUC_DEM,
    }

    # Common test split with the strictest dropna (BASE+ALL) so all models
    # are compared on the same rows
    strict = FEATURES_A_CLEAN + RENEW + NUC_DEM + [TARGET]
    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"
    df_tr_strict = df[tr_mask].dropna(subset=strict).copy()
    df_te_strict = df[te_mask].dropna(subset=strict).copy().reset_index(drop=True)
    print(f"  Train (strict): {len(df_tr_strict):,}  "
          f"Test 2024 (strict): {len(df_te_strict):,}")

    y_te = df_te_strict[TARGET].values
    y_tr = df_tr_strict[TARGET].values

    # AR benchmarks
    p_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    p_ar24 = fit_predict_ar(y_tr, y_te, lags=24)

    rows = []
    preds = {}
    for name, feats in sets.items():
        print(f"\n  Training {name}  ({len(feats)} features)...")
        pM, mM = train_predict(df_tr_strict, df_te_strict, feats, "reg:squarederror")
        pQ, mQ = train_predict(df_tr_strict, df_te_strict, feats, "reg:quantileerror")
        preds[name] = {"mean": pM, "quant": pQ, "model_q": mQ, "feats": feats}
        rows.append({"set": name, "kind": "v3-mean", "n_feats": len(feats),
                     "rmse": rmse(y_te, pM), "pinball_q95": mp(y_te, pM)})
        rows.append({"set": name, "kind": "v4-quant", "n_feats": len(feats),
                     "rmse": rmse(y_te, pQ), "pinball_q95": mp(y_te, pQ)})
    rows.append({"set": "AR(1)", "kind": "benchmark", "n_feats": 1,
                 "rmse": rmse(y_te, p_ar1), "pinball_q95": mp(y_te, p_ar1)})
    rows.append({"set": "AR(24)", "kind": "benchmark", "n_feats": 24,
                 "rmse": rmse(y_te, p_ar24), "pinball_q95": mp(y_te, p_ar24)})

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  RMSE and Pinball q=0.95 (Test 2024, common rows)")
    print("=" * 78)
    print(df_out.to_string(index=False))

    # DM mean: each +X vs BASE
    print("\n" + "=" * 78)
    print("  DM squared-error: each extended set vs BASE (v3-mean)")
    print("=" * 78)
    pBASE_mean = preds["BASE"]["mean"]
    for name in ["BASE+REN", "BASE+ND", "BASE+ALL"]:
        p_ext = preds[name]["mean"]
        dm = dm_test(squared_loss(y_te, p_ext), squared_loss(y_te, pBASE_mean))
        sign = "in favor of " + name if dm.mean_loss_diff < 0 else "in favor of BASE"
        print(f"  {name:10s} vs BASE  stat={dm.statistic:+7.3f}  "
              f"p={dm.p_value:.4f}  diff(ext-BASE)={dm.mean_loss_diff:+.4f}  "
              f"-> {sign}")

    # GW pinball quant: each +X vs BASE (v4-quant) and vs AR(1)
    print("\n" + "=" * 78)
    print("  GW pinball q=0.95: each v4-quant extended vs BASE-quant and AR(1)")
    print("=" * 78)
    pBASE_quant = preds["BASE"]["quant"]
    for name in ["BASE+REN", "BASE+ND", "BASE+ALL"]:
        p_ext = preds[name]["quant"]
        gw_b = compare_models_pinball(y_te, p_ext, pBASE_quant, q=ALPHA, conditional=True)
        diff_b = mp(y_te, p_ext) - mp(y_te, pBASE_quant)
        gw_a = compare_models_pinball(y_te, p_ext, p_ar1, q=ALPHA, conditional=True)
        diff_a = mp(y_te, p_ext) - mp(y_te, p_ar1)
        print(f"  {name:10s} v4q vs BASE-quant  stat={gw_b.statistic:+8.2f}  "
              f"p={gw_b.p_value:.4f}  diff={diff_b:+.4f}")
        print(f"  {name:10s} v4q vs AR(1)       stat={gw_a.statistic:+8.2f}  "
              f"p={gw_a.p_value:.4f}  diff={diff_a:+.4f}")

    # Top features for v4-quant BASE+ALL
    print("\n" + "=" * 78)
    print("  Top 15 features v4-quant BASE+ALL")
    print("=" * 78)
    feats_all = preds["BASE+ALL"]["feats"]
    imp = sorted(zip(feats_all, preds["BASE+ALL"]["model_q"].feature_importances_),
                 key=lambda x: -x[1])
    for f, v in imp[:15]:
        tag = ""
        if f in RENEW: tag = "[RENEW]"
        elif f in NUC_DEM: tag = "[NUC/DEM]"
        print(f"  {f:25s}  {v:.4f}  {tag}")

    out_path = RESULTS_DIR / "v5_features_comparison.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
