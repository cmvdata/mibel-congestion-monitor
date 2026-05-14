"""
compare_models_with_renewables.py
---------------------------------
Comparativa antes/despues de anadir las 4 features de forecast renovable
al feature set. Entrena 4 modelos:

    A) v3-mean    (FEATURES_A_CLEAN, 26 features, objective=reg:squarederror)
    B) v3-mean+R  (FEATURES_A_CLEAN + 4 renovables, mismo objective)
    C) v4-quant   (FEATURES_A_CLEAN, objective=reg:quantileerror alpha=0.95)
    D) v4-quant+R (FEATURES_A_CLEAN + 4 renovables, mismo objective quantile)

Salida: results/renewables_ablation.csv
        results/renewables_ablation_features.txt
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
from gw_pinball_test import (
    compare_models_pinball,
    dm_test,
    pinball_loss,
    squared_loss,
)
from benchmarks_dm import fit_predict_ar

PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
DATASET_V4 = PROC_DIR / "mibel_dataset_20190101_20241231_v4.parquet"

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
RENEWABLES = ["fr_solar_fc", "fr_wind_fc", "es_solar_fc", "es_wind_fc"]
TARGET = "spread_da"
ALPHA = 0.95


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mp(y, p, q=ALPHA):
    return float(pinball_loss(np.asarray(y), np.asarray(p), q).mean())


def base_params(objective: str):
    p = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    if objective == "reg:quantileerror":
        p["objective"] = "reg:quantileerror"
        p["quantile_alpha"] = ALPHA
    return p


def train_predict(df_tr, df_te, features, objective):
    X_tr, y_tr = df_tr[features], df_tr[TARGET]
    X_te = df_te[features]
    m = xgb.XGBRegressor(**base_params(objective))
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict(X_te), m


def main():
    print("=" * 72)
    print("  Compare: with vs without renewable forecast features")
    print("=" * 72)

    df = pd.read_parquet(DATASET_V4).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"

    feats_base = FEATURES_A_CLEAN
    feats_r = FEATURES_A_CLEAN + RENEWABLES

    df_tr_base = df[tr_mask].dropna(subset=feats_base + [TARGET]).copy()
    df_te_base = df[te_mask].dropna(subset=feats_base + [TARGET]).copy()
    df_tr_r = df[tr_mask].dropna(subset=feats_r + [TARGET]).copy()
    df_te_r = df[te_mask].dropna(subset=feats_r + [TARGET]).copy()
    print(f"  Train base: {len(df_tr_base):,}  Test base: {len(df_te_base):,}")
    print(f"  Train +R:   {len(df_tr_r):,}  Test +R:   {len(df_te_r):,}")

    # Train all four
    print("\n  Training A) v3-mean (FEATURES_A_CLEAN)...")
    pA, _ = train_predict(df_tr_base, df_te_base, feats_base, "reg:squarederror")
    print("  Training B) v3-mean +R (FEATURES_A_CLEAN + renewables)...")
    pB, mB = train_predict(df_tr_r, df_te_r, feats_r, "reg:squarederror")
    print("  Training C) v4-quant (FEATURES_A_CLEAN)...")
    pC, _ = train_predict(df_tr_base, df_te_base, feats_base, "reg:quantileerror")
    print("  Training D) v4-quant +R (FEATURES_A_CLEAN + renewables)...")
    pD, mD = train_predict(df_tr_r, df_te_r, feats_r, "reg:quantileerror")
    print("  AR(1) and AR(24) on common test...")
    # Use the +R test split for fair comparison (smaller of the two)
    y_tr_v = df_tr_r[TARGET].values
    y_te_v = df_te_r[TARGET].values
    p_ar1 = fit_predict_ar(y_tr_v, y_te_v, lags=1)
    p_ar24 = fit_predict_ar(y_tr_v, y_te_v, lags=24)

    # Note: pA, pC are on df_te_base (potentially different size). For
    # a clean comparison, retrain A and C on df_tr_r so test set matches.
    if len(df_te_r) != len(df_te_base):
        print("  [info] re-aligning A and C to the +R test split for fair compare")
        pA, _ = train_predict(df_tr_r, df_te_r, feats_base, "reg:squarederror")
        pC, _ = train_predict(df_tr_r, df_te_r, feats_base, "reg:quantileerror")

    # Metrics table
    print("\n" + "=" * 72)
    print("  RMSE (Test 2024)")
    print("=" * 72)
    for name, p in [("A v3-mean", pA), ("B v3-mean +R", pB),
                    ("C v4-quant", pC), ("D v4-quant +R", pD),
                    ("AR(1)", p_ar1), ("AR(24)", p_ar24)]:
        print(f"  {name:15s}  RMSE = {rmse(y_te_v, p):.4f} EUR/MWh")

    print("\n" + "=" * 72)
    print(f"  Mean pinball q={ALPHA}")
    print("=" * 72)
    for name, p in [("A v3-mean", pA), ("B v3-mean +R", pB),
                    ("C v4-quant", pC), ("D v4-quant +R", pD),
                    ("AR(1)", p_ar1), ("AR(24)", p_ar24)]:
        print(f"  {name:15s}  pinball = {mp(y_te_v, p):.4f}")

    # DM on squared error: A vs B (does adding R help in MSE?)
    print("\n" + "=" * 72)
    print("  DM squared-error: efecto de anadir renovables")
    print("=" * 72)
    dm_ab = dm_test(squared_loss(y_te_v, pA), squared_loss(y_te_v, pB))
    print(f"  A v3-mean vs B v3-mean+R   stat={dm_ab.statistic:+.4f}  "
          f"p={dm_ab.p_value:.4f}  mean_diff(A-B)={dm_ab.mean_loss_diff:+.4f}")

    # GW pinball: C vs D, then C vs AR(1), then D vs AR(1)
    print("\n" + "=" * 72)
    print(f"  GW conditional pinball q={ALPHA}")
    print("=" * 72)
    for left_name, left in [("C v4-quant", pC), ("D v4-quant+R", pD)]:
        for right_name, right in [("AR(1)", p_ar1), ("AR(24)", p_ar24)]:
            gw = compare_models_pinball(y_te_v, left, right, q=ALPHA, conditional=True)
            diff = mp(y_te_v, left) - mp(y_te_v, right)
            print(f"  {left_name:15s} vs {right_name:8s}  stat={gw.statistic:+8.2f}  "
                  f"p={gw.p_value:.4f}  pinball_diff(L-R)={diff:+.4f}")

    # C vs D directly
    gw_cd = compare_models_pinball(y_te_v, pD, pC, q=ALPHA, conditional=True)
    diff_cd = mp(y_te_v, pD) - mp(y_te_v, pC)
    print(f"  D v4-quant+R  vs C v4-quant  stat={gw_cd.statistic:+8.2f}  "
          f"p={gw_cd.p_value:.4f}  pinball_diff(D-C)={diff_cd:+.4f}")

    # Top features
    print("\n" + "=" * 72)
    print("  Top 12 features importance (B v3-mean+R, gain)")
    print("=" * 72)
    imp = sorted(zip(feats_r, mB.feature_importances_), key=lambda x: -x[1])
    for f, v in imp[:12]:
        tag = "[RENEWABLE]" if f in RENEWABLES else ""
        print(f"  {f:25s}  {v:.4f}  {tag}")

    print("\n  Top 12 features importance (D v4-quant+R, gain)")
    print("=" * 72)
    imp = sorted(zip(feats_r, mD.feature_importances_), key=lambda x: -x[1])
    for f, v in imp[:12]:
        tag = "[RENEWABLE]" if f in RENEWABLES else ""
        print(f"  {f:25s}  {v:.4f}  {tag}")

    # Save summary CSV
    rows = []
    for name, p in [("A_v3_mean", pA), ("B_v3_mean_R", pB),
                    ("C_v4_quant", pC), ("D_v4_quant_R", pD),
                    ("AR1", p_ar1), ("AR24", p_ar24)]:
        rows.append({
            "model": name,
            "rmse": rmse(y_te_v, p),
            "pinball_q95": mp(y_te_v, p),
        })
    out_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "renewables_ablation.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
