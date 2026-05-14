"""
train_decomposed_es_fr.py
-------------------------
Decomposed forecasting: dos XGBoost separados, uno para price_es y
otro para price_fr. El spread predicho es la diferencia:

    pred_spread = pred_es - pred_fr

Hipotesis: predecir cada precio captura senal fuerte (cada precio
esta dominado por sus fundamentales locales). El spread emerge como
diferencia limpia de dos predicciones bien especificadas.

Comparativa contra:
    - v3-mean direct (1 XGB sobre spread directo) [single model]
    - AR(1) y AR(24) sobre spread

Tambien version con renovables.

Output: results/decomposed_vs_direct.csv
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
    compare_models_pinball, dm_test, pinball_loss, squared_loss,
)
from benchmarks_dm import fit_predict_ar

PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231_v4.parquet"

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


def build_lags(df, base_col, lags):
    """Create lag features from base_col."""
    new_cols = {}
    for L in lags:
        new_cols[f"{base_col}_lag{L}h"] = df[base_col].shift(L)
    return pd.DataFrame(new_cols, index=df.index)


LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_MA = [24, 168]


def build_price_features(df, base_col):
    """Build standalone-price feature set: own lags + rolling stats."""
    cols = build_lags(df, base_col, LAGS)
    cols[f"{base_col}_ma24h"] = df[base_col].rolling(24, min_periods=12).mean().shift(1)
    cols[f"{base_col}_ma168h"] = df[base_col].rolling(168, min_periods=84).mean().shift(1)
    cols[f"{base_col}_std24h"] = df[base_col].rolling(24, min_periods=12).std().shift(1)
    return cols


SHARED_FUNDAMENTALS = [
    "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
    "ttf_lag24h", "ttf_lag168h",
    "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
    "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
]
ES_RENEWABLES = ["es_solar_fc", "es_wind_fc"]
FR_RENEWABLES = ["fr_solar_fc", "fr_wind_fc"]


def main():
    print("=" * 72)
    print("  Decomposed forecasting: price_ES + price_FR -> spread")
    print("=" * 72)

    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build lag features for price_es and price_fr
    df_es_lags = build_price_features(df, "price_es")
    df_fr_lags = build_price_features(df, "price_fr")
    df = pd.concat([df, df_es_lags, df_fr_lags], axis=1)

    # Feature sets for each component model
    es_feats_base = list(df_es_lags.columns) + SHARED_FUNDAMENTALS
    fr_feats_base = list(df_fr_lags.columns) + SHARED_FUNDAMENTALS
    es_feats_r = es_feats_base + ES_RENEWABLES
    fr_feats_r = fr_feats_base + FR_RENEWABLES

    # And for direct spread model (control)
    SPREAD_FEATURES = [
        "spread_da_lag1h", "spread_da_lag2h", "spread_da_lag3h",
        "spread_da_lag6h", "spread_da_lag12h", "spread_da_lag24h",
        "spread_da_lag48h", "spread_da_lag168h",
        "spread_da_ma24h", "spread_da_ma168h", "spread_da_std24h",
    ] + SHARED_FUNDAMENTALS

    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"

    all_needed = list(set(
        es_feats_r + fr_feats_r + SPREAD_FEATURES
        + ["price_es", "price_fr", "spread_da"]
    ))
    df_tr = df[tr_mask].dropna(subset=all_needed).copy()
    df_te = df[te_mask].dropna(subset=all_needed).copy()
    print(f"  Train: {len(df_tr):,}  Test 2024: {len(df_te):,}")

    y_es_tr = df_tr["price_es"].values
    y_fr_tr = df_tr["price_fr"].values
    y_spread_tr = df_tr["spread_da"].values
    y_es_te = df_te["price_es"].values
    y_fr_te = df_te["price_fr"].values
    y_spread_te = df_te["spread_da"].values

    # --- Train decomposed models (base features) ---
    print("\n  Training XGB_ES (price_es target, base features)...")
    m_es = xgb.XGBRegressor(**base_params()).fit(
        df_tr[es_feats_base], y_es_tr, verbose=False)
    pred_es_te = m_es.predict(df_te[es_feats_base])

    print("  Training XGB_FR (price_fr target, base features)...")
    m_fr = xgb.XGBRegressor(**base_params()).fit(
        df_tr[fr_feats_base], y_fr_tr, verbose=False)
    pred_fr_te = m_fr.predict(df_te[fr_feats_base])

    pred_spread_decomp = pred_es_te - pred_fr_te

    # --- Train decomposed +renewables ---
    print("  Training XGB_ES_R (with ES renewables)...")
    m_es_r = xgb.XGBRegressor(**base_params()).fit(
        df_tr[es_feats_r], y_es_tr, verbose=False)
    pred_es_te_r = m_es_r.predict(df_te[es_feats_r])
    print("  Training XGB_FR_R (with FR renewables)...")
    m_fr_r = xgb.XGBRegressor(**base_params()).fit(
        df_tr[fr_feats_r], y_fr_tr, verbose=False)
    pred_fr_te_r = m_fr_r.predict(df_te[fr_feats_r])
    pred_spread_decomp_r = pred_es_te_r - pred_fr_te_r

    # --- Control: direct spread model (single XGB) ---
    print("  Training XGB_direct (spread target, FEATURES_A_CLEAN)...")
    m_direct = xgb.XGBRegressor(**base_params()).fit(
        df_tr[SPREAD_FEATURES], y_spread_tr, verbose=False)
    pred_spread_direct = m_direct.predict(df_te[SPREAD_FEATURES])

    # --- AR benchmarks ---
    print("  AR(1) and AR(24) on spread...")
    p_ar1 = fit_predict_ar(y_spread_tr, y_spread_te, lags=1)
    p_ar24 = fit_predict_ar(y_spread_tr, y_spread_te, lags=24)

    # --- Diagnostics on component models ---
    print("\n" + "=" * 72)
    print("  Component-model RMSE on Test 2024")
    print("=" * 72)
    print(f"  price_ES model      RMSE = {rmse(y_es_te, pred_es_te):.4f} EUR/MWh")
    print(f"  price_FR model      RMSE = {rmse(y_fr_te, pred_fr_te):.4f} EUR/MWh")
    print(f"  price_ES +R model   RMSE = {rmse(y_es_te, pred_es_te_r):.4f} EUR/MWh")
    print(f"  price_FR +R model   RMSE = {rmse(y_fr_te, pred_fr_te_r):.4f} EUR/MWh")

    # --- Spread metrics ---
    print("\n" + "=" * 72)
    print("  Spread prediction: RMSE on Test 2024 (n={:,})".format(len(y_spread_te)))
    print("=" * 72)
    spread_models = [
        ("Decomposed (ES - FR)", pred_spread_decomp),
        ("Decomposed +R", pred_spread_decomp_r),
        ("Direct (single XGB)", pred_spread_direct),
        ("AR(1)", p_ar1),
        ("AR(24)", p_ar24),
    ]
    for name, p in spread_models:
        print(f"  {name:25s}  RMSE = {rmse(y_spread_te, p):.4f}")

    print("\n" + "=" * 72)
    print(f"  Spread prediction: Pinball q={ALPHA}")
    print("=" * 72)
    for name, p in spread_models:
        print(f"  {name:25s}  pinball = {mp(y_spread_te, p):.4f}")

    # --- DM and GW tests ---
    print("\n" + "=" * 72)
    print("  DM squared-error (Decomposed vs Direct, vs AR(1))")
    print("=" * 72)
    for left_name, left in [("Decomposed", pred_spread_decomp),
                             ("Decomposed +R", pred_spread_decomp_r)]:
        for right_name, right in [("Direct", pred_spread_direct),
                                   ("AR(1)", p_ar1)]:
            dm = dm_test(squared_loss(y_spread_te, left),
                         squared_loss(y_spread_te, right))
            print(f"  {left_name:15s} vs {right_name:8s}  stat={dm.statistic:+7.3f}  "
                  f"p={dm.p_value:.4f}  mean_diff(L-R)={dm.mean_loss_diff:+.4f}")

    print("\n" + "=" * 72)
    print(f"  GW conditional pinball q={ALPHA}")
    print("=" * 72)
    for left_name, left in [("Decomposed", pred_spread_decomp),
                             ("Decomposed +R", pred_spread_decomp_r)]:
        for right_name, right in [("Direct", pred_spread_direct),
                                   ("AR(1)", p_ar1)]:
            gw = compare_models_pinball(y_spread_te, left, right,
                                        q=ALPHA, conditional=True)
            diff = mp(y_spread_te, left) - mp(y_spread_te, right)
            print(f"  {left_name:15s} vs {right_name:8s}  stat={gw.statistic:+8.2f}  "
                  f"p={gw.p_value:.4f}  pinball_diff(L-R)={diff:+.4f}")

    # --- Top features in ES and FR models ---
    print("\n" + "=" * 72)
    print("  Top 8 features price_ES model (base)")
    print("=" * 72)
    imp = sorted(zip(es_feats_base, m_es.feature_importances_), key=lambda x: -x[1])
    for f, v in imp[:8]:
        print(f"  {f:25s}  {v:.4f}")

    print("\n  Top 8 features price_FR model (base)")
    print("=" * 72)
    imp = sorted(zip(fr_feats_base, m_fr.feature_importances_), key=lambda x: -x[1])
    for f, v in imp[:8]:
        print(f"  {f:25s}  {v:.4f}")

    # Save CSV
    out_rows = []
    for name, p in spread_models:
        out_rows.append({
            "model": name,
            "rmse_spread": rmse(y_spread_te, p),
            "pinball_spread_q95": mp(y_spread_te, p),
        })
    out_df = pd.DataFrame(out_rows)
    out_path = RESULTS_DIR / "decomposed_vs_direct.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
