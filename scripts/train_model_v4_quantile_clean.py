"""
train_model_v4_quantile_clean.py
--------------------------------
v4-QUANTILE CLEAN: XGBoost objective='reg:quantileerror' alpha=0.95
con FEATURES_A_CLEAN (las 26 originales del v3, sin leakage ATC)
mas 3 features adicionales no-leakage: price_es_lag24h,
price_fr_lag24h, spread_id_lag24h (todas con lag24h, del auction
anterior, sin contaminacion same-auction ni derivacion del spread).

Drop explicito de las features que generaron leakage en mi v4 anterior:
  - atc_fatigue, atc_fatigue_lag24h (atc_congestion deriva del spread
    en periodo pre-JAO 2019-ago 2022)
  - price_es_lag1h, price_fr_lag1h (mismo auction)
"""
import sys
from pathlib import Path
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
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

# FEATURES_A_CLEAN (identico a train_model_v3.py linea 65-78)
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

# Solo 3 features nuevas, todas lag24h+ (sin leakage same-auction)
NEW_FEATURES_CLEAN = [
    "price_es_lag24h",
    "price_fr_lag24h",
    "spread_id_lag24h",
]

TARGET = "spread_da"
ALPHA = 0.95


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mp(y, p, q=ALPHA):
    return float(pinball_loss(np.asarray(y), np.asarray(p), q).mean())


def main():
    print("=" * 72)
    print(f"  v4-QUANTILE CLEAN  alpha={ALPHA}")
    print(f"  Features: FEATURES_A_CLEAN (26) + 3 nuevas lag24h+ = 29")
    print(f"  Drop: atc_fatigue, atc_fatigue_lag24h, price_*_lag1h")
    print("=" * 72)

    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Crear los 3 lags nuevos (no en el dataset)
    df["price_es_lag24h"] = df["price_es"].shift(24)
    df["price_fr_lag24h"] = df["price_fr"].shift(24)
    df["spread_id_lag24h"] = df["spread_id"].shift(24)

    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"

    all_feats = FEATURES_A_CLEAN + NEW_FEATURES_CLEAN
    df_tr = df[tr_mask].dropna(subset=all_feats + [TARGET]).copy()
    df_te = df[te_mask].dropna(subset=all_feats + [TARGET]).copy()
    print(f"\n  Train: {len(df_tr):,} h    Test: {len(df_te):,} h")

    X_tr, y_tr = df_tr[all_feats], df_tr[TARGET]
    X_te, y_te = df_te[all_feats], df_te[TARGET]

    params = dict(
        objective="reg:quantileerror", quantile_alpha=ALPHA,
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )

    print("\n  Entrenando v4-quantile-clean...")
    model_qc = xgb.XGBRegressor(**params).fit(X_tr, y_tr, verbose=False)
    pa_qc = model_qc.predict(X_te)

    # Tambien una version SOLO con FEATURES_A_CLEAN (sin las 3 nuevas)
    # para aislar el efecto del feature set
    print("  Entrenando v4-quantile-cleanONLY (sin las 3 nuevas)...")
    X_tr_only = df_tr[FEATURES_A_CLEAN]
    X_te_only = df_te[FEATURES_A_CLEAN]
    model_qonly = xgb.XGBRegressor(**params).fit(X_tr_only, y_tr, verbose=False)
    pa_qonly = model_qonly.predict(X_te_only)

    print("  AR(1) y AR(24)...")
    p_ar1 = fit_predict_ar(y_tr.values, y_te.values, lags=1)
    p_ar24 = fit_predict_ar(y_tr.values, y_te.values, lags=24)

    print("\n" + "=" * 72)
    print("  RMSE Test 2024")
    print("=" * 72)
    for name, p in [
        ("v4-q-cleanONLY", pa_qonly),
        ("v4-q-clean (+3 new)", pa_qc),
        ("AR(1)", p_ar1),
        ("AR(24)", p_ar24),
    ]:
        print(f"  {name:25s}  RMSE = {rmse(y_te, p):.4f}")

    print("\n" + "=" * 72)
    print(f"  Mean pinball q={ALPHA}")
    print("=" * 72)
    for name, p in [
        ("v4-q-cleanONLY", pa_qonly),
        ("v4-q-clean (+3 new)", pa_qc),
        ("AR(1)", p_ar1),
        ("AR(24)", p_ar24),
    ]:
        print(f"  {name:25s}  pinball = {mp(y_te, p):.4f}")

    print("\n" + "=" * 72)
    print(f"  GW conditional pinball q={ALPHA}: each model vs AR(1)")
    print("=" * 72)
    for name, p in [("v4-q-cleanONLY", pa_qonly), ("v4-q-clean (+3 new)", pa_qc)]:
        gw = compare_models_pinball(y_te.values, p, p_ar1, q=ALPHA, conditional=True)
        diff = mp(y_te, p) - mp(y_te, p_ar1)
        winner = name if diff < 0 else "AR(1)"
        print(f"  {name:25s} vs AR(1)  stat={gw.statistic:+10.4f}  "
              f"p={gw.p_value:.4f}  pinball_diff={diff:+.4f}  -> mejor: {winner}")

    # Top features
    print("\n" + "=" * 72)
    print("  Top 12 features (v4-q-clean +3 new) gain")
    print("=" * 72)
    imp = sorted(zip(all_feats, model_qc.feature_importances_), key=lambda x: -x[1])
    for f, v in imp[:12]:
        tag = "[NUEVA]" if f in NEW_FEATURES_CLEAN else ""
        print(f"  {f:25s}  {v:.4f}  {tag}")


if __name__ == "__main__":
    main()
