"""
v2_stage_a_rich_features.py
---------------------------
Camino 1: Stage A con features fisicas extendidas.

Anade a las 18 features de v2_stage_a_classifier.py las features
disponibles en mibel_dataset_..._v5.parquet:
  - es_solar_fc, es_wind_fc  (forecast renovable ES, MW)
  - fr_solar_fc, fr_wind_fc  (forecast renovable FR, MW)
  - es_demand_fc, fr_demand_fc  (forecast demanda ES/FR, MW)
  - fr_nuclear_avail  (disponibilidad nuclear FR, MW, lag 24h sin leakage)

Y construye features DERIVADAS fisicamente justificadas:
  - solar_penetration_es = es_solar_fc / es_demand_fc
  - solar_penetration_fr = fr_solar_fc / fr_demand_fc
  - solar_pen_asym = solar_penetration_es - solar_penetration_fr
  - wind_penetration_es = es_wind_fc / es_demand_fc
  - wind_penetration_fr = fr_wind_fc / fr_demand_fc
  - wind_pen_asym = wind_penetration_es - wind_penetration_fr
  - demand_log_ratio = log(es_demand_fc / fr_demand_fc)
  - nuclear_ratio_fr = fr_nuclear_avail / fr_demand_fc

Total: 18 + 7 + 8 = 33 features.

Reentrenamos logistic + LGBM con walk-forward semanal sobre 2024 y
reportamos AUC. Comparamos contra el v2 baseline (18 features).

Output: results/v2_stage_a_rich_features_metrics.json
"""
from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import v2_stage_a_classifier as v2sa

RESULTS = BASE / "results"
DATA_V5 = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231_v5.parquet"

NEW_RAW_FEATURES = [
    "es_solar_fc", "es_wind_fc",
    "fr_solar_fc", "fr_wind_fc",
    "es_demand_fc", "fr_demand_fc",
    "fr_nuclear_avail",
]

DERIVED_FEATURES = [
    "solar_pen_es", "solar_pen_fr", "solar_pen_asym",
    "wind_pen_es", "wind_pen_fr", "wind_pen_asym",
    "demand_log_ratio",
    "nuclear_ratio_fr",
]


def add_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = v2sa.add_features(df)  # base features + state target
    # Forecast penetracion = forecast generacion / forecast demanda
    eps = 1e-3
    df["solar_pen_es"] = df["es_solar_fc"] / (df["es_demand_fc"].abs() + eps)
    df["solar_pen_fr"] = df["fr_solar_fc"] / (df["fr_demand_fc"].abs() + eps)
    df["solar_pen_asym"] = df["solar_pen_es"] - df["solar_pen_fr"]
    df["wind_pen_es"] = df["es_wind_fc"] / (df["es_demand_fc"].abs() + eps)
    df["wind_pen_fr"] = df["fr_wind_fc"] / (df["fr_demand_fc"].abs() + eps)
    df["wind_pen_asym"] = df["wind_pen_es"] - df["wind_pen_fr"]
    df["demand_log_ratio"] = np.log(
        (df["es_demand_fc"].abs() + eps) / (df["fr_demand_fc"].abs() + eps)
    )
    df["nuclear_ratio_fr"] = df["fr_nuclear_avail"] / (df["fr_demand_fc"].abs() + eps)
    return df


FEATURES = v2sa.FEATURES + NEW_RAW_FEATURES + DERIVED_FEATURES


def walk_forward_rich(df, model_fn_fit, model_fn_pred):
    """Same walk-forward as v2sa but using the extended FEATURES list."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_test = df[df["timestamp"] >= "2024-01-01"].copy()
    df_test["_week"] = df_test["timestamp"].dt.isocalendar().week
    df_test["_year"] = df_test["timestamp"].dt.year
    df_test["_week_id"] = df_test["_year"] * 100 + df_test["_week"]
    week_ids = sorted(df_test["_week_id"].unique())

    all_preds = []
    for w_id in week_ids:
        df_week = df_test[df_test["_week_id"] == w_id].copy()
        if df_week.empty:
            continue
        cutoff = df_week["timestamp"].min()
        df_tr = df[df["timestamp"] < cutoff].dropna(
            subset=FEATURES + ["state"]).copy()
        df_te = df_week.dropna(subset=FEATURES + ["state"]).copy()
        if df_tr.empty or df_te.empty:
            continue
        X_tr = df_tr[FEATURES].values
        y_tr = df_tr["state"].values
        X_te = df_te[FEATURES].values
        y_te = df_te["state"].values
        artefacts = model_fn_fit(X_tr, y_tr)
        proba = model_fn_pred(artefacts, X_te)
        all_preds.append(pd.DataFrame({
            "timestamp": df_te["timestamp"].values,
            "y_true": y_te, "proba": proba, "week_id": w_id,
        }))
    return pd.concat(all_preds, ignore_index=True)


def main():
    print("=" * 72)
    print("  v2 Stage A — RICH FEATURES (camino 1)")
    print(f"  Total features: {len(FEATURES)}")
    print("=" * 72)
    df = pd.read_parquet(DATA_V5)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_rich_features(df)
    pos_rate = float(df["state"].mean())
    print(f"  n={len(df):,}  positive rate: {pos_rate*100:.2f}%")
    print(f"  Base features:    {len(v2sa.FEATURES)}")
    print(f"  Raw v5 features:  {len(NEW_RAW_FEATURES)}: {NEW_RAW_FEATURES}")
    print(f"  Derived features: {len(DERIVED_FEATURES)}: {DERIVED_FEATURES}")

    # Logistic
    print("\n  Walk-forward weekly logistic regression...")
    pred_log = walk_forward_rich(df, v2sa.fit_logistic, v2sa.predict_logistic)
    auc_log = float(roc_auc_score(pred_log["y_true"], pred_log["proba"]))
    ap_log = float(average_precision_score(pred_log["y_true"], pred_log["proba"]))
    brier_log = float(brier_score_loss(pred_log["y_true"], pred_log["proba"]))
    lead_log = v2sa.lead_time_to_transition(pred_log)
    print(f"    AUC-ROC: {auc_log:.4f}  AUC-PR: {ap_log:.4f}  Brier: {brier_log:.4f}")
    print(f"    lead median: {lead_log['median_lead_h']}h  "
          f"frac advance: {lead_log.get('frac_predicted_in_advance', 0):.2f}")

    # LightGBM
    print("\n  Walk-forward weekly LightGBM...")
    pred_gbm = walk_forward_rich(df, v2sa.fit_lgbm, v2sa.predict_lgbm)
    auc_gbm = float(roc_auc_score(pred_gbm["y_true"], pred_gbm["proba"]))
    ap_gbm = float(average_precision_score(pred_gbm["y_true"], pred_gbm["proba"]))
    brier_gbm = float(brier_score_loss(pred_gbm["y_true"], pred_gbm["proba"]))
    lead_gbm = v2sa.lead_time_to_transition(pred_gbm)
    print(f"    AUC-ROC: {auc_gbm:.4f}  AUC-PR: {ap_gbm:.4f}  Brier: {brier_gbm:.4f}")
    print(f"    lead median: {lead_gbm['median_lead_h']}h  "
          f"frac advance: {lead_gbm.get('frac_predicted_in_advance', 0):.2f}")

    # Comparativa contra v2 baseline (18 features)
    print("\n" + "=" * 72)
    print("  Comparison: 18 features (baseline) vs 33 features (rich)")
    print("=" * 72)
    baseline_path = RESULTS / "v2_stage_a_metrics.json"
    if baseline_path.exists():
        bl = json.load(open(baseline_path))
        print(f"  Logistic:  AUC-ROC {bl['logistic']['auc_roc']:.4f} -> {auc_log:.4f}  "
              f"(delta {auc_log - bl['logistic']['auc_roc']:+.4f})")
        print(f"  Logistic:  AUC-PR  {bl['logistic']['auc_pr']:.4f} -> {ap_log:.4f}  "
              f"(delta {ap_log - bl['logistic']['auc_pr']:+.4f})")
        print(f"  LightGBM:  AUC-ROC {bl['lightgbm']['auc_roc']:.4f} -> {auc_gbm:.4f}  "
              f"(delta {auc_gbm - bl['lightgbm']['auc_roc']:+.4f})")
        print(f"  LightGBM:  AUC-PR  {bl['lightgbm']['auc_pr']:.4f} -> {ap_gbm:.4f}  "
              f"(delta {ap_gbm - bl['lightgbm']['auc_pr']:+.4f})")

    # H1 retest
    print("\n" + "=" * 72)
    print("  H1 retest (AUC-ROC > 0.95 AND AUC-PR > 0.60, logistic)")
    print("=" * 72)
    h1_log = auc_log > 0.95 and ap_log > 0.60
    h1_gbm = auc_gbm > 0.95 and ap_gbm > 0.60
    print(f"  Logistic: AUC-ROC {auc_log:.4f} > 0.95? {auc_log>0.95}    "
          f"AUC-PR {ap_log:.4f} > 0.60? {ap_log>0.60}")
    print(f"  LightGBM: AUC-ROC {auc_gbm:.4f} > 0.95? {auc_gbm>0.95}    "
          f"AUC-PR {ap_gbm:.4f} > 0.60? {ap_gbm>0.60}")
    if h1_log:
        print("  -> H1 CONFIRMED with logistic + rich features")
    elif h1_gbm:
        print("  -> H1 partial: rich features + non-linearity needed")
    elif auc_gbm > 0.90:
        print("  -> H1 not strictly met but rich features substantial gain. "
              "Stage A may still be useful operationally.")
    else:
        print("  -> H1 still fails. DGP not two-state-Markov with available data. "
              "Consider HMM or accept Stage A as imperfect filter.")

    # Top features (LGBM gain)
    print("\n  Re-training LightGBM on full pre-2024 to extract feature importance...")
    df_full_tr = df[df["timestamp"] < "2024-01-01"].dropna(
        subset=FEATURES + ["state"])
    import lightgbm as lgb
    pos_frac = float(df_full_tr["state"].mean())
    scale_pos = (1 - pos_frac) / max(pos_frac, 1e-6)
    params = dict(
        objective="binary", metric="auc",
        num_leaves=63, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_data_in_leaf=50, verbosity=-1,
        scale_pos_weight=scale_pos, random_state=42,
    )
    train_set = lgb.Dataset(df_full_tr[FEATURES].values,
                              label=df_full_tr["state"].values,
                              feature_name=FEATURES)
    booster = lgb.train(params, train_set, num_boost_round=500)
    imp = booster.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": FEATURES, "gain": imp}).sort_values(
        "gain", ascending=False)
    print("  Top 15 features by gain (LGBM, pre-2024 training):")
    for _, row in imp_df.head(15).iterrows():
        tag = ""
        if row["feature"] in NEW_RAW_FEATURES: tag = "[v5]"
        elif row["feature"] in DERIVED_FEATURES: tag = "[derived]"
        print(f"    {row['feature']:25s}  {row['gain']:>12.1f}  {tag}")

    out = {
        "n_features": len(FEATURES),
        "features": FEATURES,
        "positive_rate": pos_rate,
        "logistic": {"auc_roc": auc_log, "auc_pr": ap_log, "brier": brier_log,
                      **lead_log},
        "lightgbm": {"auc_roc": auc_gbm, "auc_pr": ap_gbm, "brier": brier_gbm,
                      **lead_gbm},
        "H1_logistic": bool(h1_log),
        "H1_lgbm": bool(h1_gbm),
    }
    with open(RESULTS / "v2_stage_a_rich_features_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    imp_df.to_csv(RESULTS / "v2_stage_a_rich_features_importance.csv", index=False)
    print(f"\n  Saved: {RESULTS / 'v2_stage_a_rich_features_metrics.json'}")
    print(f"  Saved: {RESULTS / 'v2_stage_a_rich_features_importance.csv'}")


if __name__ == "__main__":
    main()
