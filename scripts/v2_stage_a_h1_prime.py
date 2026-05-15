"""
v2_stage_a_h1_prime.py
----------------------
Test of H1' pre-registered in docs/v2_h1_prime_preregistration.md.

Adds the 3 ESIOS-derived features to the rich Stage A feature set
and re-evaluates walk-forward weekly on 2024.

New features:
  - scheduled_net_es_to_fr_mw  (positive = ES exports to FR)
  - scheduled_abs_mw           (absolute value of net schedule)
  - utilization_d1             (|scheduled| / NTC of relevant direction)

Decision (binary, pre-registered):
  - confirmed:  AUC-ROC > 0.92 AND AUC-PR > 0.45 (LightGBM)
  - partial:    0.88 <= AUC-ROC <= 0.92
  - falsified:  AUC-ROC <= 0.88

Output: results/v2_stage_a_h1_prime_metrics.json
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
import v2_stage_a_rich_features as v2sar

RESULTS = BASE / "results"
DATA_V6 = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231_v6.parquet"

NEW_ESIOS_FEATURES = [
    "scheduled_net_es_to_fr_mw",
    "scheduled_abs_mw",
    "utilization_d1",
]

FEATURES = v2sar.FEATURES + NEW_ESIOS_FEATURES  # 33 + 3 = 36


def walk_forward(df, model_fn_fit, model_fn_pred):
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
    print("  H1' test: Stage A + ESIOS scheduled-exchange features")
    print("=" * 72)
    df = pd.read_parquet(DATA_V6)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = v2sar.add_rich_features(df)
    pos_rate = float(df["state"].mean())
    print(f"  n={len(df):,}  positive rate: {pos_rate*100:.2f}%")
    print(f"  Total features: {len(FEATURES)} (rich 33 + 3 new ESIOS)")
    print(f"  New features added: {NEW_ESIOS_FEATURES}")
    print(f"  Coverage on master of new features:")
    for c in NEW_ESIOS_FEATURES:
        nn = df[c].notna().sum()
        print(f"    {c:30s}  {nn:,}/{len(df):,}  ({nn/len(df)*100:.2f}%)")

    print("\n  Walk-forward weekly logistic regression...")
    pred_log = walk_forward(df, v2sa.fit_logistic, v2sa.predict_logistic)
    auc_log = float(roc_auc_score(pred_log["y_true"], pred_log["proba"]))
    ap_log = float(average_precision_score(pred_log["y_true"], pred_log["proba"]))
    brier_log = float(brier_score_loss(pred_log["y_true"], pred_log["proba"]))
    lead_log = v2sa.lead_time_to_transition(pred_log)
    print(f"    AUC-ROC: {auc_log:.4f}  AUC-PR: {ap_log:.4f}  Brier: {brier_log:.4f}")
    print(f"    lead median: {lead_log['median_lead_h']}h  "
          f"frac advance: {lead_log.get('frac_predicted_in_advance', 0):.2f}")

    print("\n  Walk-forward weekly LightGBM (primary model)...")
    pred_gbm = walk_forward(df, v2sa.fit_lgbm, v2sa.predict_lgbm)
    auc_gbm = float(roc_auc_score(pred_gbm["y_true"], pred_gbm["proba"]))
    ap_gbm = float(average_precision_score(pred_gbm["y_true"], pred_gbm["proba"]))
    brier_gbm = float(brier_score_loss(pred_gbm["y_true"], pred_gbm["proba"]))
    lead_gbm = v2sa.lead_time_to_transition(pred_gbm)
    print(f"    AUC-ROC: {auc_gbm:.4f}  AUC-PR: {ap_gbm:.4f}  Brier: {brier_gbm:.4f}")
    print(f"    lead median: {lead_gbm['median_lead_h']}h  "
          f"frac advance: {lead_gbm.get('frac_predicted_in_advance', 0):.2f}")

    # Compare to v2 rich baseline (33 features, no ESIOS)
    print("\n  Comparison vs prior rich baseline (33 features, no ESIOS):")
    prev_path = RESULTS / "v2_stage_a_rich_features_metrics.json"
    if prev_path.exists():
        prev = json.load(open(prev_path))
        print(f"    Logistic AUC-ROC: {prev['logistic']['auc_roc']:.4f} -> {auc_log:.4f}  "
              f"(delta {auc_log - prev['logistic']['auc_roc']:+.4f})")
        print(f"    Logistic AUC-PR:  {prev['logistic']['auc_pr']:.4f} -> {ap_log:.4f}  "
              f"(delta {ap_log - prev['logistic']['auc_pr']:+.4f})")
        print(f"    LightGBM AUC-ROC: {prev['lightgbm']['auc_roc']:.4f} -> {auc_gbm:.4f}  "
              f"(delta {auc_gbm - prev['lightgbm']['auc_roc']:+.4f})")
        print(f"    LightGBM AUC-PR:  {prev['lightgbm']['auc_pr']:.4f} -> {ap_gbm:.4f}  "
              f"(delta {ap_gbm - prev['lightgbm']['auc_pr']:+.4f})")

    # H1' decision
    print("\n" + "=" * 72)
    print("  H1' decision (pre-registered)")
    print("=" * 72)
    if auc_gbm > 0.92 and ap_gbm > 0.45:
        verdict = "CONFIRMED"
        print(f"  AUC-ROC {auc_gbm:.4f} > 0.92? True  "
              f"AUC-PR {ap_gbm:.4f} > 0.45? True")
        print("  -> H1' CONFIRMED. Physical-utilization breaks the ceiling.")
    elif 0.88 <= auc_gbm <= 0.92:
        verdict = "PARTIAL"
        print(f"  AUC-ROC {auc_gbm:.4f} in [0.88, 0.92].  Modest gain over prior 0.87.")
        print("  -> H1' PARTIAL. Ceiling moved, not broken.")
    else:
        verdict = "FALSIFIED"
        print(f"  AUC-ROC {auc_gbm:.4f} <= 0.88.")
        print("  -> H1' FALSIFIED. Ceiling structural even with physical schedule.")

    # Top features (LGBM)
    print("\n  Re-training LightGBM on pre-2024 for feature importance...")
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
    print("  Top 15 by gain (LGBM, pre-2024):")
    for _, row in imp_df.head(15).iterrows():
        tag = ""
        if row["feature"] in NEW_ESIOS_FEATURES: tag = "[ESIOS]"
        elif row["feature"] in v2sar.NEW_RAW_FEATURES: tag = "[v5]"
        elif row["feature"] in v2sar.DERIVED_FEATURES: tag = "[derived]"
        print(f"    {row['feature']:30s}  {row['gain']:>12.1f}  {tag}")

    # Save
    out = {
        "n_features": len(FEATURES),
        "new_esios_features": NEW_ESIOS_FEATURES,
        "positive_rate": pos_rate,
        "logistic": {"auc_roc": auc_log, "auc_pr": ap_log, "brier": brier_log,
                      **lead_log},
        "lightgbm": {"auc_roc": auc_gbm, "auc_pr": ap_gbm, "brier": brier_gbm,
                      **lead_gbm},
        "H1_prime_verdict": verdict,
        "H1_prime_confirmed_by_lgbm": bool(auc_gbm > 0.92 and ap_gbm > 0.45),
    }
    with open(RESULTS / "v2_stage_a_h1_prime_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    imp_df.to_csv(RESULTS / "v2_stage_a_h1_prime_importance.csv", index=False)
    print(f"\n  Saved: {RESULTS / 'v2_stage_a_h1_prime_metrics.json'}")
    print(f"  Saved: {RESULTS / 'v2_stage_a_h1_prime_importance.csv'}")


if __name__ == "__main__":
    main()
