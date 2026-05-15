"""
v2_stage_a_threshold_sweep.py
-----------------------------
Camino 3: barrido de thresholds para Stage A.

Re-entrena el clasificador para varios umbrales tau en {0.5, 1, 2, 5}
EUR/MWh. Comprueba si subir tau separa mejor el modo desacoplado de la
zona de ruido transicional.

Output: results/v2_stage_a_threshold_sweep.csv
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

# Import functions, but override TARGET_THRESHOLD per iteration
import v2_stage_a_classifier as v2sa

RESULTS = BASE / "results"
DATA = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231.parquet"

THRESHOLDS = [0.5, 1.0, 2.0, 5.0]


def main():
    print("=" * 72)
    print("  Stage A threshold sensitivity sweep")
    print("=" * 72)
    df_raw = pd.read_parquet(DATA)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

    rows = []
    for tau in THRESHOLDS:
        print(f"\n  ---- THRESHOLD tau = {tau} EUR/MWh ----")
        v2sa.TARGET_THRESHOLD = tau
        df = v2sa.add_features(df_raw.copy())
        pos_rate = float(df["state"].mean())
        print(f"    positive class rate: {pos_rate*100:.2f}%")
        if pos_rate < 0.001:
            print("    too few positives, skip")
            continue

        # Logistic
        pred_log = v2sa.walk_forward(df, v2sa.fit_logistic, v2sa.predict_logistic)
        auc_log = float(roc_auc_score(pred_log["y_true"], pred_log["proba"]))
        ap_log = float(average_precision_score(pred_log["y_true"], pred_log["proba"]))
        brier_log = float(brier_score_loss(pred_log["y_true"], pred_log["proba"]))
        lead_log = v2sa.lead_time_to_transition(pred_log)
        print(f"    Logistic: AUC-ROC={auc_log:.4f}  AUC-PR={ap_log:.4f}  "
              f"Brier={brier_log:.4f}  lead_med={lead_log['median_lead_h']}h  "
              f"frac_adv={lead_log.get('frac_predicted_in_advance', 0):.2f}")

        # LightGBM
        pred_gbm = v2sa.walk_forward(df, v2sa.fit_lgbm, v2sa.predict_lgbm)
        auc_gbm = float(roc_auc_score(pred_gbm["y_true"], pred_gbm["proba"]))
        ap_gbm = float(average_precision_score(pred_gbm["y_true"], pred_gbm["proba"]))
        brier_gbm = float(brier_score_loss(pred_gbm["y_true"], pred_gbm["proba"]))
        lead_gbm = v2sa.lead_time_to_transition(pred_gbm)
        print(f"    LightGBM: AUC-ROC={auc_gbm:.4f}  AUC-PR={ap_gbm:.4f}  "
              f"Brier={brier_gbm:.4f}  lead_med={lead_gbm['median_lead_h']}h  "
              f"frac_adv={lead_gbm.get('frac_predicted_in_advance', 0):.2f}")

        rows.append({
            "tau_eur_mwh": tau,
            "positive_rate": pos_rate,
            "logistic_auc_roc": auc_log,
            "logistic_auc_pr": ap_log,
            "logistic_brier": brier_log,
            "logistic_lead_med_h": lead_log["median_lead_h"],
            "logistic_frac_adv": lead_log.get("frac_predicted_in_advance", 0),
            "lgbm_auc_roc": auc_gbm,
            "lgbm_auc_pr": ap_gbm,
            "lgbm_brier": brier_gbm,
            "lgbm_lead_med_h": lead_gbm["median_lead_h"],
            "lgbm_frac_adv": lead_gbm.get("frac_predicted_in_advance", 0),
            "n_transitions": lead_log["n_transitions"],
        })

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("  Summary across thresholds")
    print("=" * 72)
    print(df_out[[
        "tau_eur_mwh", "positive_rate",
        "logistic_auc_roc", "logistic_auc_pr",
        "lgbm_auc_roc", "lgbm_auc_pr",
        "n_transitions",
    ]].to_string(index=False))
    df_out.to_csv(RESULTS / "v2_stage_a_threshold_sweep.csv", index=False)
    print(f"\n  CSV: {RESULTS / 'v2_stage_a_threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
