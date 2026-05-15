"""
v2_stage_a_hmm.py
-----------------
Stage A via Hidden Markov Model con 3 estados gaussianos.

H2 pre-registrada en docs/v2_dgp_analysis.md §7.2:
  AUC-ROC > 0.92 AND AUC-PR > 0.45 sobre S_t = 1{|spread|>0.5}
  en walk-forward semanal 2024.

Implementacion:
  - hmmlearn.GaussianHMM(n_components=3, covariance_type='full')
  - Features de observacion: subconjunto continuo de las 33 features
    rich (excluyendo dummies categoricas y lags binarios del propio
    estado, para no inducir circularidad con el target).
  - Identificacion post-hoc de estados: ordenar por la media de
    |spread| asociada a cada estado en training -> el de mayor
    |spread| medio = "desacoplado".
  - Prediccion: forward algorithm posterior P(state=desacoplado | x_{1..t-1}).
  - Walk-forward semanal: reentrenamiento cada semana ISO 2024.

Output:
  results/v2_stage_a_hmm_metrics.json
  results/v2_stage_a_hmm_predictions.parquet
  results/v2_stage_a_hmm_state_means.csv
"""
from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import v2_stage_a_classifier as v2sa
import v2_stage_a_rich_features as v2sar

RESULTS = BASE / "results"
DATA_V5 = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231_v5.parquet"

# Continuous physical features (excluye dummies regimen, holidays, lags
# binarios del estado, run_length que es ya un agregado de estado).
HMM_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos",
    "ntc_es_fr", "ntc_fr_es",
    "es_solar_fc", "es_wind_fc",
    "fr_solar_fc", "fr_wind_fc",
    "es_demand_fc", "fr_demand_fc",
    "fr_nuclear_avail",
    "solar_pen_es", "solar_pen_fr", "solar_pen_asym",
    "wind_pen_es", "wind_pen_fr", "wind_pen_asym",
    "demand_log_ratio", "nuclear_ratio_fr",
]

N_STATES = 3


def identify_decoupled_state(model, X_scaled, y_state, n_states=N_STATES):
    """Asigna a cada estado HMM el rango por |spread| medio.

    Para cada estado k, computa la fraccion de horas con y_state=1
    cuando el HMM las asigna a k via Viterbi. El estado con mayor
    fraccion = "desacoplado".

    Returns: dict {k: rank} y el indice del estado desacoplado.
    """
    state_seq = model.predict(X_scaled)
    decoupled_frac = {}
    for k in range(n_states):
        mask = state_seq == k
        if mask.sum() == 0:
            decoupled_frac[k] = 0.0
            continue
        decoupled_frac[k] = float(y_state[mask].mean())
    # Rank by decoupled fraction
    ranked = sorted(decoupled_frac, key=decoupled_frac.get, reverse=True)
    decoupled_state = ranked[0]
    rank_map = {k: rank for rank, k in enumerate(ranked)}
    return decoupled_state, rank_map, decoupled_frac, state_seq


def fit_hmm(X_train_scaled, random_state=42):
    """Fit Gaussian HMM. Retries with different random seeds if not converged."""
    best_model = None
    best_score = -np.inf
    for seed in [random_state, random_state + 1, random_state + 2]:
        try:
            model = hmm.GaussianHMM(
                n_components=N_STATES, covariance_type="full",
                n_iter=50, tol=1e-3, random_state=seed,
            )
            model.fit(X_train_scaled)
            sc = model.score(X_train_scaled)
            if sc > best_score:
                best_score = sc
                best_model = model
        except Exception as e:
            print(f"    HMM fit seed {seed} failed: {e}")
            continue
    return best_model


def predict_decoupled_proba(model, X_te_scaled, decoupled_state):
    """Compute posterior P(state_t = decoupled_state | x_{1..t}).

    Uses the forward algorithm (predict_proba in hmmlearn).
    """
    # predict_proba returns posterior over states for each time step
    post = model.predict_proba(X_te_scaled)
    return post[:, decoupled_state]


def walk_forward_hmm(df, features, target_col="state"):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_test = df[df["timestamp"] >= "2024-01-01"].copy()
    df_test["_week"] = df_test["timestamp"].dt.isocalendar().week
    df_test["_year"] = df_test["timestamp"].dt.year
    df_test["_week_id"] = df_test["_year"] * 100 + df_test["_week"]
    week_ids = sorted(df_test["_week_id"].unique())

    all_preds = []
    convergence_log = []
    for w_id in week_ids:
        df_week = df_test[df_test["_week_id"] == w_id].copy()
        if df_week.empty:
            continue
        cutoff = df_week["timestamp"].min()
        df_tr = df[df["timestamp"] < cutoff].dropna(
            subset=features + [target_col]).copy()
        df_te = df_week.dropna(subset=features + [target_col]).copy()
        if df_tr.empty or df_te.empty:
            continue

        X_tr = df_tr[features].values
        y_tr = df_tr[target_col].values
        X_te = df_te[features].values
        y_te = df_te[target_col].values

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        model = fit_hmm(X_tr_s)
        if model is None:
            print(f"    week_id {w_id}: HMM did not converge, skipping")
            continue

        decoupled_state, rank_map, decoupled_frac, _ = identify_decoupled_state(
            model, X_tr_s, y_tr)
        proba = predict_decoupled_proba(model, X_te_s, decoupled_state)

        all_preds.append(pd.DataFrame({
            "timestamp": df_te["timestamp"].values,
            "y_true": y_te,
            "proba": proba,
            "week_id": w_id,
            "decoupled_state": decoupled_state,
        }))
        convergence_log.append({
            "week_id": w_id,
            "n_train": len(df_tr),
            "decoupled_state": int(decoupled_state),
            "decoupled_state_frac": float(decoupled_frac[decoupled_state]),
            "monotone": float(decoupled_frac[decoupled_state]) >
                         max(v for k, v in decoupled_frac.items()
                             if k != decoupled_state),
        })

    return pd.concat(all_preds, ignore_index=True), pd.DataFrame(convergence_log)


def state_means_report(model, scaler, features):
    """For interpretation: undo scaling and print mean of each state's
    emission distribution in original units."""
    means_scaled = model.means_
    means_orig = scaler.inverse_transform(means_scaled)
    df = pd.DataFrame(means_orig, columns=features)
    df.index = [f"state_{k}" for k in range(model.n_components)]
    return df


def main():
    print("=" * 72)
    print("  v2 Stage A — HMM (camino 2)")
    print(f"  Pre-registered H2: AUC-ROC > 0.92 AND AUC-PR > 0.45")
    print("=" * 72)
    df = pd.read_parquet(DATA_V5)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = v2sar.add_rich_features(df)
    pos_rate = float(df["state"].mean())
    print(f"  n={len(df):,}  positive rate (decoupled): {pos_rate*100:.2f}%")
    print(f"  HMM observation features ({len(HMM_FEATURES)}): {HMM_FEATURES}")

    # Walk-forward
    print(f"\n  Walk-forward weekly HMM (n_states={N_STATES}, "
          "covariance_type='full')...")
    pred_hmm, conv_log = walk_forward_hmm(df, HMM_FEATURES)
    print(f"    n predictions: {len(pred_hmm):,}")
    print(f"    Weeks fitted: {len(conv_log)}")
    print(f"    Decoupled-state correctly identified in "
          f"{int(conv_log['monotone'].sum())}/{len(conv_log)} weeks")

    # Metrics
    auc = float(roc_auc_score(pred_hmm["y_true"], pred_hmm["proba"]))
    ap = float(average_precision_score(pred_hmm["y_true"], pred_hmm["proba"]))
    brier = float(brier_score_loss(pred_hmm["y_true"], pred_hmm["proba"]))
    lead = v2sa.lead_time_to_transition(pred_hmm)
    print(f"\n  HMM AUC-ROC:  {auc:.4f}")
    print(f"  HMM AUC-PR:   {ap:.4f}")
    print(f"  HMM Brier:    {brier:.5f}")
    print(f"  HMM lead-time mediano: {lead['median_lead_h']}h  "
          f"frac_adv={lead.get('frac_predicted_in_advance', 0):.2f}")

    # H2 decision
    print("\n" + "=" * 72)
    print("  H2 decision")
    print("=" * 72)
    cond_strong = (auc > 0.92) and (ap > 0.45)
    cond_marginal = (0.88 <= auc <= 0.92)
    cond_falsified = auc <= 0.88
    print(f"  AUC-ROC = {auc:.4f}  > 0.92? {auc > 0.92}")
    print(f"  AUC-PR  = {ap:.4f}  > 0.45? {ap > 0.45}")
    if cond_strong:
        verdict = "CONFIRMED"
        print("  -> H2 CONFIRMED. HMM is the right tool. Proceed to Stage B.")
    elif cond_marginal:
        verdict = "MARGINAL"
        print("  -> H2 MARGINAL improvement (>0.88 but <0.92). Documentar como "
              "contribucion modesta; HMM como filtro y proceder a Stage B.")
    else:
        verdict = "FALSIFIED"
        print("  -> H2 FALSIFIED. The transition noise is structurally stochastic "
              "rather than hidden-state. Empirical ceiling for DA-only data ~ AUC 0.87.")

    # Comparison
    print("\n" + "=" * 72)
    print("  Comparison vs best previous Stage A (LGBM rich features)")
    print("=" * 72)
    prev_path = RESULTS / "v2_stage_a_rich_features_metrics.json"
    if prev_path.exists():
        prev = json.load(open(prev_path))
        prev_auc = prev["lightgbm"]["auc_roc"]
        prev_ap = prev["lightgbm"]["auc_pr"]
        print(f"  AUC-ROC: GBM rich {prev_auc:.4f} -> HMM {auc:.4f}  "
              f"(delta {auc-prev_auc:+.4f})")
        print(f"  AUC-PR:  GBM rich {prev_ap:.4f} -> HMM {ap:.4f}  "
              f"(delta {ap-prev_ap:+.4f})")

    # Save artefacts
    out = {
        "n_states": N_STATES,
        "n_features": len(HMM_FEATURES),
        "features": HMM_FEATURES,
        "positive_rate": pos_rate,
        "auc_roc": auc, "auc_pr": ap, "brier": brier,
        "lead_time": lead,
        "H2_verdict": verdict,
        "H2_confirmed": cond_strong,
        "decoupled_identification_consistency_pct": float(
            conv_log["monotone"].mean()) if len(conv_log) else None,
    }
    with open(RESULTS / "v2_stage_a_hmm_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    pred_hmm.to_parquet(RESULTS / "v2_stage_a_hmm_predictions.parquet", index=False)
    conv_log.to_csv(RESULTS / "v2_stage_a_hmm_convergence.csv", index=False)
    print(f"\n  Saved: {RESULTS / 'v2_stage_a_hmm_metrics.json'}")
    print(f"  Saved: {RESULTS / 'v2_stage_a_hmm_predictions.parquet'}")
    print(f"  Saved: {RESULTS / 'v2_stage_a_hmm_convergence.csv'}")


if __name__ == "__main__":
    main()
