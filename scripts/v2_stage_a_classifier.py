"""
v2_stage_a_classifier.py
------------------------
Stage A del v2 hurdle architecture.

Target binario: S_t = 1{|spread_da_t| > 0.5 EUR/MWh}
                (1 = desacoplado / congestion, 0 = acoplado)

Features (todas sin look-ahead, computadas con shift(1) cuando aplica):
  Temporales civiles:
    - hora seno/coseno
    - dia-semana seno/coseno
    - mes seno/coseno
    - festivo ES (binario), festivo FR (binario)
  Capacidad fisica:
    - ntc_es_fr, ntc_fr_es (publicado D-1)
    - ntc_is_observed (1 = JAO real, 0 = imputado pre-JAO)
  Lags del estado mismo:
    - state_lag24 (mismo hora ayer)
    - state_lag168 (mismo hora la semana pasada)
    - run_length_t-1 (cuantas horas consecutivas en el estado actual al cierre de t-1)
  Regimen regulatorio:
    - regime (one-hot)

Modelos:
  - Logistic regression (baseline, H1 test)
  - LightGBM (comparativa no-lineal)

Validacion: walk-forward semanal sobre 2024. Cada semana se reentrena
con todo lo anterior (train 2019-2023 + semanas 1..k-1 de 2024) y se
predice la semana k.

Metricas:
  - AUC-ROC global
  - AUC-PR global (importante con clase rara: ~4% positivos)
  - Brier score (calibracion)
  - Lead time mediano a transicion 0->1 (horas)

H1: AUC-ROC > 0.95 y AUC-PR > 0.60 en logistic.

Salida: results/v2_stage_a_metrics.json
        results/v2_stage_a_predictions.parquet
"""
from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import holidays as pyholidays

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231.parquet"
RESULTS = BASE / "results"

TARGET_THRESHOLD = 0.5
ONE_HOT_REGIMES = ["pre_crisis", "crisis_y_excepcion", "post_excepcion"]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def build_state(df: pd.DataFrame) -> pd.Series:
    return (df["spread_da"].abs() > TARGET_THRESHOLD).astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]

    # Cyclical encoding of time
    h = ts.dt.hour.values
    dow = ts.dt.dayofweek.values
    m = ts.dt.month.values
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["mon_sin"] = np.sin(2 * np.pi * m / 12)
    df["mon_cos"] = np.cos(2 * np.pi * m / 12)

    # National holidays
    years = sorted(set(ts.dt.year.unique()))
    es_h = pyholidays.country_holidays("ES", years=years)
    fr_h = pyholidays.country_holidays("FR", years=years)
    df["holiday_es"] = ts.dt.date.map(lambda d: int(d in es_h)).astype(int)
    df["holiday_fr"] = ts.dt.date.map(lambda d: int(d in fr_h)).astype(int)
    df["holiday_xor"] = (df["holiday_es"] != df["holiday_fr"]).astype(int)

    # State lags and run length, all sin look-ahead
    state = build_state(df)
    df["state"] = state.values
    df["state_lag24"] = state.shift(24).fillna(0).astype(int)
    df["state_lag168"] = state.shift(168).fillna(0).astype(int)

    # Run length of the current state, computed as the length of the
    # consecutive run that ENDS at t-1 (so usable to predict t).
    s_shift = state.shift(1).fillna(0).astype(int).values
    run = np.zeros(len(s_shift), dtype=int)
    for i in range(1, len(s_shift)):
        if s_shift[i] == s_shift[i - 1]:
            run[i] = run[i - 1] + 1
        else:
            run[i] = 1
    df["run_length_prev"] = run

    # Regime one-hot
    for r in ONE_HOT_REGIMES:
        df[f"reg_{r}"] = (df["regime"] == r).astype(int)

    return df


FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos",
    "holiday_es", "holiday_fr", "holiday_xor",
    "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
    "state_lag24", "state_lag168", "run_length_prev",
    "reg_pre_crisis", "reg_crisis_y_excepcion", "reg_post_excepcion",
]


# --------------------------------------------------------------------------
# Walk-forward weekly evaluation
# --------------------------------------------------------------------------
def fit_logistic(X_tr, y_tr):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="lbfgs", C=1.0,
    )
    model.fit(Xs, y_tr)
    return model, sc


def predict_logistic(artefacts, X_te):
    model, sc = artefacts
    Xs = sc.transform(X_te)
    return model.predict_proba(Xs)[:, 1]


def fit_lgbm(X_tr, y_tr):
    pos_frac = float(y_tr.mean())
    scale_pos = (1 - pos_frac) / max(pos_frac, 1e-6)
    params = dict(
        objective="binary", metric="auc",
        num_leaves=63, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_data_in_leaf=50, verbosity=-1,
        scale_pos_weight=scale_pos, random_state=42,
    )
    train_set = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, train_set, num_boost_round=500)
    return model


def predict_lgbm(model, X_te):
    return model.predict(X_te)


def walk_forward(df: pd.DataFrame, model_fn_fit, model_fn_pred):
    """Reentrenamiento semanal sobre 2024.

    En el inicio de cada semana ISO de 2024, se reentrena con
    todos los datos hasta el comienzo de esa semana (incluyendo
    2019-2023 + semanas anteriores de 2024) y se predice esa semana.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_train_all = df[df["timestamp"] < "2024-01-01"].copy()
    df_test_all = df[df["timestamp"] >= "2024-01-01"].copy()

    # Iso week of test_set
    df_test_all["_week"] = df_test_all["timestamp"].dt.isocalendar().week
    df_test_all["_year"] = df_test_all["timestamp"].dt.year
    # Compose week id (year*100+week) to handle 2024 isoweek 1 in 2023
    df_test_all["_week_id"] = (df_test_all["_year"] * 100
                                + df_test_all["_week"])
    week_ids = sorted(df_test_all["_week_id"].unique())

    all_preds = []
    incremental = df_train_all.copy()
    for w_id in week_ids:
        df_week = df_test_all[df_test_all["_week_id"] == w_id].copy()
        if df_week.empty:
            continue
        # Train on everything strictly before the start of this week
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

        model_artefacts = model_fn_fit(X_tr, y_tr)
        proba = model_fn_pred(model_artefacts, X_te)
        all_preds.append(pd.DataFrame({
            "timestamp": df_te["timestamp"].values,
            "y_true": y_te,
            "proba": proba,
            "week_id": w_id,
        }))

    return pd.concat(all_preds, ignore_index=True)


def lead_time_to_transition(pred_df: pd.DataFrame,
                            decision_threshold: float = 0.5) -> dict:
    """Lead time desde primera alerta (proba > threshold) hasta primera
    hora con y_true = 1 (transicion 0->1) tras un periodo de y_true=0.

    Para cada transicion 0->1, miramos las 48h previas y vemos cuanto
    tiempo antes empezo a haber alerta. Si no hubo alerta antes, lead=0
    (alerta justo en la hora). Si nunca dispara, NaN.
    """
    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    y = pred_df["y_true"].values
    p = pred_df["proba"].values
    alert = (p > decision_threshold).astype(int)
    transitions = []
    for i in range(1, len(y)):
        if y[i - 1] == 0 and y[i] == 1:
            # look back up to 48h
            lookback = max(0, i - 48)
            window_alert = alert[lookback:i]
            # lead time = number of hours alert was on consecutively
            # immediately before transition
            # easiest definition: time from first alert in lookback to i
            on_indices = np.where(window_alert == 1)[0]
            if len(on_indices) == 0:
                lead = 0.0  # alarm only at transition or never
            else:
                lead = float(len(window_alert) - on_indices[0])
            transitions.append({"i": i, "lead_h": lead})

    if not transitions:
        return {"n_transitions": 0, "median_lead_h": np.nan,
                "mean_lead_h": np.nan}
    leads = np.array([t["lead_h"] for t in transitions])
    return {
        "n_transitions": len(leads),
        "median_lead_h": float(np.median(leads)),
        "mean_lead_h": float(leads.mean()),
        "frac_predicted_in_advance": float((leads > 0).mean()),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  v2 Stage A: binary congestion classifier")
    print(f"  Target: S_t = 1{{|spread_da| > {TARGET_THRESHOLD} EUR/MWh}}")
    print("=" * 72)

    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_features(df)
    overall_rate = float(df["state"].mean())
    print(f"  total n: {len(df):,}  decoupled rate (positive class): "
          f"{overall_rate*100:.2f}%")

    print(f"\n  Features ({len(FEATURES)}): {FEATURES}")

    # --- Logistic ---
    print("\n  Walk-forward weekly logistic regression on 2024...")
    pred_log = walk_forward(df, fit_logistic, predict_logistic)
    auc_log = float(roc_auc_score(pred_log["y_true"], pred_log["proba"]))
    ap_log = float(average_precision_score(pred_log["y_true"], pred_log["proba"]))
    brier_log = float(brier_score_loss(pred_log["y_true"], pred_log["proba"]))
    lead_log = lead_time_to_transition(pred_log)
    print(f"    n predictions: {len(pred_log):,}")
    print(f"    AUC-ROC:       {auc_log:.4f}")
    print(f"    AUC-PR:        {ap_log:.4f}")
    print(f"    Brier:         {brier_log:.5f}")
    print(f"    n transitions 0->1: {lead_log['n_transitions']}")
    print(f"    median lead time (h): {lead_log['median_lead_h']}")
    print(f"    frac predicted in advance: {lead_log.get('frac_predicted_in_advance', 'n/a')}")

    # --- LightGBM ---
    print("\n  Walk-forward weekly LightGBM on 2024...")
    pred_gbm = walk_forward(df, fit_lgbm, predict_lgbm)
    auc_gbm = float(roc_auc_score(pred_gbm["y_true"], pred_gbm["proba"]))
    ap_gbm = float(average_precision_score(pred_gbm["y_true"], pred_gbm["proba"]))
    brier_gbm = float(brier_score_loss(pred_gbm["y_true"], pred_gbm["proba"]))
    lead_gbm = lead_time_to_transition(pred_gbm)
    print(f"    AUC-ROC:       {auc_gbm:.4f}")
    print(f"    AUC-PR:        {ap_gbm:.4f}")
    print(f"    Brier:         {brier_gbm:.5f}")
    print(f"    n transitions 0->1: {lead_gbm['n_transitions']}")
    print(f"    median lead time (h): {lead_gbm['median_lead_h']}")
    print(f"    frac predicted in advance: {lead_gbm.get('frac_predicted_in_advance', 'n/a')}")

    # --- H1 decision ---
    print("\n" + "=" * 72)
    print("  H1 decision: AUC-ROC > 0.95 AND AUC-PR > 0.60 (logistic baseline)")
    print("=" * 72)
    h1_auc = auc_log > 0.95
    h1_ap = ap_log > 0.60
    print(f"  logistic AUC-ROC = {auc_log:.4f}  > 0.95?  {h1_auc}")
    print(f"  logistic AUC-PR  = {ap_log:.4f}  > 0.60?  {h1_ap}")
    if h1_auc and h1_ap:
        print("  -> H1 CONFIRMED: proceed to Stage B")
    elif auc_log > 0.85 and ap_log > 0.30:
        print("  -> H1 partial: GBM also tested; check non-linear gain")
        print(f"     GBM AUC-ROC = {auc_gbm:.4f}  AUC-PR = {ap_gbm:.4f}")
    else:
        print("  -> H1 FAILED: DGP more complex than two states. "
              "Consider HMM with 3+ regimes or richer features.")

    # --- Save ---
    out = {
        "target_threshold_eur_mwh": TARGET_THRESHOLD,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "overall_decoupled_rate": overall_rate,
        "logistic": {
            "auc_roc": auc_log, "auc_pr": ap_log, "brier": brier_log,
            **lead_log,
        },
        "lightgbm": {
            "auc_roc": auc_gbm, "auc_pr": ap_gbm, "brier": brier_gbm,
            **lead_gbm,
        },
        "H1_confirmed_by_logistic": bool(h1_auc and h1_ap),
    }
    with open(RESULTS / "v2_stage_a_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    pred_log["model"] = "logistic"
    pred_gbm["model"] = "lightgbm"
    pd.concat([pred_log, pred_gbm], ignore_index=True).to_parquet(
        RESULTS / "v2_stage_a_predictions.parquet", index=False)
    print(f"\n  Saved: {RESULTS / 'v2_stage_a_metrics.json'}")
    print(f"  Saved: {RESULTS / 'v2_stage_a_predictions.parquet'}")


if __name__ == "__main__":
    main()
