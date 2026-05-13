"""
apply_cascade_v4_quantile.py
----------------------------
Cierra el loop operacional del paper. Toma el modelo v4-quantile-cleanONLY
(XGBoost con objective=reg:quantileerror alpha=0.95 sobre FEATURES_A_CLEAN),
genera residuos en test 2024, aplica la cascada Verde/Ambar/Naranja/Roja
y valida contra el panel de 40 eventos.

Compara contra el v3 baseline (V0_baseline_naranja del ablation).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

PROC_DIR = BASE_DIR / "data" / "processed"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
EVENTS_CSV = DATA_DIR / "events_panel.csv"

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
ROLLING_WINDOW = 168
TRAIN_FRACTION = 0.5
ALERT_WINDOW_START = pd.Timestamp("2024-01-01")
ALERT_WINDOW_END = pd.Timestamp("2024-12-31 23:00")


def train_v4_quantile(df_tr, df_te):
    params = dict(
        objective="reg:quantileerror", quantile_alpha=0.95,
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    m = xgb.XGBRegressor(**params).fit(df_tr[FEATURES_A_CLEAN], df_tr[TARGET],
                                       verbose=False)
    return m.predict(df_te[FEATURES_A_CLEAN]), m


def compute_rolling_z(res, window=168):
    mu = res.rolling(window, min_periods=window // 2).mean().shift(1)
    sd = res.rolling(window, min_periods=window // 2).std().shift(1)
    sigma_floor = res.dropna().std() * 0.1
    sd = sd.clip(lower=sigma_floor)
    z = ((res - mu) / sd).clip(-10, 10)
    return z


def compute_cusum(res, train_idx, k_factor=0.5, h=5.0):
    sigma_train = float(res.iloc[train_idx].std())
    if sigma_train < 1e-6:
        return np.zeros(len(res), dtype=int)
    rn = res.values / sigma_train
    cp = np.zeros(len(rn))
    cn = np.zeros(len(rn))
    active = np.zeros(len(rn), dtype=int)
    for i in range(1, len(rn)):
        cp[i] = max(0, cp[i - 1] + rn[i] - k_factor)
        cn[i] = max(0, cn[i - 1] - rn[i] - k_factor)
        if cp[i] > h or cn[i] > h:
            active[i] = active[i - 1] + 1
        else:
            active[i] = 0
    return active


def lag_per_event(df_alerts, events):
    rows = []
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev["date_start"])
        end = pd.Timestamp(ev["date_end"]) + pd.Timedelta(hours=23)
        if end < ALERT_WINDOW_START or start > ALERT_WINDOW_END:
            rows.append({"event_id": ev["event_id"],
                         "magnitude": ev["magnitude_indicator"],
                         "out_of_window": True, "detected": False,
                         "lag_hours": np.nan})
            continue
        ws = max(start, ALERT_WINDOW_START)
        we = min(end, ALERT_WINDOW_END)
        mask = ((df_alerts["timestamp"] >= ws) &
                (df_alerts["timestamp"] <= we) &
                df_alerts["alert"])
        if mask.any():
            first = df_alerts.loc[mask, "timestamp"].min()
            lag = (first - start).total_seconds() / 3600.0
        else:
            lag = np.nan
        rows.append({"event_id": ev["event_id"],
                     "magnitude": ev["magnitude_indicator"],
                     "out_of_window": False,
                     "detected": bool(mask.any()),
                     "lag_hours": float(lag) if not np.isnan(lag) else np.nan})
    return pd.DataFrame(rows)


def fpr_precision_72h(df_alerts, events, buffer_h=72):
    ts = df_alerts["timestamp"].values.astype("datetime64[ns]")
    in_period = ((df_alerts["timestamp"] >= ALERT_WINDOW_START) &
                 (df_alerts["timestamp"] <= ALERT_WINDOW_END)).values
    trig = df_alerts["alert"].astype(bool).values
    centers = []
    for _, ev in events.iterrows():
        s = pd.Timestamp(ev["date_start"])
        e = pd.Timestamp(ev["date_end"]) + pd.Timedelta(hours=23)
        centers.append(s + (e - s) / 2)
    centers_arr = pd.Series(centers).values.astype("datetime64[ns]")
    mask_active = trig & in_period
    near = np.zeros(len(ts), dtype=bool)
    if mask_active.any():
        ts_a = ts[mask_active]
        diffs = np.abs(ts_a[:, None] - centers_arr[None, :])
        near_a = (diffs <= np.timedelta64(buffer_h, "h")).any(axis=1)
        near[mask_active] = near_a
    n_total = int(mask_active.sum())
    n_inside = int(near.sum())
    return {
        "total_alerts": n_total,
        "fpr_per_month": (n_total - n_inside) / 12,  # 12 meses test 2024
        "precision_72h": n_inside / max(n_total, 1),
    }


def main():
    print("=" * 72)
    print("  Cascada de deteccion sobre v4-quantile-cleanONLY")
    print("=" * 72)

    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_tr = df[(df["timestamp"] >= "2019-01-01") &
               (df["timestamp"] < "2024-01-01")].dropna(
        subset=FEATURES_A_CLEAN + [TARGET]).copy()
    df_te = df[df["timestamp"] >= "2024-01-01"].dropna(
        subset=FEATURES_A_CLEAN + [TARGET]).copy().reset_index(drop=True)
    print(f"  Train: {len(df_tr):,}  Test 2024: {len(df_te):,}")

    print("  Entrenando v4-quantile-cleanONLY...")
    pred, _ = train_v4_quantile(df_tr, df_te)
    df_te["spread_predicted"] = pred
    df_te["residual"] = df_te[TARGET].values - pred

    print(f"  Residual stats: mean={df_te['residual'].mean():.3f}  "
          f"std={df_te['residual'].std():.3f}  "
          f"min={df_te['residual'].min():.3f}  max={df_te['residual'].max():.3f}")

    # ── z-score ──────────────────────────────────────────────────────────
    df_te["z_score"] = compute_rolling_z(df_te["residual"], window=ROLLING_WINDOW)
    df_te["abs_z"] = df_te["z_score"].abs()

    # ── Thresholds sobre primer 50% del test ─────────────────────────────
    valid = df_te.dropna(subset=["z_score"]).copy()
    cutoff = int(len(valid) * TRAIN_FRACTION)
    train_z = valid.iloc[:cutoff]["abs_z"]
    p90 = float(train_z.quantile(0.90))
    p95 = float(train_z.quantile(0.95))
    p99 = float(train_z.quantile(0.99))
    print(f"  Thresholds (primer 50% del test): "
          f"p90={p90:.3f}  p95={p95:.3f}  p99={p99:.3f}")

    # ── CUSUM ────────────────────────────────────────────────────────────
    cutoff_idx = df_te.index[:int(len(df_te) * TRAIN_FRACTION)].tolist()
    df_te["cusum_active"] = compute_cusum(df_te["residual"], cutoff_idx)

    # ── Cascada NARANJA-equivalente (igual definicion que V0 ablation) ──
    abs_z = df_te["abs_z"].values
    cusum = df_te["cusum_active"].values
    naranja = (abs_z >= p95) & ((cusum >= 1))
    roja = (abs_z >= p99) & (cusum >= 3)
    df_te["alert"] = naranja | roja
    df_te["alert_level"] = "VERDE"
    df_te.loc[(abs_z >= p90) & (abs_z < p95), "alert_level"] = "AMBAR"
    df_te.loc[naranja, "alert_level"] = "NARANJA"
    df_te.loc[roja, "alert_level"] = "ROJA"

    print(f"\n  Distribucion alertas:")
    for lvl in ["VERDE", "AMBAR", "NARANJA", "ROJA"]:
        n = int((df_te["alert_level"] == lvl).sum())
        print(f"    {lvl:8s}  {n:>5,}")

    # ── Validacion contra events_panel ───────────────────────────────────
    events = pd.read_csv(EVENTS_CSV, encoding="utf-8")
    pe = lag_per_event(df_te[["timestamp", "alert"]], events)
    iw = pe[~pe["out_of_window"]]
    det = iw[iw["detected"]]
    recall_event = len(det) / len(iw) if len(iw) else np.nan
    median_lag = float(det["lag_hours"].median()) if len(det) else np.nan
    recall_24h = float((det["lag_hours"] <= 24).mean()) if len(det) else 0.0
    recall_168h = float((det["lag_hours"] <= 168).mean()) if len(det) else 0.0

    fpr_prec = fpr_precision_72h(df_te[["timestamp", "alert"]], events)

    # ── Reporte ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Resultados — v4-quantile-cleanONLY cascada (test 2024)")
    print("=" * 72)
    print(f"  n_events_in_window     = {len(iw)}")
    print(f"  n_detected             = {len(det)}")
    print(f"  recall@event           = {recall_event:.3f}")
    print(f"  median_lag (h)         = {median_lag:.1f}")
    print(f"  recall@24h             = {recall_24h:.3f}")
    print(f"  recall@168h            = {recall_168h:.3f}")
    print(f"  total_alerts           = {fpr_prec['total_alerts']}")
    print(f"  fpr_per_month          = {fpr_prec['fpr_per_month']:.2f}")
    print(f"  precision@72h          = {fpr_prec['precision_72h']:.3f}")

    print("\n" + "=" * 72)
    print("  Comparativa contra V0_baseline_naranja (v3 cascada, del ablation):")
    print("=" * 72)
    print(f"                            v4-quantile    v3-mean (V0)")
    print(f"  recall@event              {recall_event:.3f}          0.833")
    print(f"  median_lag (h)            {median_lag:.1f}          180.0")
    print(f"  recall@24h                {recall_24h:.3f}          0.200")
    print(f"  fpr_per_month             {fpr_prec['fpr_per_month']:.2f}          17.78")
    print(f"  precision@72h             {fpr_prec['precision_72h']:.3f}          0.148")

    # Guardar
    out_csv = RESULTS_DIR / "cascada_v4_quantile_metrics.json"
    with open(out_csv, "w") as fh:
        json.dump({
            "model": "v4-quantile-cleanONLY",
            "features": FEATURES_A_CLEAN,
            "n_test_2024": len(df_te),
            "recall_event": recall_event,
            "median_lag_h": median_lag,
            "recall_at_24h": recall_24h,
            "recall_at_168h": recall_168h,
            **fpr_prec,
            "thresholds": {"p90": p90, "p95": p95, "p99": p99},
        }, fh, indent=2)
    print(f"\n  Metrics: {out_csv}")


if __name__ == "__main__":
    main()
