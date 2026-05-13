"""
ablate_lag_components.py
------------------------
Ablation empírico de las componentes del lag de detección de 180h.

Variantes evaluadas (todas sobre el mismo conjunto de residuos v3):
  V0_baseline    : sistema actual (|z|>=p99 AND IF>=p95 AND CUSUM>=3h)
  V1_no_persist  : CUSUM persistence relajada a >=1h
  V2_no_AND      : sin IF y sin CUSUM (solo |z|>=p99)
  V3_window24h   : z-score con ventana 24h en lugar de 168h
  V4_EWMA        : EWMA lambda_mu=0.94, lambda_sig=0.97 + persistence 2h

Para cada variante se computa el lag (horas) entre date_start de cada
evento del panel y el primer trigger dentro de [date_start, date_end].
Se aplica la MISMA mascara de in-alert-window que event_panel_validation.

Salidas:
    results/ablation_lag_components.csv      (un registro por variante)
    results/ablation_lag_per_event.csv       (detalle variante x evento)
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from anomaly_detection_v3 import (
    compute_rolling_zscore,
    compute_regime_thresholds,
)

DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

EVENTS_CSV = DATA_DIR / "events_panel.csv"
ALERTS_CSV = RESULTS_DIR / "alerts_registry_v3.csv"

ALERT_WINDOW_START = pd.Timestamp("2022-01-01")
ALERT_WINDOW_END = pd.Timestamp("2024-12-31 23:00")


# ════════════════════════════════════════════════════════════════════════════
#  z-score variants
# ════════════════════════════════════════════════════════════════════════════

def zscore_rolling_window(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling z por régimen con la misma lógica que anomaly_detection_v3."""
    z = pd.Series(np.nan, index=df.index, dtype=float)
    for regime, g in df.groupby("regime"):
        idx = g.sort_values("timestamp").index
        z.loc[idx] = compute_rolling_zscore(df.loc[idx, "residual"],
                                            window=window).values
    return z


def zscore_ewma(df: pd.DataFrame,
                lambda_mu: float = 0.94,
                lambda_sig: float = 0.97,
                init_window: int = 168,
                sigma_floor: float = None) -> pd.Series:
    """EWMA recursiva por régimen con shift(1) implícito.

    mu_t = (1-l_mu) * r_{t-1} + l_mu * mu_{t-1}
    sig2_t = (1-l_s) * (r_{t-1}-mu_{t-1})^2 + l_s * sig2_{t-1}
    z_t = (r_t - mu_t) / sqrt(sig2_t)

    sigma_floor: si None, usa proxy de pre-crisis (primer trimestre de
    residuos disponibles) * 0.1. Si se pasa, se respeta.
    """
    z = pd.Series(np.nan, index=df.index, dtype=float)
    if sigma_floor is None:
        # Proxy de pre-crisis: primer trimestre 2022 (antes de Excepcion Iberica)
        ts = pd.to_datetime(df["timestamp"])
        mask_proxy = (ts >= pd.Timestamp("2022-01-01")) & (ts < pd.Timestamp("2022-04-01"))
        proxy_sigma = float(df.loc[mask_proxy, "residual"].dropna().std())
        sigma_floor = proxy_sigma * 0.1

    for regime, g in df.groupby("regime"):
        idx_list = g.sort_values("timestamp").index.tolist()
        r = df.loc[idx_list, "residual"].values
        n = len(r)
        if n < init_window + 10:
            continue
        mu = np.zeros(n)
        sig2 = np.zeros(n)
        # Init con primeras init_window observaciones (training puro)
        valid_init = r[:init_window]
        valid_init = valid_init[~np.isnan(valid_init)]
        if len(valid_init) < 10:
            continue
        mu[init_window - 1] = float(np.mean(valid_init))
        sig2[init_window - 1] = float(np.var(valid_init))
        z_arr = np.full(n, np.nan)
        for t in range(init_window, n):
            if np.isnan(r[t - 1]):
                mu[t] = mu[t - 1]
                sig2[t] = sig2[t - 1]
            else:
                mu[t] = (1 - lambda_mu) * r[t - 1] + lambda_mu * mu[t - 1]
                sig2[t] = (1 - lambda_sig) * (r[t - 1] - mu[t - 1]) ** 2 \
                          + lambda_sig * sig2[t - 1]
            if np.isnan(r[t]):
                continue
            sigma_t = max(np.sqrt(max(sig2[t], 0.0)), sigma_floor)
            z_arr[t] = (r[t] - mu[t]) / sigma_t
        # Clip simetrico igual que v3
        z_arr = np.clip(z_arr, -10, 10)
        z.loc[idx_list] = z_arr
    return z


# ════════════════════════════════════════════════════════════════════════════
#  Trigger logic por variante
# ════════════════════════════════════════════════════════════════════════════

def trigger_baseline(df: pd.DataFrame, thresholds: dict, p95_if: float,
                     cusum_min: int = 3, require_if: bool = True,
                     require_cusum: bool = True) -> pd.Series:
    """Trigger = condiciones equivalentes a NARANJA o ROJA del v3.

    Convención: trigger si (|z| >= p95 del régimen) y opcionalmente IF y CUSUM.
    Esto coincide con el umbral NARANJA del v3 (|z|>=p95 AND (IF>=p95 OR CUSUM>=1))
    cuando require_if=True con OR. Aquí lo simplificamos a AND para coherencia
    con el lenguaje del paper sobre la cascada AND.
    """
    abs_z = df["z_score"].abs()
    z_p95 = df["regime"].map({r: t["p95"] for r, t in thresholds.items()})
    z_p99 = df["regime"].map({r: t["p99"] for r, t in thresholds.items()})

    # Trigger = condicion NARANJA o ROJA del v3
    # Equivalente operativo: alerta operacional (no ámbar)
    cond_z = abs_z >= z_p95
    cond_if = df["if_score"] >= p95_if if require_if else True
    if require_cusum:
        cond_cusum = df["cusum_active"] >= cusum_min
        return cond_z & (cond_if | cond_cusum)  # NARANJA-equivalente con OR
    return cond_z & cond_if if require_if else cond_z


def trigger_simple(df: pd.DataFrame, thresholds: dict,
                   percentile: str = "p99") -> pd.Series:
    """Trigger simple: |z| >= percentil del régimen, sin IF ni CUSUM."""
    abs_z = df["z_score"].abs()
    z_thr = df["regime"].map({r: t[percentile] for r, t in thresholds.items()})
    return abs_z >= z_thr


def trigger_ewma(df: pd.DataFrame, k_threshold: float = 3.0,
                 persistence: int = 2,
                 k_by_regime: dict = None) -> pd.Series:
    """Trigger EWMA: |z_ewma| > k durante 'persistence' horas consecutivas.

    Si k_by_regime se pasa, el umbral es régimen-específico (V4b).
    """
    trig = pd.Series(False, index=df.index)
    for regime, g in df.groupby("regime"):
        idx = g.sort_values("timestamp").index
        k_r = k_by_regime[regime] if k_by_regime else k_threshold
        f = (df.loc[idx, "z_score"].abs() > k_r).astype(int).values
        run = np.zeros(len(f), dtype=int)
        for i in range(len(f)):
            run[i] = run[i - 1] + 1 if (f[i] == 1 and i > 0) else int(f[i] == 1)
        trig.loc[idx] = run >= persistence
    return trig


def trigger_naive_spread(df: pd.DataFrame, train_fraction: float = 0.5) -> pd.Series:
    """V_naive: |spread_observed| > p99 régimen-específico (calculado sobre
    primer 50% de cada régimen). Referencia del techo de manifestación.
    """
    trig = pd.Series(False, index=df.index)
    for regime, g in df.groupby("regime"):
        idx = g.sort_values("timestamp").index
        cutoff = int(len(idx) * train_fraction)
        train_idx = idx[:cutoff]
        p99_spread = float(df.loc[train_idx, "spread_observed"].abs().quantile(0.99))
        trig.loc[idx] = df.loc[idx, "spread_observed"].abs() >= p99_spread
    return trig


# ════════════════════════════════════════════════════════════════════════════
#  Lag por evento
# ════════════════════════════════════════════════════════════════════════════

def lag_per_event(df_trig: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Para cada evento, primer trigger dentro de [date_start, date_end].
    Devuelve DataFrame con event_id, magnitud, lag_hours, detected.
    """
    rows = []
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev["date_start"])
        end = pd.Timestamp(ev["date_end"]) + pd.Timedelta(hours=23)
        if end < ALERT_WINDOW_START or start > ALERT_WINDOW_END:
            rows.append({
                "event_id": ev["event_id"],
                "magnitude": ev["magnitude_indicator"],
                "event_type": ev["event_type"],
                "out_of_window": True,
                "detected": False,
                "lag_hours": np.nan,
            })
            continue
        ws = max(start, ALERT_WINDOW_START)
        we = min(end, ALERT_WINDOW_END)
        mask = (df_trig["timestamp"] >= ws) & (df_trig["timestamp"] <= we) \
               & df_trig["trigger"]
        if mask.any():
            first = df_trig.loc[mask, "timestamp"].min()
            lag = (first - start).total_seconds() / 3600.0
        else:
            lag = np.nan
        rows.append({
            "event_id": ev["event_id"],
            "magnitude": ev["magnitude_indicator"],
            "event_type": ev["event_type"],
            "out_of_window": False,
            "detected": bool(mask.any()),
            "lag_hours": float(lag) if not np.isnan(lag) else np.nan,
        })
    return pd.DataFrame(rows)


def fpr_and_precision(df_trig: pd.DataFrame, events: pd.DataFrame,
                       buffer_h: int = 72) -> dict:
    """FPR (alertas/mes a más de buffer_h del CENTRO de cualquier evento) y
    precision@buffer.

    Usa el centro del evento como referencia (no el intervalo) para no caer
    en la tautología que el paper marca como DEPRECATED: eventos de 1+ años
    de duración cubren todo el periodo si se usa intervalo.

    Misma definición que event_panel_validation.py compute_precision_at_window.
    """
    ts = pd.to_datetime(df_trig["timestamp"])
    in_period = (ts >= ALERT_WINDOW_START) & (ts <= ALERT_WINDOW_END)
    trig = df_trig["trigger"].astype(bool)

    # Centros de evento (todos, incluso fuera del periodo de alerta)
    centers = []
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev["date_start"])
        end = pd.Timestamp(ev["date_end"]) + pd.Timedelta(hours=23)
        centers.append(start + (end - start) / 2)
    centers_s = pd.Series(centers).values.astype("datetime64[ns]")

    # Vectorizado: para cada trigger activo en periodo, distancia mínima al centro
    mask = (trig & in_period).values
    ts_arr = ts.values
    near = np.zeros(len(ts_arr), dtype=bool)
    if mask.any():
        ts_active = ts_arr[mask]
        # diff matrix: (n_active, n_centers)
        diffs = np.abs(ts_active[:, None] - centers_s[None, :])
        near_active = (diffs <= np.timedelta64(buffer_h, "h")).any(axis=1)
        near[mask] = near_active

    n_total = int(mask.sum())
    n_inside = int(near.sum())
    n_outside = n_total - n_inside
    n_months = 36
    return {
        "total_triggers": n_total,
        "fpr_per_month": n_outside / n_months,
        "precision_72h": n_inside / max(n_total, 1),
    }


def summarize(per_event: pd.DataFrame, variant: str,
              df_trig: pd.DataFrame = None,
              events: pd.DataFrame = None) -> dict:
    """Métricas agregadas sobre eventos in-window + FPR si df_trig se pasa."""
    iw = per_event[~per_event["out_of_window"]]
    detected = iw[iw["detected"]]
    n_iw = len(iw)
    n_det = len(detected)
    out = {
        "variant": variant,
        "n_events_in_window": n_iw,
        "n_detected": n_det,
        "recall_event": n_det / n_iw if n_iw else np.nan,
        "median_lag_h": float(detected["lag_hours"].median()) if n_det else np.nan,
        "mean_lag_h": float(detected["lag_hours"].mean()) if n_det else np.nan,
        "p25_lag_h": float(detected["lag_hours"].quantile(0.25)) if n_det else np.nan,
        "p75_lag_h": float(detected["lag_hours"].quantile(0.75)) if n_det else np.nan,
        "recall_at_24h": float((detected["lag_hours"] <= 24).mean()) if n_det else 0.0,
        "recall_at_48h": float((detected["lag_hours"] <= 48).mean()) if n_det else 0.0,
        "recall_at_168h": float((detected["lag_hours"] <= 168).mean()) if n_det else 0.0,
    }
    if df_trig is not None and events is not None:
        out.update(fpr_and_precision(df_trig, events))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Ablation del lag de deteccion — MIBEL Congestion Monitor")
    print("=" * 72)

    df_alerts = pd.read_csv(ALERTS_CSV)
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
    df_alerts = df_alerts.sort_values(["regime", "timestamp"]).reset_index(drop=True)
    df_events = pd.read_csv(EVENTS_CSV, encoding="utf-8")
    print(f"  Alerts registry: {len(df_alerts):,} filas | Eventos panel: {len(df_events)}")

    with open(RESULTS_DIR / "regime_thresholds_v3.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    thresholds_base = meta["thresholds"]
    p95_if = meta["p95_if"]

    # Base columns we need: timestamp, regime, residual, z_score, if_score, cusum_active
    df_base = df_alerts.rename(columns={"z_score": "z_score_base"}).copy()
    df_base["z_score"] = df_base["z_score_base"]

    variants = {}

    # ── V0 baseline (replica del sistema actual: NARANJA-equiv) ───────────
    print("\n[V0] baseline (NARANJA-equiv: |z|>=p95 AND (IF>=p95 OR CUSUM>=1))")
    trig = trigger_baseline(df_base, thresholds_base, p95_if,
                            cusum_min=1, require_if=True, require_cusum=True)
    variants["V0_baseline_naranja"] = trig

    # Baseline puro ROJA (la cascada AND completa)
    print("[V0b] baseline ROJA (|z|>=p99 AND IF>=p95 AND CUSUM>=3)")
    abs_z = df_base["z_score"].abs()
    z_p99 = df_base["regime"].map({r: t["p99"] for r, t in thresholds_base.items()})
    trig_red = (abs_z >= z_p99) & (df_base["if_score"] >= p95_if) \
               & (df_base["cusum_active"] >= 3)
    variants["V0b_baseline_roja"] = trig_red

    # ── V1 CUSUM persistence relajada ──────────────────────────────────────
    print("[V1] CUSUM persistence relajada a >=1h (resto igual a V0b)")
    trig_v1 = (abs_z >= z_p99) & (df_base["if_score"] >= p95_if) \
              & (df_base["cusum_active"] >= 1)
    variants["V1_no_persist"] = trig_v1

    # ── V2 sin AND (solo |z|>=p99) ─────────────────────────────────────────
    print("[V2] solo |z|>=p99 régimen, sin IF, sin CUSUM")
    trig_v2 = trigger_simple(df_base, thresholds_base, percentile="p99")
    variants["V2_no_AND"] = trig_v2

    # ── V3 ventana 24h en lugar de 168h ────────────────────────────────────
    print("[V3] recomputando z con ventana 24h, cascada AND completa")
    df_v3 = df_base.copy()
    df_v3["z_score"] = zscore_rolling_window(df_v3, window=24)
    # Recompute thresholds for the new z
    thr_v3 = compute_regime_thresholds(df_v3, train_fraction=0.5)
    abs_z_v3 = df_v3["z_score"].abs()
    z_p99_v3 = df_v3["regime"].map({r: t["p99"] for r, t in thr_v3.items()})
    trig_v3 = (abs_z_v3 >= z_p99_v3) & (df_v3["if_score"] >= p95_if) \
              & (df_v3["cusum_active"] >= 3)
    variants["V3_window24h"] = trig_v3

    # ── V4 EWMA + persistence 2h, threshold k global ───────────────────────
    print("[V4] EWMA lambda_mu=0.94, lambda_sig=0.97, persistence=2h, k global p99")
    df_v4 = df_base.copy()
    df_v4["z_score"] = zscore_ewma(df_v4, lambda_mu=0.94, lambda_sig=0.97)
    k_thresh = float(df_v4.loc[
        df_v4["timestamp"] < pd.Timestamp("2023-07-01"), "z_score"
    ].abs().quantile(0.99))
    print(f"     k calibrado global (p99 train) = {k_thresh:.3f}")
    trig_v4 = trigger_ewma(df_v4, k_threshold=k_thresh, persistence=2)
    variants["V4_EWMA"] = trig_v4

    # ── V4b: EWMA con k régime-specific (p95) y dos niveles de persistencia ──
    print("\n[V4b] EWMA con sigma_floor proxy pre-crisis + k régime-specific p95")
    # σ_floor proxy explícito: primer trimestre de 2022 antes de Excepción
    ts_all = pd.to_datetime(df_base["timestamp"])
    mask_proxy = (ts_all >= pd.Timestamp("2022-01-01")) & (ts_all < pd.Timestamp("2022-04-01"))
    proxy_sigma = float(df_base.loc[mask_proxy, "residual"].dropna().std())
    sigma_floor_v4b = proxy_sigma * 0.1
    print(f"     sigma_proxy_pre_crisis (Q1 2022) = {proxy_sigma:.3f}")
    print(f"     sigma_floor (10%) = {sigma_floor_v4b:.4f}")

    df_v4b = df_base.copy()
    df_v4b["z_score"] = zscore_ewma(df_v4b, lambda_mu=0.94, lambda_sig=0.97,
                                    sigma_floor=sigma_floor_v4b)
    # k régime-specific al p95 de |z_ewma| sobre primer 50% del régimen
    k_by_regime = {}
    for regime, g in df_v4b.groupby("regime"):
        idx = g.sort_values("timestamp").index
        cutoff = int(len(idx) * 0.5)
        train_idx = idx[:cutoff]
        k_r = float(df_v4b.loc[train_idx, "z_score"].abs().quantile(0.95))
        k_by_regime[regime] = k_r
        print(f"     k_{regime} (p95 train) = {k_r:.3f}")

    print("[V4b_p1] persistence=1h")
    trig_v4b_p1 = trigger_ewma(df_v4b, k_by_regime=k_by_regime, persistence=1)
    variants["V4b_p1_ewma_k95_persist1"] = trig_v4b_p1

    print("[V4b_p2] persistence=2h")
    trig_v4b_p2 = trigger_ewma(df_v4b, k_by_regime=k_by_regime, persistence=2)
    variants["V4b_p2_ewma_k95_persist2"] = trig_v4b_p2

    # ── V_naive: |spread_observed|>p99 régimen (referencia manifestación) ───
    print("[V_naive] |spread_observed| > p99 régimen (techo de manifestación)")
    trig_naive = trigger_naive_spread(df_base, train_fraction=0.5)
    variants["V_naive_spread_p99"] = trig_naive

    # ── Computar lag por variante ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Resultados por variante")
    print("=" * 72)

    summary_rows = []
    per_event_rows = []
    for name, trig in variants.items():
        df_t = df_alerts[["timestamp"]].copy()
        df_t["trigger"] = trig.values if hasattr(trig, "values") else trig
        pe = lag_per_event(df_t, df_events)
        pe["variant"] = name
        per_event_rows.append(pe)
        summary_rows.append(summarize(pe, name, df_trig=df_t, events=df_events))

    df_summary = pd.DataFrame(summary_rows)
    df_per_event = pd.concat(per_event_rows, ignore_index=True)

    # ── Pretty print summary ───────────────────────────────────────────────
    cols = ["variant", "n_detected", "recall_event", "median_lag_h",
            "recall_at_24h", "recall_at_168h",
            "total_triggers", "fpr_per_month", "precision_72h"]
    df_print = df_summary[cols].copy()
    for c in df_print.columns:
        if c not in ("variant", "n_detected", "total_triggers"):
            df_print[c] = df_print[c].round(3)
    print("\n" + df_print.to_string(index=False))

    df_summary.to_csv(RESULTS_DIR / "ablation_lag_components.csv", index=False)
    df_per_event.to_csv(RESULTS_DIR / "ablation_lag_per_event.csv", index=False)
    print(f"\nGuardado: {RESULTS_DIR / 'ablation_lag_components.csv'}")
    print(f"Guardado: {RESULTS_DIR / 'ablation_lag_per_event.csv'}")


if __name__ == "__main__":
    main()
