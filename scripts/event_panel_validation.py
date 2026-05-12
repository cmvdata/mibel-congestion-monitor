"""
event_panel_validation.py
-------------------------
T8 — Validación sistemática contra panel de eventos documentados.

Lógica:
1. Para cada evento del panel (40 eventos verificados, 2019-2024) se verifica
   si AL MENOS UNA hora dentro de [date_start, date_end] tiene alerta NARANJA
   o ROJA del sistema (Modelo A v3 + cascada v3).
2. Se calculan métricas globales y desagregadas (por magnitud y tipo).
3. Comparativa contra benchmark naïf: alerta si |spread_da| > p99 del régimen.
4. Análisis de fallos (eventos MUY_ALTA/ALTA no detectados).

Salidas:
    data/event_validation.csv
    results/event_validation_summary.json
    results/recall_by_magnitude.csv      (Tabla 5)
    results/recall_by_type.csv           (Tabla 6)
    results/system_vs_naive_benchmark.csv (Tabla 7)
    plots/event_validation.png

LIMITACIÓN: el sistema solo produce alertas para 2022-2024 (residuos
out-of-sample). Eventos de 2019-2021 (~14 eventos) NO pueden ser detectados
por construcción y se reportan como "out_of_alert_window=True". La precisión
del recall se reporta también condicional al periodo con alertas.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

EVENTS_CSV = DATA_DIR / "events_panel.csv"
ALERTS_CSV = RESULTS_DIR / "alerts_registry_v3.csv"
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

# Régime-specific p99 de |spread_da| (calculados sobre el régimen entero) ──
# para benchmark naïf
def regime_spread_p99(df: pd.DataFrame) -> dict:
    out = {}
    for r, g in df.groupby("regime"):
        out[r] = float(g["spread_da"].abs().quantile(0.99))
    return out


def detect_event(event: pd.Series, alerts: pd.DataFrame, df_full: pd.DataFrame,
                 system_levels=("NARANJA", "ROJA"),
                 alert_window_start=pd.Timestamp("2022-01-01"),
                 alert_window_end=pd.Timestamp("2024-12-31 23:00")) -> dict:
    start = pd.Timestamp(event["date_start"])
    end = pd.Timestamp(event["date_end"]) + pd.Timedelta(hours=23)
    out_of_window = (end < alert_window_start) or (start > alert_window_end)

    info = {
        "event_id": event["event_id"],
        "date_start": str(start.date()),
        "date_end": str(end.date()),
        "duration_h": int((end - start).total_seconds() // 3600 + 1),
        "event_type": event["event_type"],
        "magnitude": event["magnitude_indicator"],
        "direction": event["direction_impact"],
        "out_of_alert_window": bool(out_of_window),
    }

    if out_of_window:
        info.update({
            "system_detected": False,
            "system_n_amber_or_higher": 0,
            "system_n_orange_or_higher": 0,
            "system_n_red": 0,
            "system_max_abs_z": np.nan,
            "system_tier_max": "OUT_OF_WINDOW",
            "system_lag_hours": np.nan,
            "naive_detected": False,
            "naive_n_alert_hours": 0,
        })
        return info

    # Sistema: ventana del evento intersección con periodo con alertas
    win_start = max(start, alert_window_start)
    win_end = min(end, alert_window_end)
    win_alerts = alerts[(alerts["timestamp"] >= win_start) &
                        (alerts["timestamp"] <= win_end)]

    n_amber = int(win_alerts["alert_level"].isin(["ÁMBAR", "NARANJA", "ROJA"]).sum())
    n_orange = int(win_alerts["alert_level"].isin(["NARANJA", "ROJA"]).sum())
    n_red = int((win_alerts["alert_level"] == "ROJA").sum())
    detected = n_orange > 0  # Definición: detectado si ≥1 hora NARANJA/ROJA

    if not win_alerts.empty:
        max_abs_z = float(win_alerts["abs_z"].max()) if "abs_z" in win_alerts.columns else np.nan
    else:
        max_abs_z = np.nan

    if detected:
        first_alert = win_alerts[win_alerts["alert_level"].isin(system_levels)] \
                                .sort_values("timestamp").iloc[0]["timestamp"]
        lag_hours = (pd.Timestamp(first_alert) - start).total_seconds() / 3600
        if n_red > 0:
            tier_max = "ROJA"
        else:
            tier_max = "NARANJA"
    else:
        lag_hours = np.nan
        tier_max = "AMBAR" if n_amber > 0 else "VERDE"

    # Naive benchmark: |spread| > p99 régimen ──────────────────────────────
    df_win = df_full[(df_full["timestamp"] >= win_start) &
                     (df_full["timestamp"] <= win_end)]
    p99_by_regime = regime_spread_p99(df_full)
    if df_win.empty:
        naive_detected = False
        naive_n = 0
    else:
        p99_arr = df_win["regime"].map(p99_by_regime).values
        flag = np.abs(df_win["spread_da"].values) > p99_arr
        naive_n = int(flag.sum())
        naive_detected = naive_n > 0

    info.update({
        "system_detected": bool(detected),
        "system_n_amber_or_higher": n_amber,
        "system_n_orange_or_higher": n_orange,
        "system_n_red": n_red,
        "system_max_abs_z": max_abs_z,
        "system_tier_max": tier_max,
        "system_lag_hours": float(lag_hours) if not np.isnan(lag_hours) else np.nan,
        "naive_detected": naive_detected,
        "naive_n_alert_hours": naive_n,
    })
    return info


def main():
    print("=" * 64)
    print("  T8 — Validación sistemática contra panel de eventos")
    print("=" * 64)

    df_events = pd.read_csv(EVENTS_CSV, encoding="utf-8")
    print(f"  Panel cargado: {len(df_events)} eventos")
    print(f"  Magnitudes: {df_events['magnitude_indicator'].value_counts().to_dict()}")
    print(f"  Tipos:      {df_events['event_type'].value_counts().to_dict()}")

    df_alerts = pd.read_csv(ALERTS_CSV)
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
    print(f"  Alertas:    {len(df_alerts):,} horas con alerta clasificada")
    print(f"  Rango alertas: {df_alerts['timestamp'].min()} → {df_alerts['timestamp'].max()}")

    df_full = pd.read_parquet(DATASET)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
    print(f"  Dataset:    {len(df_full):,} horas")

    alert_start = df_alerts["timestamp"].min()
    alert_end = df_alerts["timestamp"].max()

    # ── Detectar cada evento ───────────────────────────────────────────────
    rows = []
    for _, ev in df_events.iterrows():
        rows.append(detect_event(ev, df_alerts, df_full,
                                 alert_window_start=alert_start,
                                 alert_window_end=alert_end))
    df_val = pd.DataFrame(rows)
    df_val.to_csv(DATA_DIR / "event_validation.csv", index=False)
    print(f"\n  event_validation.csv guardado ({len(df_val)} eventos)")

    # ── Métricas globales ──────────────────────────────────────────────────
    in_window = df_val[~df_val["out_of_alert_window"]]
    print(f"\n  Eventos in alert window (2022-2024): {len(in_window)} de {len(df_val)}")

    recall_global_in = in_window["system_detected"].mean()
    recall_global_all = df_val["system_detected"].mean()
    naive_recall_in = in_window["naive_detected"].mean()
    print(f"  Recall global (in-window):     {recall_global_in:.3f} ({int(in_window['system_detected'].sum())}/{len(in_window)})")
    print(f"  Recall global (all events):    {recall_global_all:.3f} ({int(df_val['system_detected'].sum())}/{len(df_val)})")
    print(f"  Naive recall (in-window):      {naive_recall_in:.3f} ({int(in_window['naive_detected'].sum())}/{len(in_window)})")

    # ── Tabla 5 — Recall por magnitud ──────────────────────────────────────
    by_mag_in = (in_window.groupby("magnitude")
                 .agg(n=("event_id", "count"),
                      n_detected=("system_detected", "sum"),
                      recall=("system_detected", "mean"),
                      n_naive_detected=("naive_detected", "sum"),
                      recall_naive=("naive_detected", "mean"))
                 .reset_index())
    by_mag_all = (df_val.groupby("magnitude")
                  .agg(n_total=("event_id", "count"),
                       n_in_window=("out_of_alert_window", lambda s: int((~s).sum())))
                  .reset_index())
    by_mag = by_mag_all.merge(by_mag_in, on="magnitude", how="left").fillna(0)
    by_mag = by_mag[["magnitude", "n_total", "n_in_window", "n_detected",
                     "recall", "n_naive_detected", "recall_naive"]]
    by_mag.to_csv(RESULTS_DIR / "recall_by_magnitude.csv", index=False)
    print("\n  Tabla 5 — Recall por magnitud (in-window)")
    print(by_mag.to_string(index=False))

    # ── Tabla 6 — Recall por tipo ──────────────────────────────────────────
    by_type_in = (in_window.groupby("event_type")
                  .agg(n=("event_id", "count"),
                       n_detected=("system_detected", "sum"),
                       recall=("system_detected", "mean"),
                       n_naive_detected=("naive_detected", "sum"),
                       recall_naive=("naive_detected", "mean"))
                  .reset_index())
    by_type_all = (df_val.groupby("event_type")
                   .agg(n_total=("event_id", "count"),
                        n_in_window=("out_of_alert_window", lambda s: int((~s).sum())))
                   .reset_index())
    by_type = by_type_all.merge(by_type_in, on="event_type", how="left").fillna(0)
    by_type = by_type[["event_type", "n_total", "n_in_window", "n_detected",
                       "recall", "n_naive_detected", "recall_naive"]]
    by_type.to_csv(RESULTS_DIR / "recall_by_type.csv", index=False)
    print("\n  Tabla 6 — Recall por tipo de evento (in-window)")
    print(by_type.to_string(index=False))

    # ── Tabla 7 — Sistema vs naive ─────────────────────────────────────────
    cmp = pd.DataFrame({
        "metric": ["recall_in_window", "n_detected", "n_total_in_window"],
        "system_v3": [recall_global_in, int(in_window["system_detected"].sum()), len(in_window)],
        "naive_p99_spread": [naive_recall_in, int(in_window["naive_detected"].sum()), len(in_window)],
    })
    # ¿coinciden o difieren?
    both = in_window[in_window["system_detected"] & in_window["naive_detected"]]
    only_sys = in_window[in_window["system_detected"] & ~in_window["naive_detected"]]
    only_naive = in_window[~in_window["system_detected"] & in_window["naive_detected"]]
    neither = in_window[~in_window["system_detected"] & ~in_window["naive_detected"]]
    cmp_extra = pd.DataFrame({
        "category": ["both detect", "system only", "naive only", "neither"],
        "n": [len(both), len(only_sys), len(only_naive), len(neither)],
    })
    cmp.to_csv(RESULTS_DIR / "system_vs_naive_benchmark.csv", index=False)
    cmp_extra.to_csv(RESULTS_DIR / "system_vs_naive_breakdown.csv", index=False)
    print("\n  Tabla 7 — Sistema (cascada Modelo A v3) vs naive (|spread| > p99 régimen)")
    print(cmp.to_string(index=False))
    print("\n  Solapamiento detección:")
    print(cmp_extra.to_string(index=False))

    # ── Análisis de fallos: MUY_ALTA / ALTA no detectados in-window ────────
    failures = in_window[
        (in_window["magnitude"].isin(["MUY_ALTA", "ALTA"])) &
        (~in_window["system_detected"])
    ].merge(df_events[["event_id", "event_name", "description_es"]],
            on="event_id", how="left")
    failures.to_csv(RESULTS_DIR / "high_magnitude_misses.csv", index=False)
    print(f"\n  Fallos MUY_ALTA/ALTA in-window no detectados: {len(failures)}")
    if len(failures) > 0:
        print(failures[["event_id", "magnitude", "event_type",
                        "system_n_amber_or_higher", "system_n_red", "event_name"]].to_string(index=False))

    # ── Precisión con ventana estrecha (no tautológica) ────────────────────
    df_alerts_high = df_alerts[
        df_alerts["alert_level"].isin(["NARANJA", "ROJA"])
    ].copy()

    # Centro del evento como punto medio del intervalo documentado
    event_centers = []
    for _, r in df_events.iterrows():
        start = pd.Timestamp(r["date_start"])
        end = pd.Timestamp(r["date_end"]) + pd.Timedelta(hours=23)
        event_centers.append(start + (end - start) / 2)

    def compute_precision_at_window(window_hours):
        """TP si la alerta está a <= window_hours del centro de algún evento."""
        delta_sec = window_hours * 3600
        def near_any(t):
            return any(
                abs((t - c).total_seconds()) <= delta_sec for c in event_centers
            )
        col = f"near_event_{window_hours}h"
        df_alerts_high[col] = df_alerts_high["timestamp"].apply(near_any)
        n_total = len(df_alerts_high)
        n_tp = int(df_alerts_high[col].sum())
        return n_tp / n_total if n_total else 0.0

    precision_72h = compute_precision_at_window(72)
    precision_7d = compute_precision_at_window(7 * 24)
    precision_30d = compute_precision_at_window(30 * 24)

    # FP rate por mes fuera de la ventana estrecha (72h)
    alerts_not_near_72h = int((~df_alerts_high["near_event_72h"]).sum())
    if len(df_alerts_high) > 0:
        span_days = (
            df_alerts_high["timestamp"].max() - df_alerts_high["timestamp"].min()
        ).days
        n_months = max(span_days / 30, 1)
        fp_rate_per_month = alerts_not_near_72h / n_months
    else:
        fp_rate_per_month = 0.0

    # Métrica tautológica anterior, mantenida con etiqueta DEPRECATED
    def in_any_event_full(t):
        for _, r in df_events.iterrows():
            s = pd.Timestamp(r["date_start"])
            e = pd.Timestamp(r["date_end"]) + pd.Timedelta(hours=23)
            if s <= t <= e:
                return True
        return False
    df_alerts_high["in_event"] = df_alerts_high["timestamp"].apply(in_any_event_full)
    n_alert_total = len(df_alerts_high)
    n_alert_in_event = int(df_alerts_high["in_event"].sum())
    precision = n_alert_in_event / n_alert_total if n_alert_total else 0.0
    print(f"\n  Precisión (alertas Naranja/Roja):")
    print(f"    precision_at_72h  = {precision_72h:.3f}")
    print(f"    precision_at_7d   = {precision_7d:.3f}")
    print(f"    precision_at_30d  = {precision_30d:.3f}")
    print(f"    fp_rate/mes fuera 72h = {fp_rate_per_month:.2f}")
    print(f"    [DEPRECATED] precision tautológica = {precision:.3f}")

    # ── JSON resumen ───────────────────────────────────────────────────────
    summary = {
        "n_events_total": int(len(df_val)),
        "n_events_in_alert_window": int(len(in_window)),
        "alert_window": {"start": str(alert_start), "end": str(alert_end)},
        "recall_global_in_window": float(recall_global_in),
        "recall_global_all_events": float(recall_global_all),
        "naive_recall_in_window": float(naive_recall_in),
        "precision_at_72h": float(precision_72h),
        "precision_at_7d": float(precision_7d),
        "precision_at_30d": float(precision_30d),
        "fp_rate_per_month_outside_72h": float(fp_rate_per_month),
        "n_alerts_high": int(len(df_alerts_high)),
        "n_alerts_tp_72h": int(df_alerts_high["near_event_72h"].sum()),
        "alert_precision_tautological_DEPRECATED": float(precision),
        "_note_precision_tautological": (
            "DEPRECATED: precision sobre intervalos largos cubre casi toda la "
            "ventana. Use precision_at_72h o precision_at_7d como metricas no "
            "tautologicas."
        ),
        "n_alerts_orange_or_red": int(n_alert_total),
        "n_alerts_in_documented_event": int(n_alert_in_event),
        "by_magnitude": by_mag.to_dict(orient="records"),
        "by_type": by_type.to_dict(orient="records"),
        "system_vs_naive_breakdown": cmp_extra.to_dict(orient="records"),
        "n_high_magnitude_misses": int(len(failures)),
    }

    # ── A12: Wilson CI y test McNemar pareado ──────────────────────────────
    try:
        from statsmodels.stats.proportion import proportion_confint
        from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

        if len(in_window) > 0:
            n_iw = len(in_window)
            sys_det = int(in_window["system_detected"].sum())
            naive_det = int(in_window["naive_detected"].sum()) \
                if "naive_detected" in in_window.columns else None

            sys_ci_low, sys_ci_high = proportion_confint(sys_det, n_iw, method="wilson")
            summary["system_recall_n"] = n_iw
            summary["system_recall_ci95_low"] = float(sys_ci_low)
            summary["system_recall_ci95_high"] = float(sys_ci_high)

            if naive_det is not None:
                naive_ci_low, naive_ci_high = proportion_confint(
                    naive_det, n_iw, method="wilson"
                )
                summary["naive_recall_n"] = n_iw
                summary["naive_recall_ci95_low"] = float(naive_ci_low)
                summary["naive_recall_ci95_high"] = float(naive_ci_high)

                # McNemar pareado: tabla 2x2 garantizada
                full_table = pd.DataFrame(
                    [[0, 0], [0, 0]],
                    index=[False, True],
                    columns=[False, True],
                )
                ct = pd.crosstab(
                    in_window["system_detected"], in_window["naive_detected"]
                )
                for i in ct.index:
                    for j in ct.columns:
                        full_table.loc[i, j] = ct.loc[i, j]

                mc = mcnemar_test(full_table.values, exact=True)
                summary["mcnemar_statistic"] = (
                    float(mc.statistic) if mc.statistic is not None else None
                )
                summary["mcnemar_pvalue"] = float(mc.pvalue)
                summary["mcnemar_discordant_sys_only"] = int(full_table.loc[True, False])
                summary["mcnemar_discordant_naive_only"] = int(full_table.loc[False, True])
    except ImportError:
        summary["_note_statsmodels"] = (
            "statsmodels not installed; Wilson CI / McNemar skipped"
        )

    # ── A13: lead time agregado (detección operativa vs retrospectiva) ────
    if "system_lag_hours" in in_window.columns:
        detected = in_window[in_window["system_lag_hours"].notna()]
        if len(detected) > 0 and len(in_window) > 0:
            n_24 = int((detected["system_lag_hours"] <= 24).sum())
            n_48 = int((detected["system_lag_hours"] <= 48).sum())
            n_168 = int((detected["system_lag_hours"] <= 168).sum())
            summary["recall_within_24h"] = n_24 / len(in_window)
            summary["recall_within_48h"] = n_48 / len(in_window)
            summary["recall_within_168h"] = n_168 / len(in_window)
            summary["median_lag_hours_detected"] = float(
                detected["system_lag_hours"].median()
            )
            summary["p75_lag_hours_detected"] = float(
                detected["system_lag_hours"].quantile(0.75)
            )

    with open(RESULTS_DIR / "event_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Resumen: {RESULTS_DIR / 'event_validation_summary.json'}")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Systematic validation against the 40-event panel",
                 fontsize=12, fontweight="bold")

    # Recall por magnitud (in-window)
    ax = axes[0]
    by_mag_plot = by_mag[by_mag["n_in_window"] > 0].copy()
    x = np.arange(len(by_mag_plot)); w = 0.4
    ax.bar(x - w/2, by_mag_plot["recall"], w, label="System v3", color="#27AE60")
    ax.bar(x + w/2, by_mag_plot["recall_naive"], w, label="Naive |spread|>p99", color="#7F8C8D")
    ax.set_xticks(x); ax.set_xticklabels(by_mag_plot["magnitude"], rotation=15)
    ax.set_ylabel("Recall"); ax.set_title("Recall by magnitude (in-window)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    for i, (rec, n) in enumerate(zip(by_mag_plot["recall"], by_mag_plot["n_in_window"])):
        ax.text(i - w/2, rec + 0.02, f"n={int(n)}", ha="center", fontsize=8)

    # Recall por tipo (in-window)
    ax = axes[1]
    by_type_plot = by_type[by_type["n_in_window"] > 0].copy()
    x = np.arange(len(by_type_plot)); w = 0.4
    ax.bar(x - w/2, by_type_plot["recall"], w, label="System v3", color="#27AE60")
    ax.bar(x + w/2, by_type_plot["recall_naive"], w, label="Naive |spread|>p99", color="#7F8C8D")
    ax.set_xticks(x); ax.set_xticklabels(by_type_plot["event_type"], rotation=20, ha="right")
    ax.set_ylabel("Recall"); ax.set_title("Recall by event type (in-window)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    for i, (rec, n) in enumerate(zip(by_type_plot["recall"], by_type_plot["n_in_window"])):
        ax.text(i - w/2, rec + 0.02, f"n={int(n)}", ha="center", fontsize=8)

    # Solapamiento sistema vs naive
    ax = axes[2]
    cats = cmp_extra["category"].tolist()
    ns = cmp_extra["n"].tolist()
    cols = ["#27AE60", "#3498DB", "#7F8C8D", "#C0392B"]
    ax.bar(cats, ns, color=cols, alpha=0.85)
    ax.set_ylabel("Events (in-window)")
    ax.set_title("System vs naïve overlap")
    ax.tick_params(axis="x", rotation=15)
    for i, v in enumerate(ns):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOTS_DIR / "event_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {out}")
    print("\n  T8 COMPLETADA")


if __name__ == "__main__":
    main()
