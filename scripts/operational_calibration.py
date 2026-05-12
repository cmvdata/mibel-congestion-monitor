"""
operational_calibration.py
--------------------------
TANDA 2 — Calibración operativa del trade-off precision/coverage.

Análisis PARALELO al pipeline base: NO modifica la cascada del sistema
operativo (Verde/Ámbar/Naranja/Roja) ya commiteada en anomaly_detection_v3.py.

Para cada combinación de umbrales (z_threshold, if_threshold, cusum_min_h)
recalcula una cascada binaria "alerta_high" sobre los residuos del
Modelo A v3 ya generados, y reporta:
  - n_alertas, alertas_per_month
  - precision_at_72h / 7d / 30d (TP = alerta a ≤window del centro del evento)
  - recall_at_event (de los 24 eventos in-window, % con ≥1 alerta dentro
    de [date_start, date_end + 23h])
  - fp_rate_per_month_outside_72h
  - median_lag_hours_detected (para los eventos detectados)

Definición operativa de "alerta_high" en el grid:
    alerta_high(t) = |z(t)| >= z_threshold
                     AND if_score(t) >= if_threshold
                     AND cusum_active(t) >= cusum_min_h

z_threshold y if_threshold son percentiles del training fraction (50%
temporal temprano de cada régimen), agnósticos a la cascada actual.

Grid: 4 (z) × 3 (if) × 4 (cusum) = 48 combinaciones.

Salidas:
    results/operational_calibration_grid.csv
    results/operational_calibration_summary.json
    plots/10_operational_calibration_pareto.png
    plots/11_operational_calibration_table.png
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
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

ALERTS_CSV = RESULTS_DIR / "alerts_registry_v3.csv"
EVENTS_CSV = DATA_DIR / "events_panel.csv"
THRESHOLDS_JSON = RESULTS_DIR / "regime_thresholds_v3.json"

# Periodo donde hay residuos out-of-sample (cascada operativa)
ALERT_START = pd.Timestamp("2022-01-01")
ALERT_END = pd.Timestamp("2024-12-31 23:00")

# Capacidades operativas canónicas (alertas/mes target)
CAPACITY_TARGETS = {
    "muy_baja": 3,
    "baja": 5,
    "media": 10,
    "alta": 20,
}

# Z-threshold: percentiles del |z| sobre training fraction (régime-específico)
# Grid extendido para cubrir zona laxa (15-30 alertas/mes) y alcanzar Pareto
Z_PERCENTILES = [93, 95, 97, 99, 99.5]

# IF-threshold: percentiles del if_score sobre training fraction (global)
# Grid extendido a percentiles bajos para cubrir cascada permisiva del sistema actual
IF_PERCENTILES = [80, 85, 90, 95, 98]

# CUSUM min consecutive hours
CUSUM_MIN_H = [1, 2, 3, 5]


def compute_training_percentile(df: pd.DataFrame, col: str, q: float,
                                by_regime: bool, train_frac: float = 0.5) -> dict | float:
    """Calcula el percentil q de `col` sobre el 50% temporal temprano.

    Si by_regime=True: devuelve dict {regime: percentil}.
    Si by_regime=False: devuelve un único valor (calculado sobre el global
    de todos los regímenes, manteniendo coherencia con el cascada actual).
    """
    if by_regime:
        out = {}
        for regime, g in df.groupby("regime"):
            g_sorted = g.sort_values("timestamp").reset_index(drop=True)
            cutoff = int(len(g_sorted) * train_frac)
            train = g_sorted.iloc[:cutoff]
            out[regime] = float(train[col].abs().quantile(q / 100))
        return out
    else:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        train_parts = []
        for regime, g in df_sorted.groupby("regime"):
            g_r = g.sort_values("timestamp").reset_index(drop=True)
            cutoff = int(len(g_r) * train_frac)
            train_parts.append(g_r.iloc[:cutoff])
        train_all = pd.concat(train_parts, ignore_index=True)
        return float(train_all[col].quantile(q / 100))


def evaluate_grid_point(df_alerts: pd.DataFrame, df_events: pd.DataFrame,
                        z_by_regime: dict, if_global: float, cusum_min: int) -> dict:
    """Para una combinación de umbrales, calcula métricas operativas."""
    df = df_alerts.copy()
    # Mapear umbral z por régimen
    df["z_thr"] = df["regime"].map(z_by_regime)

    # Alerta binaria conjunta
    df["alert_high"] = (
        (df["abs_z"] >= df["z_thr"])
        & (df["if_score"] >= if_global)
        & (df["cusum_active"] >= cusum_min)
    )

    df_high = df[df["alert_high"]].copy()
    n_alerts = int(len(df_high))

    # Periodo en meses
    span_days = (ALERT_END - ALERT_START).days
    n_months = span_days / 30.4375  # promedio incluyendo bisiestos
    alerts_per_month = n_alerts / n_months if n_months > 0 else 0.0

    # ── Precision con ventanas ±72h, ±7d, ±30d desde el centro del evento ──
    event_centers = []
    for _, r in df_events.iterrows():
        start = pd.Timestamp(r["date_start"])
        end = pd.Timestamp(r["date_end"]) + pd.Timedelta(hours=23)
        event_centers.append(start + (end - start) / 2)
    # Forzar nanosecond precision para ambos arrays (pandas 3 default cambió a microsegundos)
    centers_ns = np.array([c.to_datetime64().astype("datetime64[ns]").astype("int64")
                           for c in event_centers])

    def near_any(ts_ns: int, window_h: int) -> bool:
        delta_ns = int(window_h) * 3600 * 10**9
        return bool(np.any(np.abs(centers_ns - ts_ns) <= delta_ns))

    if n_alerts > 0:
        alerts_ts_ns = df_high["timestamp"].to_numpy().astype("datetime64[ns]").astype("int64")
        deltas = np.abs(alerts_ts_ns[:, None] - centers_ns[None, :])  # (n_alerts, n_events)
        min_delta = deltas.min(axis=1)  # ns hasta evento más cercano
        n_tp_72h = int((min_delta <= 72 * 3600 * 10**9).sum())
        n_tp_7d = int((min_delta <= 7 * 24 * 3600 * 10**9).sum())
        n_tp_30d = int((min_delta <= 30 * 24 * 3600 * 10**9).sum())
        precision_72h = n_tp_72h / n_alerts
        precision_7d = n_tp_7d / n_alerts
        precision_30d = n_tp_30d / n_alerts
        fp_outside_72h = n_alerts - n_tp_72h
        fp_rate_per_month = fp_outside_72h / n_months if n_months > 0 else 0.0
    else:
        precision_72h = precision_7d = precision_30d = 0.0
        fp_rate_per_month = 0.0
        n_tp_72h = n_tp_7d = n_tp_30d = 0

    # ── Recall@event: eventos in-window con ≥1 alerta_high dentro ──────────
    in_window_events = []
    for _, r in df_events.iterrows():
        start = pd.Timestamp(r["date_start"])
        end = pd.Timestamp(r["date_end"]) + pd.Timedelta(hours=23)
        if end < ALERT_START or start > ALERT_END:
            continue
        in_window_events.append((r["event_id"], max(start, ALERT_START), min(end, ALERT_END)))

    n_in_window = len(in_window_events)
    detected = 0
    lags_hours = []
    for eid, s, e in in_window_events:
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e) & df["alert_high"]
        sub = df.loc[mask].sort_values("timestamp")
        if len(sub) > 0:
            detected += 1
            first_alert = sub.iloc[0]["timestamp"]
            lag_h = (pd.Timestamp(first_alert) - s).total_seconds() / 3600
            lags_hours.append(lag_h)
    recall_event = detected / n_in_window if n_in_window > 0 else 0.0
    median_lag = float(np.median(lags_hours)) if lags_hours else float("nan")

    return {
        "n_alerts": n_alerts,
        "alerts_per_month": alerts_per_month,
        "n_tp_72h": n_tp_72h,
        "n_tp_7d": n_tp_7d,
        "precision_72h": precision_72h,
        "precision_7d": precision_7d,
        "precision_30d": precision_30d,
        "recall_event": recall_event,
        "n_events_in_window": n_in_window,
        "n_events_detected": detected,
        "fp_rate_per_month_outside_72h": fp_rate_per_month,
        "median_lag_hours": median_lag,
    }


def main():
    print("=" * 64)
    print("  TANDA 2 — Calibración operativa precision/coverage")
    print("=" * 64)

    df_alerts = pd.read_csv(ALERTS_CSV, parse_dates=["timestamp"])
    df_events = pd.read_csv(EVENTS_CSV)
    print(f"  Alertas (residuos OOS): {len(df_alerts):,}")
    print(f"  Eventos en panel: {len(df_events)}")

    # ── Pre-calcular percentiles régime-específicos del |z| sobre training ─
    print("\n  Calculando umbrales sobre training fraction (50% temporal):")
    z_thresholds_grid = {}  # {p: {regime: value}}
    for p in Z_PERCENTILES:
        z_thresholds_grid[p] = compute_training_percentile(
            df_alerts, "abs_z", p, by_regime=True
        )
        print(f"    z_p{p}: {z_thresholds_grid[p]}")

    if_thresholds_grid = {}
    for p in IF_PERCENTILES:
        if_thresholds_grid[p] = compute_training_percentile(
            df_alerts, "if_score", p, by_regime=False
        )
        print(f"    if_p{p}: {if_thresholds_grid[p]:.4f}")

    # ── Barrido del grid ───────────────────────────────────────────────────
    print(f"\n  Evaluando {len(Z_PERCENTILES) * len(IF_PERCENTILES) * len(CUSUM_MIN_H)} combinaciones del grid...")
    rows = []
    for z_p in Z_PERCENTILES:
        for if_p in IF_PERCENTILES:
            for cusum_h in CUSUM_MIN_H:
                metrics = evaluate_grid_point(
                    df_alerts, df_events,
                    z_by_regime=z_thresholds_grid[z_p],
                    if_global=if_thresholds_grid[if_p],
                    cusum_min=cusum_h,
                )
                row = {
                    "z_percentile": z_p,
                    "if_percentile": if_p,
                    "cusum_min_h": cusum_h,
                    "z_thr_crisis": z_thresholds_grid[z_p].get("crisis_y_excepcion", np.nan),
                    "z_thr_post": z_thresholds_grid[z_p].get("post_excepcion", np.nan),
                    "if_thr": if_thresholds_grid[if_p],
                    **metrics,
                }
                rows.append(row)

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(RESULTS_DIR / "operational_calibration_grid.csv", index=False)
    print(f"\n  Grid guardado: {RESULTS_DIR / 'operational_calibration_grid.csv'}")
    print(f"  Rango alertas/mes: {df_grid['alerts_per_month'].min():.2f} → {df_grid['alerts_per_month'].max():.2f}")

    # ── Selección de 4 puntos canónicos ────────────────────────────────────
    # Estrategia: bandas DISJUNTAS por capacidad para evitar duplicados entre
    # 'media' y 'alta'. Dentro de cada banda, Pareto-óptimo por (recall+P@72h)/2.
    cap_bands = {
        "muy_baja": (0.5, 3.5),
        "baja":     (3.5, 7.0),
        "media":    (7.0, 14.0),
        "alta":     (14.0, 30.0),
    }
    print("\n  Seleccionando puntos canónicos por capacidad operativa (bandas disjuntas):")
    canonical = {}
    for cap_name, target in CAPACITY_TARGETS.items():
        lo, hi = cap_bands[cap_name]
        band = df_grid[
            (df_grid["alerts_per_month"] >= lo)
            & (df_grid["alerts_per_month"] <= hi)
        ].copy()
        if len(band) == 0:
            band = df_grid.copy()
            band["diff"] = (band["alerts_per_month"] - target).abs()
            best = band.sort_values("diff").iloc[0]
        else:
            # Score Pareto compuesto: media de recall y precision
            band["pareto_score"] = (band["recall_event"] + band["precision_72h"]) / 2
            best = band.sort_values(
                ["pareto_score", "recall_event"], ascending=[False, False]
            ).iloc[0]
        canonical[cap_name] = {
            "target_alerts_per_month": target,
            "actual_alerts_per_month": float(best["alerts_per_month"]),
            "z_percentile": float(best["z_percentile"]),
            "if_percentile": float(best["if_percentile"]),
            "cusum_min_h": int(best["cusum_min_h"]),
            "z_thr_crisis": float(best["z_thr_crisis"]),
            "z_thr_post": float(best["z_thr_post"]),
            "if_thr": float(best["if_thr"]),
            "n_alerts": int(best["n_alerts"]),
            "recall_event": float(best["recall_event"]),
            "precision_72h": float(best["precision_72h"]),
            "precision_7d": float(best["precision_7d"]),
            "precision_30d": float(best["precision_30d"]),
            "fp_rate_per_month_outside_72h": float(best["fp_rate_per_month_outside_72h"]),
            "median_lag_hours": float(best["median_lag_hours"]) if not np.isnan(best["median_lag_hours"]) else None,
        }
        print(f"    {cap_name:9s} (target {target:>3}/mes): z_p{best['z_percentile']}, "
              f"if_p{best['if_percentile']}, cusum>={best['cusum_min_h']}h | "
              f"actual {best['alerts_per_month']:.1f}/mes, "
              f"recall {best['recall_event']:.3f}, "
              f"p@72h {best['precision_72h']:.3f}")

    # ── Punto de operación ACTUAL (sistema vigente) ────────────────────────
    df_actual = df_alerts[df_alerts["alert_level"].isin(["NARANJA", "ROJA"])].copy()
    n_actual = len(df_actual)
    span_days = (ALERT_END - ALERT_START).days
    n_months = span_days / 30.4375
    actual = {
        "alerts_per_month": n_actual / n_months,
        "n_alerts": n_actual,
        "cascade": "current (Verde/Ámbar/Naranja/Roja with regime p99 z, global p95 IF, cusum>=3)",
    }

    summary = {
        "alert_window": {"start": str(ALERT_START), "end": str(ALERT_END), "n_months": n_months},
        "grid_size": len(df_grid),
        "z_percentiles": Z_PERCENTILES,
        "if_percentiles": IF_PERCENTILES,
        "cusum_min_h": CUSUM_MIN_H,
        "canonical_operating_points": canonical,
        "current_system_operating_point": actual,
    }
    with open(RESULTS_DIR / "operational_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Summary guardado: {RESULTS_DIR / 'operational_calibration_summary.json'}")

    # ── Plot Pareto frontier ───────────────────────────────────────────────
    # Sistema actual usa percentil IF in-sample (p95_if global) y la cascada
    # Verde/Ámbar/Naranja/Roja completa. Los puntos del grid usan percentiles
    # OOS régime-específicos sobre training fraction. Por construcción no son
    # estrictamente comparables, pero sí útiles para situar al sistema actual
    # en el espacio (alertas/mes × precision × recall).
    with open(RESULTS_DIR / "event_validation_summary.json", encoding="utf-8") as f:
        ev_sum = json.load(f)
    actual_p72 = ev_sum.get("precision_at_72h", np.nan)
    actual_recall = ev_sum.get("recall_global_in_window", np.nan)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    cusum_colors = {1: "#3498DB", 2: "#27AE60", 3: "#E67E22", 5: "#C0392B"}
    for cusum_h in CUSUM_MIN_H:
        sub = df_grid[df_grid["cusum_min_h"] == cusum_h].sort_values("alerts_per_month")
        ax1.plot(sub["alerts_per_month"], sub["precision_72h"],
                 marker="o", markersize=4.5, alpha=0.75,
                 label=f"Grid OOS · CUSUM ≥ {cusum_h}h",
                 color=cusum_colors[cusum_h], linewidth=1.2)
    ax1.set_xlabel("Alerts per month (operational load)", fontsize=11)
    ax1.set_ylabel("Precision @ 72h  (TP within ≤72h of event center)",
                   fontsize=11, color="#1F2937")
    ax1.set_title("Operational Pareto: precision vs coverage tradeoff\n"
                  "OOS grid 5×5×4 (100 pts) + current system (in-sample cascade) · residuals 2022-2024",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xlim(left=0)

    # Eje derecho: recall_event en línea punteada
    ax2 = ax1.twinx()
    for cusum_h in CUSUM_MIN_H:
        sub = df_grid[df_grid["cusum_min_h"] == cusum_h].sort_values("alerts_per_month")
        ax2.plot(sub["alerts_per_month"], sub["recall_event"],
                 marker="s", markersize=3.5, alpha=0.32, linestyle="--",
                 color=cusum_colors[cusum_h], linewidth=0.8)
    ax2.set_ylabel("Recall @ event (dotted lines, right axis)",
                   fontsize=10, color="#6B7280")
    ax2.set_ylim(0, 1.05)

    # Marcar puntos canónicos como estrellas grandes
    cap_colors = {"muy_baja": "#7F1D1D", "baja": "#92400E",
                  "media": "#065F46", "alta": "#1E3A8A"}
    cap_label = {"muy_baja": "Very low", "baja": "Low (elbow)",
                 "media": "Medium", "alta": "High"}
    for cap_name, c in canonical.items():
        ax1.scatter(c["actual_alerts_per_month"], c["precision_72h"],
                    s=200, marker="*", color=cap_colors[cap_name],
                    edgecolor="black", linewidth=1.3, zorder=5,
                    label=f"Canonical {cap_label[cap_name]} "
                          f"({c['actual_alerts_per_month']:.1f}/mo)")

    # ── Punto ACTUAL (sistema en cascada in-sample) ────────────────────────
    ax1.scatter(actual["alerts_per_month"], actual_p72,
                s=320, marker="X", color="#DC2626",
                edgecolor="black", linewidth=1.8, zorder=6,
                label=f"Current system (in-sample IF) "
                      f"({actual['alerts_per_month']:.1f}/mo)")
    # Anotación clara
    ax1.annotate(
        f"Current system\n  {actual['alerts_per_month']:.1f} alerts/month\n"
        f"  precision@72h = {actual_p72:.3f}\n"
        f"  recall@event = {actual_recall:.3f}\n"
        f"  (in-sample cascade, global p95_if)",
        xy=(actual["alerts_per_month"], actual_p72),
        xytext=(40, -45), textcoords="offset points",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2",
                  edgecolor="#DC2626", linewidth=1),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
    )

    # Caja explicativa con caveat metodológico
    caveat_text = (
        "Methodological note:\n"
        "• Grid points (lines): régime-specific OOS percentiles\n"
        "   on training fraction (first 50% of each regime)\n"
        "• Current system (red X): operational cascade with global\n"
        "   in-sample p95_if → more permissive in IF, similar in z régime"
    )
    ax1.text(0.02, 0.98, caveat_text, transform=ax1.transAxes,
             fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3F4F6",
                       edgecolor="#9CA3AF", linewidth=0.6))

    ax1.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.95)
    plt.tight_layout()
    out_p = PLOTS_DIR / "10_operational_calibration_pareto.png"
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pareto plot: {out_p}")

    # ── Identificar fila "CODO" (mayor pareto_score entre canónicos) ───────
    codo_cap = max(
        canonical.keys(),
        key=lambda k: (canonical[k]["recall_event"] + canonical[k]["precision_72h"]) / 2,
    )

    # ── Plot tabla canónicos ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")
    headers = ["Capacity", "Alerts/mo.", "z_thr (crisis/post)", "if_thr",
               "cusum≥", "Recall@event", "P@72h", "P@7d", "FP/mo."]
    cap_label_full = {"muy_baja": "Very low", "baja": "Low",
                      "media": "Medium", "alta": "High"}
    rows_tab = []
    for cap_name, c in canonical.items():
        label = cap_label_full[cap_name]
        if cap_name == codo_cap:
            label = f"{label} (ELBOW)"
        rows_tab.append([
            label,
            f"{c['actual_alerts_per_month']:.1f}",
            f"{c['z_thr_crisis']:.2f} / {c['z_thr_post']:.2f} (p{c['z_percentile']:g})",
            f"{c['if_thr']:.3f} (p{c['if_percentile']:g})",
            f"≥{c['cusum_min_h']}h",
            f"{c['recall_event']:.3f}",
            f"{c['precision_72h']:.3f}",
            f"{c['precision_7d']:.3f}",
            f"{c['fp_rate_per_month_outside_72h']:.2f}",
        ])
    # Fila ACTUAL (sistema in-sample)
    rows_tab.append([
        "Current (in-sample)",
        f"{actual['alerts_per_month']:.1f}",
        "5.74 / 6.12 (p99 OOS)",
        "0.608 (p95 global IS)",
        "≥3h",
        f"{actual_recall:.3f}",
        f"{actual_p72:.3f}",
        f"{ev_sum.get('precision_at_7d', 0):.3f}",
        f"{ev_sum.get('fp_rate_per_month_outside_72h', 0):.2f}",
    ])

    table = ax.table(cellText=rows_tab, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#1F2937")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows_tab) + 1):
        cap_key = list(canonical.keys())[i - 1] if i <= len(canonical) else None
        if i == len(rows_tab):  # fila ACTUAL
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#FEE2E2")
        elif cap_key == codo_cap:  # CODO en verde claro
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#D1FAE5")
                table[(i, j)].set_text_props(fontweight="bold")
        elif i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#F3F4F6")
    ax.set_title("Operational calibration: 4 canonical points from OOS grid + current system\n"
                 "(Grid 5×5×4 = 100 combinations · residuals 2022-2024 · ELBOW = best recall/precision tradeoff)",
                 fontsize=10.5, fontweight="bold", pad=15)
    plt.tight_layout()
    out_t = PLOTS_DIR / "11_operational_calibration_table.png"
    plt.savefig(out_t, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tabla plot: {out_t}")

    print("\n  TANDA 2 COMPLETADA")


if __name__ == "__main__":
    main()
