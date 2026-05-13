"""
anomaly_detection_v3.py
-----------------------
Detección de anomalías sobre residuos del Modelo A v3 (sin leakage ATC).

Implementa la metodología descrita textualmente en §4.1 del paper, que el
script v2 NO implementaba:

  1. z_t = (res_t - mu_{t-168:t-1}) / sigma_{t-168:t-1}
     - Ventana rolling 168h con shift(1) obligatorio (no look-ahead).
     - El estadístico se calcula sobre el residuo *firmado* (no |residual|),
       porque el signo importa para distinguir spread ES>>FR de FR>>ES.

  2. Percentiles p90/p95/p99 calculados SOBRE EL TRAINING SET DEL RÉGIMEN
     únicamente. El "training set" de cada régimen se define como el primer
     50% cronológico de las observaciones con z-score válido en ese régimen.
     Esto evita look-ahead: los umbrales se fijan a partir de la mitad
     temprana del régimen y se aplican al periodo posterior.

  3. Reglas operativas (cascada exacta del PDF):
       Verde:    z < p90 AND IF_score < p95
       Ámbar:    p90 <= z < p95  OR  IF_score >= p95
       Naranja:  z >= p95 AND (IF_score >= p95 OR cusum_active >= 1)
       Roja:     z >= p99 AND IF_score >= p95 AND cusum_active >= 3

Salidas:
    results/alerts_registry_v3.csv
    results/top_episodes_v3.csv
    results/regime_thresholds_v3.json
    plots/06_anomaly_detection_v3.png
    plots/07_alert_concentration_v3.png
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

RESIDUALS_PATH = PROC_DIR / "residuals_v3.parquet"
DATASET_PATH = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

ROLLING_WINDOW = 168  # horas (1 semana)
TRAIN_FRACTION = 0.5  # 50% temprano de cada régimen para fijar umbrales


# ════════════════════════════════════════════════════════════════════════════
#  z-score rolling con shift(1)
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_zscore(res: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """z_t = (res_t - mu_{t-window:t-1}) / sigma_{t-window:t-1}.
    Implementado como rolling(...).shift(1) para no usar res_t en el estadístico.
    """
    rolling_mu = res.rolling(window, min_periods=window // 2).mean().shift(1)
    rolling_sigma = res.rolling(window, min_periods=window // 2).std().shift(1)

    # Floor numérico: evitar sigma rolling colapsando a valores ínfimos
    # que disparan z a magnitudes artifactuales (|z|=7591 en EVT017).
    sigma_floor = res.dropna().std() * 0.1
    rolling_sigma = rolling_sigma.clip(lower=sigma_floor)

    # Z-score con clip simétrico: cualquier |z|>10 ya es trivialmente ROJA.
    z_raw = (res - rolling_mu) / rolling_sigma
    z = z_raw.clip(-10, 10)

    n_clipped_sigma = int((rolling_sigma == sigma_floor).sum())
    n_clipped_z = int(((z_raw > 10) | (z_raw < -10)).sum())
    print(f"[A5] sigma_floor={sigma_floor:.4f}, "
          f"clipped_sigma={n_clipped_sigma}, clipped_z={n_clipped_z}")
    return z


# ════════════════════════════════════════════════════════════════════════════
#  Umbrales p90/p95/p99 sobre training set de cada régimen
# ════════════════════════════════════════════════════════════════════════════

def compute_regime_thresholds(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> dict:
    """Para cada régimen del df de residuos, toma el primer train_fraction
    cronológico como training set y calcula p90/p95/p99 de |z|.
    Devuelve dict regime → thresholds + tamaño del training.
    """
    out = {}
    for regime, g in df.groupby("regime"):
        g_sorted = g.sort_values("timestamp")
        g_valid = g_sorted.dropna(subset=["z_score"])
        if len(g_valid) < 200:
            print(f"  ⚠ Régimen {regime}: pocas observaciones ({len(g_valid)}), saltando.")
            continue
        cutoff = int(len(g_valid) * train_fraction)
        train = g_valid.iloc[:cutoff]
        abs_z = train["z_score"].abs()
        out[regime] = {
            "p90": float(abs_z.quantile(0.90)),
            "p95": float(abs_z.quantile(0.95)),
            "p99": float(abs_z.quantile(0.99)),
            "n_train": int(len(train)),
            "n_total": int(len(g_valid)),
            "train_period": [str(train["timestamp"].min()),
                             str(train["timestamp"].max())],
        }
        print(f"  {regime}: p90={out[regime]['p90']:.3f}  "
              f"p95={out[regime]['p95']:.3f}  p99={out[regime]['p99']:.3f}  "
              f"(n_train={out[regime]['n_train']:,})")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Isolation Forest (entrenado en training set de cada régimen)
# ════════════════════════════════════════════════════════════════════════════

def run_isolation_forest_per_regime(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> pd.DataFrame:
    df = df.copy()
    df["if_score"] = np.nan
    feats = ["residual", "ntc_es_fr", "ntc_is_observed", "spark_spread",
             "hour", "dow", "month"]
    feats = [f for f in feats if f in df.columns]
    print("\n  Isolation Forest (entrenado en training set, evaluado en todo)")
    print(f"  Features: {feats}")

    for regime, g in df.groupby("regime"):
        idx = g.sort_values("timestamp").index
        cutoff = int(len(idx) * train_fraction)
        train_idx = idx[:cutoff]
        all_idx = idx

        X_train = df.loc[train_idx, feats].fillna(0).values
        X_all = df.loc[all_idx, feats].fillna(0).values

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_all_s = scaler.transform(X_all)

        iso = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=42, n_jobs=-1)
        iso.fit(X_train_s)
        scores_all = iso.score_samples(X_all_s)
        # Normalizar a [0,1] dentro del régimen, 1 = más anómalo
        scores_norm = 1 - (scores_all - scores_all.min()) / (scores_all.max() - scores_all.min() + 1e-12)
        df.loc[all_idx, "if_score"] = scores_norm

    return df


# ════════════════════════════════════════════════════════════════════════════
#  CUSUM bilateral con sigma del training set
# ════════════════════════════════════════════════════════════════════════════

def run_cusum_per_regime(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION,
                         k_factor: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    df = df.copy().sort_values(["regime", "timestamp"]).reset_index(drop=True)
    df["cusum_pos"] = 0.0
    df["cusum_neg"] = 0.0
    df["cusum_active"] = 0
    print("\n  CUSUM bilateral (sigma estimada en training set)")
    for regime, g in df.groupby("regime"):
        idx = g.index.tolist()
        cutoff = int(len(idx) * train_fraction)
        train_idx = idx[:cutoff]
        sigma_train = float(df.loc[train_idx, "residual"].std())
        if sigma_train < 1e-6:
            continue
        res_norm = (df.loc[idx, "residual"].values / sigma_train)
        cp = np.zeros(len(res_norm)); cn = np.zeros(len(res_norm))
        active = np.zeros(len(res_norm), dtype=int)
        for i in range(1, len(res_norm)):
            cp[i] = max(0, cp[i-1] + res_norm[i] - k_factor)
            cn[i] = max(0, cn[i-1] - res_norm[i] - k_factor)
            if cp[i] > h or cn[i] > h:
                active[i] = active[i-1] + 1
            else:
                active[i] = 0
        df.loc[idx, "cusum_pos"] = cp
        df.loc[idx, "cusum_neg"] = cn
        df.loc[idx, "cusum_active"] = active
        print(f"    {regime}: sigma_train={sigma_train:.3f}, "
              f"horas con CUSUM≥3: {int((active >= 3).sum()):,}")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  Cascada Verde / Ámbar / Naranja / Roja según el PDF
# ════════════════════════════════════════════════════════════════════════════

def assign_alerts(df: pd.DataFrame, thresholds: dict, p95_if: float) -> pd.DataFrame:
    df = df.copy()
    df["alert_level"] = "VERDE"
    abs_z = df["z_score"].abs()
    df["abs_z"] = abs_z

    df["z_p90"] = df["regime"].map({r: t["p90"] for r, t in thresholds.items()})
    df["z_p95"] = df["regime"].map({r: t["p95"] for r, t in thresholds.items()})
    df["z_p99"] = df["regime"].map({r: t["p99"] for r, t in thresholds.items()})

    # ÁMBAR: p90 ≤ |z| < p95  OR  IF >= p95
    amber = ((abs_z >= df["z_p90"]) & (abs_z < df["z_p95"])) | (df["if_score"] >= p95_if)
    df.loc[amber, "alert_level"] = "ÁMBAR"

    # NARANJA: |z| >= p95  AND  (IF activo OR CUSUM >= 1)
    orange = (abs_z >= df["z_p95"]) & ((df["if_score"] >= p95_if) | (df["cusum_active"] >= 1))
    df.loc[orange, "alert_level"] = "NARANJA"

    # ROJA: |z| >= p99  AND  IF >= p95  AND  CUSUM >= 3
    red = (abs_z >= df["z_p99"]) & (df["if_score"] >= p95_if) & (df["cusum_active"] >= 3)
    df.loc[red, "alert_level"] = "ROJA"

    return df


# ════════════════════════════════════════════════════════════════════════════
#  Episodios y reporte
# ════════════════════════════════════════════════════════════════════════════

def extract_episodes(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    high = df[df["alert_level"].isin(["NARANJA", "ROJA"])].copy()
    if high.empty:
        return pd.DataFrame()
    high = high.sort_values("abs_z", ascending=False)
    used = set()
    episodes = []
    for _, row in high.iterrows():
        ts = row["timestamp"]
        if ts in used:
            continue
        win = (df["timestamp"] >= ts - pd.Timedelta(hours=2)) & \
              (df["timestamp"] <= ts + pd.Timedelta(hours=12)) & \
              (df["alert_level"].isin(["NARANJA", "ROJA"]))
        ep = df.loc[win]
        used.update(ep["timestamp"].tolist())
        episodes.append({
            "start": ep["timestamp"].min(),
            "end": ep["timestamp"].max(),
            "duration_h": int(len(ep)),
            "max_abs_z": float(ep["abs_z"].max()),
            "max_residual": float(ep["residual"].abs().max()),
            "alert_max": "ROJA" if (ep["alert_level"] == "ROJA").any() else "NARANJA",
            "regime": row["regime"],
            "max_spread_observed": float(ep["spread_observed"].abs().max()),
        })
        if len(episodes) >= top_n:
            break
    return pd.DataFrame(episodes).sort_values("max_abs_z", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_anomalies(df: pd.DataFrame, episodes: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("MIBEL Congestion Monitor v3 — Anomaly detection\n"
                 "(168h rolling z-score, shift(1); thresholds on training set, régime-specific)",
                 fontsize=12, fontweight="bold")
    palette = {"VERDE": "#27AE60", "ÁMBAR": "#F1C40F",
               "NARANJA": "#E67E22", "ROJA": "#C0392B"}
    # Plot-side English mapping (data values stay in Spanish for downstream scripts)
    label_en = {"VERDE": "Green", "ÁMBAR": "Amber",
                "NARANJA": "Orange", "ROJA": "Red"}

    ax = axes[0]
    for lvl in ["VERDE", "ÁMBAR", "NARANJA", "ROJA"]:
        m = df["alert_level"] == lvl
        ax.scatter(df.loc[m, "timestamp"], df.loc[m, "residual"],
                   alpha=0.4 if lvl == "VERDE" else 0.8,
                   s=2 if lvl == "VERDE" else 9,
                   color=palette[lvl],
                   label=f"{label_en[lvl]} ({int(m.sum()):,})")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Residuals (Model A v3) colored by alert level")
    ax.set_ylabel("Residual (EUR/MWh)")
    ax.legend(fontsize=8, markerscale=2.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["timestamp"], df["abs_z"], color="#2C3E50", lw=0.4, alpha=0.6)
    red_mask = df["alert_level"] == "ROJA"
    ax.scatter(df.loc[red_mask, "timestamp"], df.loc[red_mask, "abs_z"],
               color="#C0392B", s=15, zorder=5, label="Red alert")
    ax.set_title("|z-score| rolling (168h window, shift(1))")
    ax.set_ylabel("|z|")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.fill_between(df["timestamp"], df["cusum_active"],
                    color="#9B59B6", alpha=0.45, label="CUSUM active (consec. hours)")
    ax.axhline(3, color="#C0392B", lw=1, ls="--", label="Red threshold (≥3h)")
    ax.set_title("Two-sided CUSUM — deviation persistence")
    ax.set_ylabel("Consecutive hours")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "06_anomaly_detection_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot guardado: {out}")

    # Concentración de alertas
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Alert concentration — Model A v3", fontsize=12)
    high = df[df["alert_level"].isin(["NARANJA", "ROJA"])].copy()
    high["month_year"] = high["timestamp"].dt.to_period("M")
    monthly = high.groupby(["month_year", "alert_level"]).size().unstack(fill_value=0)
    if not monthly.empty:
        # Rename columns to English for the legend
        monthly = monthly.rename(columns=label_en)
        color_en = {label_en[k]: v for k, v in
                    {"NARANJA": "#E67E22", "ROJA": "#C0392B"}.items()}
        monthly.plot(kind="bar", ax=axes2[0],
                     color=color_en,
                     alpha=0.85, width=0.85)
        axes2[0].set_title("Orange / Red alerts per month")
        axes2[0].set_xlabel("")
        axes2[0].set_ylabel("Hours")
        axes2[0].tick_params(axis="x", rotation=90, labelsize=6)
        axes2[0].grid(True, alpha=0.3, axis="y")

    if not episodes.empty:
        top = episodes.head(10)
        cols = [palette[l] for l in top["alert_max"]]
        axes2[1].barh(range(len(top)), top["max_residual"].values, color=cols, alpha=0.9)
        axes2[1].set_yticks(range(len(top)))
        axes2[1].set_yticklabels([str(d)[:10] for d in top["start"]], fontsize=8)
        axes2[1].set_title("Top 10 episodes (max residual)")
        axes2[1].set_xlabel("Max |Residual| (EUR/MWh)")
        axes2[1].grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(top["max_residual"].values):
            axes2[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    out2 = PLOTS_DIR / "07_alert_concentration_v3.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot concentración: {out2}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  MIBEL Congestion Monitor — Anomaly Detection v3")
    print("  (z-score rolling 168h shift(1) + umbrales train-only)")
    print("=" * 64)

    df_res = pd.read_parquet(RESIDUALS_PATH)
    df_res["timestamp"] = pd.to_datetime(df_res["timestamp"])
    df_full = pd.read_parquet(DATASET_PATH)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])

    df = df_res.merge(
        df_full[["timestamp", "ntc_es_fr", "ntc_is_observed", "spark_spread",
                 "hour", "dow", "month", "ttf_eur_mwh"]],
        on="timestamp", how="left"
    )
    df = df.sort_values(["regime", "timestamp"]).reset_index(drop=True)
    print(f"\n  Residuos cargados: {len(df):,} filas")
    print(f"  Rango: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print("  Distribución por régimen:")
    for r, n in df["regime"].value_counts().items():
        print(f"    {r:25s}: {n:>7,}")

    # ── 1) z-score rolling shift(1) por régimen ────────────────────────────
    print(f"\n  Calculando z-score rolling (ventana={ROLLING_WINDOW}h, shift(1))...")
    df["z_score"] = np.nan
    for regime, g in df.groupby("regime"):
        idx = g.sort_values("timestamp").index
        df.loc[idx, "z_score"] = compute_rolling_zscore(
            df.loc[idx, "residual"]
        ).values

    # ── 2) Umbrales p90/p95/p99 sobre training set por régimen ─────────────
    print(f"\n  Umbrales (sobre primer {int(TRAIN_FRACTION*100)}% temporal de cada régimen):")
    thresholds = compute_regime_thresholds(df, TRAIN_FRACTION)

    # ── 3) Isolation Forest ────────────────────────────────────────────────
    df = run_isolation_forest_per_regime(df, TRAIN_FRACTION)
    p95_if = float(df["if_score"].quantile(0.95))
    print(f"  IF p95 global: {p95_if:.3f}")

    # ── 4) CUSUM ───────────────────────────────────────────────────────────
    df = run_cusum_per_regime(df, TRAIN_FRACTION)

    # ── 5) Cascada de alertas ──────────────────────────────────────────────
    df = assign_alerts(df, thresholds, p95_if)
    print("\n  Distribución de alertas (Tabla 4):")
    counts = df["alert_level"].value_counts().reindex(["VERDE", "ÁMBAR", "NARANJA", "ROJA"], fill_value=0)
    total = int(counts.sum())
    for lvl in ["VERDE", "ÁMBAR", "NARANJA", "ROJA"]:
        n = int(counts[lvl])
        print(f"    {lvl:8s}: {n:>7,}  ({n/total*100:5.2f}% del total clasificado)")
    print(f"    TOTAL clasificado: {total:,}")

    # ── 6) Episodios y artefactos ──────────────────────────────────────────
    episodes = extract_episodes(df, top_n=25)

    out_cols = ["timestamp", "regime", "spread_observed", "spread_predicted",
                "residual", "z_score", "abs_z", "if_score", "cusum_active",
                "z_p90", "z_p95", "z_p99", "alert_level", "wf_step"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(RESULTS_DIR / "alerts_registry_v3.csv", index=False)
    episodes.to_csv(RESULTS_DIR / "top_episodes_v3.csv", index=False)
    with open(RESULTS_DIR / "regime_thresholds_v3.json", "w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds,
                   "p95_if": p95_if,
                   "rolling_window": ROLLING_WINDOW,
                   "train_fraction": TRAIN_FRACTION,
                   "alert_counts": {k: int(v) for k, v in counts.items()}},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Registro: {RESULTS_DIR / 'alerts_registry_v3.csv'}")
    print(f"  Episodios: {RESULTS_DIR / 'top_episodes_v3.csv'}")
    print(f"  Umbrales:  {RESULTS_DIR / 'regime_thresholds_v3.json'}")

    plot_anomalies(df, episodes)
    print("\n  T2 COMPLETADA")


if __name__ == "__main__":
    main()
