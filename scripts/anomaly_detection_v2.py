"""
anomaly_detection_v2.py
-----------------------
Detección de anomalías sobre residuos del Modelo A (limpio, sin leakage).

Módulos:
1. Isolation Forest — detección de observaciones atípicas multivariante
2. CUSUM — detección de acumulaciones sostenidas del residuo
3. k-NN forense — identificación de episodios históricos similares
4. Sistema de alertas con definición operativa formal

Definición operativa de alertas:
    ROJA:    |residuo| > p99(régimen) AND IF_score > p95 AND CUSUM_activo >= 3h
    NARANJA: |residuo| > p95(régimen) AND (IF_score > p90 OR CUSUM_activo >= 2h)
    ÁMBAR:   |residuo| > p90(régimen) OR IF_score > p85
    VERDE:   resto

Salidas:
    results/alerts_registry_v2.csv
    results/top_episodes_v2.csv
    plots/06_anomaly_detection_v2.png
    plots/07_alert_concentration_v2.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

RESIDUALS_PATH = DATA_DIR / "processed" / "residuals_v2.parquet"
DATASET_PATH   = DATA_DIR / "processed" / "mibel_dataset_20190101_20241231.parquet"


def load_data():
    """Carga residuos y dataset completo, hace merge."""
    df_res = pd.read_parquet(RESIDUALS_PATH)
    df_res["timestamp"] = pd.to_datetime(df_res["timestamp"])

    df_full = pd.read_parquet(DATASET_PATH)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])

    # Merge para tener features adicionales en el contexto de anomalías
    df = df_res.merge(
        df_full[["timestamp", "atc_implicit", "atc_congestion", "atc_fatigue",
                 "ttf_eur_mwh", "spark_spread", "hour", "dow", "month",
                 "spread_id", "price_id_es", "price_id_fr"]],
        on="timestamp", how="left"
    )
    return df


def compute_regime_percentiles(df):
    """Calcula percentiles del residuo por régimen."""
    percentiles = {}
    for regime in df["regime"].unique():
        mask = df["regime"] == regime
        abs_res = df[mask]["residual"].abs()
        percentiles[regime] = {
            "p90": abs_res.quantile(0.90),
            "p95": abs_res.quantile(0.95),
            "p99": abs_res.quantile(0.99),
        }
        print(f"  {regime}: p90={percentiles[regime]['p90']:.2f} "
              f"p95={percentiles[regime]['p95']:.2f} "
              f"p99={percentiles[regime]['p99']:.2f} €/MWh")
    return percentiles


def run_isolation_forest(df):
    """Isolation Forest sobre residuos normalizados por régimen."""
    print("\n[1/3] Isolation Forest...")
    df["if_score"] = np.nan
    df["if_anomaly"] = False

    features_if = ["residual", "atc_fatigue", "spark_spread", "hour", "dow", "month"]
    available = [f for f in features_if if f in df.columns]

    for regime in df["regime"].unique():
        mask = df["regime"] == regime
        X = df[mask][available].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # contamination=auto: deja que el algoritmo decida, no forzamos un %
        iso = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )
        iso.fit(X_scaled)

        scores = iso.score_samples(X_scaled)  # más negativo = más anómalo
        # Normalizar a [0,1] donde 1 = más anómalo
        scores_norm = 1 - (scores - scores.min()) / (scores.max() - scores.min())

        df.loc[mask, "if_score"] = scores_norm
        df.loc[mask, "if_anomaly"] = iso.predict(X_scaled) == -1

    n_anomalies = df["if_anomaly"].sum()
    pct = n_anomalies / len(df) * 100
    print(f"  Anomalías IF: {n_anomalies:,} ({pct:.1f}%) — determinado por el algoritmo")
    return df


def run_cusum(df, k_factor=0.5):
    """
    CUSUM bilateral sobre residuos normalizados por régimen.
    k_factor: sensibilidad (0.5 = estándar)
    """
    print("\n[2/3] CUSUM...")
    df["cusum_pos"] = 0.0
    df["cusum_neg"] = 0.0
    df["cusum_active"] = 0  # horas consecutivas con CUSUM activo

    for regime in df["regime"].unique():
        mask = df["regime"] == regime
        idx = df[mask].index
        res = df.loc[idx, "residual"].values

        # Normalizar por desviación típica del régimen
        sigma = np.std(res)
        if sigma < 1e-6:
            continue
        res_norm = res / sigma

        k = k_factor  # umbral de referencia
        h = 5.0       # umbral de alarma (5 sigmas acumuladas)

        cusum_pos = np.zeros(len(res_norm))
        cusum_neg = np.zeros(len(res_norm))
        active = np.zeros(len(res_norm), dtype=int)

        for i in range(1, len(res_norm)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + res_norm[i] - k)
            cusum_neg[i] = max(0, cusum_neg[i-1] - res_norm[i] - k)
            if cusum_pos[i] > h or cusum_neg[i] > h:
                active[i] = active[i-1] + 1
            else:
                active[i] = 0

        df.loc[idx, "cusum_pos"] = cusum_pos
        df.loc[idx, "cusum_neg"] = cusum_neg
        df.loc[idx, "cusum_active"] = active

    n_cusum = (df["cusum_active"] >= 3).sum()
    print(f"  Horas con CUSUM activo >= 3h: {n_cusum:,}")
    return df


def run_knn_forensic(df, n_neighbors=5):
    """
    k-NN forense: para cada anomalía, encuentra los episodios históricos más similares.
    Solo se aplica sobre las horas con residuo > p95.
    """
    print("\n[3/3] k-NN forense...")
    df["knn_mean_dist"] = np.nan
    df["knn_nearest_date"] = pd.NaT

    features_knn = ["residual", "atc_fatigue", "spark_spread", "hour", "month"]
    available = [f for f in features_knn if f in df.columns]

    # Solo sobre el conjunto de validación (excepcion_iberica + post_excepcion)
    mask_val = df["regime"].isin(["excepcion_iberica", "post_excepcion"])
    mask_train = df["regime"] == "pre_crisis"

    if mask_train.sum() < n_neighbors:
        print("  ⚠ No hay suficientes datos de entrenamiento para k-NN")
        return df

    X_train = df[mask_train][available].fillna(0).values
    X_val   = df[mask_val][available].fillna(0).values

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_s)

    distances, indices = knn.kneighbors(X_val_s)
    mean_dists = distances.mean(axis=1)
    nearest_dates = df[mask_train].iloc[indices[:, 0]]["timestamp"].values

    df.loc[mask_val, "knn_mean_dist"] = mean_dists
    df.loc[mask_val, "knn_nearest_date"] = nearest_dates

    print(f"  k-NN aplicado a {mask_val.sum():,} horas de validación")
    return df


def assign_alert_level(df, percentiles):
    """
    Asigna nivel de alerta con definición operativa formal.

    ROJA:    |residuo| > p99(régimen) AND IF_score > p95_global AND CUSUM_activo >= 3h
    NARANJA: |residuo| > p95(régimen) AND (IF_score > p90_global OR CUSUM_activo >= 2h)
    ÁMBAR:   |residuo| > p90(régimen) OR IF_score > p85_global
    VERDE:   resto
    """
    print("\n[Alertas] Asignando niveles...")

    # Percentiles globales del IF score
    p85_if = df["if_score"].quantile(0.85)
    p90_if = df["if_score"].quantile(0.90)
    p95_if = df["if_score"].quantile(0.95)

    df["alert_level"] = "VERDE"
    df["abs_residual"] = df["residual"].abs()

    # Añadir columna de percentil del residuo por régimen
    df["res_p90"] = df["regime"].map({r: v["p90"] for r, v in percentiles.items()})
    df["res_p95"] = df["regime"].map({r: v["p95"] for r, v in percentiles.items()})
    df["res_p99"] = df["regime"].map({r: v["p99"] for r, v in percentiles.items()})

    # ÁMBAR
    amber = (df["abs_residual"] > df["res_p90"]) | (df["if_score"] > p85_if)
    df.loc[amber, "alert_level"] = "ÁMBAR"

    # NARANJA
    orange = (df["abs_residual"] > df["res_p95"]) & \
             ((df["if_score"] > p90_if) | (df["cusum_active"] >= 2))
    df.loc[orange, "alert_level"] = "NARANJA"

    # ROJA
    red = (df["abs_residual"] > df["res_p99"]) & \
          (df["if_score"] > p95_if) & \
          (df["cusum_active"] >= 3)
    df.loc[red, "alert_level"] = "ROJA"

    # Resumen
    counts = df["alert_level"].value_counts()
    print(f"  VERDE:   {counts.get('VERDE', 0):,}")
    print(f"  ÁMBAR:   {counts.get('ÁMBAR', 0):,}")
    print(f"  NARANJA: {counts.get('NARANJA', 0):,}")
    print(f"  ROJA:    {counts.get('ROJA', 0):,}")

    return df


def extract_top_episodes(df, n=20):
    """Extrae los top N episodios por residuo absoluto."""
    red_orange = df[df["alert_level"].isin(["ROJA", "NARANJA"])].copy()
    red_orange = red_orange.sort_values("abs_residual", ascending=False)

    # Agrupar horas consecutivas en episodios
    episodes = []
    used = set()

    for _, row in red_orange.iterrows():
        ts = row["timestamp"]
        if ts in used:
            continue
        # Buscar horas consecutivas
        episode_mask = (
            (df["timestamp"] >= ts - pd.Timedelta(hours=2)) &
            (df["timestamp"] <= ts + pd.Timedelta(hours=6)) &
            (df["alert_level"].isin(["ROJA", "NARANJA"]))
        )
        episode = df[episode_mask]
        for t in episode["timestamp"]:
            used.add(t)

        episodes.append({
            "start": episode["timestamp"].min(),
            "end": episode["timestamp"].max(),
            "duration_h": len(episode),
            "max_residual": episode["residual"].abs().max(),
            "mean_residual": episode["residual"].abs().mean(),
            "alert_level": row["alert_level"],
            "regime": row["regime"],
            "spread_observed": episode["spread_observed"].iloc[0],
            "spread_predicted": episode["spread_predicted"].iloc[0],
        })

        if len(episodes) >= n:
            break

    return pd.DataFrame(episodes)


def plot_anomalies(df, top_episodes):
    """Genera gráficos de detección de anomalías."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("MIBEL Congestion Monitor — Detección de Anomalías (modelo limpio)",
                 fontsize=13, fontweight="bold")

    colors_alert = {"VERDE": "#2ECC71", "ÁMBAR": "#F39C12",
                    "NARANJA": "#E67E22", "ROJA": "#E74C3C"}

    # ── Panel 1: Residuos coloreados por nivel de alerta ──────────────────
    ax1 = axes[0]
    for level in ["VERDE", "ÁMBAR", "NARANJA", "ROJA"]:
        mask = df["alert_level"] == level
        ax1.scatter(df[mask]["timestamp"], df[mask]["residual"],
                    alpha=0.4 if level == "VERDE" else 0.7,
                    s=2 if level == "VERDE" else 8,
                    color=colors_alert[level], label=f"{level} ({mask.sum():,})")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Residuos por nivel de alerta (Modelo A — sin leakage)")
    ax1.set_ylabel("Residuo (€/MWh)")
    ax1.legend(fontsize=9, markerscale=3)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: IF Score a lo largo del tiempo ───────────────────────────
    ax2 = axes[1]
    ax2.plot(df["timestamp"], df["if_score"], color="#3498DB", alpha=0.4, linewidth=0.5)
    ax2.axhline(df["if_score"].quantile(0.95), color="red", linewidth=1,
                linestyle="--", label="p95 IF score")
    # Marcar alertas rojas
    red_mask = df["alert_level"] == "ROJA"
    ax2.scatter(df[red_mask]["timestamp"], df[red_mask]["if_score"],
                color="#E74C3C", s=15, zorder=5, label="Alerta Roja")
    ax2.set_title("Isolation Forest Score (1 = más anómalo)")
    ax2.set_ylabel("IF Score")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: CUSUM activo ─────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(df["timestamp"], df["cusum_active"],
                     alpha=0.5, color="#9B59B6", label="Horas CUSUM activo")
    ax3.axhline(3, color="red", linewidth=1, linestyle="--", label="Umbral Alerta Roja (3h)")
    ax3.set_title("CUSUM — Persistencia de desviaciones")
    ax3.set_ylabel("Horas consecutivas activo")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "06_anomaly_detection_v2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Gráfico anomalías: {out}")

    # ── Gráfico de concentración de alertas ───────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Concentración de alertas — Validación externa", fontsize=12)

    # Alertas por mes
    df_alerts = df[df["alert_level"].isin(["ROJA", "NARANJA"])].copy()
    df_alerts["month_year"] = df_alerts["timestamp"].dt.to_period("M")
    monthly = df_alerts.groupby(["month_year", "alert_level"]).size().unstack(fill_value=0)

    ax_m = axes2[0]
    if not monthly.empty:
        monthly.plot(kind="bar", ax=ax_m, color=["#E67E22", "#E74C3C"],
                     alpha=0.8, width=0.8)
        ax_m.set_title("Alertas NARANJA/ROJA por mes")
        ax_m.set_xlabel("")
        ax_m.set_ylabel("Número de horas")
        ax_m.tick_params(axis="x", rotation=90, labelsize=6)
        ax_m.legend(fontsize=9)
        ax_m.grid(True, alpha=0.3, axis="y")

    # Top 10 episodios
    ax_e = axes2[1]
    if not top_episodes.empty:
        top10 = top_episodes.head(10)
        colors_ep = [colors_alert.get(l, "gray") for l in top10["alert_level"]]
        bars = ax_e.barh(range(len(top10)),
                         top10["max_residual"].values,
                         color=colors_ep, alpha=0.8)
        ax_e.set_yticks(range(len(top10)))
        ax_e.set_yticklabels([str(d)[:10] for d in top10["start"]], fontsize=8)
        ax_e.set_title("Top 10 episodios por residuo máximo")
        ax_e.set_xlabel("|Residuo| máximo (€/MWh)")
        for bar, val in zip(bars, top10["max_residual"].values):
            ax_e.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                      f"{val:.1f}", va="center", fontsize=8)
        ax_e.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out2 = PLOTS_DIR / "07_alert_concentration_v2.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico concentración: {out2}")


def main():
    print("=" * 60)
    print("  MIBEL Congestion Monitor — Detección de Anomalías")
    print("  (sobre residuos del Modelo A — sin leakage)")
    print("=" * 60)

    df = load_data()
    print(f"\nResíduos cargados: {len(df):,} filas")
    print(f"Rango: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # Percentiles por régimen
    print("\n[Percentiles del residuo absoluto por régimen]")
    percentiles = compute_regime_percentiles(df)

    # Módulos de detección
    df = run_isolation_forest(df)
    df = run_cusum(df)
    df = run_knn_forensic(df)

    # Sistema de alertas
    df = assign_alert_level(df, percentiles)

    # Top episodios
    top_episodes = extract_top_episodes(df, n=20)
    print(f"\n[Top episodios detectados]")
    if not top_episodes.empty:
        print(top_episodes[["start", "end", "duration_h", "max_residual",
                             "alert_level", "regime"]].to_string(index=False))

    # Guardar resultados
    alert_cols = ["timestamp", "regime", "spread_observed", "spread_predicted",
                  "residual", "abs_residual", "if_score", "if_anomaly",
                  "cusum_active", "knn_mean_dist", "alert_level"]
    available_cols = [c for c in alert_cols if c in df.columns]
    df[available_cols].to_csv(RESULTS_DIR / "alerts_registry_v2.csv", index=False)
    top_episodes.to_csv(RESULTS_DIR / "top_episodes_v2.csv", index=False)
    print(f"\n  Registro guardado: results/alerts_registry_v2.csv")
    print(f"  Top episodios: results/top_episodes_v2.csv")

    # Gráficos
    plot_anomalies(df, top_episodes)

    print(f"\n{'='*60}")
    print("  DETECCIÓN COMPLETADA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
