"""
validate_dataset.py
-------------------
Validación y análisis exploratorio del dataset MIBEL.
Genera gráficos de calidad, distribuciones y estadísticas por régimen.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET = DATA_DIR / "processed" / "mibel_dataset_20190101_20241231.parquet"

# Colores por régimen
REGIME_COLORS = {
    "pre_crisis": "#2196F3",
    "excepcion_iberica": "#FF5722",
    "post_excepcion": "#4CAF50",
}

def load_data():
    df = pd.read_parquet(DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_spread_series(df):
    """Serie temporal del spread ES-FR con regímenes coloreados."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    fig.suptitle("Spread Eléctrico España-Francia (Day-Ahead) 2019-2024\nMIBEL Congestion Monitor",
                 fontsize=14, fontweight="bold", y=0.98)

    # Resample a media diaria para visualización
    df_daily = df.set_index("timestamp")[["spread_da", "regime"]].copy()
    df_daily["spread_da_daily"] = df_daily["spread_da"].resample("D").mean()
    df_daily = df_daily.resample("D").agg({"spread_da": "mean", "regime": "first"}).dropna()

    # Panel 1: Serie completa
    ax1 = axes[0]
    for regime, color in REGIME_COLORS.items():
        mask = df_daily["regime"] == regime
        ax1.fill_between(df_daily.index[mask], df_daily["spread_da"][mask],
                         alpha=0.3, color=color, label=regime.replace("_", " ").title())
        ax1.plot(df_daily.index[mask], df_daily["spread_da"][mask],
                 color=color, linewidth=0.5, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Spread DA (€/MWh)", fontsize=10)
    ax1.set_title("Serie completa 2019-2024 (media diaria)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distribución por régimen (boxplot)
    ax2 = axes[1]
    regimes = ["pre_crisis", "excepcion_iberica", "post_excepcion"]
    labels = ["Pre-crisis\n(2019-2021)", "Excepción Ibérica\n(2022-2023)", "Post-excepción\n(2024)"]
    data_by_regime = [df[df["regime"] == r]["spread_da"].dropna() for r in regimes]
    bp = ax2.boxplot(data_by_regime, labels=labels, patch_artist=True,
                     showfliers=False, widths=0.5)
    for patch, regime in zip(bp["boxes"], regimes):
        patch.set_facecolor(REGIME_COLORS[regime])
        patch.set_alpha(0.7)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Spread DA (€/MWh)", fontsize=10)
    ax2.set_title("Distribución del spread por régimen de mercado", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Perfil horario del spread
    ax3 = axes[2]
    hourly = df.groupby(["hour", "regime"])["spread_da"].mean().reset_index()
    for regime, color in REGIME_COLORS.items():
        mask = hourly["regime"] == regime
        ax3.plot(hourly[mask]["hour"], hourly[mask]["spread_da"],
                 color=color, linewidth=2, marker="o", markersize=3,
                 label=regime.replace("_", " ").title())
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.set_xlabel("Hora del día (UTC)", fontsize=10)
    ax3.set_ylabel("Spread DA medio (€/MWh)", fontsize=10)
    ax3.set_title("Perfil horario del spread por régimen", fontsize=11)
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "01_spread_series.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def plot_ntc_series(df):
    """Serie temporal del NTC ES-FR (JAO real + ATC implícito)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle("Capacidad de Interconexión España-Francia (NTC/ATC)\nMIBEL Congestion Monitor",
                 fontsize=14, fontweight="bold")

    # Panel 1: NTC real (sep 2022 - dic 2024)
    ax1 = axes[0]
    df_jao = df[df["atc_source"] == "jao_real"].copy()
    df_jao_daily = df_jao.set_index("timestamp")["ntc_es_fr"].resample("D").mean()
    ax1.plot(df_jao_daily.index, df_jao_daily.values, color="#1565C0", linewidth=0.8, alpha=0.8)
    ax1.axhline(3700, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="Capacidad nominal (3700 MW)")
    ax1.axhline(3700 * 0.90, color="orange", linewidth=1.5, linestyle=":", alpha=0.7, label="Umbral congestión (90%)")
    ax1.fill_between(df_jao_daily.index, df_jao_daily.values, 3700 * 0.90,
                     where=df_jao_daily.values < 3700 * 0.90,
                     alpha=0.3, color="red", label="Congestión detectada")
    ax1.set_ylabel("NTC ES→FR (MW)", fontsize=10)
    ax1.set_title("NTC real España→Francia (JAO SWE CCR, sep 2022 - dic 2024)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ATC implícito (2019 - ago 2022)
    ax2 = axes[1]
    df_impl = df[df["atc_source"] == "implicit"].copy()
    df_impl_daily = df_impl.set_index("timestamp")["atc_congestion"].resample("D").mean() * 100
    ax2.fill_between(df_impl_daily.index, df_impl_daily.values, alpha=0.6, color="#FF5722")
    ax2.plot(df_impl_daily.index, df_impl_daily.values, color="#BF360C", linewidth=0.5)
    ax2.set_ylabel("% horas con congestión implícita", fontsize=10)
    ax2.set_xlabel("Fecha", fontsize=10)
    ax2.set_title("ATC implícito: % horas diarias con congestión (2019 - ago 2022)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "02_ntc_series.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def plot_ttf_spread_correlation(df):
    """Correlación TTF Gas vs Spread ES-FR."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("TTF Gas, CO2 y Spreads Eléctricos 2019-2024\nMIBEL Congestion Monitor",
                 fontsize=13, fontweight="bold")

    # Panel 1: TTF y CO2 en el tiempo
    ax1 = axes[0]
    df_daily = df.set_index("timestamp")[["ttf_eur_mwh", "co2_eur_t"]].resample("W").mean()
    ax1_twin = ax1.twinx()
    l1, = ax1.plot(df_daily.index, df_daily["ttf_eur_mwh"], color="#FF6F00", linewidth=1.5, label="TTF Gas (€/MWh)")
    l2, = ax1_twin.plot(df_daily.index, df_daily["co2_eur_t"], color="#388E3C", linewidth=1.5,
                         linestyle="--", label="CO2 EUA (€/t)")
    ax1.set_ylabel("TTF Gas (€/MWh)", color="#FF6F00", fontsize=9)
    ax1_twin.set_ylabel("CO2 EUA (€/t)", color="#388E3C", fontsize=9)
    ax1.set_title("Precios TTF Gas y CO2 (media semanal)", fontsize=10)
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Spark Spread en el tiempo
    ax2 = axes[1]
    df_spark = df.set_index("timestamp")[["spark_spread", "clean_spark_spread"]].resample("W").mean()
    ax2.plot(df_spark.index, df_spark["spark_spread"], color="#1565C0", linewidth=1.2, label="Spark Spread")
    ax2.plot(df_spark.index, df_spark["clean_spark_spread"], color="#7B1FA2", linewidth=1.2,
             linestyle="--", label="Clean Spark Spread")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("€/MWh", fontsize=9)
    ax2.set_title("Spark Spread y Clean Spark Spread (semanal)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Scatter TTF vs |Spread DA| por régimen
    ax3 = axes[2]
    df_scatter = df[["ttf_eur_mwh", "spread_da", "regime"]].dropna()
    df_scatter["abs_spread"] = df_scatter["spread_da"].abs()
    # Muestra aleatoria para no saturar el scatter
    sample = df_scatter.sample(min(5000, len(df_scatter)), random_state=42)
    for regime, color in REGIME_COLORS.items():
        mask = sample["regime"] == regime
        ax3.scatter(sample[mask]["ttf_eur_mwh"], sample[mask]["abs_spread"],
                    alpha=0.15, s=3, color=color, label=regime.replace("_", " ").title())
    ax3.set_xlabel("TTF Gas (€/MWh)", fontsize=9)
    ax3.set_ylabel("|Spread DA| (€/MWh)", fontsize=9)
    ax3.set_title("TTF Gas vs |Spread| por régimen", fontsize=10)
    ax3.legend(fontsize=8, markerscale=5)
    ax3.set_ylim(0, 50)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "03_ttf_spread_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def plot_data_quality(df):
    """Mapa de calidad de datos: huecos y cobertura."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Calidad del Dataset MIBEL (2019-2024)\nCobertura y Huecos por Variable",
                 fontsize=13, fontweight="bold")

    # Panel 1: Cobertura mensual
    ax1 = axes[0, 0]
    df_monthly = df.set_index("timestamp").resample("ME").agg(
        spread_coverage=("spread_da", lambda x: x.notna().mean() * 100),
        ntc_coverage=("ntc_es_fr", lambda x: x.notna().mean() * 100),
        ttf_coverage=("ttf_eur_mwh", lambda x: x.notna().mean() * 100),
    )
    ax1.plot(df_monthly.index, df_monthly["spread_coverage"], color="#2196F3", linewidth=1.5, label="Spread DA")
    ax1.plot(df_monthly.index, df_monthly["ntc_coverage"], color="#FF5722", linewidth=1.5, label="NTC ES-FR")
    ax1.plot(df_monthly.index, df_monthly["ttf_coverage"], color="#4CAF50", linewidth=1.5, label="TTF Gas")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Cobertura (%)", fontsize=9)
    ax1.set_title("Cobertura mensual por variable", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distribución del spread (histograma)
    ax2 = axes[0, 1]
    for regime, color in REGIME_COLORS.items():
        data = df[df["regime"] == regime]["spread_da"].dropna()
        ax2.hist(data, bins=100, alpha=0.5, color=color,
                 label=f"{regime.replace('_', ' ').title()} (n={len(data):,})",
                 density=True, range=(-30, 30))
    ax2.set_xlabel("Spread DA (€/MWh)", fontsize=9)
    ax2.set_ylabel("Densidad", fontsize=9)
    ax2.set_title("Distribución del spread por régimen", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Heatmap spread por hora y mes
    ax3 = axes[1, 0]
    pivot = df.pivot_table(values="spread_da", index="hour", columns="month", aggfunc="mean")
    im = ax3.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], fontsize=8)
    ax3.set_yticks(range(0, 24, 2))
    ax3.set_yticklabels(range(0, 24, 2), fontsize=8)
    ax3.set_xlabel("Mes", fontsize=9)
    ax3.set_ylabel("Hora del día", fontsize=9)
    ax3.set_title("Spread medio por hora y mes (2019-2024)", fontsize=10)
    plt.colorbar(im, ax=ax3, label="€/MWh")

    # Panel 4: Estadísticas por régimen (tabla)
    ax4 = axes[1, 1]
    ax4.axis("off")
    stats = []
    for regime in ["pre_crisis", "excepcion_iberica", "post_excepcion"]:
        d = df[df["regime"] == regime]["spread_da"].dropna()
        stats.append([
            regime.replace("_", " ").title(),
            f"{len(d):,}",
            f"{d.mean():.2f}",
            f"{d.std():.2f}",
            f"{(d.abs() > 0.5).mean() * 100:.1f}%",
            f"{d.min():.1f}",
            f"{d.max():.1f}",
        ])
    table = ax4.table(
        cellText=stats,
        colLabels=["Régimen", "Horas", "Media\n(€/MWh)", "Std\n(€/MWh)", "% Cong.", "Min", "Max"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#E3F2FD")
    ax4.set_title("Estadísticas del spread por régimen", fontsize=10, pad=20)

    plt.tight_layout()
    out = PLOTS_DIR / "04_data_quality.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def generate_validation_report(df):
    """Genera el informe de validación en texto."""
    report = []
    report.append("=" * 60)
    report.append("MIBEL Congestion Monitor — Informe de Validación")
    report.append("=" * 60)
    report.append(f"\nDataset: {DATASET.name}")
    report.append(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    report.append(f"Período: {df.timestamp.min()} → {df.timestamp.max()}")

    report.append("\n--- Cobertura temporal ---")
    total_expected = 6 * 365 * 24 + 2 * 24  # 6 años + 2 bisiestos
    coverage = len(df) / total_expected * 100
    report.append(f"Horas esperadas: {total_expected:,}")
    report.append(f"Horas reales: {len(df):,}")
    report.append(f"Cobertura: {coverage:.2f}%")

    report.append("\n--- Huecos por variable ---")
    for col in ["spread_da", "price_es", "price_fr", "ttf_eur_mwh", "co2_eur_t",
                "ntc_es_fr", "spark_spread", "atc_fatigue"]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct = n_missing / len(df) * 100
            report.append(f"  {col}: {n_missing:,} huecos ({pct:.2f}%)")

    report.append("\n--- Estadísticas por régimen ---")
    for regime in ["pre_crisis", "excepcion_iberica", "post_excepcion"]:
        d = df[df["regime"] == regime]
        spread = d["spread_da"].dropna()
        report.append(f"\n  {regime.upper()} ({len(d):,} horas):")
        report.append(f"    Spread DA: media={spread.mean():.2f}, std={spread.std():.2f}, "
                      f"min={spread.min():.1f}, max={spread.max():.1f}")
        report.append(f"    % congestión (|spread|>0.5): {(spread.abs()>0.5).mean()*100:.1f}%")
        if "ntc_es_fr" in d.columns:
            ntc = d["ntc_es_fr"].dropna()
            if len(ntc) > 0:
                report.append(f"    NTC ES→FR: media={ntc.mean():.0f} MW, min={ntc.min():.0f} MW")

    report.append("\n--- Fuentes de ATCs ---")
    for src, cnt in df["atc_source"].value_counts().items():
        report.append(f"  {src}: {cnt:,} horas ({cnt/len(df)*100:.1f}%)")

    report.append("\n--- Correlaciones clave ---")
    corr_ttf = df["ttf_eur_mwh"].corr(df["spread_da"].abs())
    corr_ntc = df["ntc_es_fr"].corr(df["spread_da"].abs())
    report.append(f"  Corr(TTF, |spread_DA|): {corr_ttf:.3f}")
    report.append(f"  Corr(NTC_ES_FR, |spread_DA|): {corr_ntc:.3f}")

    report_text = "\n".join(report)
    out = BASE_DIR / "data" / "processed" / "validation_report.txt"
    with open(out, "w") as f:
        f.write(report_text)
    print(f"\nInforme guardado: {out}")
    print(report_text)
    return report_text


if __name__ == "__main__":
    print("Cargando dataset...")
    df = load_data()
    print(f"Dataset cargado: {df.shape}")

    print("\nGenerando gráficos...")
    plot_spread_series(df)
    plot_ntc_series(df)
    plot_ttf_spread_correlation(df)
    plot_data_quality(df)

    print("\nGenerando informe de validación...")
    generate_validation_report(df)

    print("\n✓ Validación completada. Gráficos en:", PLOTS_DIR)
