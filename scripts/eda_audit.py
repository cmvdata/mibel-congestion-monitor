"""
eda_audit.py
------------
Audit retrospectivo del dataset y de los residuos del v3-mean.
Cosas que debiamos haber chequeado antes de modelar:

  - Distribucion de spread_da, price_es, price_fr (mean, std, skew,
    kurt, quantiles, outliers extremos)
  - NaN patterns por columna y por regimen
  - Timestamps duplicados o gaps
  - Sign convention del spread (sanity: spread_da = price_es - price_fr?)
  - Residuos v3-mean: mean ~ 0? autocorrelacion? heteroscedasticidad?
  - Atypical values in fundamentals (TTF, CO2)

Output: results/eda_audit_report.txt
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
RESIDUALS = PROC_DIR / "residuals_v3.parquet"


def section(title, lines):
    lines.append("\n" + "=" * 72)
    lines.append(f"  {title}")
    lines.append("=" * 72)


def main():
    lines = []
    df = pd.read_parquet(DATASET)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    section("Dataset shape and timestamp integrity", lines)
    lines.append(f"  shape: {df.shape}")
    lines.append(f"  timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    expected_hours = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 3600) + 1
    lines.append(f"  expected hours (continuous): {expected_hours:,}")
    lines.append(f"  observed rows: {len(df):,}")
    lines.append(f"  duplicate timestamps: {int(df['timestamp'].duplicated().sum())}")
    diffs = df["timestamp"].diff().dropna()
    gaps_over_1h = (diffs > pd.Timedelta(hours=1)).sum()
    lines.append(f"  gaps > 1h: {int(gaps_over_1h)} (expected DST and missing hours)")
    if gaps_over_1h > 0:
        biggest = diffs.nlargest(5)
        lines.append(f"  top 5 gaps: {biggest.tolist()}")

    section("Target spread_da: distribution and outliers", lines)
    s = df["spread_da"].dropna()
    lines.append(f"  n_non_null: {len(s):,}  (NaN: {df['spread_da'].isna().sum()})")
    lines.append(f"  mean = {s.mean():.4f}  std = {s.std():.4f}")
    lines.append(f"  min  = {s.min():.4f}    max = {s.max():.4f}")
    lines.append(f"  skewness = {stats.skew(s):.3f}    kurtosis = {stats.kurtosis(s):.3f}")
    lines.append(f"  quantiles: p1={s.quantile(0.01):.2f}  p5={s.quantile(0.05):.2f}  "
                 f"p25={s.quantile(0.25):.2f}  p50={s.quantile(0.50):.2f}  "
                 f"p75={s.quantile(0.75):.2f}  p95={s.quantile(0.95):.2f}  "
                 f"p99={s.quantile(0.99):.2f}")
    for thr in [50, 100, 200, 500]:
        n_above = int((s.abs() > thr).sum())
        lines.append(f"  |spread| > {thr} EUR/MWh: {n_above} obs "
                     f"({n_above/len(s)*100:.3f}%)")
    # Extreme rows
    top5 = s.abs().nlargest(5)
    lines.append("  top 5 |spread| values:")
    for idx, v in top5.items():
        t = df.loc[idx, "timestamp"]
        sp = df.loc[idx, "spread_da"]
        es = df.loc[idx, "price_es"]
        fr = df.loc[idx, "price_fr"]
        lines.append(f"    {t}  spread={sp:.2f}  price_es={es:.2f}  price_fr={fr:.2f}")

    section("Sign convention check: spread_da == price_es - price_fr?", lines)
    diff = (df["price_es"] - df["price_fr"]) - df["spread_da"]
    diff_clean = diff.dropna()
    lines.append(f"  max |(price_es - price_fr) - spread_da| = {diff_clean.abs().max():.6f}")
    lines.append(f"  mean diff = {diff_clean.mean():.6f}")
    lines.append("  -> CONVENTION OK" if diff_clean.abs().max() < 1e-3 else
                 "  -> WARNING: sign or definition mismatch")

    section("price_es and price_fr: distribution and negatives", lines)
    for col in ["price_es", "price_fr"]:
        p = df[col].dropna()
        n_neg = int((p < 0).sum())
        n_zero = int((p == 0).sum())
        lines.append(f"  {col}: mean={p.mean():.2f}  std={p.std():.2f}  "
                     f"min={p.min():.2f}  max={p.max():.2f}")
        lines.append(f"    negative prices: {n_neg} ({n_neg/len(p)*100:.3f}%)  "
                     f"zero prices: {n_zero}")

    section("NaN patterns by feature", lines)
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    lines.append(f"  columns with any NaN: {len(nan_counts)}")
    for col, n in nan_counts.head(15).items():
        lines.append(f"    {col:30s} {n:>6,} NaN ({n/len(df)*100:5.2f}%)")

    section("Fundamentals sanity: TTF and CO2 outliers", lines)
    for col in ["ttf_eur_mwh", "co2_eur_t"]:
        p = df[col].dropna()
        lines.append(f"  {col}: range [{p.min():.2f}, {p.max():.2f}]  "
                     f"mean={p.mean():.2f}  std={p.std():.2f}  "
                     f"p99={p.quantile(0.99):.2f}")

    # --- Residuals -----------------------------------------------------
    if RESIDUALS.exists():
        section("Residuals v3-mean: mean, autocorr, heteroscedasticity", lines)
        res = pd.read_parquet(RESIDUALS)
        res["timestamp"] = pd.to_datetime(res["timestamp"])
        r = res["residual"].dropna()
        lines.append(f"  n: {len(r):,}")
        lines.append(f"  mean = {r.mean():.4f}    median = {r.median():.4f}")
        lines.append(f"  std  = {r.std():.4f}    iqr = {r.quantile(0.75) - r.quantile(0.25):.4f}")
        lines.append(f"  skewness = {stats.skew(r):.3f}   kurtosis = {stats.kurtosis(r):.3f}")
        # Mean should be ~ 0 if model is unbiased
        # t-test against zero
        t_stat, t_p = stats.ttest_1samp(r, 0.0)
        lines.append(f"  one-sample t-test (H0: mean=0): t={t_stat:.3f}  p={t_p:.4f}")
        if t_p < 0.05:
            lines.append(f"    -> RESIDUALS HAVE NON-ZERO MEAN (bias of {r.mean():.4f}); "
                         "consider intercept correction")
        # Autocorrelation lag 1, 24
        for L in [1, 24, 168]:
            acf = pd.Series(r.values).autocorr(lag=L)
            lines.append(f"  autocorr lag {L}h: {acf:.4f}")
        # Ljung-Box
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(r.values, lags=[24], return_df=True)
            lines.append(f"  Ljung-Box at 24 lags: stat={lb['lb_stat'].iloc[0]:.2f}  "
                         f"p={lb['lb_pvalue'].iloc[0]:.4f}")
            if lb['lb_pvalue'].iloc[0] < 0.05:
                lines.append("    -> RESIDUALS ARE AUTOCORRELATED (model leaves structure)")
        except Exception as e:
            lines.append(f"  Ljung-Box failed: {e}")
        # Heteroscedasticity: Engle ARCH-LM test on r^2
        try:
            from statsmodels.stats.diagnostic import het_arch
            lm_stat, lm_p, _, _ = het_arch(r.values, nlags=24)
            lines.append(f"  Engle ARCH-LM (24 lags): stat={lm_stat:.2f}  p={lm_p:.4f}")
            if lm_p < 0.05:
                lines.append("    -> RESIDUALS HETEROSCEDASTIC (conditional vol changes "
                             "over time; expected in EPF)")
        except Exception as e:
            lines.append(f"  ARCH-LM failed: {e}")
    else:
        lines.append("  residuals_v3.parquet not found; skipping residual diagnostics")

    section("Per-regime breakdown of spread_da", lines)
    if "regime" in df.columns:
        for reg, g in df.groupby("regime"):
            s = g["spread_da"].dropna()
            lines.append(f"  {reg:25s}  n={len(s):>6,}  "
                         f"mean={s.mean():+6.2f}  std={s.std():5.2f}  "
                         f"p99|abs|={s.abs().quantile(0.99):5.2f}  "
                         f"max|abs|={s.abs().max():6.2f}")
    else:
        lines.append("  no regime column")

    text = "\n".join(lines)
    print(text)
    out = RESULTS_DIR / "eda_audit_report.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n  Guardado: {out}")


if __name__ == "__main__":
    main()
