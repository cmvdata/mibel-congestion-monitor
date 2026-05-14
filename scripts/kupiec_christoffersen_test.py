"""
kupiec_christoffersen_test.py
-----------------------------
Tests de calibracion de cuantil sobre las predicciones q=0.95 del
modelo A v4-quantile-cleanONLY (XGBoost objective=reg:quantileerror,
alpha=0.95, FEATURES_A_CLEAN).

  - Kupiec (1995) "Techniques for verifying the accuracy of risk
    measurement models": Likelihood-ratio test de coverage
    unconditional. H0: la tasa empirica de exceedances es igual al
    nivel nominal alpha = 0.05. LR_uc ~ chi2(1).

  - Christoffersen (1998) "Evaluating interval forecasts",
    International Economic Review 39: LR_ind test de independencia de
    exceedances via Markov chain transitions. H0: exceedances son
    independientes en el tiempo (no clustering). LR_ind ~ chi2(1).

  - Conditional coverage: LR_cc = LR_uc + LR_ind ~ chi2(2).
    H0: coverage correcto AND exceedances independientes.

Estos tests son el estandar de evaluacion de probabilistic forecasts
en Nowotarski-Weron 2018 (Renewable & Sustainable Energy Reviews) y
en el GEFCom2014 framework.

Output: results/kupiec_christoffersen_results.txt
"""
from pathlib import Path
import sys
import warnings

import math
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from benchmarks_dm import fit_predict_ar

PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"

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
ALPHA_QUANTILE = 0.95
NOMINAL_EXCEEDANCE_RATE = 1 - ALPHA_QUANTILE  # 0.05 for upper q=0.95


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else -1e10


def kupiec_test(I: np.ndarray, alpha_exc: float = NOMINAL_EXCEEDANCE_RATE) -> dict:
    """Kupiec (1995) unconditional coverage LR test.

    Parameters
    ----------
    I : array of 0/1 exceedances (I_t = 1 if y_t exceeded predicted quantile)
    alpha_exc : nominal exceedance rate (= 1 - quantile level for upper q)
    """
    n = int(I.size)
    x = int(I.sum())
    pi_hat = x / n if n else 0.0

    # LR_uc = -2 * [x*ln(alpha) + (n-x)*ln(1-alpha) - x*ln(pi_hat) - (n-x)*ln(1-pi_hat)]
    L_null = x * _safe_log(alpha_exc) + (n - x) * _safe_log(1 - alpha_exc)
    L_alt = x * _safe_log(pi_hat) + (n - x) * _safe_log(1 - pi_hat)
    lr_uc = -2.0 * (L_null - L_alt)
    p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
    return {
        "test": "Kupiec_unconditional_coverage",
        "n": n,
        "exceedances_observed": x,
        "exceedances_expected": n * alpha_exc,
        "rate_observed": pi_hat,
        "rate_nominal": alpha_exc,
        "LR_stat": float(lr_uc),
        "df": 1,
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "interpretation": (
            "reject H0: coverage rate differs from nominal alpha"
            if p_value < 0.05 else
            "do NOT reject H0: empirical coverage consistent with nominal alpha"
        ),
    }


def christoffersen_independence_test(I: np.ndarray) -> dict:
    """Christoffersen (1998) LR_ind test of exceedance independence
    via Markov chain transitions.
    """
    I = np.asarray(I, dtype=int).ravel()
    # Build transition counts
    n_00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
    n_01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
    n_10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
    n_11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))
    n_total = n_00 + n_01 + n_10 + n_11

    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0
    pi = (n_01 + n_11) / n_total if n_total > 0 else 0.0

    # L_ind: assumed independence
    L_ind = ((n_00 + n_10) * _safe_log(1 - pi)
             + (n_01 + n_11) * _safe_log(pi))
    # L_dep: first-order Markov dependence
    L_dep = (n_00 * _safe_log(1 - pi_01)
             + n_01 * _safe_log(pi_01)
             + n_10 * _safe_log(1 - pi_11)
             + n_11 * _safe_log(pi_11))
    lr_ind = -2.0 * (L_ind - L_dep)
    p_value = 1.0 - stats.chi2.cdf(lr_ind, df=1)
    return {
        "test": "Christoffersen_independence",
        "n_00": n_00, "n_01": n_01, "n_10": n_10, "n_11": n_11,
        "pi_01": pi_01, "pi_11": pi_11, "pi_marginal": pi,
        "LR_stat": float(lr_ind),
        "df": 1,
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "interpretation": (
            "reject H0: exceedances cluster (NOT independent)"
            if p_value < 0.05 else
            "do NOT reject H0: exceedances appear independent in time"
        ),
    }


def conditional_coverage_test(I: np.ndarray,
                              alpha_exc: float = NOMINAL_EXCEEDANCE_RATE) -> dict:
    """Christoffersen (1998) joint LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    uc = kupiec_test(I, alpha_exc)
    ind = christoffersen_independence_test(I)
    lr_cc = uc["LR_stat"] + ind["LR_stat"]
    p_value = 1.0 - stats.chi2.cdf(lr_cc, df=2)
    return {
        "test": "Christoffersen_conditional_coverage",
        "LR_uc": uc["LR_stat"],
        "LR_ind": ind["LR_stat"],
        "LR_cc": float(lr_cc),
        "df": 2,
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "interpretation": (
            "reject H0: coverage incorrect OR exceedances cluster"
            if p_value < 0.05 else
            "do NOT reject H0: forecast is correctly conditionally calibrated"
        ),
    }


# ---------------------------------------------------------------------------
# Train v4-quantile and produce upper q=0.95 predictions
# ---------------------------------------------------------------------------
def train_v4_quantile(df_tr, df_te, features, alpha=ALPHA_QUANTILE):
    params = dict(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    m = xgb.XGBRegressor(**params).fit(df_tr[features], df_tr[TARGET], verbose=False)
    return m.predict(df_te[features]), m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print(f"  Kupiec + Christoffersen tests on v4-quantile q={ALPHA_QUANTILE}")
    print(f"  Nominal upper-tail exceedance rate = {NOMINAL_EXCEEDANCE_RATE:.4f}")
    print("=" * 72)

    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"
    df_tr = df[tr_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy()
    df_te = df[te_mask].dropna(subset=FEATURES_A_CLEAN + [TARGET]).copy()
    print(f"\n  Train: {len(df_tr):,}  Test 2024: {len(df_te):,}")

    print("  Training XGBoost v4-quantile (cleanONLY)...")
    pred_q95, _ = train_v4_quantile(df_tr, df_te, FEATURES_A_CLEAN)
    y_te = df_te[TARGET].values

    # Exceedance indicator: y_t > predicted q=0.95 quantile
    I = (y_te > pred_q95).astype(int)
    n_exc = int(I.sum())
    rate = n_exc / len(I)
    expected = len(I) * NOMINAL_EXCEEDANCE_RATE
    print(f"\n  Empirical exceedances: {n_exc}/{len(I)} = {rate*100:.3f}%  "
          f"(expected at 5% nominal: {expected:.1f})")

    # Run all three tests
    uc = kupiec_test(I)
    ind = christoffersen_independence_test(I)
    cc = conditional_coverage_test(I)

    # Pretty print
    def _print_result(r):
        print(f"\n  [{r['test']}]")
        for k, v in r.items():
            if k == "test":
                continue
            if isinstance(v, float):
                print(f"    {k:25s} = {v:.4f}")
            else:
                print(f"    {k:25s} = {v}")

    _print_result(uc)
    _print_result(ind)
    _print_result(cc)

    # ── Auxiliary: same tests on AR(1) parametric q=0.95 (Gaussian residuals)
    print("\n" + "=" * 72)
    print("  AUXILIARY: AR(1) parametric q=0.95 = pred + 1.645 * sigma_residuals")
    print("  (assumes Gaussian residuals; not the right model but useful for context)")
    print("=" * 72)
    y_tr = df_tr[TARGET].values
    p_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    # In-sample residual std
    pred_ar1_train = fit_predict_ar(y_tr[:-len(y_te)], y_tr[-len(y_te):], lags=1) \
        if len(y_tr) > len(y_te) else None
    # Use full training residual stdev for the test set parametric bound
    p_ar1_train = fit_predict_ar(y_tr[: int(0.8 * len(y_tr))],
                                  y_tr[int(0.8 * len(y_tr)):], lags=1)
    sigma_train_res = float(np.std(y_tr[int(0.8 * len(y_tr)):] - p_ar1_train))
    z_95 = 1.6449
    pred_ar1_q95 = p_ar1 + z_95 * sigma_train_res
    print(f"  sigma(residuos AR1 holdout train) = {sigma_train_res:.3f}  "
          f"=> upper q=0.95 = pred + {z_95}*sigma")
    I_ar1 = (y_te > pred_ar1_q95).astype(int)
    print(f"  Exceedances AR(1) param: {int(I_ar1.sum())}/{len(I_ar1)} = "
          f"{I_ar1.mean()*100:.3f}%")

    uc_ar1 = kupiec_test(I_ar1)
    ind_ar1 = christoffersen_independence_test(I_ar1)
    cc_ar1 = conditional_coverage_test(I_ar1)
    print(f"\n  AR(1) param Kupiec: LR={uc_ar1['LR_stat']:.3f}  p={uc_ar1['p_value']:.4f}  "
          f"-> {uc_ar1['interpretation']}")
    print(f"  AR(1) param Christoffersen ind: LR={ind_ar1['LR_stat']:.3f}  "
          f"p={ind_ar1['p_value']:.4f}  -> {ind_ar1['interpretation']}")
    print(f"  AR(1) param conditional cov: LR={cc_ar1['LR_cc']:.3f}  "
          f"p={cc_ar1['p_value']:.4f}  -> {cc_ar1['interpretation']}")

    # Save text
    out = RESULTS_DIR / "kupiec_christoffersen_results.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"Kupiec + Christoffersen calibration tests\n")
        fh.write(f"Model: A v4 (XGBoost quantile, alpha={ALPHA_QUANTILE})\n")
        fh.write(f"Test set: 2024 (n={len(I)})\n")
        fh.write(f"Empirical exceedance rate: {rate*100:.3f}%  "
                 f"(nominal 5.00%, expected {expected:.0f})\n\n")
        for r in [uc, ind, cc]:
            fh.write(f"\n[{r['test']}]\n")
            for k, v in r.items():
                if k == "test":
                    continue
                fh.write(f"  {k}: {v}\n")
        fh.write("\n\n--- Auxiliary: AR(1) parametric q=0.95 ---\n")
        for r in [uc_ar1, ind_ar1, cc_ar1]:
            fh.write(f"\n[{r['test']}]\n")
            for k, v in r.items():
                if k == "test":
                    continue
                fh.write(f"  {k}: {v}\n")
    print(f"\n  Resultados guardados: {out}")


if __name__ == "__main__":
    main()
