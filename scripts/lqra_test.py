"""
lqra_test.py
------------
LASSO-Regularized Quantile Regression Averaging (LQRA), following
Uniejewski & Weron (2021), Energy Economics 95:105121.

Idea: take K point forecasts of the conditional mean and combine them
into a probabilistic forecast via L1-regularized quantile regression
at each target quantile. The LASSO selects which point forecasts are
informative for each quantile; the regularization controls the
"low-quality predictor" vulnerability of plain QRA (Marcjasz et al.
2020).

Inputs to LQRA (point forecasts of conditional mean):
  1. v3-mean   XGBoost MSE on FEATURES_A_CLEAN
  2. AR(1)
  3. AR(24)
  4. LASSO regression (from benchmarks_dm.py recipe)
  5. Decomposed (XGB_ES - XGB_FR)

Target quantiles evaluated: 0.05, 0.25, 0.50, 0.75, 0.95.

Splitting (clean OOS for LQRA):
  - Train point models on 2019-2023, predict ALL 2024.
  - Period A = first half 2024 = LQRA training (alpha selection + final fit).
  - Period B = second half 2024 = LQRA evaluation.

Alpha selection: 80/20 split inside period A; pick alpha minimizing
pinball loss on the 20% hold-out. Then refit on full A and evaluate on B.

Output:
  results/lqra_results.csv      one row per target quantile
  results/lqra_coefficients.csv L1 coefficients per quantile
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from benchmarks_dm import fit_predict_ar
from gw_pinball_test import pinball_loss

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

POINT_MODEL_NAMES = ["v3_mean", "AR1", "AR24", "LASSO", "decomposed"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]


def xgb_mean(df_tr, df_te, features):
    params = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    m = xgb.XGBRegressor(**params).fit(
        df_tr[features], df_tr[TARGET], verbose=False)
    return m.predict(df_te[features])


def lasso_predict(df_tr, df_te, features):
    sc = StandardScaler()
    X_tr = sc.fit_transform(df_tr[features].fillna(0))
    X_te = sc.transform(df_te[features].fillna(0))
    m = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=20000)
    m.fit(X_tr, df_tr[TARGET].values)
    return m.predict(X_te)


def decomposed_predict(df_tr, df_te, common_feats):
    """Train XGB on price_es and XGB on price_fr, return pred_es - pred_fr."""
    base = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    # build price lags
    def lags_for(df, base_col):
        cols = {}
        for L in [1, 2, 3, 6, 12, 24, 48, 168]:
            cols[f"{base_col}_lag{L}h"] = df[base_col].shift(L)
        cols[f"{base_col}_ma24h"] = df[base_col].rolling(24, min_periods=12).mean().shift(1)
        cols[f"{base_col}_ma168h"] = df[base_col].rolling(168, min_periods=84).mean().shift(1)
        cols[f"{base_col}_std24h"] = df[base_col].rolling(24, min_periods=12).std().shift(1)
        return pd.DataFrame(cols, index=df.index)

    # We need the full unfiltered df because lags require history beyond the
    # training mask. The caller passes df_tr, df_te already with these merged
    # via the helper above; but here we recompute on the fly.
    # Reuse: the caller should provide df_full to compute lags properly.
    raise NotImplementedError("Use the integrated path in main()")


def pinball_mean(y, p, q):
    return float(pinball_loss(np.asarray(y), np.asarray(p), q).mean())


def fit_lqra(X_tr, y_tr, q, alpha):
    """Fit LASSO-regularized quantile regressor."""
    qr = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
    qr.fit(X_tr, y_tr)
    return qr


def select_alpha(X_a, y_a, q, alpha_grid):
    """Pick alpha minimizing pinball on inner 20% hold-out of period A."""
    n = len(X_a)
    cut = int(n * 0.8)
    X_inner_tr, X_inner_val = X_a[:cut], X_a[cut:]
    y_inner_tr, y_inner_val = y_a[:cut], y_a[cut:]
    best_alpha, best_pinball = None, np.inf
    for alpha in alpha_grid:
        try:
            qr = fit_lqra(X_inner_tr, y_inner_tr, q, alpha)
            pred = qr.predict(X_inner_val)
            pb = pinball_mean(y_inner_val, pred, q)
            if pb < best_pinball:
                best_pinball, best_alpha = pb, alpha
        except Exception:
            continue
    return best_alpha, best_pinball


def main():
    print("=" * 72)
    print("  LQRA (LASSO-Regularized Quantile Regression Averaging)")
    print("  Uniejewski-Weron 2021, Energy Economics 95")
    print("=" * 72)

    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ----- Build features needed for decomposed model -----
    for base in ["price_es", "price_fr"]:
        for L in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"{base}_lag{L}h"] = df[base].shift(L)
        df[f"{base}_ma24h"] = df[base].rolling(24, min_periods=12).mean().shift(1)
        df[f"{base}_ma168h"] = df[base].rolling(168, min_periods=84).mean().shift(1)
        df[f"{base}_std24h"] = df[base].rolling(24, min_periods=12).std().shift(1)

    ES_PRICE_FEATS = [c for c in df.columns if c.startswith("price_es_")]
    FR_PRICE_FEATS = [c for c in df.columns if c.startswith("price_fr_")]
    SHARED = [
        "ttf_eur_mwh", "co2_eur_t", "spark_spread", "clean_spark_spread",
        "ttf_lag24h", "ttf_lag168h",
        "ntc_es_fr", "ntc_fr_es", "ntc_is_observed",
        "hour", "dow", "month", "is_weekend", "is_night", "is_peak",
    ]
    es_feats = ES_PRICE_FEATS + SHARED
    fr_feats = FR_PRICE_FEATS + SHARED

    # ----- Split -----
    tr_mask = (df["timestamp"] >= "2019-01-01") & (df["timestamp"] < "2024-01-01")
    te_mask = df["timestamp"] >= "2024-01-01"
    need = FEATURES_A_CLEAN + es_feats + fr_feats + [
        TARGET, "price_es", "price_fr"
    ]
    df_tr = df[tr_mask].dropna(subset=list(set(need))).copy()
    df_te = df[te_mask].dropna(subset=list(set(need))).copy().reset_index(drop=True)
    print(f"  Train: {len(df_tr):,}  Test 2024 (clean): {len(df_te):,}")

    # ----- 5 point predictors on the full test 2024 -----
    print("\n  Training point predictors...")
    pred_v3 = xgb_mean(df_tr, df_te, FEATURES_A_CLEAN)
    print(f"    v3_mean ready")
    pred_ar1 = fit_predict_ar(df_tr[TARGET].values, df_te[TARGET].values, lags=1)
    pred_ar24 = fit_predict_ar(df_tr[TARGET].values, df_te[TARGET].values, lags=24)
    print(f"    AR(1), AR(24) ready")
    pred_lasso = lasso_predict(df_tr, df_te, FEATURES_A_CLEAN)
    print(f"    LASSO ready")
    # Decomposed: train two XGB, return diff
    base = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    m_es = xgb.XGBRegressor(**base).fit(df_tr[es_feats], df_tr["price_es"], verbose=False)
    m_fr = xgb.XGBRegressor(**base).fit(df_tr[fr_feats], df_tr["price_fr"], verbose=False)
    pred_dec = m_es.predict(df_te[es_feats]) - m_fr.predict(df_te[fr_feats])
    print(f"    decomposed (ES - FR) ready")

    y_te = df_te[TARGET].values
    X_te = np.column_stack([pred_v3, pred_ar1, pred_ar24, pred_lasso, pred_dec])

    # ----- Period A / Period B split -----
    cut_ab = len(df_te) // 2
    X_A, y_A = X_te[:cut_ab], y_te[:cut_ab]
    X_B, y_B = X_te[cut_ab:], y_te[cut_ab:]
    print(f"\n  Period A (LQRA train): {len(y_A):,}  "
          f"Period B (LQRA eval): {len(y_B):,}")

    # ----- LQRA for each target quantile -----
    print("\n" + "=" * 72)
    print("  LQRA: alpha selection + evaluation per target quantile")
    print("=" * 72)
    rows = []
    coef_rows = []
    for q in QUANTILES:
        alpha_star, val_pb = select_alpha(X_A, y_A, q, ALPHA_GRID)
        qr = fit_lqra(X_A, y_A, q, alpha_star)
        pred_B_lqra = qr.predict(X_B)
        pb_B = pinball_mean(y_B, pred_B_lqra, q)

        # Compare to best single point predictor on period B
        single_pb = {}
        for j, name in enumerate(POINT_MODEL_NAMES):
            single_pb[name] = pinball_mean(y_B, X_B[:, j], q)
        best_single = min(single_pb, key=single_pb.get)
        best_single_pb = single_pb[best_single]

        # Coverage (only meaningful for the predicted upper-tail)
        if q > 0.5:
            cov = float((y_B > pred_B_lqra).mean())
            nominal = 1 - q
        elif q < 0.5:
            cov = float((y_B < pred_B_lqra).mean())
            nominal = q
        else:
            cov = float((y_B > pred_B_lqra).mean())
            nominal = 0.5

        print(f"\n  q = {q:.2f}:")
        print(f"    alpha* (BIC-like CV) = {alpha_star}")
        print(f"    pinball LQRA (period B) = {pb_B:.4f}")
        print(f"    pinball best single (period B, {best_single}) = {best_single_pb:.4f}")
        print(f"    LQRA - best single   = {pb_B - best_single_pb:+.4f}")
        print(f"    coverage (vs nominal {nominal:.3f}) = {cov:.4f}")
        print(f"    coefficients: intercept={qr.intercept_:.3f}  "
              f"{dict(zip(POINT_MODEL_NAMES, [round(float(c), 3) for c in qr.coef_]))}")

        rows.append({
            "quantile": q,
            "alpha_selected": alpha_star,
            "pinball_LQRA_B": pb_B,
            "pinball_best_single_B": best_single_pb,
            "best_single_name": best_single,
            "delta_LQRA_minus_best": pb_B - best_single_pb,
            "coverage_B": cov,
            "nominal_coverage": nominal,
            "n_B": len(y_B),
        })
        coef_rows.append({
            "quantile": q,
            "alpha": alpha_star,
            "intercept": float(qr.intercept_),
            **dict(zip(POINT_MODEL_NAMES, [float(c) for c in qr.coef_])),
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "lqra_results.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(RESULTS_DIR / "lqra_coefficients.csv", index=False)
    print(f"\n  Resultados: {RESULTS_DIR / 'lqra_results.csv'}")
    print(f"  Coeficientes: {RESULTS_DIR / 'lqra_coefficients.csv'}")


if __name__ == "__main__":
    main()
