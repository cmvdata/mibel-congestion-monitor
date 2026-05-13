"""
apply_gw_to_residuals.py
------------------------
Aplica el test Giacomini-White (2006) con pinball loss asimétrica
(q = 0.95) a los residuos reales del Modelo A v3 vs benchmarks AR(1)
y AR(24) sobre el test 2024.

Motivación: el DM clásico sobre squared-error (benchmarks_dm.py)
no rechaza H0 vs AR(1) con p = 0.078. La hipótesis del paper es que
Modelo A bate al benchmark especialmente en la cola superior, donde
viven los eventos de congestión. Pinball loss a q = 0.95 + test GW
es el procedimiento correcto bajo rolling re-estimation.

Reutiliza:
  - scripts/benchmarks_dm.py :: fit_predict_ar    (AR(p) rolling)
  - scripts/gw_pinball_test.py :: compare_models_pinball, dm_test,
                                   pinball_loss, squared_loss

Salida: results/gw_pinball_real_data.csv
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from benchmarks_dm import fit_predict_ar
from gw_pinball_test import (
    compare_models_pinball,
    dm_test,
    pinball_loss,
    squared_loss,
)

PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

DATASET = PROC_DIR / "mibel_dataset_20190101_20241231.parquet"
PRED_TEST2024 = PROC_DIR / "predictions_test2024_v3.parquet"
TARGET = "spread_da"


def main() -> None:
    print("=" * 72)
    print("  Giacomini-White + pinball q=0.95 sobre residuos reales")
    print("  Test 2024 — Modelo A v3 (XGBoost) vs AR(1) y AR(24)")
    print("=" * 72)

    # ── Datos ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(DATASET).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train_mask = df["timestamp"] < pd.Timestamp("2024-01-01")
    test_mask = df["timestamp"] >= pd.Timestamp("2024-01-01")
    y_tr = df.loc[train_mask, TARGET].astype(float).values
    y_te = df.loc[test_mask, TARGET].astype(float).values
    print(f"  Train: {len(y_tr):,} h     Test 2024: {len(y_te):,} h")

    # ── Modelo A v3 predicciones ───────────────────────────────────────────
    pred = pd.read_parquet(PRED_TEST2024).sort_values("timestamp")
    pa = pred["spread_predicted_modelA"].astype(float).values
    assert len(pa) == len(y_te), (
        f"Predicciones A ({len(pa)}) != test ({len(y_te)})"
    )

    # ── Benchmarks ─────────────────────────────────────────────────────────
    print("\n  Ajustando AR(1) y AR(24) sobre y_train (rolling 1-step)...")
    p_ar1 = fit_predict_ar(y_tr, y_te, lags=1)
    p_ar24 = fit_predict_ar(y_tr, y_te, lags=24)
    print("    AR(1) y AR(24) listos")

    # ── Tests para cada benchmark ──────────────────────────────────────────
    rows = []
    for name, p_b in [("AR(1)", p_ar1), ("AR(24)", p_ar24)]:
        print("\n" + "-" * 72)
        print(f"  Modelo A v3  vs  {name}")
        print("-" * 72)

        # DM sobre squared error (replica de la métrica del paper)
        L_A_sq = squared_loss(y_te, pa)
        L_B_sq = squared_loss(y_te, p_b)
        dm = dm_test(L_A_sq, L_B_sq)
        print("\n  [DM] squared-error loss:")
        print(f"    stat = {dm.statistic:+.4f}  p = {dm.p_value:.4f}  "
              f"mean_diff (A-B) = {dm.mean_loss_diff:+.4f}")

        # GW conditional sobre pinball q=0.95
        gw = compare_models_pinball(y_te, pa, p_b, q=0.95, conditional=True)
        pl_A = float(pinball_loss(y_te, pa, 0.95).mean())
        pl_B = float(pinball_loss(y_te, p_b, 0.95).mean())
        print("\n  [GW] conditional pinball loss q=0.95:")
        print(f"    mean pinball A = {pl_A:.4f}")
        print(f"    mean pinball B = {pl_B:.4f}  (A - B = {pl_A - pl_B:+.4f})")
        print(f"    stat = {gw.statistic:+.4f}  df = {gw.df}  "
              f"p = {gw.p_value:.4f}")
        winner = "A" if pl_A < pl_B else "B"
        verdict = (
            f"A mejor que {name} en q=0.95 (rechaza H0)"
            if (gw.p_value < 0.05 and winner == "A")
            else f"No rechazo H0 (o B gana)"
        )
        print(f"    veredicto: {verdict}")

        rows.append({
            "benchmark": name,
            "n": int(dm.n),
            # DM (sq error) — para contraste con el paper actual
            "dm_sq_stat": float(dm.statistic),
            "dm_sq_pvalue": float(dm.p_value),
            "dm_sq_mean_diff_A_minus_B": float(dm.mean_loss_diff),
            # GW (pinball q=0.95)
            "gw_pinball_stat": float(gw.statistic),
            "gw_pinball_pvalue": float(gw.p_value),
            "gw_pinball_df": int(gw.df),
            "mean_pinball_A": pl_A,
            "mean_pinball_B": pl_B,
            "pinball_diff_A_minus_B": pl_A - pl_B,
            "winner_q95": winner,
            "rejects_H0_at_5pct": bool(gw.p_value < 0.05),
        })

    # ── CSV resumen ────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    out = RESULTS_DIR / "gw_pinball_real_data.csv"
    df_out.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("  Resumen final")
    print("=" * 72)
    cols = ["benchmark", "n", "dm_sq_pvalue", "gw_pinball_pvalue",
            "mean_pinball_A", "mean_pinball_B",
            "winner_q95", "rejects_H0_at_5pct"]
    df_print = df_out[cols].copy()
    for c in ["dm_sq_pvalue", "gw_pinball_pvalue",
              "mean_pinball_A", "mean_pinball_B"]:
        df_print[c] = df_print[c].round(4)
    print(df_print.to_string(index=False))
    print(f"\n  CSV: {out}")


if __name__ == "__main__":
    main()
