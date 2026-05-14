"""
v2_dgp_analysis.py
------------------
Caracterizacion empirica del DGP del spread ES-FR para v2.

Salida: numeros que alimentan docs/v2_dgp_analysis.md. NO entrena
modelos. Solo describe los datos para JUSTIFICAR la arquitectura
two-stage hurdle.

Output: results/v2_dgp_analysis.json + prints
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed" / "mibel_dataset_20190101_20241231.parquet"
RESULTS = BASE / "results"


def run_length_stats(states):
    """Compute mean / median run length per state value (0 or 1)."""
    if len(states) == 0:
        return {}
    runs_by_state = {0: [], 1: []}
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            runs_by_state[int(current)].append(length)
            current = s
            length = 1
    runs_by_state[int(current)].append(length)
    out = {}
    for k, lst in runs_by_state.items():
        if not lst:
            continue
        arr = np.array(lst)
        out[f"state{k}_n_runs"] = int(len(arr))
        out[f"state{k}_mean_h"] = float(arr.mean())
        out[f"state{k}_median_h"] = float(np.median(arr))
        out[f"state{k}_p95_h"] = float(np.quantile(arr, 0.95))
        out[f"state{k}_max_h"] = int(arr.max())
    return out


def transition_matrix(states):
    """Empirical 2x2 transition matrix for binary state sequence."""
    s = np.asarray(states, dtype=int)
    transitions = np.zeros((2, 2), dtype=int)
    for a, b in zip(s[:-1], s[1:]):
        transitions[a, b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    probs = transitions / np.where(row_sums == 0, 1, row_sums)
    return {
        "counts": {
            "00": int(transitions[0, 0]),
            "01": int(transitions[0, 1]),
            "10": int(transitions[1, 0]),
            "11": int(transitions[1, 1]),
        },
        "probs": {
            "P(0->0)": float(probs[0, 0]),
            "P(0->1)": float(probs[0, 1]),
            "P(1->0)": float(probs[1, 0]),
            "P(1->1)": float(probs[1, 1]),
        },
    }


def main():
    df = pd.read_parquet(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    s = df["spread_da"].values
    out = {}

    # --- 1. Zero-inflation magnitude -------------------------------
    print("=" * 72)
    print("  1. Zero-inflation magnitude")
    print("=" * 72)
    n = len(s)
    n_exact_zero = int(np.sum(s == 0.0))
    fracs = {}
    for thr in [0.0, 0.1, 0.5, 1.0, 2.0]:
        cnt = int(np.sum(np.abs(s) <= thr))
        fracs[f"|spread|<={thr}"] = {"n": cnt, "frac": cnt / n}
    out["zero_inflation"] = {
        "n_total": n,
        "n_exact_zero": n_exact_zero,
        "frac_exact_zero": n_exact_zero / n,
        "by_threshold": fracs,
    }
    for thr, v in fracs.items():
        print(f"  {thr}: n={v['n']:,} ({v['frac']*100:.2f}%)")
    print(f"  exact zero: {n_exact_zero:,} ({n_exact_zero/n*100:.2f}%)")

    # --- 2. Per-regime breakdown -----------------------------------
    print("\n" + "=" * 72)
    print("  2. Zero-inflation per regime")
    print("=" * 72)
    by_regime = {}
    for reg, g in df.groupby("regime"):
        sv = g["spread_da"].values
        ez = int(np.sum(sv == 0.0))
        within_1 = int(np.sum(np.abs(sv) <= 1.0))
        by_regime[reg] = {
            "n": len(sv),
            "frac_exact_zero": ez / len(sv) if len(sv) else 0,
            "frac_within_1": within_1 / len(sv) if len(sv) else 0,
            "mean_abs": float(np.mean(np.abs(sv))),
            "p99_abs": float(np.quantile(np.abs(sv), 0.99)),
            "max_abs": float(np.max(np.abs(sv))),
        }
        print(f"  {reg:25s}  n={by_regime[reg]['n']:>6,}  "
              f"exact_zero={by_regime[reg]['frac_exact_zero']*100:5.2f}%  "
              f"|s|<=1: {by_regime[reg]['frac_within_1']*100:5.2f}%  "
              f"p99|s|={by_regime[reg]['p99_abs']:5.2f}")
    out["per_regime"] = by_regime

    # --- 3. Binary state series + transitions ----------------------
    print("\n" + "=" * 72)
    print("  3. Binary state series (decoupled = |spread| > 0.5)")
    print("=" * 72)
    THR = 0.5
    state = (np.abs(s) > THR).astype(int)
    out["state_threshold_eur_mwh"] = THR
    out["state_overall_decoupled_frac"] = float(state.mean())
    print(f"  threshold = {THR} EUR/MWh")
    print(f"  decoupled fraction = {state.mean()*100:.2f}%")

    trans = transition_matrix(state)
    print(f"  Transition counts: {trans['counts']}")
    print(f"  P(0->0)={trans['probs']['P(0->0)']:.4f}  "
          f"P(0->1)={trans['probs']['P(0->1)']:.4f}")
    print(f"  P(1->0)={trans['probs']['P(1->0)']:.4f}  "
          f"P(1->1)={trans['probs']['P(1->1)']:.4f}")
    out["transition_matrix"] = trans

    # --- 4. Run lengths --------------------------------------------
    print("\n" + "=" * 72)
    print("  4. Run-length distribution of states")
    print("=" * 72)
    rl = run_length_stats(state)
    out["run_lengths"] = rl
    for k, v in rl.items():
        print(f"  {k}: {v}")

    # --- 5. Decoupling rate per hour-of-day ------------------------
    print("\n" + "=" * 72)
    print("  5. Decoupling rate by hour-of-day")
    print("=" * 72)
    df["_state"] = state
    by_hour = df.groupby(df["timestamp"].dt.hour)["_state"].mean()
    out["decouple_by_hour"] = {int(h): float(v) for h, v in by_hour.items()}
    print(f"  min hour: {int(by_hour.idxmin()):02d}h -> {by_hour.min()*100:.2f}%")
    print(f"  max hour: {int(by_hour.idxmax()):02d}h -> {by_hour.max()*100:.2f}%")
    print(f"  range: {(by_hour.max() - by_hour.min())*100:.2f} percentage points")

    # --- 6. Conditional magnitude distribution ---------------------
    print("\n" + "=" * 72)
    print("  6. Conditional distribution given decoupled (|spread| > 0.5)")
    print("=" * 72)
    cond = s[state == 1]
    abs_cond = np.abs(cond)
    out["conditional_magnitude"] = {
        "n": int(len(cond)),
        "mean": float(cond.mean()),
        "median": float(np.median(cond)),
        "std": float(cond.std()),
        "mean_abs": float(abs_cond.mean()),
        "p50_abs": float(np.median(abs_cond)),
        "p75_abs": float(np.quantile(abs_cond, 0.75)),
        "p95_abs": float(np.quantile(abs_cond, 0.95)),
        "p99_abs": float(np.quantile(abs_cond, 0.99)),
        "max_abs": float(np.max(abs_cond)),
        "frac_positive": float((cond > 0).mean()),
        "frac_negative": float((cond < 0).mean()),
    }
    cm = out["conditional_magnitude"]
    print(f"  n decoupled = {cm['n']:,}")
    print(f"  |spread| given decoupled: median={cm['p50_abs']:.2f}  "
          f"p75={cm['p75_abs']:.2f}  p95={cm['p95_abs']:.2f}  "
          f"p99={cm['p99_abs']:.2f}  max={cm['max_abs']:.2f}")
    print(f"  direction: +{cm['frac_positive']*100:.1f}% / "
          f"-{cm['frac_negative']*100:.1f}%")

    # --- 7. NTC saturation proxy and decoupling --------------------
    print("\n" + "=" * 72)
    print("  7. NTC saturation proxy vs decoupling")
    print("=" * 72)
    if "ntc_es_fr" in df.columns:
        ntc_p95 = df["ntc_es_fr"].quantile(0.95)
        df["_low_ntc"] = (df["ntc_es_fr"] < df["ntc_es_fr"].quantile(0.5)).astype(int)
        ct = pd.crosstab(df["_low_ntc"], df["_state"], normalize="index")
        print(f"  Conditional decoupling rate given NTC below median:")
        print(ct.to_string())
        out["ntc_low_vs_decouple"] = ct.to_dict()

    # --- 8. Spark spread asymmetry hint ----------------------------
    print("\n" + "=" * 72)
    print("  8. Conditional |spread| by TTF tertile")
    print("=" * 72)
    if "ttf_eur_mwh" in df.columns:
        df["_ttf_tertile"] = pd.qcut(df["ttf_eur_mwh"], 3,
                                     labels=["low", "mid", "high"],
                                     duplicates="drop")
        agg = df.groupby("_ttf_tertile")["_state"].mean()
        out["decouple_by_ttf_tertile"] = {str(k): float(v) for k, v in agg.items()}
        for k, v in agg.items():
            print(f"  TTF {k}: decoupled rate = {v*100:.2f}%")

    # Save
    json_path = RESULTS / "v2_dgp_analysis.json"
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
