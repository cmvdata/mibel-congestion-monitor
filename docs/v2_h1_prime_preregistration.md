# H1' pre-registration — Stage A with ESIOS scheduled-exchange features

**Date**: 2026-05-15
**Status**: pre-registered, NOT yet tested.

## Background

H1 and H2 of `docs/v2_dgp_analysis.md` were falsified empirically: the
binary classifier of congestion regime $S_t = \mathbb{1}\{|spread| >
0.5\,\text{€/MWh}\}$ reached an empirical ceiling of AUC-ROC $\approx 0.87$
with 33 features (LightGBM, rich set), and a 3-state Gaussian HMM on
the continuous-feature subset underperformed (AUC-ROC $= 0.60$).

The empirical-limits section (§6) of the paper closed with the
identification of data not available at the time: ENTSO-E transparency
flows, REE/RTE physical flows, intraday OMIE. Of these, the
**ESIOS scheduled commercial exchange ES↔FR** is now obtained
(indicators 28 import FR→ES and 32 export ES→FR, combined into
`scheduled_net_es_to_fr_mw` with derived `utilization_d1 = |scheduled|
/ NTC` in `data/processed/mibel_dataset_..._v6.parquet`). Coverage on
master timestamps is 99.92%.

This is exactly the physical-utilization signal that was hypothesised
to be missing for ex-ante prediction of congestion onset.

## H1'

> **H1'**: adding the ESIOS-derived features
> `scheduled_net_es_to_fr_mw`, `scheduled_abs_mw` and `utilization_d1`
> to the existing 33-feature rich set raises Stage A performance to
> **AUC-ROC $> 0.92$ AND AUC-PR $> 0.45$** for the LightGBM
> classifier of $S_t$ in walk-forward weekly evaluation on 2024.

## Specification

- Target: $S_t = \mathbb{1}\{|\text{spread\_da}_t| > 0.5\,\text{€/MWh}\}$.
- Features: `FEATURES_A_CLEAN` (26) + 7 v5 features (renewables,
  demand, nuclear) + 8 derived ratios + **3 new ESIOS features** = 36.
- Models tested: logistic regression (baseline, secondary) and
  LightGBM (primary).
- Validation: walk-forward weekly, retraining at the start of every
  ISO week on 2024 using all prior data.
- Metrics: AUC-ROC, AUC-PR, Brier, median lead time to $0 \to 1$
  transition, fraction predicted in advance.

## Decision rules

- **H1' confirmed**: LightGBM AUC-ROC $> 0.92$ AND AUC-PR $> 0.45$.
  Implication: the physical-utilization signal breaks the ceiling and
  ex-ante prediction is feasible with the right open data. The paper
  conclusion updates accordingly.
- **H1' partial**: $0.88 \leq$ AUC-ROC $\leq 0.92$, marginal gain over
  prior 0.87 ceiling. Document as modest contribution; the ceiling
  moves but is not broken.
- **H1' falsified**: AUC-ROC $\leq 0.88$. The empirical ceiling is
  confirmed structural even with explicit physical-utilization data;
  remaining gains require intraday or transactional data.

## Why pre-register

Across this project I (CV) have repeatedly observed predictions of
"high probability of improvement" fail empirically (0/5 in the
preceding paper iteration). Pre-registration of decision thresholds
before running the test removes the temptation to retrofit the
narrative to the result. The hypothesis is binary and the thresholds
are numerical; the result will be reported whichever way it falls.
