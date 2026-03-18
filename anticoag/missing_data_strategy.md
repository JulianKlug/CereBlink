# Missing Data Strategy

## Mechanism Assumption

Missing covariates are assumed **Missing at Random (MAR)** — missingness depends on observed variables (e.g., disease severity, era of data collection) but not on the missing values themselves. Sensitivity analyses address potential MNAR violations.

## Variable Triage

| Tier | Missingness | Action | Variables |
|---|---|---|---|
| Low | <10% | Impute (MICE) | Age, sex, HTN, DM, smoking, WFNS, HH, GCS, Fisher, IVH, ICH, intubated, prior meds, pre-morbid mRS, EVD, coiling/clipping, MAP, SBP, temperature |
| Moderate | 10–25% | Impute (MICE) | Aneurysm location/size (12.6%), time ictus-to-treatment (16.3%), WBC (23.6%), hematocrit (23.6%) |
| High | ~49% | **Drop** from primary model | CRP — imputing at ~50% is unreliable; include in sensitivity analysis only |
| Unavailable | 100% | Acknowledge as limitation | BMI, cardiovascular disease, heart rate, respiratory rate, potassium, pH/HCO3, thick SAH burden |

## Primary Method: MI + Within-method IPTW

Multiple imputation by chained equations (MICE), combined with propensity score estimation using the **within** (MI-te) approach.

### Workflow

```
For m = 1 to M imputed datasets:
  1. Impute missing covariates via MICE
  2. Fit propensity model (logistic regression) on imputed covariates
  3. Compute IPTW weights (with stabilization and trimming)
  4. Estimate treatment effect from weighted outcome model
  5. Store point estimate + SE
Combine M estimates via Rubin's rules → final estimate + CI
```

### MICE Specification

- **M = 30 imputations** (sufficient for up to ~25% missingness in retained variables)
- **Imputation models**: predictive mean matching (PMM) for continuous variables, logistic regression for binary, proportional odds for ordinal
- **Predictor set**: all covariates + treatment indicator + outcome
- **Including the outcome** in the imputation model is required for congeniality with the analysis model — it does not introduce circularity because the propensity model conditions only on covariates, never on the outcome

### Why "within" and not "across"

The across method (averaging propensity scores across imputed datasets) produces biased estimates and unreliable confidence intervals. The within method is the only approach consistently shown to be unbiased across simulation scenarios (Leyrat et al. 2019, Loh et al. 2024).

## Sensitivity Analyses

1. **Complete case analysis** — valid comparator even under MNAR if no effect modification; trades power for robustness
2. **MI + missing indicator** — for lab variables (WBC, hematocrit) where MNAR is plausible (not ordered → less severe patient), add a binary missingness indicator alongside the imputed value
3. **CRP included** — secondary model with CRP imputed via MICE to test whether its exclusion changes results

## References

- Leyrat C et al. Propensity score analysis with partially observed covariates: How should multiple imputation be used? *Stat Methods Med Res*. 2019;28(1):3–19.
- Loh WW et al. Multiple imputation for propensity score analysis with covariates missing at random. *Am J Epidemiol*. 2024;193(10):1470–1480.
- Granger E et al. Avoiding pitfalls when combining multiple imputation and propensity scores. *Stat Med*. 2019;38(26):5120–5132.
- Choi J et al. A comparison of different methods to handle missing data in the context of propensity score analysis. *Eur J Epidemiol*. 2019;34(1):23–36.
- Moons KGM et al. Using the outcome for imputation of missing predictor values was preferred. *J Clin Epidemiol*. 2006;59(10):1092–1101.
