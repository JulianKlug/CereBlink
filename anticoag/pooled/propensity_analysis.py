"""
IPTW propensity analysis: heparin vs enoxaparin after aSAH.

Reads the pooled cohort (KSSG + MIMIC-IV), estimates propensity scores,
computes stabilized IPTW weights, and produces:
  - Table 1 (unweighted and weighted)
  - Treatment effect estimates (OR, 95% CI, p) for binary outcomes
  - Love plot, PS distribution, forest plot
  - Narrative report (Markdown)

Outputs go to pooled/artifacts/.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

CONTINUOUS_COVARIATES = [
    "age", "gcs_admission", "hunt_hess", "admission_map",
    "admission_temp", "admission_wbc", "admission_hemoglobin",
    "admission_glucose", "admission_sodium",
]

BINARY_COVARIATES = [
    "sex", "hypertension", "diabetes", "smoking",
    "prior_anticoagulant", "prior_antiplatelet",
    "treatment_clipping", "treatment_coiling", "evd",
]

ALL_COVARIATES = CONTINUOUS_COVARIATES + BINARY_COVARIATES  # 18 shared

BINARY_OUTCOMES = ["dci", "rebleeding", "mortality"]

# Display labels for covariates
COVARIATE_LABELS = {
    "age": "Age",
    "gcs_admission": "GCS at admission",
    "hunt_hess": "Hunt-Hess grade",
    "admission_map": "MAP (mmHg)",
    "admission_temp": "Temperature (°C)",
    "admission_wbc": "WBC (×10⁹/L)",
    "admission_hemoglobin": "Hemoglobin (g/dL)",
    "admission_glucose": "Glucose (mmol/L)",
    "admission_sodium": "Sodium (mmol/L)",
    "sex": "Male sex",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "smoking": "Smoking history",
    "prior_anticoagulant": "Prior anticoagulant",
    "prior_antiplatelet": "Prior antiplatelet",
    "treatment_clipping": "Surgical clipping",
    "treatment_coiling": "Endovascular coiling",
    "evd": "EVD placement",
    "site_mimic4": "Site (MIMIC-IV)",
}

OUTCOME_LABELS = {
    "dci": "Delayed cerebral ischemia",
    "rebleeding": "Rebleeding",
    "mortality": "In-hospital mortality",
}


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------
def load_cohort():
    """Load pooled cohort and encode treatment as binary."""
    path = os.path.join(ARTIFACTS_DIR, "pooled_cohort.csv")
    df = pd.read_csv(path)
    df["treatment"] = (df["group"] == "enoxaparin").astype(int)
    df["site_mimic4"] = (df["site"] == "mimic4").astype(int)
    print(f"Loaded {len(df)} patients  "
          f"(heparin={int((df['treatment']==0).sum())}, "
          f"enoxaparin={int((df['treatment']==1).sum())})")
    return df


def impute_missing(df):
    """Simple imputation: median (continuous), mode (binary). Returns df copy."""
    df = df.copy()
    impute_log = []
    for var in CONTINUOUS_COVARIATES:
        n_miss = int(df[var].isna().sum())
        if n_miss > 0:
            med = df[var].median()
            df[var] = df[var].fillna(med)
            impute_log.append(f"  {var}: {n_miss} imputed with median={med:.2f}")
    for var in BINARY_COVARIATES:
        n_miss = int(df[var].isna().sum())
        if n_miss > 0:
            mode_val = df[var].mode().iloc[0]
            df[var] = df[var].fillna(mode_val)
            impute_log.append(f"  {var}: {n_miss} imputed with mode={mode_val:.0f}")
    if impute_log:
        print("Imputation:")
        for line in impute_log:
            print(line)
    else:
        print("No missing values in covariates — no imputation needed.")
    return df, impute_log


# ---------------------------------------------------------------------------
# Propensity score estimation
# ---------------------------------------------------------------------------
def estimate_propensity_scores(df):
    """Fit logistic regression for P(enoxaparin | covariates + site)."""
    model_vars = ALL_COVARIATES + ["site_mimic4"]
    X = sm.add_constant(df[model_vars].astype(float))
    y = df["treatment"].astype(float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.Logit(y, X).fit(disp=0, maxiter=300)

    converged = model.mle_retvals["converged"]
    ps = model.predict(X)
    auc = roc_auc_score(y, ps)

    print(f"\nPropensity model: converged={converged}, "
          f"c-statistic (AUC)={auc:.3f}")
    if not converged:
        print("  WARNING: model did not converge — coefficients may be unreliable")
    if caught:
        for w in caught:
            print(f"  Warning: {w.message}")

    # Print site coefficient
    site_idx = list(X.columns).index("site_mimic4")
    print(f"  site_mimic4 coefficient: {model.params.iloc[site_idx]:.3f} "
          f"(SE={model.bse.iloc[site_idx]:.3f})")

    df = df.copy()
    df["ps"] = ps
    return df, model, auc, converged


# ---------------------------------------------------------------------------
# IPTW weights
# ---------------------------------------------------------------------------
def compute_stabilized_iptw(df, trim_lower=0.01, trim_upper=0.99):
    """Compute stabilized IPTW weights with propensity score trimming."""
    df = df.copy()
    ps_raw = df["ps"].copy()

    # Trim propensity scores
    n_trimmed_low = int((ps_raw < trim_lower).sum())
    n_trimmed_high = int((ps_raw > trim_upper).sum())
    df["ps_trimmed"] = ps_raw.clip(trim_lower, trim_upper)
    print(f"\nPS trimming [{trim_lower}, {trim_upper}]: "
          f"{n_trimmed_low} trimmed low, {n_trimmed_high} trimmed high")

    # Stabilized weights
    p_treat = df["treatment"].mean()  # marginal P(A=1)
    ps_t = df["ps_trimmed"]

    treated = df["treatment"] == 1
    df["iptw"] = np.where(
        treated,
        p_treat / ps_t,
        (1 - p_treat) / (1 - ps_t),
    )

    # Report weight diagnostics per group
    for grp, label in [(1, "enoxaparin"), (0, "heparin")]:
        mask = df["treatment"] == grp
        w = df.loc[mask, "iptw"]
        ess = w.sum() ** 2 / (w ** 2).sum()
        print(f"  {label}: n={int(mask.sum())}, "
              f"weight range=[{w.min():.2f}, {w.max():.2f}], "
              f"mean={w.mean():.2f}, ESS={ess:.1f}")

    return df, {
        "n_trimmed_low": n_trimmed_low,
        "n_trimmed_high": n_trimmed_high,
        "p_treat": p_treat,
        "ess_treated": (df.loc[treated, "iptw"].sum() ** 2
                        / (df.loc[treated, "iptw"] ** 2).sum()),
        "ess_control": (df.loc[~treated, "iptw"].sum() ** 2
                        / (df.loc[~treated, "iptw"] ** 2).sum()),
        "max_weight": float(df["iptw"].max()),
    }


# ---------------------------------------------------------------------------
# Standardized mean differences
# ---------------------------------------------------------------------------
def compute_smd(df, weighted=False):
    """Compute SMD for all covariates + site_mimic4."""
    vars_to_check = ALL_COVARIATES + ["site_mimic4"]
    results = {}

    treated = df["treatment"] == 1
    w = df["iptw"] if weighted else pd.Series(1.0, index=df.index)

    for var in vars_to_check:
        x = df[var].astype(float)
        w1 = w[treated]
        w0 = w[~treated]
        x1 = x[treated]
        x0 = x[~treated]

        # Weighted means
        mean1 = np.average(x1, weights=w1)
        mean0 = np.average(x0, weights=w0)

        if var in BINARY_COVARIATES or var == "site_mimic4":
            # Binary SMD
            p1, p0 = mean1, mean0
            denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
            smd = (p1 - p0) / denom if denom > 0 else 0.0
        else:
            # Continuous SMD: weighted variances
            var1 = np.average((x1 - mean1) ** 2, weights=w1)
            var0 = np.average((x0 - mean0) ** 2, weights=w0)
            denom = np.sqrt((var1 + var0) / 2)
            smd = (mean1 - mean0) / denom if denom > 0 else 0.0

        results[var] = {
            "smd": smd,
            "abs_smd": abs(smd),
            "mean_treated": mean1,
            "mean_control": mean0,
        }

    return results


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def generate_table1(df, weighted=False):
    """Generate baseline characteristics table."""
    treated = df["treatment"] == 1
    w = df["iptw"] if weighted else pd.Series(1.0, index=df.index)

    smd_dict = compute_smd(df, weighted=weighted)
    rows = []

    for var in ALL_COVARIATES + ["site_mimic4"]:
        x = df[var].astype(float)
        w1, w0 = w[treated], w[~treated]
        x1, x0 = x[treated], x[~treated]

        mean1 = np.average(x1, weights=w1)
        mean0 = np.average(x0, weights=w0)

        label = COVARIATE_LABELS.get(var, var)
        smd_val = smd_dict[var]["abs_smd"]

        if var in BINARY_COVARIATES or var == "site_mimic4":
            # Weighted counts for display: sum of weights * proportion
            n_eff_1 = w1.sum()
            n_eff_0 = w0.sum()
            rows.append({
                "Variable": label,
                "Enoxaparin": f"{mean1 * 100:.1f}%",
                "Heparin": f"{mean0 * 100:.1f}%",
                "SMD": f"{smd_val:.3f}",
            })
        else:
            sd1 = np.sqrt(np.average((x1 - mean1) ** 2, weights=w1))
            sd0 = np.sqrt(np.average((x0 - mean0) ** 2, weights=w0))
            rows.append({
                "Variable": label,
                "Enoxaparin": f"{mean1:.1f} ({sd1:.1f})",
                "Heparin": f"{mean0:.1f} ({sd0:.1f})",
                "SMD": f"{smd_val:.3f}",
            })

    table = pd.DataFrame(rows)
    return table, smd_dict


# ---------------------------------------------------------------------------
# Outcome analysis
# ---------------------------------------------------------------------------
def analyze_binary_outcome(df, outcome):
    """IPTW-weighted logistic regression for a binary outcome."""
    y = df[outcome].astype(float)
    valid = y.notna()
    y = y[valid]
    X = sm.add_constant(df.loc[valid, "treatment"].astype(float))
    w = df.loc[valid, "iptw"]

    model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w)
    result = model.fit(cov_type="HC1")

    coef = result.params.iloc[1]
    ci = result.conf_int().iloc[1]
    p_val = result.pvalues.iloc[1]

    or_est = np.exp(coef)
    or_lower = np.exp(ci[0])
    or_upper = np.exp(ci[1])

    # Unweighted event rates
    treated = df.loc[valid, "treatment"] == 1
    rate_treated = y[treated].mean()
    rate_control = y[~treated].mean()

    return {
        "outcome": outcome,
        "label": OUTCOME_LABELS[outcome],
        "or": or_est,
        "or_lower": or_lower,
        "or_upper": or_upper,
        "p_value": p_val,
        "coef": coef,
        "se": result.bse.iloc[1],
        "n_events": int(y.sum()),
        "n_total": int(valid.sum()),
        "rate_enoxaparin": rate_treated,
        "rate_heparin": rate_control,
    }


def run_sensitivity_interaction(df, outcome):
    """Test treatment × site interaction in IPTW model."""
    y = df[outcome].astype(float)
    valid = y.notna()
    y = y[valid]
    treat = df.loc[valid, "treatment"].astype(float)
    site = df.loc[valid, "site_mimic4"].astype(float)
    interaction = treat * site

    X = sm.add_constant(pd.DataFrame({
        "treatment": treat,
        "site_mimic4": site,
        "treatment_x_site": interaction,
    }))
    w = df.loc[valid, "iptw"]

    model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w)
    result = model.fit(cov_type="HC1")

    ix_idx = list(X.columns).index("treatment_x_site")
    return {
        "outcome": outcome,
        "interaction_coef": result.params.iloc[ix_idx],
        "interaction_se": result.bse.iloc[ix_idx],
        "interaction_p": result.pvalues.iloc[ix_idx],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_love_plot(smd_before, smd_after):
    """Love plot: SMD before and after IPTW."""
    vars_ordered = ALL_COVARIATES + ["site_mimic4"]
    labels = [COVARIATE_LABELS.get(v, v) for v in vars_ordered]

    smd_b = [smd_before[v]["abs_smd"] for v in vars_ordered]
    smd_a = [smd_after[v]["abs_smd"] for v in vars_ordered]

    y_pos = np.arange(len(vars_ordered))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(smd_b, y_pos, marker="o", color="#d62728", s=60, label="Unweighted", zorder=3)
    ax.scatter(smd_a, y_pos, marker="s", color="#1f77b4", s=60, label="IPTW-weighted", zorder=3)

    # Connect before/after with lines
    for i in range(len(vars_ordered)):
        ax.plot([smd_b[i], smd_a[i]], [y_pos[i], y_pos[i]],
                color="gray", linewidth=0.8, zorder=2)

    ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="SMD = 0.1")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate Balance: Before and After IPTW")
    ax.legend(loc="lower right")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(ARTIFACTS_DIR, "love_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def plot_ps_distribution(df):
    """Propensity score distribution by treatment group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(df.loc[df["treatment"] == 0, "ps"], bins=bins, alpha=0.6,
            color="#1f77b4", label="Heparin", density=True)
    ax.hist(df.loc[df["treatment"] == 1, "ps"], bins=bins, alpha=0.6,
            color="#d62728", label="Enoxaparin", density=True)
    ax.axvline(0.01, color="black", linestyle=":", linewidth=1, label="Trim thresholds")
    ax.axvline(0.99, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution")
    ax.legend()

    # Mirrored histogram (treated up, control down)
    ax = axes[1]
    ps_treat = df.loc[df["treatment"] == 1, "ps"]
    ps_ctrl = df.loc[df["treatment"] == 0, "ps"]
    ax.hist(ps_treat, bins=bins, alpha=0.7, color="#d62728",
            label="Enoxaparin", density=True)
    ax.hist(ps_ctrl, bins=bins, alpha=0.7, color="#1f77b4",
            label="Heparin", density=True,
            weights=-np.ones(len(ps_ctrl)) / len(ps_ctrl) * len(bins))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0.01, color="black", linestyle=":", linewidth=1)
    ax.axvline(0.99, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density (mirrored)")
    ax.set_title("Mirrored PS Distribution")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "ps_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def plot_forest(results):
    """Forest plot of treatment effects (OR with 95% CI)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [r["label"] for r in results]
    ors = [r["or"] for r in results]
    lowers = [r["or_lower"] for r in results]
    uppers = [r["or_upper"] for r in results]
    pvals = [r["p_value"] for r in results]

    y_pos = np.arange(len(results))

    for i in range(len(results)):
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]],
                color="#1f77b4", linewidth=2, zorder=2)
        ax.scatter([ors[i]], [y_pos[i]], color="#1f77b4", s=80, zorder=3)

        # Annotate with OR (CI) p
        p_str = f"p={pvals[i]:.3f}" if pvals[i] >= 0.001 else f"p<0.001"
        ax.text(max(uppers) * 1.15, y_pos[i],
                f"OR {ors[i]:.2f} ({lowers[i]:.2f}–{uppers[i]:.2f})  {p_str}",
                va="center", fontsize=9)

    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Odds Ratio (enoxaparin vs heparin)")
    ax.set_title("IPTW-Adjusted Treatment Effects")
    ax.set_xscale("log")

    # Extend x-axis for annotations
    ax.set_xlim(left=min(lowers) * 0.5, right=max(uppers) * 4)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(ARTIFACTS_DIR, "forest_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(df, model_info, weight_info, smd_before, smd_after,
                    table1_uw, table1_w, outcome_results, sensitivity_results):
    """Generate Markdown analysis report."""
    auc, converged = model_info

    # Identify residual imbalance
    imbalanced = [v for v in ALL_COVARIATES + ["site_mimic4"]
                  if smd_after[v]["abs_smd"] >= 0.1]

    lines = [
        "# Propensity Analysis Report: Heparin vs Enoxaparin after aSAH",
        "",
        "## Methods",
        "",
        "### Study design",
        "Inverse probability of treatment weighting (IPTW) was used to compare "
        "heparin vs enoxaparin at thromboprophylactic doses in a pooled cohort "
        "of aSAH patients from KSSG and MIMIC-IV.",
        "",
        "### Propensity score model",
        f"- Logistic regression with {len(ALL_COVARIATES)} clinical covariates + site indicator (19 total)",
        f"- Model convergence: {'Yes' if converged else '**No — interpret with caution**'}",
        f"- C-statistic (AUC): {auc:.3f}",
        "",
        "### IPTW weights",
        "- Stabilized weights: w = P(A) / PS for treated, w = P(1-A) / (1-PS) for controls",
        f"- Propensity score trimming: [0.01, 0.99]",
        f"  - Trimmed low: {weight_info['n_trimmed_low']}",
        f"  - Trimmed high: {weight_info['n_trimmed_high']}",
        f"- Maximum weight: {weight_info['max_weight']:.2f}",
        f"- Effective sample size — enoxaparin: {weight_info['ess_treated']:.1f}, "
        f"heparin: {weight_info['ess_control']:.1f}",
        "",
        "### Outcome analysis",
        "- IPTW-weighted generalized linear model (binomial family, logit link)",
        "- Robust (HC1) standard errors",
        "- Results reported as odds ratios (OR) with 95% confidence intervals",
        "",
        "## Results",
        "",
        "### Covariate balance",
        "",
    ]

    if imbalanced:
        lines.append(f"**{len(imbalanced)} covariate(s) with residual SMD ≥ 0.1 after weighting:**")
        for v in imbalanced:
            label = COVARIATE_LABELS.get(v, v)
            lines.append(f"- {label}: SMD = {smd_after[v]['abs_smd']:.3f}")
        lines.append("")
    else:
        lines.append("All covariates achieved SMD < 0.1 after IPTW weighting.")
        lines.append("")

    # Table 1 unweighted
    lines += [
        "### Table 1: Baseline Characteristics (Unweighted)",
        "",
        table1_uw.to_markdown(index=False),
        "",
        "### Table 1: Baseline Characteristics (IPTW-Weighted)",
        "",
        table1_w.to_markdown(index=False),
        "",
        "### Treatment Effects",
        "",
        "| Outcome | Enoxaparin rate | Heparin rate | OR (95% CI) | p-value |",
        "|---------|----------------|-------------|-------------|---------|",
    ]
    for r in outcome_results:
        p_str = f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "<0.001"
        lines.append(
            f"| {r['label']} | {r['rate_enoxaparin']:.1%} | {r['rate_heparin']:.1%} | "
            f"{r['or']:.2f} ({r['or_lower']:.2f}–{r['or_upper']:.2f}) | {p_str} |"
        )

    # Sensitivity
    lines += [
        "",
        "### Sensitivity: Treatment × Site Interaction",
        "",
        "| Outcome | Interaction coefficient | SE | p-value |",
        "|---------|----------------------|------|---------|",
    ]
    for s in sensitivity_results:
        p_str = f"{s['interaction_p']:.3f}" if s["interaction_p"] >= 0.001 else "<0.001"
        lines.append(
            f"| {OUTCOME_LABELS[s['outcome']]} | {s['interaction_coef']:.3f} | "
            f"{s['interaction_se']:.3f} | {p_str} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "1. **Site-treatment confounding**: KSSG patients predominantly received "
        "enoxaparin (172/179) while MIMIC-IV patients predominantly received "
        "heparin (868/870). Despite including site as a covariate, this near-perfect "
        "separation limits the propensity model's ability to fully balance groups. "
        "Treatment effects may reflect residual site-level confounding.",
        "",
        "2. **ICD-based outcome proxies**: DCI and rebleeding in MIMIC-IV are "
        "identified via ICD codes, which have lower sensitivity compared to "
        "clinician-verified diagnoses in KSSG. This may introduce differential "
        "outcome misclassification.",
        "",
        "3. **Effective sample size reduction**: Stabilized IPTW with extreme "
        f"propensity scores reduces the effective sample size substantially "
        f"(ESS: enoxaparin={weight_info['ess_treated']:.1f}, "
        f"heparin={weight_info['ess_control']:.1f}), "
        "widening confidence intervals.",
        "",
        "4. **Unmeasured confounders**: Institutional practice patterns, "
        "local protocols, and temporal trends may differ between sites in ways "
        "not captured by measured covariates.",
        "",
    ]

    path = os.path.join(ARTIFACTS_DIR, "propensity_analysis_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load and impute
    df = load_cohort()
    df, impute_log = impute_missing(df)

    # 2. Propensity scores
    df, ps_model, auc, converged = estimate_propensity_scores(df)

    # 3. IPTW weights
    df, weight_info = compute_stabilized_iptw(df)

    # 4. Balance assessment
    smd_before = compute_smd(df, weighted=False)
    smd_after = compute_smd(df, weighted=True)

    print("\nCovariate balance (|SMD|):")
    print(f"  {'Variable':<28} {'Before':>8} {'After':>8}")
    for var in ALL_COVARIATES + ["site_mimic4"]:
        label = COVARIATE_LABELS.get(var, var)
        b = smd_before[var]["abs_smd"]
        a = smd_after[var]["abs_smd"]
        flag = " ***" if a >= 0.1 else ""
        print(f"  {label:<28} {b:>8.3f} {a:>8.3f}{flag}")

    # 5. Table 1
    table1_uw, _ = generate_table1(df, weighted=False)
    table1_w, _ = generate_table1(df, weighted=True)

    table1_uw.to_csv(os.path.join(ARTIFACTS_DIR, "table1_unweighted.csv"), index=False)
    table1_w.to_csv(os.path.join(ARTIFACTS_DIR, "table1_weighted.csv"), index=False)
    print(f"\nSaved table1_unweighted.csv and table1_weighted.csv")

    # 6. Outcome analyses
    outcome_results = []
    for outcome in BINARY_OUTCOMES:
        r = analyze_binary_outcome(df, outcome)
        outcome_results.append(r)
        p_str = f"p={r['p_value']:.3f}" if r["p_value"] >= 0.001 else "p<0.001"
        print(f"\n{r['label']}:")
        print(f"  Events: {r['n_events']}/{r['n_total']}")
        print(f"  Enoxaparin: {r['rate_enoxaparin']:.1%}, Heparin: {r['rate_heparin']:.1%}")
        print(f"  OR={r['or']:.2f} (95% CI {r['or_lower']:.2f}–{r['or_upper']:.2f}), {p_str}")

    results_df = pd.DataFrame(outcome_results)
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "propensity_results.csv"), index=False)
    print(f"\nSaved propensity_results.csv")

    # 7. Sensitivity: treatment × site interaction
    sensitivity_results = []
    print("\nSensitivity — Treatment × Site interaction:")
    for outcome in BINARY_OUTCOMES:
        s = run_sensitivity_interaction(df, outcome)
        sensitivity_results.append(s)
        p_str = f"p={s['interaction_p']:.3f}" if s["interaction_p"] >= 0.001 else "p<0.001"
        print(f"  {OUTCOME_LABELS[outcome]}: coef={s['interaction_coef']:.3f}, {p_str}")

    # 8. Figures
    print()
    plot_love_plot(smd_before, smd_after)
    plot_ps_distribution(df)
    plot_forest(outcome_results)

    # 9. Report
    generate_report(df, (auc, converged), weight_info,
                    smd_before, smd_after,
                    table1_uw, table1_w,
                    outcome_results, sensitivity_results)

    # Verification summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Propensity model converged: {converged}")
    print(f"  C-statistic: {auc:.3f}")
    n_imbalanced = sum(1 for v in ALL_COVARIATES + ["site_mimic4"]
                       if smd_after[v]["abs_smd"] >= 0.1)
    print(f"  Covariates with SMD ≥ 0.1 after IPTW: {n_imbalanced}")
    print(f"  ESS (enoxaparin): {weight_info['ess_treated']:.1f}")
    print(f"  ESS (heparin): {weight_info['ess_control']:.1f}")
    for r in outcome_results:
        print(f"  {r['label']}: OR={r['or']:.2f} "
              f"({r['or_lower']:.2f}–{r['or_upper']:.2f}), p={r['p_value']:.3f}")

    artifacts = [
        "table1_unweighted.csv", "table1_weighted.csv",
        "propensity_results.csv", "love_plot.png",
        "ps_distribution.png", "forest_plot.png",
        "propensity_analysis_report.md",
    ]
    print(f"\nOutputs ({len(artifacts)} files):")
    for a in artifacts:
        path = os.path.join(ARTIFACTS_DIR, a)
        exists = os.path.exists(path)
        print(f"  {'OK' if exists else 'MISSING'}: {a}")


if __name__ == "__main__":
    main()
