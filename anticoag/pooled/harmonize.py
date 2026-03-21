"""
Pool MIMIC-IV + KSSG datasets for propensity analysis.

Reads both patient tables, harmonizes variables, applies consistent
exclusions, assigns groups, and outputs a pooled dataset ready for
IPTW propensity analysis of heparin vs enoxaparin after aSAH.

Outputs (in pooled/artifacts/):
  - pooled_cohort.csv       — one row per patient
  - harmonization_report.md — variable mapping, coverage, pooling summary
  - pooled_flowchart.png    — CONSORT-style flow from both sources
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
KSSG_DIR = os.path.join(SCRIPT_DIR, "..", "kssg", "artifacts")
MIMIC_DIR = os.path.join(SCRIPT_DIR, "..", "mimic4", "artifacts")

# Creatinine conversion: MIMIC uses mg/dL, threshold 150 µmol/L ≈ 1.70 mg/dL
CREATININE_THRESHOLD_MGDL = 150 / 88.4  # ≈ 1.697
HEPARIN_THERAPEUTIC_THRESHOLD = 10000  # UI/24h
ENOXAPARIN_THERAPEUTIC_THRESHOLD = 40  # mg/d


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_kssg_data():
    """Load KSSG cohort assignments and patient table."""
    cohort = pd.read_csv(os.path.join(KSSG_DIR, "cohort.csv"))
    pt = pd.read_csv(os.path.join(KSSG_DIR, "patient_table.csv"))

    # Merge group assignment into patient table
    pt = pt.merge(cohort[["pNr", "group"]], on="pNr", how="inner")

    print(f"KSSG: loaded {len(pt)} patients from cohort")
    print(f"  Groups: {pt['group'].value_counts().to_dict()}")
    return pt


def load_mimic4_data():
    """Load MIMIC-IV patient table."""
    pt = pd.read_csv(os.path.join(MIMIC_DIR, "patient_table.csv"))
    print(f"MIMIC-IV: loaded {len(pt)} patients")
    return pt


# ---------------------------------------------------------------------------
# MIMIC-IV exclusions & group assignment
# ---------------------------------------------------------------------------
def apply_mimic4_exclusions(df):
    """Apply exclusion criteria to MIMIC-IV patients.

    Returns (included_df, flow_dict) where flow_dict tracks counts.
    """
    flow = {"total": len(df)}

    # 1. Received both heparin and enoxaparin
    mask_both = df["received_both"] == 1
    flow["excluded_both"] = int(mask_both.sum())
    df = df[~mask_both].copy()

    # 2. Therapeutic heparin (>10,000 UI/24h)
    mask_therapeutic_hep = df["heparin_max_daily_dose"] > HEPARIN_THERAPEUTIC_THRESHOLD
    flow["excluded_therapeutic_heparin"] = int(mask_therapeutic_hep.sum())
    df = df[~mask_therapeutic_hep].copy()

    # 3. Therapeutic enoxaparin (>40 mg/d)
    mask_therapeutic_enox = df["enoxaparin_max_daily_dose"] > ENOXAPARIN_THERAPEUTIC_THRESHOLD
    flow["excluded_therapeutic_enoxaparin"] = int(mask_therapeutic_enox.sum())
    df = df[~mask_therapeutic_enox].copy()

    # 4. No anticoagulant received
    mask_no_ac = (df["heparin_ever"] == 0) & (df["enoxaparin_ever"] == 0)
    flow["excluded_no_anticoagulant"] = int(mask_no_ac.sum())
    df = df[~mask_no_ac].copy()

    # 5. Renal failure (creatinine > 1.7 mg/dL ≈ 150 µmol/L)
    mask_renal = df["admission_creatinine"] > CREATININE_THRESHOLD_MGDL
    flow["excluded_renal_failure"] = int(mask_renal.sum())
    df = df[~mask_renal].copy()

    flow["after_exclusions"] = len(df)
    print(f"MIMIC-IV exclusions: {flow['total']} → {flow['after_exclusions']}")
    for k, v in flow.items():
        if k.startswith("excluded_"):
            print(f"  {k}: {v}")

    return df, flow


def assign_mimic4_groups(df):
    """Assign heparin vs enoxaparin group for MIMIC-IV patients."""
    df = df.copy()
    df["group"] = np.where(df["enoxaparin_ever"] == 1, "enoxaparin", "heparin")
    print(f"MIMIC-IV groups: {df['group'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Variable harmonization
# ---------------------------------------------------------------------------
# Shared propensity model variables
SHARED_VARS = [
    "age", "sex", "hypertension", "diabetes", "smoking",
    "prior_anticoagulant", "prior_antiplatelet",
    "gcs_admission", "hunt_hess",
    "admission_map", "admission_temp",
    "admission_wbc", "admission_hemoglobin", "admission_glucose",
    "admission_sodium",
    "treatment_clipping", "treatment_coiling", "evd",
]

# Outcome variables
OUTCOME_VARS = [
    "dci", "rebleeding", "mortality",
    "icu_los_days", "hospital_los_days",
]

# KSSG-only variables (NaN for MIMIC)
KSSG_ONLY_VARS = [
    "admission_sbp",
    "fisher_score", "wfns", "ivh", "ich", "intubated_admission",
    "aneurysm_anterior", "aneurysm_size", "multiple_aneurysms",
    "time_ictus_to_treatment_days", "premorbid_mrs",
    "mrs_discharge", "mrs_1y", "admission_crp", "admission_hematocrit",
]

# MIMIC-only variables (NaN for KSSG)
MIMIC_ONLY_VARS = [
    "race", "renal_disease", "liver_disease",
    "admission_creatinine", "admission_potassium", "admission_platelets",
    "admission_inr", "admission_ptt", "admission_lactate",
    "admission_heart_rate",
]


def harmonize_kssg(df):
    """Rename/recode KSSG variables to common schema."""
    h = pd.DataFrame()
    h["patient_id"] = df["pNr"].astype(str)
    h["group"] = df["group"]

    # Shared variables
    h["age"] = df["age"]
    h["sex"] = df["sex"]
    h["hypertension"] = df["hypertension"]
    h["diabetes"] = df["diabetes"]
    # Smoking: 0=no, 1=yes, 2=ex → recode 1,2 → 1
    h["smoking"] = df["smoking"].apply(lambda x: 1 if x in (1, 2) else (0 if x == 0 else np.nan))
    h["prior_anticoagulant"] = df["prior_oac"]
    # Prior antiplatelet: max(ASS, Clopidogrel)
    h["prior_antiplatelet"] = df[["prior_ass", "prior_clopidogrel"]].max(axis=1)
    h["gcs_admission"] = df["gcs_admission"]
    h["hunt_hess"] = df["hunt_hess"]
    h["admission_map"] = df["admission_map"]
    h["admission_sbp"] = df["admission_sbp"]
    h["admission_temp"] = df["admission_temp"]
    h["admission_wbc"] = df["admission_wbc"]
    h["admission_hemoglobin"] = df["admission_hb"]  # rename
    h["admission_glucose"] = df["admission_glucose"]
    h["admission_sodium"] = df["admission_sodium"]
    h["treatment_clipping"] = df["treatment_clipping"]
    h["treatment_coiling"] = df["treatment_coiling"]
    h["evd"] = df["evd"]

    # Outcomes
    h["dci"] = df["dci"]
    h["rebleeding"] = df["rebleeding"]
    h["mortality"] = df["death"]
    h["icu_los_days"] = np.nan
    h["hospital_los_days"] = np.nan

    # KSSG-only
    for col in KSSG_ONLY_VARS:
        h[col] = df[col] if col in df.columns else np.nan

    # MIMIC-only → NaN
    for col in MIMIC_ONLY_VARS:
        h[col] = np.nan

    h["site"] = "kssg"
    return h


def harmonize_mimic4(df):
    """Rename/recode MIMIC-IV variables to common schema."""
    h = pd.DataFrame()
    h["patient_id"] = df["subject_id"].astype(str)
    h["group"] = df["group"]

    # Shared variables
    h["age"] = df["age"]
    h["sex"] = df["sex"]
    h["hypertension"] = df["hypertension"]
    h["diabetes"] = df["diabetes"]
    h["smoking"] = df["smoking_history"]  # already binary
    h["prior_anticoagulant"] = df["prior_anticoagulant"]
    h["prior_antiplatelet"] = df["prior_antiplatelet"]
    h["gcs_admission"] = df["gcs_admission"]
    h["hunt_hess"] = df["hunt_hess_derived"]  # rename
    h["admission_map"] = df["admission_map"]
    h["admission_sbp"] = df["admission_sbp"]
    h["admission_temp"] = df["admission_temp"]
    h["admission_wbc"] = df["admission_wbc"]
    h["admission_hemoglobin"] = df["admission_hemoglobin"]
    h["admission_glucose"] = df["admission_glucose"]
    h["admission_sodium"] = df["admission_sodium"]
    h["treatment_clipping"] = df["treatment_clipping"]
    h["treatment_coiling"] = df["treatment_coiling"]
    h["evd"] = df["evd_placement"]  # rename

    # Outcomes
    h["dci"] = df["dci_icd"]
    h["rebleeding"] = df["rebleeding_icd"]
    h["mortality"] = df["hospital_mortality"]
    h["icu_los_days"] = df["icu_los_days"]
    h["hospital_los_days"] = df["hospital_los_days"]

    # KSSG-only → NaN
    for col in KSSG_ONLY_VARS:
        h[col] = np.nan

    # MIMIC-only
    h["race"] = df["race"] if "race" in df.columns else np.nan
    h["renal_disease"] = df["renal_disease"] if "renal_disease" in df.columns else np.nan
    h["liver_disease"] = df["liver_disease"] if "liver_disease" in df.columns else np.nan
    h["admission_creatinine"] = df["admission_creatinine"]
    h["admission_potassium"] = df["admission_potassium"]
    h["admission_platelets"] = df["admission_platelets"]
    h["admission_inr"] = df["admission_inr"]
    h["admission_ptt"] = df["admission_ptt"]
    h["admission_lactate"] = df["admission_lactate"]
    h["admission_heart_rate"] = df["admission_heart_rate"]

    h["site"] = "mimic4"
    return h


# ---------------------------------------------------------------------------
# Build pooled cohort
# ---------------------------------------------------------------------------
def build_pooled_cohort():
    """Main pipeline: load, exclude, harmonize, pool."""
    # Load
    kssg = load_kssg_data()
    mimic = load_mimic4_data()

    # KSSG flow (exclusions already applied in cohort selection)
    kssg_flow = {
        "total_patient_table": 246,
        "in_cohort": len(kssg),
        "heparin": int((kssg["group"] == "heparin").sum()),
        "enoxaparin": int((kssg["group"] == "enoxaparin").sum()),
    }

    # MIMIC exclusions
    mimic, mimic_flow = apply_mimic4_exclusions(mimic)
    mimic = assign_mimic4_groups(mimic)
    mimic_flow["heparin"] = int((mimic["group"] == "heparin").sum())
    mimic_flow["enoxaparin"] = int((mimic["group"] == "enoxaparin").sum())

    # Harmonize
    h_kssg = harmonize_kssg(kssg)
    h_mimic = harmonize_mimic4(mimic)

    # Pool
    pooled = pd.concat([h_kssg, h_mimic], ignore_index=True)

    # Verify no duplicate patient_ids within site
    for site in ["kssg", "mimic4"]:
        site_df = pooled[pooled["site"] == site]
        n_dup = site_df["patient_id"].duplicated().sum()
        if n_dup > 0:
            print(f"WARNING: {n_dup} duplicate patient_ids in {site}")
        else:
            print(f"  {site}: no duplicate patient_ids")

    print(f"\nPooled cohort: {len(pooled)} patients")
    print(f"  Site: {pooled['site'].value_counts().to_dict()}")
    print(f"  Group: {pooled['group'].value_counts().to_dict()}")

    return pooled, kssg_flow, mimic_flow


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(pooled, kssg_flow, mimic_flow):
    """Generate harmonization_report.md."""
    all_vars = SHARED_VARS + OUTCOME_VARS + KSSG_ONLY_VARS + MIMIC_ONLY_VARS

    lines = [
        "# Harmonization Report: Pooled KSSG + MIMIC-IV Cohort",
        "",
        "## Cohort Summary",
        "",
        f"| Site | Total | Heparin | Enoxaparin |",
        f"|------|-------|---------|------------|",
        f"| KSSG | {kssg_flow['in_cohort']} | {kssg_flow['heparin']} | {kssg_flow['enoxaparin']} |",
        f"| MIMIC-IV | {mimic_flow['after_exclusions']} | {mimic_flow['heparin']} | {mimic_flow['enoxaparin']} |",
        f"| **Pooled** | **{len(pooled)}** | **{int((pooled['group']=='heparin').sum())}** | **{int((pooled['group']=='enoxaparin').sum())}** |",
        "",
        "## MIMIC-IV Exclusion Flow",
        "",
        f"- Starting patients: {mimic_flow['total']}",
        f"- Excluded (received both): {mimic_flow['excluded_both']}",
        f"- Excluded (therapeutic heparin >10,000 UI/24h): {mimic_flow['excluded_therapeutic_heparin']}",
        f"- Excluded (therapeutic enoxaparin >40 mg/d): {mimic_flow['excluded_therapeutic_enoxaparin']}",
        f"- Excluded (no anticoagulant): {mimic_flow['excluded_no_anticoagulant']}",
        f"- Excluded (renal failure, creatinine >{CREATININE_THRESHOLD_MGDL:.2f} mg/dL): {mimic_flow['excluded_renal_failure']}",
        f"- **After exclusions: {mimic_flow['after_exclusions']}**",
        "",
        "## Variable Harmonization",
        "",
        "### Shared variables (propensity model)",
        "",
        "| Variable | N available | % missing | KSSG avail | MIMIC avail |",
        "|----------|------------|-----------|------------|-------------|",
    ]

    kssg_mask = pooled["site"] == "kssg"
    mimic_mask = pooled["site"] == "mimic4"

    for var in SHARED_VARS:
        n_avail = int(pooled[var].notna().sum())
        pct_miss = 100 * (1 - n_avail / len(pooled))
        kssg_avail = int(pooled.loc[kssg_mask, var].notna().sum())
        mimic_avail = int(pooled.loc[mimic_mask, var].notna().sum())
        lines.append(f"| `{var}` | {n_avail} | {pct_miss:.1f}% | {kssg_avail} | {mimic_avail} |")

    lines += [
        "",
        "### Outcome variables",
        "",
        "| Variable | N available | % missing | Notes |",
        "|----------|------------|-----------|-------|",
    ]

    outcome_notes = {
        "dci": "KSSG: clinician-verified; MIMIC: ICD proxy",
        "rebleeding": "KSSG: clinician-verified; MIMIC: ICD proxy",
        "mortality": "KSSG: death; MIMIC: hospital_mortality",
        "icu_los_days": "MIMIC only",
        "hospital_los_days": "MIMIC only",
    }
    for var in OUTCOME_VARS:
        n_avail = int(pooled[var].notna().sum())
        pct_miss = 100 * (1 - n_avail / len(pooled))
        lines.append(f"| `{var}` | {n_avail} | {pct_miss:.1f}% | {outcome_notes.get(var, '')} |")

    lines += [
        "",
        "### Site-specific variables",
        "",
        "| Variable | Source | N available |",
        "|----------|--------|------------|",
    ]
    for var in KSSG_ONLY_VARS:
        n_avail = int(pooled[var].notna().sum())
        lines.append(f"| `{var}` | KSSG | {n_avail} |")
    for var in MIMIC_ONLY_VARS:
        n_avail = int(pooled[var].notna().sum())
        lines.append(f"| `{var}` | MIMIC-IV | {n_avail} |")

    # Binary variable validation
    lines += [
        "",
        "## Binary Variable Validation",
        "",
        "| Variable | Unique values | Valid (0/1/NaN only) |",
        "|----------|--------------|---------------------|",
    ]
    binary_vars = [
        "sex", "hypertension", "diabetes", "smoking",
        "prior_anticoagulant", "prior_antiplatelet",
        "treatment_clipping", "treatment_coiling", "evd",
        "dci", "rebleeding", "mortality",
    ]
    for var in binary_vars:
        unique_vals = sorted(pooled[var].dropna().unique())
        valid = all(v in (0, 1) for v in unique_vals)
        lines.append(f"| `{var}` | {unique_vals} | {'Yes' if valid else '**NO**'} |")

    # Propensity feasibility
    n_enox = int((pooled["group"] == "enoxaparin").sum())
    n_shared = len(SHARED_VARS)
    lines += [
        "",
        "## Propensity Model Feasibility",
        "",
        f"- Smaller group (enoxaparin): {n_enox}",
        f"- Number of covariates: {n_shared}",
        f"- Ratio (events per covariate): {n_enox / n_shared:.1f}",
        f"- Feasible (≥10 per covariate): {'Yes' if n_enox / n_shared >= 10 else 'No'}",
    ]

    report_path = os.path.join(ARTIFACTS_DIR, "harmonization_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {report_path}")


# ---------------------------------------------------------------------------
# Flowchart generation
# ---------------------------------------------------------------------------
def generate_flowchart(pooled, kssg_flow, mimic_flow):
    """Generate CONSORT-style pooled flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_kwargs = dict(
        boxstyle="round,pad=0.4", edgecolor="black", linewidth=1.5
    )

    def draw_box(x, y, text, width=4.5, height=0.7, color="#E8F4FD"):
        rect = mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            **box_kwargs, facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold", wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=1.5))

    def draw_exclusion_box(x, y, text, width=4.5, height=0.7):
        draw_box(x, y, text, width=width, height=height, color="#FDE8E8")

    # Title
    ax.text(7, 9.7, "Pooled Cohort Selection: KSSG + MIMIC-IV",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # === KSSG side (left) ===
    draw_box(3.5, 9.0, f"KSSG patient table\nn = {kssg_flow['total_patient_table']}")
    draw_arrow(3.5, 8.65, 3.5, 8.1)
    draw_box(3.5, 7.7, f"KSSG cohort (exclusions applied)\nn = {kssg_flow['in_cohort']}")
    draw_arrow(3.5, 7.35, 3.5, 6.8)

    # KSSG groups
    draw_box(2.0, 6.4, f"Heparin\nn = {kssg_flow['heparin']}", width=2.5, color="#D4EDDA")
    draw_box(5.0, 6.4, f"Enoxaparin\nn = {kssg_flow['enoxaparin']}", width=2.5, color="#D4EDDA")
    draw_arrow(3.5, 7.35, 2.0, 6.75)
    draw_arrow(3.5, 7.35, 5.0, 6.75)

    # === MIMIC-IV side (right) ===
    draw_box(10.5, 9.0, f"MIMIC-IV aSAH cohort\nn = {mimic_flow['total']}")
    draw_arrow(10.5, 8.65, 10.5, 8.1)

    # Exclusions
    y_exc = 7.7
    n_exc = (mimic_flow["excluded_both"] + mimic_flow["excluded_therapeutic_heparin"]
             + mimic_flow["excluded_therapeutic_enoxaparin"]
             + mimic_flow["excluded_no_anticoagulant"] + mimic_flow["excluded_renal_failure"])
    exc_text = (
        f"Excluded (n = {n_exc}):\n"
        f"  Both agents: {mimic_flow['excluded_both']}\n"
        f"  Therapeutic heparin: {mimic_flow['excluded_therapeutic_heparin']}\n"
        f"  Therapeutic enoxaparin: {mimic_flow['excluded_therapeutic_enoxaparin']}\n"
        f"  No anticoagulant: {mimic_flow['excluded_no_anticoagulant']}\n"
        f"  Renal failure: {mimic_flow['excluded_renal_failure']}"
    )
    draw_exclusion_box(10.5, y_exc, exc_text, height=1.8)
    draw_arrow(10.5, 8.65, 10.5, y_exc + 0.9)

    draw_arrow(10.5, y_exc - 0.9, 10.5, 6.0)
    draw_box(10.5, 5.6, f"After exclusions\nn = {mimic_flow['after_exclusions']}")

    # MIMIC groups
    draw_arrow(10.5, 5.25, 9.0, 4.75)
    draw_arrow(10.5, 5.25, 12.0, 4.75)
    draw_box(9.0, 4.4, f"Heparin\nn = {mimic_flow['heparin']}", width=2.5, color="#D4EDDA")
    draw_box(12.0, 4.4, f"Enoxaparin\nn = {mimic_flow['enoxaparin']}", width=2.5, color="#D4EDDA")

    # === Pooled ===
    n_hep_total = kssg_flow["heparin"] + mimic_flow["heparin"]
    n_enox_total = kssg_flow["enoxaparin"] + mimic_flow["enoxaparin"]
    draw_arrow(3.5, 6.05, 7.0, 2.5)
    draw_arrow(10.5, 4.05, 7.0, 2.5)

    draw_box(7.0, 2.0, f"Pooled cohort\nn = {len(pooled)}", width=5, height=0.8, color="#FFF3CD")
    draw_arrow(5.5, 1.6, 5.5, 0.8)
    draw_arrow(8.5, 1.6, 8.5, 0.8)
    draw_box(5.5, 0.4, f"Heparin\nn = {n_hep_total}", width=3, color="#D4EDDA")
    draw_box(8.5, 0.4, f"Enoxaparin\nn = {n_enox_total}", width=3, color="#D4EDDA")

    fig_path = os.path.join(ARTIFACTS_DIR, "pooled_flowchart.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    pooled, kssg_flow, mimic_flow = build_pooled_cohort()

    # Save pooled cohort
    out_path = os.path.join(ARTIFACTS_DIR, "pooled_cohort.csv")
    pooled.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(pooled)} rows x {len(pooled.columns)} cols)")

    # Generate report and flowchart
    generate_report(pooled, kssg_flow, mimic_flow)
    generate_flowchart(pooled, kssg_flow, mimic_flow)

    # Summary statistics
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    # Shared variable missingness
    print("\nShared variable missingness:")
    for var in SHARED_VARS:
        n = pooled[var].notna().sum()
        pct = 100 * n / len(pooled)
        flag = " *** >50% MISSING ***" if pct < 50 else ""
        print(f"  {var:<25} {n:>5}/{len(pooled)}  ({100-pct:>5.1f}% missing){flag}")

    # Binary validation
    print("\nBinary variable check:")
    for var in ["sex", "hypertension", "diabetes", "smoking",
                "prior_anticoagulant", "prior_antiplatelet",
                "treatment_clipping", "treatment_coiling", "evd",
                "dci", "rebleeding", "mortality"]:
        vals = pooled[var].dropna().unique()
        ok = all(v in (0, 1) for v in vals)
        print(f"  {var:<25} {'OK' if ok else 'INVALID: ' + str(sorted(vals))}")

    # Group counts
    print(f"\nFinal group counts:")
    for grp in ["heparin", "enoxaparin"]:
        n = int((pooled["group"] == grp).sum())
        print(f"  {grp}: {n}")

    # Site counts
    print(f"\nSite distribution:")
    for site in ["kssg", "mimic4"]:
        n = int((pooled["site"] == site).sum())
        print(f"  {site}: {n}")

    # Propensity feasibility
    n_enox = int((pooled["group"] == "enoxaparin").sum())
    n_covariates = len(SHARED_VARS)
    ratio = n_enox / n_covariates
    print(f"\nPropensity feasibility: {n_enox} enoxaparin / {n_covariates} covariates = {ratio:.1f} per covariate")
    print(f"  ≥10 per covariate: {'YES' if ratio >= 10 else 'NO'}")


if __name__ == "__main__":
    main()
