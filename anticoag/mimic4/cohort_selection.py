"""
Step 1: MIMIC-IV cohort selection for aSAH anticoagulation study.

Identifies aneurysmal subarachnoid hemorrhage patients from MIMIC-IV,
applies exclusion criteria, and flags aneurysm confirmation status.

Outputs:
  - cohort.csv             One row per included patient
  - cohort_flow.csv        Exclusion counts at each step
  - cohort_flowchart.png   CONSORT flow diagram
"""

import os
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOSP_DIR = "/mnt/hdd1/datasets/mimiciv_3.1/hosp"
ICU_DIR = "/mnt/hdd1/datasets/mimiciv_3.1/icu"
OUTPUT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

# ---------------------------------------------------------------------------
# ICD codes (dual ICD-9 + ICD-10)
# ---------------------------------------------------------------------------
SAH_ICD9 = "430"
SAH_ICD10_PREFIX = "I60"

TRAUMA_ICD9_PREFIXES = [
    "800", "801", "802", "803", "804",  # skull fractures
    "850", "851", "852", "853", "854",  # head injury
]
TRAUMA_ICD10_PREFIXES = [
    "S02",   # skull fractures
    "S06",   # TBI
    "S066",  # traumatic SAH
]

# Aneurysm confirmation codes
ANEURYSM_DX_ICD9 = "4373"
ANEURYSM_DX_ICD10 = "I671"

CLIP_ICD9 = ["3951"]
CLIP_ICD10 = ["03VG0CZ", "03VG0ZZ", "03LG0CZ", "03LG0ZZ"]

COIL_ICD9 = ["3972", "3975", "3976"]
COIL_ICD10 = ["03VG3DZ", "03VG3BZ", "03LG3DZ", "03LG3BZ"]

ANGIOGRAPHY_ICD9 = "8841"
ANGIOGRAPHY_ICD10_PREFIX = "B31R"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tables() -> dict:
    """Load required MIMIC-IV tables."""
    tables = {}
    files = {
        "diagnoses": (HOSP_DIR, "diagnoses_icd.csv.gz"),
        "procedures": (HOSP_DIR, "procedures_icd.csv.gz"),
        "patients": (HOSP_DIR, "patients.csv.gz"),
        "admissions": (HOSP_DIR, "admissions.csv.gz"),
        "icustays": (ICU_DIR, "icustays.csv.gz"),
    }
    for key, (base_dir, fname) in files.items():
        path = os.path.join(base_dir, fname)
        df = pd.read_csv(path, compression="gzip")
        # Ensure icd_code is string for consistent matching
        if "icd_code" in df.columns:
            df["icd_code"] = df["icd_code"].astype(str).str.strip()
        tables[key] = df
        print(f"  Loaded {fname}: {len(df):,} rows")

    # Parse date columns
    tables["patients"]["dod"] = pd.to_datetime(
        tables["patients"]["dod"], errors="coerce"
    )
    tables["admissions"]["admittime"] = pd.to_datetime(
        tables["admissions"]["admittime"], errors="coerce"
    )
    tables["admissions"]["dischtime"] = pd.to_datetime(
        tables["admissions"]["dischtime"], errors="coerce"
    )
    tables["icustays"]["intime"] = pd.to_datetime(
        tables["icustays"]["intime"], errors="coerce"
    )
    tables["icustays"]["outtime"] = pd.to_datetime(
        tables["icustays"]["outtime"], errors="coerce"
    )

    return tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _code_matches_any(code, version, icd9_prefixes, icd10_prefixes):
    """Check if code matches any prefix for its ICD version."""
    if version == 9:
        return any(code.startswith(p) for p in icd9_prefixes)
    elif version == 10:
        return any(code.startswith(p) for p in icd10_prefixes)
    return False


def has_trauma_code(diag_df: pd.DataFrame, hadm_ids: set) -> set:
    """Return hadm_ids that have a concurrent traumatic brain injury code."""
    trauma = diag_df[diag_df["hadm_id"].isin(hadm_ids)].copy()
    mask = trauma.apply(
        lambda r: _code_matches_any(
            r["icd_code"], r["icd_version"],
            TRAUMA_ICD9_PREFIXES, TRAUMA_ICD10_PREFIXES
        ), axis=1
    )
    return set(trauma.loc[mask, "hadm_id"].unique())


def compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age at admission for MIMIC-IV.

    anchor_age is the patient's age at anchor_year;
    actual age = anchor_age + (admittime.year - anchor_year).
    Cap at 91 (MIMIC-IV privacy threshold).
    """
    merged = admissions[["subject_id", "hadm_id", "admittime"]].merge(
        patients[["subject_id", "anchor_age", "anchor_year"]],
        on="subject_id", how="left"
    )
    merged["age"] = (
        merged["anchor_age"] + (merged["admittime"].dt.year - merged["anchor_year"])
    )
    # Cap age at 91 for MIMIC-IV privacy-shifted patients
    merged.loc[merged["age"] > 91, "age"] = 91.0
    return merged[["subject_id", "hadm_id", "age"]]


def get_aneurysm_flags(
    diag_df: pd.DataFrame, proc_df: pd.DataFrame, hadm_ids: set
) -> pd.DataFrame:
    """
    Flag admissions with evidence of aneurysmal etiology.

    aneurysm_confirmed = 1 if clipping, coiling, or aneurysm dx present.
    angiography = 1 if cerebral angiography present.
    """
    # Procedure-based flags (dual ICD)
    proc_sub = proc_df[proc_df["hadm_id"].isin(hadm_ids)].copy()

    # Treatment codes (clipping + coiling)
    clip_all = set(CLIP_ICD9 + CLIP_ICD10)
    coil_all = set(COIL_ICD9 + COIL_ICD10)
    treatment_codes = clip_all | coil_all

    # Exact match for procedure codes
    has_treatment = set(
        proc_sub.loc[proc_sub["icd_code"].isin(treatment_codes), "hadm_id"].unique()
    )
    # Also check prefix matching for ICD-10 procedure codes
    proc_icd10 = proc_sub[proc_sub["icd_version"] == 10]
    clip_prefix_mask = proc_icd10["icd_code"].apply(
        lambda c: c.startswith("03VG0") or c.startswith("03LG0")
    )
    coil_prefix_mask = proc_icd10["icd_code"].apply(
        lambda c: c.startswith("03VG3") or c.startswith("03LG3")
    )
    has_treatment.update(
        proc_icd10.loc[clip_prefix_mask | coil_prefix_mask, "hadm_id"].unique()
    )

    # Angiography
    has_angiography = set(
        proc_sub.loc[
            (proc_sub["icd_code"] == ANGIOGRAPHY_ICD9) &
            (proc_sub["icd_version"] == 9),
            "hadm_id"
        ].unique()
    )
    # ICD-10 angiography prefix
    angio_icd10 = proc_icd10[
        proc_icd10["icd_code"].str.startswith(ANGIOGRAPHY_ICD10_PREFIX)
    ]
    has_angiography.update(angio_icd10["hadm_id"].unique())

    # Diagnosis-based flag (cerebral aneurysm)
    diag_sub = diag_df[diag_df["hadm_id"].isin(hadm_ids)].copy()
    has_aneurysm_dx = set(
        diag_sub.loc[
            ((diag_sub["icd_code"] == ANEURYSM_DX_ICD9) & (diag_sub["icd_version"] == 9)) |
            ((diag_sub["icd_code"] == ANEURYSM_DX_ICD10) & (diag_sub["icd_version"] == 10)),
            "hadm_id"
        ].unique()
    )

    rows = []
    for hadm_id in hadm_ids:
        confirmed = 1 if (hadm_id in has_treatment or hadm_id in has_aneurysm_dx) else 0
        angio = 1 if hadm_id in has_angiography else 0
        rows.append({
            "hadm_id": hadm_id,
            "aneurysm_confirmed": confirmed,
            "angiography": angio,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cohort selection pipeline
# ---------------------------------------------------------------------------
def select_sah_cohort(tables: dict) -> tuple:
    """
    Apply sequential inclusion/exclusion criteria.

    Returns (cohort_df, flow_steps).
    """
    diag = tables["diagnoses"]
    proc = tables["procedures"]
    patients = tables["patients"]
    admissions = tables["admissions"]
    icustays = tables["icustays"]

    flow = []

    # --- Step 0: All admissions with SAH (ICD-9 430 or ICD-10 I60x) ---
    sah_mask = (
        ((diag["icd_code"] == SAH_ICD9) & (diag["icd_version"] == 9)) |
        ((diag["icd_code"].str.startswith(SAH_ICD10_PREFIX)) & (diag["icd_version"] == 10))
    )
    sah_hadm_ids = set(diag.loc[sah_mask, "hadm_id"].unique())
    cohort = admissions[admissions["hadm_id"].isin(sah_hadm_ids)].copy()
    n0 = len(cohort)
    flow.append({
        "step": 0,
        "description": "SAH admissions (ICD-9 430 / ICD-10 I60x)",
        "n_excluded": 0,
        "n_remaining": n0,
        "n_subjects": cohort["subject_id"].nunique(),
    })
    print(f"  Step 0: SAH admissions: {n0} admissions, "
          f"{cohort['subject_id'].nunique()} subjects")

    # --- Step 1: Exclude traumatic SAH ---
    trauma_hadm = has_trauma_code(diag, sah_hadm_ids)
    before = len(cohort)
    cohort = cohort[~cohort["hadm_id"].isin(trauma_hadm)]
    n_excl = before - len(cohort)
    flow.append({
        "step": 1,
        "description": "Traumatic SAH (skull fracture / TBI codes)",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["subject_id"].nunique(),
    })
    print(f"  Step 1: Exclude traumatic SAH — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 2: Exclude age < 18 ---
    age_df = compute_age(patients, cohort)
    cohort = cohort.merge(age_df[["hadm_id", "age"]], on="hadm_id", how="left")
    before = len(cohort)
    cohort = cohort[cohort["age"] >= 18]
    n_excl = before - len(cohort)
    flow.append({
        "step": 2,
        "description": "Age < 18",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["subject_id"].nunique(),
    })
    print(f"  Step 2: Exclude age < 18 — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 3: Require ICU stay ---
    icu_hadm_ids = set(icustays["hadm_id"].unique())
    before = len(cohort)
    cohort = cohort[cohort["hadm_id"].isin(icu_hadm_ids)]
    n_excl = before - len(cohort)
    flow.append({
        "step": 3,
        "description": "No ICU stay",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["subject_id"].nunique(),
    })
    print(f"  Step 3: Require ICU stay — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 4: First admission per patient only ---
    before = len(cohort)
    cohort = cohort.sort_values("admittime")
    cohort = cohort.drop_duplicates(subset=["subject_id"], keep="first")
    n_excl = before - len(cohort)
    flow.append({
        "step": 4,
        "description": "Subsequent SAH admissions (keep first only)",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["subject_id"].nunique(),
    })
    print(f"  Step 4: First admission per patient — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Attach ICU stay info (first ICU stay per admission) ---
    first_icu = (
        icustays.sort_values("intime")
        .drop_duplicates(subset=["hadm_id"], keep="first")
        [["hadm_id", "stay_id", "intime", "outtime", "los", "first_careunit"]]
    )
    cohort = cohort.merge(first_icu, on="hadm_id", how="left")

    # --- Attach patient demographics ---
    cohort = cohort.merge(
        patients[["subject_id", "gender", "dod"]], on="subject_id", how="left"
    )

    # --- Flag aneurysm confirmation ---
    aneurysm_flags = get_aneurysm_flags(
        diag, proc, set(cohort["hadm_id"].unique())
    )
    cohort = cohort.merge(aneurysm_flags, on="hadm_id", how="left")

    n_confirmed = (cohort["aneurysm_confirmed"] == 1).sum()
    n_angio = (cohort["angiography"] == 1).sum()
    print(f"\n  Aneurysm confirmed (treatment/dx): {n_confirmed}")
    print(f"  Cerebral angiography performed: {n_angio}")

    # Select and order output columns
    output_cols = [
        "subject_id", "hadm_id", "stay_id",
        "aneurysm_confirmed", "angiography",
        "age", "gender", "race",
        "admittime", "dischtime",
        "intime", "outtime", "los", "first_careunit",
        "hospital_expire_flag", "dod",
    ]
    cohort = cohort[output_cols].reset_index(drop=True)

    return cohort, flow


# ---------------------------------------------------------------------------
# Flow chart
# ---------------------------------------------------------------------------
def generate_flowchart(flow_steps: list, output_path: str):
    """Generate a CONSORT-style flow chart."""
    steps = flow_steps
    excl_steps = [s for s in steps[1:] if s["n_excluded"] > 0]

    n_boxes = 2 + len(excl_steps)
    fig_h = max(10, 2.5 * n_boxes + 3)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(0, 10)
    y_max = 2.5 * n_boxes + 4
    ax.set_ylim(0, y_max)
    ax.axis("off")

    MAIN_COLOR = "#D6EAF8"
    EXCL_COLOR = "#FADBD8"
    FINAL_COLOR = "#D5F5E3"
    BORDER = "#2C3E50"

    box_w = 3.5
    box_h = 0.8
    excl_w = 3.5
    excl_h = 0.6
    main_x = 4.0
    excl_x = 8.2
    y_step = 2.2

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=BORDER, linewidth=1.2,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                weight=weight)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2),
        )

    y = y_max - 1.5

    # Title
    ax.text(5, y + 1.0, "MIMIC-IV aSAH Cohort Selection",
            ha="center", fontsize=14, weight="bold")

    # Step 0: Starting population
    draw_box(main_x, y, box_w, box_h,
             f"SAH admissions (ICD-9 430 / ICD-10 I60x)\n"
             f"(N = {steps[0]['n_remaining']}, "
             f"{steps[0]['n_subjects']} subjects)",
             MAIN_COLOR, fontsize=10, bold=True)

    # Exclusion steps
    for step in excl_steps:
        prev_y = y
        y -= y_step

        draw_arrow(main_x, prev_y - box_h / 2, main_x, y + box_h / 2)

        draw_box(main_x, y, box_w, box_h,
                 f"N = {step['n_remaining']}\n"
                 f"({step['n_subjects']} subjects)",
                 MAIN_COLOR, fontsize=10)

        mid_y = (prev_y + y) / 2
        excl_text = f"{step['description']}\n(n = {step['n_excluded']})"
        draw_box(excl_x, mid_y, excl_w, excl_h, excl_text,
                 EXCL_COLOR, fontsize=8)
        draw_arrow(main_x + box_w / 2, mid_y,
                   excl_x - excl_w / 2, mid_y)

    # Final cohort
    final = steps[-1]
    prev_y = y
    y -= y_step
    draw_arrow(main_x, prev_y - box_h / 2, main_x, y + box_h / 2)
    draw_box(main_x, y, box_w, box_h,
             f"Final SAH cohort\n"
             f"(N = {final['n_remaining']}, "
             f"{final['n_subjects']} subjects)",
             FINAL_COLOR, fontsize=10, bold=True)

    # Note about aneurysm flag
    ax.text(5, y - 0.7,
            "Patients flagged for aneurysm confirmation "
            "(clipping/coiling/dx) — available for sensitivity analyses",
            ha="center", fontsize=8, style="italic", color="#666666")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFlow chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COHORT SELECTION — MIMIC-IV aSAH Study")
    print("=" * 80)

    print("\nLoading MIMIC-IV tables...")
    tables = load_tables()

    print("\nApplying inclusion/exclusion criteria...")
    cohort, flow_steps = select_sah_cohort(tables)

    print(f"\n{'=' * 80}")
    print(f"FINAL COHORT: {len(cohort)} patients")
    print(f"  Aneurysm confirmed: {(cohort['aneurysm_confirmed'] == 1).sum()}")
    print(f"  Aneurysm not confirmed: {(cohort['aneurysm_confirmed'] == 0).sum()}")
    print(f"  Age range: {cohort['age'].min():.0f} – {cohort['age'].max():.0f}")
    print(f"  Gender: {cohort['gender'].value_counts().to_dict()}")
    print(f"{'=' * 80}")

    # Verify no duplicated subjects
    assert cohort["subject_id"].is_unique, "Duplicate subject_ids in final cohort!"
    assert (cohort["age"] >= 18).all(), "Underage patients in final cohort!"
    assert cohort["stay_id"].notna().all(), "Missing stay_ids in final cohort!"

    # Save outputs
    cohort_path = ARTIFACTS_DIR / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"\nSaved {cohort_path} ({len(cohort)} patients)")

    flow_df = pd.DataFrame(flow_steps)
    flow_path = ARTIFACTS_DIR / "cohort_flow.csv"
    flow_df.to_csv(flow_path, index=False)
    print(f"Saved {flow_path} ({len(flow_df)} steps)")

    flowchart_path = ARTIFACTS_DIR / "cohort_flowchart.png"
    generate_flowchart(flow_steps, str(flowchart_path))


if __name__ == "__main__":
    main()
