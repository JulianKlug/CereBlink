"""
Step 1: MIMIC-III cohort selection for aSAH anticoagulation study.

Identifies aneurysmal subarachnoid hemorrhage patients from MIMIC-III,
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
DATA_DIR = "/mnt/hdd1/datasets/mimiciii_1.4"
OUTPUT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

# ---------------------------------------------------------------------------
# ICD-9 codes
# ---------------------------------------------------------------------------
SAH_CODE = "430"

TRAUMA_PREFIXES = [
    "800", "801", "802", "803", "804",  # skull fractures
    "850", "851", "852", "853", "854",  # head injury
]

# Aneurysm confirmation codes
ANEURYSM_DX = "4373"           # cerebral aneurysm nonruptured
CLIP_CODES = ["3951"]           # clipping of cerebral aneurysm
COIL_CODES = ["3972", "3975", "3976"]  # endovascular embolization
ANGIOGRAPHY_CODE = "8841"      # cerebral angiography


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tables() -> dict:
    """Load required MIMIC-III tables."""
    tables = {}
    files = {
        "diagnoses": "DIAGNOSES_ICD.csv.gz",
        "procedures": "PROCEDURES_ICD.csv.gz",
        "patients": "PATIENTS.csv.gz",
        "admissions": "ADMISSIONS.csv.gz",
        "icustays": "ICUSTAYS.csv.gz",
    }
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, compression="gzip")
        # Ensure ICD9_CODE is string for consistent matching
        if "ICD9_CODE" in df.columns:
            df["ICD9_CODE"] = df["ICD9_CODE"].astype(str).str.strip()
        tables[key] = df
        print(f"  Loaded {fname}: {len(df):,} rows")

    # Parse date columns
    tables["patients"]["DOB"] = pd.to_datetime(
        tables["patients"]["DOB"], errors="coerce"
    )
    tables["patients"]["DOD"] = pd.to_datetime(
        tables["patients"]["DOD"], errors="coerce"
    )
    tables["admissions"]["ADMITTIME"] = pd.to_datetime(
        tables["admissions"]["ADMITTIME"], errors="coerce"
    )
    tables["admissions"]["DISCHTIME"] = pd.to_datetime(
        tables["admissions"]["DISCHTIME"], errors="coerce"
    )
    tables["icustays"]["INTIME"] = pd.to_datetime(
        tables["icustays"]["INTIME"], errors="coerce"
    )
    tables["icustays"]["OUTTIME"] = pd.to_datetime(
        tables["icustays"]["OUTTIME"], errors="coerce"
    )

    return tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_trauma_code(diag_df: pd.DataFrame, hadm_ids: set) -> set:
    """Return HADM_IDs that have a concurrent traumatic brain injury code."""
    trauma = diag_df[diag_df["HADM_ID"].isin(hadm_ids)].copy()
    mask = trauma["ICD9_CODE"].apply(
        lambda c: any(c.startswith(p) for p in TRAUMA_PREFIXES)
    )
    return set(trauma.loc[mask, "HADM_ID"].unique())


def compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age at admission. MIMIC-III shifts DOB for patients >89 to ~300 years;
    cap those at 89.
    """
    merged = admissions[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]].merge(
        patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="left"
    )
    # Use Python-level subtraction to avoid int64 overflow from shifted DOBs
    merged["age"] = merged.apply(
        lambda r: (r["ADMITTIME"].year - r["DOB"].year
                   - ((r["ADMITTIME"].month, r["ADMITTIME"].day)
                      < (r["DOB"].month, r["DOB"].day)))
        if pd.notna(r["ADMITTIME"]) and pd.notna(r["DOB"]) else np.nan,
        axis=1,
    )
    # Cap age at 89 for MIMIC privacy-shifted patients (DOB shifted to ~300 yrs)
    merged.loc[merged["age"] > 89, "age"] = 89.0
    return merged[["SUBJECT_ID", "HADM_ID", "age"]]


def get_aneurysm_flags(
    diag_df: pd.DataFrame, proc_df: pd.DataFrame, hadm_ids: set
) -> pd.DataFrame:
    """
    Flag admissions with evidence of aneurysmal etiology.

    aneurysm_confirmed = 1 if clipping, coiling, or aneurysm dx (4373) present.
    angiography = 1 if cerebral angiography (8841) present.
    """
    treatment_codes = set(CLIP_CODES + COIL_CODES)

    # Procedure-based flags
    proc_sub = proc_df[proc_df["HADM_ID"].isin(hadm_ids)].copy()
    has_treatment = set(
        proc_sub.loc[proc_sub["ICD9_CODE"].isin(treatment_codes), "HADM_ID"].unique()
    )
    has_angiography = set(
        proc_sub.loc[proc_sub["ICD9_CODE"] == ANGIOGRAPHY_CODE, "HADM_ID"].unique()
    )

    # Diagnosis-based flag (cerebral aneurysm)
    diag_sub = diag_df[diag_df["HADM_ID"].isin(hadm_ids)].copy()
    has_aneurysm_dx = set(
        diag_sub.loc[diag_sub["ICD9_CODE"] == ANEURYSM_DX, "HADM_ID"].unique()
    )

    rows = []
    for hadm_id in hadm_ids:
        confirmed = 1 if (hadm_id in has_treatment or hadm_id in has_aneurysm_dx) else 0
        angio = 1 if hadm_id in has_angiography else 0
        rows.append({
            "HADM_ID": hadm_id,
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

    # --- Step 0: All admissions with ICD-9 430 (SAH) ---
    sah_hadm_ids = set(diag.loc[diag["ICD9_CODE"] == SAH_CODE, "HADM_ID"].unique())
    sah_subject_ids = set(
        diag.loc[diag["HADM_ID"].isin(sah_hadm_ids), "SUBJECT_ID"].unique()
    )
    cohort = admissions[admissions["HADM_ID"].isin(sah_hadm_ids)].copy()
    n0 = len(cohort)
    flow.append({
        "step": 0,
        "description": "SAH admissions (ICD-9 430)",
        "n_excluded": 0,
        "n_remaining": n0,
        "n_subjects": cohort["SUBJECT_ID"].nunique(),
    })
    print(f"  Step 0: SAH admissions (ICD-9 430): {n0} admissions, "
          f"{cohort['SUBJECT_ID'].nunique()} subjects")

    # --- Step 1: Exclude traumatic SAH ---
    trauma_hadm = has_trauma_code(diag, sah_hadm_ids)
    before = len(cohort)
    cohort = cohort[~cohort["HADM_ID"].isin(trauma_hadm)]
    n_excl = before - len(cohort)
    flow.append({
        "step": 1,
        "description": "Traumatic SAH (skull fracture / head injury codes)",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["SUBJECT_ID"].nunique(),
    })
    print(f"  Step 1: Exclude traumatic SAH — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 2: Exclude age < 18 ---
    age_df = compute_age(patients, cohort)
    cohort = cohort.merge(age_df[["HADM_ID", "age"]], on="HADM_ID", how="left")
    before = len(cohort)
    cohort = cohort[cohort["age"] >= 18]
    n_excl = before - len(cohort)
    flow.append({
        "step": 2,
        "description": "Age < 18",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["SUBJECT_ID"].nunique(),
    })
    print(f"  Step 2: Exclude age < 18 — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 3: Require ICU stay ---
    icu_hadm_ids = set(icustays["HADM_ID"].unique())
    before = len(cohort)
    cohort = cohort[cohort["HADM_ID"].isin(icu_hadm_ids)]
    n_excl = before - len(cohort)
    flow.append({
        "step": 3,
        "description": "No ICU stay",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["SUBJECT_ID"].nunique(),
    })
    print(f"  Step 3: Require ICU stay — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Step 4: First admission per patient only ---
    before = len(cohort)
    cohort = cohort.sort_values("ADMITTIME")
    cohort = cohort.drop_duplicates(subset=["SUBJECT_ID"], keep="first")
    n_excl = before - len(cohort)
    flow.append({
        "step": 4,
        "description": "Subsequent SAH admissions (keep first only)",
        "n_excluded": n_excl,
        "n_remaining": len(cohort),
        "n_subjects": cohort["SUBJECT_ID"].nunique(),
    })
    print(f"  Step 4: First admission per patient — excluded {n_excl}, "
          f"remaining {len(cohort)}")

    # --- Attach ICU stay info (first ICU stay per admission) ---
    first_icu = (
        icustays.sort_values("INTIME")
        .drop_duplicates(subset=["HADM_ID"], keep="first")
        [["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS", "FIRST_CAREUNIT"]]
    )
    cohort = cohort.merge(first_icu, on="HADM_ID", how="left")

    # --- Attach patient demographics ---
    cohort = cohort.merge(
        patients[["SUBJECT_ID", "GENDER", "DOB", "DOD"]], on="SUBJECT_ID", how="left"
    )

    # --- Flag aneurysm confirmation ---
    aneurysm_flags = get_aneurysm_flags(
        diag, proc, set(cohort["HADM_ID"].unique())
    )
    cohort = cohort.merge(aneurysm_flags, on="HADM_ID", how="left")

    n_confirmed = (cohort["aneurysm_confirmed"] == 1).sum()
    n_angio = (cohort["angiography"] == 1).sum()
    print(f"\n  Aneurysm confirmed (treatment/dx): {n_confirmed}")
    print(f"  Cerebral angiography performed: {n_angio}")

    # Select and order output columns
    output_cols = [
        "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID",
        "aneurysm_confirmed", "angiography",
        "age", "GENDER", "ETHNICITY",
        "ADMITTIME", "DISCHTIME",
        "INTIME", "OUTTIME", "LOS", "FIRST_CAREUNIT",
        "HOSPITAL_EXPIRE_FLAG", "DOD",
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
    FLAG_COLOR = "#FEF9E7"
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
    ax.text(5, y + 1.0, "MIMIC-III aSAH Cohort Selection",
            ha="center", fontsize=14, weight="bold")

    # Step 0: Starting population
    draw_box(main_x, y, box_w, box_h,
             f"SAH admissions (ICD-9 430)\n"
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
            "(clipping/coiling/dx 4373) — available for sensitivity analyses",
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
    print("COHORT SELECTION — MIMIC-III aSAH Study")
    print("=" * 80)

    print("\nLoading MIMIC-III tables...")
    tables = load_tables()

    print("\nApplying inclusion/exclusion criteria...")
    cohort, flow_steps = select_sah_cohort(tables)

    print(f"\n{'=' * 80}")
    print(f"FINAL COHORT: {len(cohort)} patients")
    print(f"  Aneurysm confirmed: {(cohort['aneurysm_confirmed'] == 1).sum()}")
    print(f"  Aneurysm not confirmed: {(cohort['aneurysm_confirmed'] == 0).sum()}")
    print(f"  Age range: {cohort['age'].min():.0f} – {cohort['age'].max():.0f}")
    print(f"  Gender: {cohort['GENDER'].value_counts().to_dict()}")
    print(f"{'=' * 80}")

    # Verify no duplicated subjects
    assert cohort["SUBJECT_ID"].is_unique, "Duplicate SUBJECT_IDs in final cohort!"
    assert (cohort["age"] >= 18).all(), "Underage patients in final cohort!"
    assert cohort["ICUSTAY_ID"].notna().all(), "Missing ICUSTAY_IDs in final cohort!"

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
