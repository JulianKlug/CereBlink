"""
Step 2: Build patient-level baseline table and compute missingness.

Loads registry, outcomes, and PDMS time-series data, extracts baseline
(day 0–2) values for each propensity covariate, and exports a
missingness summary table (missingness_table.csv).
"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from utils import load_encrypted_xlsx, safe_conversion_to_datetime

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/mnt/data1/klug/datasets/kssg/SAH"
REGISTRY_PATH = f"{DATA_DIR}/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx"
OUTCOMES_PATH = f"{DATA_DIR}/outcomes_aSAH_DATA_2009_2024_18122024.xlsx"
CORRESPONDENCE_PATH = f"{DATA_DIR}/registry_pdms_correspondence.csv"

PDMS = f"{DATA_DIR}/extracted_data"
BGA_PATH = f"{PDMS}/20240116_SAH_SOS_BGA.csv"
LABOR_PATH = f"{PDMS}/20240116_SAH_SOS_Labor.csv"
BP_PATH = f"{PDMS}/20240116_SAH_SOS_Blutdruecke.csv"
TEMP_PATH = f"{PDMS}/20240116_SAH_SOS_Temperatur.csv"
GCS_PATH = f"{PDMS}/20240117_SAH_SOS_GCS.csv"

SECRETS_PATH = "/home/klug/icu_projects/CereBlink/.secrets"
BASELINE_WINDOW_DAYS = 2

# Aneurysm location code mapping
ANTERIOR_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 31]
POSTERIOR_CODES = [10, 11, 12, 13, 14, 15, 16, 17, 23, 28]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_passwords():
    """Read passwords from .secrets file."""
    with open(SECRETS_PATH) as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_all_sources():
    """Load registry, outcomes, correspondence, and PDMS DataFrames."""
    passwords = load_passwords()

    # Registry: second password (index 1)
    registry = load_encrypted_xlsx(REGISTRY_PATH, password=passwords[1])

    # Outcomes: first password (index 0)
    outcomes = load_encrypted_xlsx(OUTCOMES_PATH, password=passwords[0])

    # Correspondence
    corr = pd.read_csv(CORRESPONDENCE_PATH)
    corr.rename(columns={"JoinedName": "Name"}, inplace=True)
    corr["Date_birth"] = pd.to_datetime(corr["Date_birth"], format="%d.%m.%Y", errors="coerce")

    # Merge registry ↔ correspondence to get pNr
    registry["Date_birth"] = pd.to_datetime(registry["Date_birth"], errors="coerce")
    registry = registry.merge(
        corr[["SOS-CENTER-YEAR-NO.", "Name", "Date_birth", "pNr"]],
        on=["SOS-CENTER-YEAR-NO.", "Name", "Date_birth"],
        how="left",
    )

    # Merge outcomes ↔ correspondence
    outcomes["Date_birth"] = pd.to_datetime(outcomes["Date_birth"], errors="coerce")
    outcomes = outcomes.merge(
        corr[["SOS-CENTER-YEAR-NO.", "Name", "Date_birth", "pNr"]],
        on=["SOS-CENTER-YEAR-NO.", "Name", "Date_birth"],
        how="left",
    )

    # PDMS CSVs
    bga = pd.read_csv(BGA_PATH, sep=";", encoding="utf-8-sig")
    labor = pd.read_csv(LABOR_PATH, sep=";", encoding="utf-8-sig")
    bp = pd.read_csv(BP_PATH, sep=";", encoding="utf-8-sig",
                      usecols=["pNr", "systole", "mitteldruck", "timeBd"])
    temp = pd.read_csv(TEMP_PATH, sep=";", encoding="utf-8-sig",
                        usecols=["pNr", "temperatur", "timeTemp"])
    gcs = pd.read_csv(GCS_PATH, sep=";")

    return {
        "registry": registry,
        "outcomes": outcomes,
        "bga": bga,
        "labor": labor,
        "bp": bp,
        "temp": temp,
        "gcs": gcs,
    }


# ---------------------------------------------------------------------------
# Admission map & baseline extraction
# ---------------------------------------------------------------------------
def build_admission_map(registry):
    """Build {pNr: admission_datetime} from registry."""
    df = registry.dropna(subset=["pNr", "Date_admission"]).copy()
    df["pNr"] = df["pNr"].astype(int)
    df["Date_admission"] = pd.to_datetime(df["Date_admission"], errors="coerce")
    return df.set_index("pNr")["Date_admission"].to_dict()


def extract_baseline_timeseries(ts_df, admission_map, time_col, value_col,
                                 agg="first", filters=None):
    """
    Extract one baseline value per patient from a time-series DataFrame.

    Parameters
    ----------
    ts_df : DataFrame with pNr, time_col, value_col columns
    admission_map : {pNr: admission_datetime}
    time_col : name of timestamp column
    value_col : name of value column
    agg : 'first' (earliest) or 'max' (worst)
    filters : dict of {col: value} to pre-filter rows

    Returns
    -------
    Series indexed by pNr with one value per patient
    """
    df = ts_df.copy()
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]

    df["pNr"] = pd.to_numeric(df["pNr"], errors="coerce")
    df = df.dropna(subset=["pNr", value_col])
    df["pNr"] = df["pNr"].astype(int)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Filter to patients with known admission date
    valid_pnrs = set(admission_map.keys())
    df = df[df["pNr"].isin(valid_pnrs)]

    # Filter to baseline window
    df["_adm"] = df["pNr"].map(admission_map)
    df["_end"] = df["_adm"] + pd.Timedelta(days=BASELINE_WINDOW_DAYS)
    df = df[(df[time_col] >= df["_adm"]) & (df[time_col] <= df["_end"])]

    if len(df) == 0:
        return pd.Series(dtype=float, name=value_col)

    if agg == "first":
        result = df.sort_values(time_col).groupby("pNr")[value_col].first()
    elif agg == "max":
        result = df.groupby("pNr")[value_col].max()
    elif agg == "min":
        result = df.groupby("pNr")[value_col].min()
    else:
        raise ValueError(f"Unknown agg: {agg}")

    return result


# ---------------------------------------------------------------------------
# Patient table
# ---------------------------------------------------------------------------
def build_patient_table(sources):
    """Build one-row-per-patient DataFrame with all covariates."""
    reg = sources["registry"].copy()
    out = sources["outcomes"].copy()

    # Keep only patients with pNr (i.e., matched to PDMS)
    reg = reg.dropna(subset=["pNr"]).copy()
    reg["pNr"] = reg["pNr"].astype(int)
    out["pNr"] = pd.to_numeric(out["pNr"], errors="coerce")
    out = out.dropna(subset=["pNr"]).copy()
    out["pNr"] = out["pNr"].astype(int)

    admission_map = build_admission_map(reg)
    pt = pd.DataFrame(index=pd.Index(sorted(reg["pNr"].unique()), name="pNr"))

    # === Demographics ===
    reg_idx = reg.set_index("pNr")
    pt["age"] = reg_idx["Age"]
    pt["sex"] = reg_idx["Sex"].str.upper().map({"M": 0, "F": 1, "W": 1})

    # === Comorbidities ===
    pt["hypertension"] = pd.to_numeric(reg_idx["HTN"], errors="coerce")
    pt["diabetes"] = pd.to_numeric(reg_idx["DM"], errors="coerce")
    pt["smoking"] = pd.to_numeric(reg_idx["Smoker_0no_1yes_2ex"], errors="coerce")
    pt["prior_ass"] = pd.to_numeric(reg_idx["ASS"], errors="coerce")
    pt["prior_clopidogrel"] = pd.to_numeric(reg_idx["Clopidogrel"], errors="coerce")
    pt["prior_oac"] = pd.to_numeric(reg_idx["OAC"], errors="coerce")
    pt["premorbid_mrs"] = pd.to_numeric(reg_idx["mRS_before_ictus"], errors="coerce")

    # === Clinical grades ===
    pt["hunt_hess"] = pd.to_numeric(reg_idx["HH"], errors="coerce")
    pt["wfns"] = pd.to_numeric(reg_idx["WFNS"], errors="coerce")
    pt["fisher_score"] = pd.to_numeric(reg_idx["Fisher_Score"], errors="coerce")
    pt["intubated_admission"] = pd.to_numeric(reg_idx["Intubated_on_admission_YN"], errors="coerce")
    pt["ivh"] = pd.to_numeric(reg_idx["IVH"], errors="coerce")
    pt["ich"] = pd.to_numeric(reg_idx["ICH"], errors="coerce")

    # GCS: registry primary, PDMS fallback
    pt["gcs_admission"] = pd.to_numeric(reg_idx["GCS_admission"], errors="coerce")
    gcs_df = sources["gcs"].copy()
    gcs_df["pNr"] = pd.to_numeric(gcs_df["pNr"], errors="coerce")
    gcs_df = gcs_df.dropna(subset=["pNr"])
    gcs_df["pNr"] = gcs_df["pNr"].astype(int)
    gcs_df["GCS"] = (pd.to_numeric(gcs_df["eyes"], errors="coerce")
                     + pd.to_numeric(gcs_df["movement"], errors="coerce")
                     + pd.to_numeric(gcs_df["verbal"], errors="coerce"))
    gcs_df["timeGCS"] = pd.to_datetime(gcs_df["timeGCS"], errors="coerce")
    first_gcs = gcs_df.sort_values("timeGCS").groupby("pNr")["GCS"].first()
    pt["gcs_admission"] = pt["gcs_admission"].fillna(first_gcs)

    # Intubated: PDMS fallback
    first_intub = gcs_df.sort_values("timeGCS").groupby("pNr")["intubated"].first()
    pt["intubated_admission"] = pt["intubated_admission"].fillna(
        pd.to_numeric(first_intub, errors="coerce")
    )

    # === Aneurysm characteristics ===
    art_code = pd.to_numeric(reg_idx["Aneurysm_Artery_Code"], errors="coerce")
    pt["aneurysm_anterior"] = art_code.isin(ANTERIOR_CODES).astype(float)
    pt.loc[art_code.isna(), "aneurysm_anterior"] = np.nan
    pt["aneurysm_size"] = pd.to_numeric(reg_idx["Aneurysm_diameter"], errors="coerce")
    pt["multiple_aneurysms"] = pd.to_numeric(reg_idx["Multiple_Aneurysms_2unk"], errors="coerce")

    # === Treatment ===
    coiling = pd.to_numeric(reg_idx["Coiling"], errors="coerce")
    clipping = pd.to_numeric(reg_idx["Clipping"], errors="coerce")
    stenting = pd.to_numeric(reg_idx["Stenting"], errors="coerce")
    pt["treatment_coiling"] = coiling
    pt["treatment_clipping"] = clipping
    pt["treatment_stenting"] = stenting

    # Time ictus to treatment (days)
    date_ictus = reg_idx["Date_Ictus"].apply(safe_conversion_to_datetime)
    date_th = reg_idx["Date_First_Th"].apply(safe_conversion_to_datetime)
    pt["time_ictus_to_treatment_days"] = (date_th - date_ictus).dt.total_seconds() / 86400

    # EVD
    pt["evd"] = pd.to_numeric(reg_idx["EVD_YN"], errors="coerce")

    # === Admission physiology (PDMS, baseline window day 0–2) ===
    pt["admission_map"] = extract_baseline_timeseries(
        sources["bp"], admission_map, "timeBd", "mitteldruck", agg="first"
    )
    pt["admission_sbp"] = extract_baseline_timeseries(
        sources["bp"], admission_map, "timeBd", "systole", agg="first"
    )
    pt["admission_temp"] = extract_baseline_timeseries(
        sources["temp"], admission_map, "timeTemp", "temperatur", agg="max"
    )
    pt["admission_wbc"] = extract_baseline_timeseries(
        sources["labor"], admission_map, "DatumLabor", "Wert",
        agg="first", filters={"Labor": "Leukozyten"}
    )
    pt["admission_hematocrit"] = extract_baseline_timeseries(
        sources["labor"], admission_map, "DatumLabor", "Wert",
        agg="first", filters={"Labor": "HKT"}
    )
    pt["admission_crp"] = extract_baseline_timeseries(
        sources["labor"], admission_map, "DatumLabor", "Wert",
        agg="first", filters={"Labor": "CRP"}
    )
    pt["admission_glucose"] = extract_baseline_timeseries(
        sources["bga"], admission_map, "timeBGA", "glc", agg="first"
    )
    pt["admission_sodium"] = extract_baseline_timeseries(
        sources["bga"], admission_map, "timeBGA", "na", agg="first"
    )
    pt["admission_hb"] = extract_baseline_timeseries(
        sources["bga"], admission_map, "timeBGA", "hb", agg="first"
    )
    pt["admission_pao2"] = extract_baseline_timeseries(
        sources["bga"], admission_map, "timeBGA", "pO2",
        agg="first", filters={"bgaOrt": "arteriell"}
    )

    # === Outcomes ===
    pt["dci"] = pd.to_numeric(reg_idx["DCI_YN_verified"], errors="coerce")
    pt["dci"] = pt["dci"].fillna(pd.to_numeric(reg_idx["DCI_YN"], errors="coerce"))
    pt["rebleeding"] = pd.to_numeric(reg_idx["Rebleeding_YN"], errors="coerce")
    pt["death"] = pd.to_numeric(reg_idx["Death"], errors="coerce")

    # Outcomes from outcomes file
    out_idx = out.drop_duplicates(subset=["pNr"]).set_index("pNr")
    pt["mrs_discharge"] = pd.to_numeric(out_idx["mRS_discharge"], errors="coerce")
    pt["mrs_1y"] = pd.to_numeric(out_idx["mRS_FU_1y"], errors="coerce")

    # === NOT available — add as NaN columns ===
    for col in ["bmi", "cardiovascular_disease", "thick_sah_burden",
                "heart_rate", "respiratory_rate", "potassium", "ph_hco3"]:
        pt[col] = np.nan

    return pt


# ---------------------------------------------------------------------------
# Missingness computation
# ---------------------------------------------------------------------------
VARIABLE_META = [
    # (column_name, display_name, category, available_in_data)
    # Demographics
    ("age", "Age", "Demographics", True),
    ("sex", "Sex", "Demographics", True),
    ("bmi", "BMI / obesity", "Demographics", False),
    # Comorbidities
    ("hypertension", "Hypertension", "Comorbidities", True),
    ("diabetes", "Diabetes mellitus", "Comorbidities", True),
    ("smoking", "Smoking history", "Comorbidities", True),
    ("prior_ass", "Prior antiplatelet (ASS)", "Comorbidities", True),
    ("prior_clopidogrel", "Prior antiplatelet (Clopidogrel)", "Comorbidities", True),
    ("prior_oac", "Prior oral anticoagulation", "Comorbidities", True),
    ("premorbid_mrs", "Pre-morbid mRS", "Comorbidities", True),
    ("cardiovascular_disease", "Cardiovascular disease", "Comorbidities", False),
    # Clinical grades
    ("hunt_hess", "Hunt-Hess grade", "Clinical Grades", True),
    ("wfns", "WFNS grade", "Clinical Grades", True),
    ("gcs_admission", "GCS on admission", "Clinical Grades", True),
    ("fisher_score", "Modified Fisher grade", "Clinical Grades", True),
    ("intubated_admission", "Intubated on admission", "Clinical Grades", True),
    ("ivh", "IVH", "Clinical Grades", True),
    ("ich", "ICH", "Clinical Grades", True),
    ("thick_sah_burden", "Thick SAH burden", "Clinical Grades", False),
    # Physiology
    ("admission_map", "Admission MAP", "Physiology", True),
    ("admission_sbp", "Admission SBP", "Physiology", True),
    ("admission_temp", "Admission temperature", "Physiology", True),
    ("admission_wbc", "Admission WBC", "Physiology", True),
    ("admission_hematocrit", "Admission hematocrit", "Physiology", True),
    ("admission_crp", "Admission CRP", "Physiology", True),
    ("admission_glucose", "Admission glucose", "Physiology", True),
    ("admission_sodium", "Admission sodium", "Physiology", True),
    ("admission_hb", "Admission hemoglobin", "Physiology", True),
    ("admission_pao2", "Admission PaO2", "Physiology", True),
    ("heart_rate", "Heart rate", "Physiology", False),
    ("respiratory_rate", "Respiratory rate", "Physiology", False),
    ("potassium", "Potassium", "Physiology", False),
    ("ph_hco3", "pH / HCO3", "Physiology", False),
    # Aneurysm
    ("aneurysm_anterior", "Aneurysm location (anterior)", "Aneurysm", True),
    ("aneurysm_size", "Aneurysm size", "Aneurysm", True),
    ("multiple_aneurysms", "Multiple aneurysms", "Aneurysm", True),
    # Treatment
    ("treatment_coiling", "Coiling", "Treatment", True),
    ("treatment_clipping", "Clipping", "Treatment", True),
    ("treatment_stenting", "Stenting", "Treatment", True),
    ("time_ictus_to_treatment_days", "Time ictus-to-treatment", "Treatment", True),
    ("evd", "EVD placement", "Treatment", True),
    # Outcomes
    ("dci", "DCI", "Outcome", True),
    ("rebleeding", "Rebleeding", "Outcome", True),
    ("death", "Death", "Outcome", True),
    ("mrs_discharge", "mRS at discharge", "Outcome", True),
    ("mrs_1y", "mRS at 1 year", "Outcome", True),
]


def compute_missingness(patient_df):
    """Compute per-variable missingness statistics."""
    n_total = len(patient_df)
    rows = []
    for col, name, category, available in VARIABLE_META:
        if col in patient_df.columns:
            n_available = int(patient_df[col].notna().sum())
        else:
            n_available = 0
        n_missing = n_total - n_available
        pct_missing = round(100 * n_missing / n_total, 1) if n_total > 0 else 0.0
        rows.append({
            "variable": name,
            "column": col,
            "category": category,
            "available_in_data": available,
            "n_total": n_total,
            "n_available": n_available,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data sources...")
    sources = load_all_sources()

    n_reg = len(sources["registry"])
    n_with_pnr = sources["registry"]["pNr"].notna().sum()
    print(f"  Registry: {n_reg} patients, {n_with_pnr} with PDMS linkage (pNr)")

    print("Building patient-level baseline table...")
    pt = build_patient_table(sources)
    print(f"  Patient table: {len(pt)} patients x {len(pt.columns)} variables")

    print("Computing missingness...")
    miss = compute_missingness(pt)

    miss.to_csv("missingness_table.csv", index=False)
    print(f"\nSaved missingness_table.csv ({len(miss)} variables)")

    # Print summary
    print(f"\n{'='*80}")
    print(f"MISSINGNESS SUMMARY (N = {len(pt)} patients with PDMS linkage)")
    print(f"{'='*80}")
    for cat in miss["category"].unique():
        cat_df = miss[miss["category"] == cat]
        print(f"\n--- {cat} ---")
        for _, row in cat_df.iterrows():
            status = "AVAILABLE" if row["available_in_data"] else "NOT IN DATA"
            bar = f"{row['n_available']:>4}/{row['n_total']}"
            print(f"  {row['variable']:<35} {bar}  ({row['pct_missing']:>5.1f}% missing)  [{status}]")


if __name__ == "__main__":
    main()
