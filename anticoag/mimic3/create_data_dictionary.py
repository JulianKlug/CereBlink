"""
Step 2: MIMIC-III variable extraction for aSAH anticoagulation study.

Extracts treatment exposure, baseline confounders (within 48h of ICU admission),
and outcomes from MIMIC-III. Produces patient_table.csv, data_dictionary.md,
and missingness_table.csv.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "/mnt/hdd1/datasets/mimiciii_1.4"
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
COHORT_PATH = ARTIFACTS_DIR / "cohort.csv"

BASELINE_WINDOW_HOURS = 48

# ---------------------------------------------------------------------------
# ITEMID maps
# ---------------------------------------------------------------------------

# Treatment ITEMIDs
HEPARIN_MV_ITEMIDS = {225152, 225975}
HEPARIN_CV_ITEMIDS = {30025}
ENOXAPARIN_MV_ITEMIDS = {225906}

# Vitals (CHARTEVENTS)
VITALS_ITEMIDS = {
    "map": {"mv": [220052, 220181], "cv": [52, 456, 443]},
    "sbp": {"mv": [220050, 220179], "cv": [51, 455, 442]},
    "heart_rate": {"mv": [220045], "cv": [211]},
    "temp_c": {"mv": [223762], "cv": [676, 677]},
    "temp_f": {"mv": [223761], "cv": [678, 679]},
    "gcs_eye": {"mv": [220739], "cv": []},
    "gcs_verbal": {"mv": [223900], "cv": []},
    "gcs_motor": {"mv": [223901], "cv": []},
    "gcs_total": {"mv": [], "cv": [198]},
}

# Labs (LABEVENTS)
LAB_ITEMIDS = {
    "wbc": [51300, 51301],
    "hemoglobin": [51222, 50811],
    "creatinine": [50912],
    "glucose": [50931, 50809],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "platelets": [51265],
    "inr": [51237],
    "ptt": [51275],
    "lactate": [50813],
    "albumin": [50862],
    "bilirubin": [50885],
}

# ICD-9 prefix maps for comorbidities
COMORBIDITY_ICD9 = {
    "hypertension": ["401", "402", "403", "404", "405"],
    "diabetes": ["250"],
    "smoking_history": ["V1582", "3051"],
    "renal_disease": ["585", "586", "V420", "V451"],
    "liver_disease": ["5712", "5713", "5714", "5715", "5716", "5717",
                      "5718", "5719", "456", "572"],
    "prior_anticoagulant": ["V5861"],
    "prior_antiplatelet": ["V5863"],
}

# Procedure ICD-9 codes
PROCEDURE_ICD9 = {
    "treatment_clipping": ["3951"],
    "treatment_coiling": ["3972", "3975", "3976"],
    "evd_placement": ["0231", "231"],  # MIMIC strips leading zeros
}

# Outcome ICD-9 codes
DCI_ICD9 = ["43401", "43411", "43491", "4371"]
REBLEEDING_ICD9 = ["4320", "4321", "4329", "99811", "99812"]

# GCS text-to-numeric mappings (MV component charted as text)
GCS_EYE_MAP = {
    "1 No Response": 1, "No Response": 1, "None": 1,
    "2 To pain": 2, "To Pain": 2,
    "3 To speech": 3, "To Speech": 3,
    "4 Spontaneously": 4, "Spontaneously": 4,
}

GCS_VERBAL_MAP = {
    "1 No Response": 1, "No Response": 1, "None": 1,
    "1.0 ET/Trach": 1, "ET/Trach": 1,
    "2 Incomp sounds": 2, "Incomprehensible sounds": 2,
    "3 Inapprop words": 3, "Inappropriate Words": 3,
    "4 Confused": 4, "Confused": 4,
    "5 Oriented": 5, "Oriented": 5,
}

GCS_MOTOR_MAP = {
    "1 No Response": 1, "No Response": 1, "None": 1,
    "2 Abnorm extensn": 2, "Abnormal extension": 2, "Extension": 2,
    "3 Abnorm flexion": 3, "Abnormal Flexion": 3, "Flexion": 3,
    "4 Flex-withdraws": 4, "Flex-Withdrawal": 4,
    "5 Localizes Pain": 5, "Localizes": 5,
    "6 Obeys Commands": 6, "Obeys Commands": 6,
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_cohort():
    """Load cohort.csv and build admission maps."""
    cohort = pd.read_csv(COHORT_PATH)
    cohort["INTIME"] = pd.to_datetime(cohort["INTIME"], errors="coerce")
    cohort["OUTTIME"] = pd.to_datetime(cohort["OUTTIME"], errors="coerce")
    cohort["ADMITTIME"] = pd.to_datetime(cohort["ADMITTIME"], errors="coerce")
    cohort["DISCHTIME"] = pd.to_datetime(cohort["DISCHTIME"], errors="coerce")
    print(f"  Cohort: {len(cohort)} patients")
    return cohort


def _collect_all_chart_itemids():
    """Collect all CHARTEVENTS ITEMIDs needed in a single set."""
    ids = set()
    for group in VITALS_ITEMIDS.values():
        ids.update(group["mv"])
        ids.update(group["cv"])
    return ids


def _collect_all_lab_itemids():
    """Collect all LABEVENTS ITEMIDs needed in a single set."""
    ids = set()
    for items in LAB_ITEMIDS.values():
        ids.update(items)
    return ids


def load_filtered_events(path, id_col, cohort_ids, target_itemids,
                         usecols=None, chunksize=5_000_000):
    """
    Load events table in chunks, filtering to cohort and target ITEMIDs.

    Single-pass through potentially huge tables (CHARTEVENTS ~330M rows).
    """
    cohort_set = set(cohort_ids)
    itemid_set = set(target_itemids)
    chunks = []

    for chunk in pd.read_csv(path, compression="gzip", chunksize=chunksize,
                             usecols=usecols):
        filtered = chunk[
            (chunk[id_col].isin(cohort_set)) &
            (chunk["ITEMID"].isin(itemid_set))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {path.split('/')[-1]}: {len(result):,} relevant rows")
    return result


def load_inputevents_mv(hadm_ids):
    """Load INPUTEVENTS_MV for heparin/enoxaparin."""
    path = os.path.join(DATA_DIR, "INPUTEVENTS_MV.csv.gz")
    target_ids = HEPARIN_MV_ITEMIDS | ENOXAPARIN_MV_ITEMIDS
    usecols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME",
               "ITEMID", "AMOUNT", "AMOUNTUOM", "RATE", "RATEUOM",
               "STATUSDESCRIPTION", "ORDERCATEGORYNAME"]

    chunks = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=2_000_000,
                             usecols=usecols):
        filtered = chunk[
            (chunk["HADM_ID"].isin(set(hadm_ids))) &
            (chunk["ITEMID"].isin(target_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    # Exclude rewritten orders
    result = result[result["STATUSDESCRIPTION"] != "Rewritten"]
    print(f"  INPUTEVENTS_MV (heparin/enoxaparin): {len(result):,} rows")
    return result


def load_inputevents_cv(hadm_ids):
    """Load INPUTEVENTS_CV for heparin."""
    path = os.path.join(DATA_DIR, "INPUTEVENTS_CV.csv.gz")
    target_ids = HEPARIN_CV_ITEMIDS
    usecols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME",
               "ITEMID", "AMOUNT", "AMOUNTUOM", "RATE", "RATEUOM",
               "ORIGINALAMOUNT", "ORIGINALAMOUNTUOM"]

    chunks = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=2_000_000,
                             usecols=usecols):
        filtered = chunk[
            (chunk["HADM_ID"].isin(set(hadm_ids))) &
            (chunk["ITEMID"].isin(target_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    print(f"  INPUTEVENTS_CV (heparin): {len(result):,} rows")
    return result


def load_prescriptions(hadm_ids):
    """Load PRESCRIPTIONS for heparin/enoxaparin."""
    path = os.path.join(DATA_DIR, "PRESCRIPTIONS.csv.gz")
    usecols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "ENDDATE",
               "DRUG", "DRUG_NAME_GENERIC", "DOSE_VAL_RX", "DOSE_UNIT_RX",
               "ROUTE"]

    chunks = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=2_000_000,
                             usecols=usecols):
        filtered = chunk[chunk["HADM_ID"].isin(set(hadm_ids))]
        if len(filtered) > 0:
            # Filter for heparin/enoxaparin by drug name
            drug_lower = filtered["DRUG"].fillna("").str.lower()
            generic_lower = filtered["DRUG_NAME_GENERIC"].fillna("").str.lower()

            is_heparin = (
                drug_lower.str.contains("heparin", na=False) |
                generic_lower.str.contains("heparin", na=False)
            ) & ~(
                drug_lower.str.contains("flush|lock|enoxaparin|dalteparin|"
                                        "tinzaparin|fondaparinux", na=False) |
                generic_lower.str.contains("flush|lock|enoxaparin|dalteparin|"
                                           "tinzaparin|fondaparinux", na=False)
            )

            is_enoxaparin = (
                drug_lower.str.contains("enoxaparin|lovenox", na=False) |
                generic_lower.str.contains("enoxaparin|lovenox", na=False)
            )

            matched = filtered[is_heparin | is_enoxaparin].copy()
            matched["_is_heparin_rx"] = is_heparin[matched.index]
            matched["_is_enoxaparin_rx"] = is_enoxaparin[matched.index]
            if len(matched) > 0:
                chunks.append(matched)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    print(f"  PRESCRIPTIONS (heparin/enoxaparin): {len(result):,} rows")
    return result


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------
def extract_baseline_values(events_df, intime_map, time_col, value_col,
                            id_col, itemids, agg="first"):
    """
    Extract one baseline value per patient from an events DataFrame.

    Parameters
    ----------
    events_df : filtered events
    intime_map : {id: INTIME datetime}
    time_col : timestamp column name
    value_col : numeric value column name
    id_col : patient ID column (ICUSTAY_ID or HADM_ID)
    itemids : list of ITEMIDs to include
    agg : 'first' or 'max'
    """
    if events_df.empty:
        return pd.Series(dtype=float)

    df = events_df[events_df["ITEMID"].isin(set(itemids))].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[id_col, time_col, value_col])

    # Filter to baseline window
    df["_intime"] = df[id_col].map(intime_map)
    df = df.dropna(subset=["_intime"])
    df["_hours"] = (df[time_col] - df["_intime"]).dt.total_seconds() / 3600
    df = df[(df["_hours"] >= 0) & (df["_hours"] <= BASELINE_WINDOW_HOURS)]

    if df.empty:
        return pd.Series(dtype=float)

    if agg == "first":
        result = df.sort_values(time_col).groupby(id_col)[value_col].first()
    elif agg == "max":
        result = df.groupby(id_col)[value_col].max()
    elif agg == "min":
        result = df.groupby(id_col)[value_col].min()
    else:
        raise ValueError(f"Unknown agg: {agg}")

    return result


def extract_vitals(chartevents, intime_map):
    """Extract admission vitals from CHARTEVENTS."""
    result = pd.DataFrame(index=pd.Index(sorted(intime_map.keys()),
                                         name="ICUSTAY_ID"))

    all_map_ids = (VITALS_ITEMIDS["map"]["mv"] +
                   VITALS_ITEMIDS["map"]["cv"])
    result["admission_map"] = extract_baseline_values(
        chartevents, intime_map, "CHARTTIME", "VALUENUM",
        "ICUSTAY_ID", all_map_ids, agg="first"
    )

    all_sbp_ids = (VITALS_ITEMIDS["sbp"]["mv"] +
                   VITALS_ITEMIDS["sbp"]["cv"])
    result["admission_sbp"] = extract_baseline_values(
        chartevents, intime_map, "CHARTTIME", "VALUENUM",
        "ICUSTAY_ID", all_sbp_ids, agg="first"
    )

    all_hr_ids = (VITALS_ITEMIDS["heart_rate"]["mv"] +
                  VITALS_ITEMIDS["heart_rate"]["cv"])
    result["admission_heart_rate"] = extract_baseline_values(
        chartevents, intime_map, "CHARTTIME", "VALUENUM",
        "ICUSTAY_ID", all_hr_ids, agg="first"
    )

    # Temperature: get max, combine C and F (convert F→C)
    all_temp_c_ids = (VITALS_ITEMIDS["temp_c"]["mv"] +
                      VITALS_ITEMIDS["temp_c"]["cv"])
    temp_c = extract_baseline_values(
        chartevents, intime_map, "CHARTTIME", "VALUENUM",
        "ICUSTAY_ID", all_temp_c_ids, agg="max"
    )

    all_temp_f_ids = (VITALS_ITEMIDS["temp_f"]["mv"] +
                      VITALS_ITEMIDS["temp_f"]["cv"])
    temp_f = extract_baseline_values(
        chartevents, intime_map, "CHARTTIME", "VALUENUM",
        "ICUSTAY_ID", all_temp_f_ids, agg="max"
    )
    # Convert F to C
    temp_f_as_c = (temp_f - 32) * 5 / 9

    # Take max of both
    result["admission_temp"] = pd.concat([temp_c, temp_f_as_c], axis=1).max(axis=1)

    return result


def extract_labs(labevents, hadm_intime_map):
    """Extract admission labs from LABEVENTS."""
    result = pd.DataFrame(index=pd.Index(sorted(hadm_intime_map.keys()),
                                         name="HADM_ID"))

    for var_name, itemids in LAB_ITEMIDS.items():
        result[f"admission_{var_name}"] = extract_baseline_values(
            labevents, hadm_intime_map, "CHARTTIME", "VALUENUM",
            "HADM_ID", itemids, agg="first"
        )

    return result


def _parse_gcs_component(chartevents, intime_map, itemids, value_map):
    """Parse a GCS component from CHARTEVENTS, mapping text to numeric."""
    if not itemids:
        return pd.Series(dtype=float)

    df = chartevents[chartevents["ITEMID"].isin(set(itemids))].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], errors="coerce")

    # Try VALUENUM first; fall back to text mapping
    df["_numeric"] = pd.to_numeric(df["VALUENUM"], errors="coerce")
    mask_missing = df["_numeric"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "_numeric"] = (
            df.loc[mask_missing, "VALUE"].map(value_map)
        )
    df = df.dropna(subset=["ICUSTAY_ID", "CHARTTIME", "_numeric"])

    # Filter to baseline window
    df["_intime"] = df["ICUSTAY_ID"].map(intime_map)
    df = df.dropna(subset=["_intime"])
    df["_hours"] = (df["CHARTTIME"] - df["_intime"]).dt.total_seconds() / 3600
    df = df[(df["_hours"] >= 0) & (df["_hours"] <= BASELINE_WINDOW_HOURS)]

    if df.empty:
        return pd.Series(dtype=float)

    return df.sort_values("CHARTTIME").groupby("ICUSTAY_ID")["_numeric"].first()


def extract_gcs(chartevents, intime_map):
    """Extract GCS and derive Hunt-Hess / WFNS grades."""
    result = pd.DataFrame(index=pd.Index(sorted(intime_map.keys()),
                                         name="ICUSTAY_ID"))

    # MV components
    eye = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_eye"]["mv"], GCS_EYE_MAP
    )
    verbal = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_verbal"]["mv"], GCS_VERBAL_MAP
    )
    motor = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_motor"]["mv"], GCS_MOTOR_MAP
    )

    # MV total = sum of components
    mv_total = (eye.add(verbal, fill_value=0)
                   .add(motor, fill_value=0))
    # Only valid if all 3 components present
    mv_valid_mask = eye.notna() & verbal.reindex(eye.index, fill_value=np.nan).notna()
    mv_valid_mask = mv_valid_mask & motor.reindex(eye.index, fill_value=np.nan).notna()
    mv_total = mv_total.where(
        eye.notna() &
        verbal.reindex(eye.index).notna() &
        motor.reindex(eye.index).notna()
    )

    # CV total (ITEMID 198)
    cv_total_ids = VITALS_ITEMIDS["gcs_total"]["cv"]
    if cv_total_ids:
        cv_total = extract_baseline_values(
            chartevents, intime_map, "CHARTTIME", "VALUENUM",
            "ICUSTAY_ID", cv_total_ids, agg="first"
        )
    else:
        cv_total = pd.Series(dtype=float)

    # Combine: prefer MV (more granular), fill with CV
    gcs = mv_total.reindex(result.index)
    if not cv_total.empty:
        gcs = gcs.fillna(cv_total.reindex(result.index))

    result["gcs_admission"] = gcs.clip(3, 15)

    # Derive Hunt-Hess from GCS
    def gcs_to_hh(g):
        if pd.isna(g):
            return np.nan
        if g >= 15:
            return 2
        elif g >= 13:
            return 3
        elif g >= 8:
            return 4
        else:
            return 5

    result["hunt_hess_derived"] = result["gcs_admission"].apply(gcs_to_hh)

    # Derive WFNS from GCS (same mapping for this proxy)
    result["wfns_derived"] = result["gcs_admission"].apply(gcs_to_hh)

    return result


def extract_comorbidities(cohort, diagnoses_path):
    """Extract comorbidity flags from DIAGNOSES_ICD."""
    diag = pd.read_csv(diagnoses_path, compression="gzip")
    diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.strip()

    hadm_ids = set(cohort["HADM_ID"])
    diag = diag[diag["HADM_ID"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="HADM_ID"))

    for var_name, prefixes in COMORBIDITY_ICD9.items():
        mask = diag["ICD9_CODE"].apply(
            lambda c: any(c.startswith(p) for p in prefixes)
        )
        positive_hadm = set(diag.loc[mask, "HADM_ID"])
        result[var_name] = result.index.isin(positive_hadm).astype(int)

    return result


def extract_treatment(inputevents_mv, inputevents_cv, prescriptions,
                      cohort):
    """Extract heparin/enoxaparin treatment variables."""
    hadm_ids = sorted(cohort["HADM_ID"].unique())
    result = pd.DataFrame(index=pd.Index(hadm_ids, name="HADM_ID"))

    # --- Heparin from INPUTEVENTS_MV ---
    hep_mv = pd.DataFrame()
    if not inputevents_mv.empty:
        hep_mv = inputevents_mv[
            inputevents_mv["ITEMID"].isin(HEPARIN_MV_ITEMIDS)
        ].copy()
        # Exclude flushes
        if "ORDERCATEGORYNAME" in hep_mv.columns:
            hep_mv = hep_mv[
                ~hep_mv["ORDERCATEGORYNAME"].fillna("").str.lower()
                .str.contains("flush|lock")
            ]
        hep_mv["AMOUNT"] = pd.to_numeric(hep_mv["AMOUNT"], errors="coerce")
        hep_mv = hep_mv[hep_mv["AMOUNT"] >= 100]  # exclude flush doses

    # --- Heparin from INPUTEVENTS_CV ---
    hep_cv = pd.DataFrame()
    if not inputevents_cv.empty:
        hep_cv = inputevents_cv[
            inputevents_cv["ITEMID"].isin(HEPARIN_CV_ITEMIDS)
        ].copy()
        hep_cv["AMOUNT"] = pd.to_numeric(hep_cv["AMOUNT"], errors="coerce")
        hep_cv = hep_cv[hep_cv["AMOUNT"] >= 100]

    # --- Enoxaparin from INPUTEVENTS_MV ---
    enox_mv = pd.DataFrame()
    if not inputevents_mv.empty:
        enox_mv = inputevents_mv[
            inputevents_mv["ITEMID"].isin(ENOXAPARIN_MV_ITEMIDS)
        ].copy()
        enox_mv["AMOUNT"] = pd.to_numeric(enox_mv["AMOUNT"], errors="coerce")

    # --- Prescriptions ---
    hep_rx = pd.DataFrame()
    enox_rx = pd.DataFrame()
    if not prescriptions.empty:
        hep_rx = prescriptions[
            prescriptions["_is_heparin_rx"] == True  # noqa: E712
        ].copy()
        enox_rx = prescriptions[
            prescriptions["_is_enoxaparin_rx"] == True  # noqa: E712
        ].copy()

    # Combine heparin sources
    hep_hadm = set()
    if not hep_mv.empty:
        hep_hadm.update(hep_mv["HADM_ID"].unique())
    if not hep_cv.empty:
        hep_hadm.update(hep_cv["HADM_ID"].unique())
    if not hep_rx.empty:
        hep_hadm.update(hep_rx["HADM_ID"].unique())

    # Combine enoxaparin sources
    enox_hadm = set()
    if not enox_mv.empty:
        enox_hadm.update(enox_mv["HADM_ID"].unique())
    if not enox_rx.empty:
        enox_hadm.update(enox_rx["HADM_ID"].unique())

    result["heparin_ever"] = result.index.isin(hep_hadm).astype(int)
    result["enoxaparin_ever"] = result.index.isin(enox_hadm).astype(int)
    result["received_both"] = (
        (result["heparin_ever"] == 1) & (result["enoxaparin_ever"] == 1)
    ).astype(int)

    # Max daily heparin dose (from inputevents + prescriptions)
    hep_dose_parts = []
    if not hep_mv.empty:
        hep_dose_parts.append(
            hep_mv[["HADM_ID", "AMOUNT"]].rename(columns={"AMOUNT": "dose"})
        )
    if not hep_cv.empty:
        hep_dose_parts.append(
            hep_cv[["HADM_ID", "AMOUNT"]].rename(columns={"AMOUNT": "dose"})
        )
    if not hep_rx.empty:
        rx_dose = hep_rx[["HADM_ID", "DOSE_VAL_RX"]].copy()
        rx_dose["dose"] = pd.to_numeric(rx_dose["DOSE_VAL_RX"], errors="coerce")
        hep_dose_parts.append(rx_dose[["HADM_ID", "dose"]])

    if hep_dose_parts:
        hep_all = pd.concat(hep_dose_parts, ignore_index=True)
        hep_all["dose"] = pd.to_numeric(hep_all["dose"], errors="coerce")
        hep_max = hep_all.groupby("HADM_ID")["dose"].max()
        result["heparin_max_daily_dose"] = hep_max
    else:
        result["heparin_max_daily_dose"] = np.nan

    # Max daily enoxaparin dose
    enox_all = pd.concat([
        enox_mv[["HADM_ID", "AMOUNT"]].rename(columns={"AMOUNT": "dose"})
        if not enox_mv.empty else pd.DataFrame(columns=["HADM_ID", "dose"]),
    ], ignore_index=True)
    # Also try prescriptions for dose
    if not enox_rx.empty:
        enox_rx_dose = enox_rx[["HADM_ID", "DOSE_VAL_RX"]].copy()
        enox_rx_dose["dose"] = pd.to_numeric(enox_rx_dose["DOSE_VAL_RX"],
                                             errors="coerce")
        enox_all = pd.concat([enox_all, enox_rx_dose[["HADM_ID", "dose"]]],
                             ignore_index=True)
    if not enox_all.empty:
        enox_all["dose"] = pd.to_numeric(enox_all["dose"], errors="coerce")
        enox_max = enox_all.groupby("HADM_ID")["dose"].max()
        result["enoxaparin_max_daily_dose"] = enox_max
    else:
        result["enoxaparin_max_daily_dose"] = np.nan

    # First dose timestamps
    def _first_timestamp(df, time_col, hadm_col="HADM_ID"):
        if df.empty:
            return pd.Series(dtype="datetime64[ns]")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        return df.dropna(subset=[time_col]).sort_values(time_col).groupby(hadm_col)[time_col].first()

    hep_times = []
    if not hep_mv.empty:
        hep_times.append(_first_timestamp(hep_mv, "STARTTIME"))
    if not hep_cv.empty:
        hep_times.append(_first_timestamp(hep_cv, "CHARTTIME"))
    if hep_times:
        hep_first = pd.concat(hep_times).groupby(level=0).min()
        result["heparin_first_dose_time"] = hep_first
    else:
        result["heparin_first_dose_time"] = pd.NaT

    enox_times = []
    if not enox_mv.empty:
        enox_times.append(_first_timestamp(enox_mv, "STARTTIME"))
    if enox_times:
        enox_first = pd.concat(enox_times).groupby(level=0).min()
        result["enoxaparin_first_dose_time"] = enox_first
    else:
        result["enoxaparin_first_dose_time"] = pd.NaT

    return result


def extract_procedures(cohort, procedures_path):
    """Extract procedure flags from PROCEDURES_ICD."""
    proc = pd.read_csv(procedures_path, compression="gzip")
    proc["ICD9_CODE"] = proc["ICD9_CODE"].astype(str).str.strip()

    hadm_ids = set(cohort["HADM_ID"])
    proc = proc[proc["HADM_ID"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="HADM_ID"))

    for var_name, codes in PROCEDURE_ICD9.items():
        positive_hadm = set(
            proc.loc[proc["ICD9_CODE"].isin(codes), "HADM_ID"]
        )
        result[var_name] = result.index.isin(positive_hadm).astype(int)

    return result


def extract_outcomes(cohort, diagnoses_path):
    """Extract outcome variables."""
    diag = pd.read_csv(diagnoses_path, compression="gzip")
    diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.strip()

    hadm_ids = set(cohort["HADM_ID"])
    diag = diag[diag["HADM_ID"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="HADM_ID"))

    # Hospital mortality (from cohort)
    mort_map = cohort.set_index("HADM_ID")["HOSPITAL_EXPIRE_FLAG"]
    result["hospital_mortality"] = mort_map

    # DCI (ICD-9 proxy)
    dci_hadm = set(diag.loc[diag["ICD9_CODE"].isin(DCI_ICD9), "HADM_ID"])
    result["dci_icd9"] = result.index.isin(dci_hadm).astype(int)

    # Rebleeding (ICD-9 proxy)
    rebleed_hadm = set(
        diag.loc[diag["ICD9_CODE"].isin(REBLEEDING_ICD9), "HADM_ID"]
    )
    result["rebleeding_icd9"] = result.index.isin(rebleed_hadm).astype(int)

    # ICU LOS
    cohort_idx = cohort.set_index("HADM_ID")
    result["icu_los_days"] = (
        (cohort_idx["OUTTIME"] - cohort_idx["INTIME"])
        .dt.total_seconds() / 86400
    )

    # Hospital LOS
    result["hospital_los_days"] = (
        (cohort_idx["DISCHTIME"] - cohort_idx["ADMITTIME"])
        .dt.total_seconds() / 86400
    )

    return result


# ---------------------------------------------------------------------------
# Build patient table
# ---------------------------------------------------------------------------
def build_patient_table(cohort, vitals, labs, gcs, comorbidities,
                        treatment, procedures, outcomes):
    """Combine all extractions into one-row-per-patient table."""
    pt = cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "age", "GENDER",
                 "ETHNICITY"]].copy()
    pt = pt.set_index("HADM_ID")

    # Demographics
    pt["sex"] = pt["GENDER"].str.upper().map({"M": 0, "F": 1})
    pt.drop(columns=["GENDER"], inplace=True)

    # Vitals (indexed by ICUSTAY_ID → map to HADM_ID)
    icu_to_hadm = cohort.set_index("ICUSTAY_ID")["HADM_ID"].to_dict()
    vitals_reindexed = vitals.copy()
    vitals_reindexed.index = vitals_reindexed.index.map(
        lambda x: icu_to_hadm.get(x, x)
    )
    for col in vitals_reindexed.columns:
        pt[col] = vitals_reindexed[col]

    # Labs (indexed by HADM_ID)
    for col in labs.columns:
        pt[col] = labs[col]

    # GCS (indexed by ICUSTAY_ID → map to HADM_ID)
    gcs_reindexed = gcs.copy()
    gcs_reindexed.index = gcs_reindexed.index.map(
        lambda x: icu_to_hadm.get(x, x)
    )
    for col in gcs_reindexed.columns:
        pt[col] = gcs_reindexed[col]

    # Comorbidities (indexed by HADM_ID)
    for col in comorbidities.columns:
        pt[col] = comorbidities[col]

    # Treatment (indexed by HADM_ID)
    for col in treatment.columns:
        pt[col] = treatment[col]

    # Procedures (indexed by HADM_ID)
    for col in procedures.columns:
        pt[col] = procedures[col]

    # Outcomes (indexed by HADM_ID)
    for col in outcomes.columns:
        pt[col] = outcomes[col]

    # Unavailable variables — NaN columns
    for col in ["fisher_grade", "wfns_raw", "ivh", "ich",
                "aneurysm_size", "aneurysm_location", "aneurysm_multiplicity",
                "time_ictus_to_treatment", "premorbid_mrs",
                "mrs_discharge", "mrs_1y", "crp"]:
        if col not in pt.columns:
            pt[col] = np.nan

    pt = pt.reset_index()
    return pt


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------
VARIABLE_META = [
    # (column_name, display_name, category, available_in_data)
    # Demographics
    ("age", "Age", "Demographics", True),
    ("sex", "Sex", "Demographics", True),
    ("ETHNICITY", "Ethnicity", "Demographics", True),
    # Comorbidities
    ("hypertension", "Hypertension", "Comorbidities", True),
    ("diabetes", "Diabetes mellitus", "Comorbidities", True),
    ("smoking_history", "Smoking history", "Comorbidities", True),
    ("renal_disease", "Renal disease", "Comorbidities", True),
    ("liver_disease", "Liver disease", "Comorbidities", True),
    ("prior_anticoagulant", "Prior anticoagulant use", "Comorbidities", True),
    ("prior_antiplatelet", "Prior antiplatelet use", "Comorbidities", True),
    # Clinical Grades
    ("gcs_admission", "GCS on admission", "Clinical Grades", True),
    ("hunt_hess_derived", "Hunt-Hess (derived from GCS)", "Clinical Grades", True),
    ("wfns_derived", "WFNS (derived from GCS)", "Clinical Grades", True),
    ("fisher_grade", "Fisher grade", "Clinical Grades", False),
    ("ivh", "IVH", "Clinical Grades", False),
    ("ich", "ICH", "Clinical Grades", False),
    # Admission Physiology
    ("admission_map", "Admission MAP", "Physiology", True),
    ("admission_sbp", "Admission SBP", "Physiology", True),
    ("admission_heart_rate", "Admission heart rate", "Physiology", True),
    ("admission_temp", "Admission temperature", "Physiology", True),
    ("admission_wbc", "Admission WBC", "Physiology", True),
    ("admission_hemoglobin", "Admission hemoglobin", "Physiology", True),
    ("admission_creatinine", "Admission creatinine", "Physiology", True),
    ("admission_glucose", "Admission glucose", "Physiology", True),
    ("admission_sodium", "Admission sodium", "Physiology", True),
    ("admission_potassium", "Admission potassium", "Physiology", True),
    ("admission_platelets", "Admission platelets", "Physiology", True),
    ("admission_inr", "Admission INR", "Physiology", True),
    ("admission_ptt", "Admission PTT", "Physiology", True),
    ("admission_lactate", "Admission lactate", "Physiology", True),
    ("admission_albumin", "Admission albumin", "Physiology", True),
    ("admission_bilirubin", "Admission bilirubin", "Physiology", True),
    ("crp", "CRP", "Physiology", False),
    # Treatment
    ("heparin_ever", "Heparin exposure", "Treatment", True),
    ("enoxaparin_ever", "Enoxaparin exposure", "Treatment", True),
    ("received_both", "Received both agents", "Treatment", True),
    ("heparin_max_daily_dose", "Heparin max daily dose", "Treatment", True),
    ("enoxaparin_max_daily_dose", "Enoxaparin max daily dose", "Treatment", True),
    ("treatment_clipping", "Clipping", "Procedures", True),
    ("treatment_coiling", "Coiling", "Procedures", True),
    ("evd_placement", "EVD placement", "Procedures", True),
    # Aneurysm (unavailable)
    ("aneurysm_size", "Aneurysm size", "Aneurysm", False),
    ("aneurysm_location", "Aneurysm location", "Aneurysm", False),
    ("aneurysm_multiplicity", "Multiple aneurysms", "Aneurysm", False),
    ("time_ictus_to_treatment", "Time ictus-to-treatment", "Aneurysm", False),
    ("premorbid_mrs", "Pre-morbid mRS", "Aneurysm", False),
    # Outcomes
    ("hospital_mortality", "Hospital mortality", "Outcomes", True),
    ("dci_icd9", "DCI (ICD-9 proxy)", "Outcomes", True),
    ("rebleeding_icd9", "Rebleeding (ICD-9 proxy)", "Outcomes", True),
    ("icu_los_days", "ICU length of stay", "Outcomes", True),
    ("hospital_los_days", "Hospital length of stay", "Outcomes", True),
    ("mrs_discharge", "mRS at discharge", "Outcomes", False),
    ("mrs_1y", "mRS at 1 year", "Outcomes", False),
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
# Data dictionary generation
# ---------------------------------------------------------------------------
def generate_data_dictionary(patient_df, missingness_df):
    """Generate data_dictionary.md."""
    lines = [
        "# Data Dictionary: MIMIC-III aSAH Anticoagulation Study",
        "",
        "Maps each propensity covariate and outcome to its MIMIC-III source, "
        "extraction logic, and availability.",
        "",
        "## Data Sources",
        "",
        "| Abbreviation | MIMIC-III Table | Notes |",
        "|---|---|---|",
        "| CHARTEVENTS | `CHARTEVENTS.csv.gz` | Vitals, GCS (~330M rows, chunked) |",
        "| LABEVENTS | `LABEVENTS.csv.gz` | Laboratory results (~27M rows, chunked) |",
        "| INPUTEVENTS_MV | `INPUTEVENTS_MV.csv.gz` | Medication inputs (Metavision) |",
        "| INPUTEVENTS_CV | `INPUTEVENTS_CV.csv.gz` | Medication inputs (CareVue) |",
        "| PRESCRIPTIONS | `PRESCRIPTIONS.csv.gz` | Prescription orders |",
        "| DIAGNOSES_ICD | `DIAGNOSES_ICD.csv.gz` | ICD-9 diagnosis codes |",
        "| PROCEDURES_ICD | `PROCEDURES_ICD.csv.gz` | ICD-9 procedure codes |",
        "",
        f"Baseline window: first {BASELINE_WINDOW_HOURS} hours after ICU admission (INTIME).",
        "",
        "---",
        "",
    ]

    # Group by category
    categories = {}
    for col, name, category, available in VARIABLE_META:
        if category not in categories:
            categories[category] = []
        # Find missingness info
        miss_row = missingness_df[missingness_df["column"] == col]
        pct = miss_row["pct_missing"].values[0] if len(miss_row) > 0 else 100.0
        categories[category].append((col, name, available, pct))

    for cat, vars_list in categories.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Variable | Column | Available | Missingness |")
        lines.append("|---|---|---|---|")
        for col, name, available, pct in vars_list:
            status = "Yes" if available else "No"
            lines.append(f"| {name} | `{col}` | {status} | {pct:.1f}% |")
        lines.append("")

    # ITEMID reference
    lines.extend([
        "---",
        "",
        "## ITEMID Reference",
        "",
        "### Vitals (CHARTEVENTS)",
        "",
        "| Variable | Metavision ITEMIDs | CareVue ITEMIDs | Aggregation |",
        "|---|---|---|---|",
    ])
    vital_aggs = {"map": "first", "sbp": "first", "heart_rate": "first",
                  "temp_c": "max", "temp_f": "max (→°C)"}
    for var, ids in VITALS_ITEMIDS.items():
        if var.startswith("gcs"):
            continue
        mv_str = ", ".join(str(i) for i in ids["mv"])
        cv_str = ", ".join(str(i) for i in ids["cv"])
        agg = vital_aggs.get(var, "first")
        lines.append(f"| {var} | {mv_str} | {cv_str} | {agg} |")

    lines.extend([
        "",
        "### GCS (CHARTEVENTS)",
        "",
        "| Component | Metavision ITEMIDs | CareVue ITEMIDs |",
        "|---|---|---|",
        "| Eye | 220739 | — |",
        "| Verbal | 223900 | — |",
        "| Motor | 223901 | — |",
        "| Total | — | 198 |",
        "",
        "### Labs (LABEVENTS)",
        "",
        "| Variable | ITEMIDs |",
        "|---|---|",
    ])
    for var, ids in LAB_ITEMIDS.items():
        ids_str = ", ".join(str(i) for i in ids)
        lines.append(f"| {var} | {ids_str} |")

    lines.extend([
        "",
        "### Treatment (INPUTEVENTS_MV / CV / PRESCRIPTIONS)",
        "",
        "| Drug | MV ITEMIDs | CV ITEMIDs | PRESCRIPTIONS filter |",
        "|---|---|---|---|",
        "| Heparin | 225152, 225975 | 30025 | DRUG ilike '%heparin%' "
        "(excl. flush/lock/LMWH) |",
        "| Enoxaparin | 225906 | — | DRUG ilike '%enoxaparin%' or '%lovenox%' |",
        "",
        "### Comorbidities (DIAGNOSES_ICD, prefix matching)",
        "",
        "| Variable | ICD-9 prefixes |",
        "|---|---|",
    ])
    for var, prefixes in COMORBIDITY_ICD9.items():
        pref_str = ", ".join(prefixes)
        lines.append(f"| {var} | {pref_str} |")

    lines.extend([
        "",
        "### Procedures (PROCEDURES_ICD)",
        "",
        "| Variable | ICD-9 codes |",
        "|---|---|",
    ])
    for var, codes in PROCEDURE_ICD9.items():
        codes_str = ", ".join(codes)
        lines.append(f"| {var} | {codes_str} |")

    lines.extend([
        "",
        "### Outcomes (DIAGNOSES_ICD)",
        "",
        "| Variable | ICD-9 codes |",
        "|---|---|",
        f"| DCI (proxy) | {', '.join(DCI_ICD9)} |",
        f"| Rebleeding | {', '.join(REBLEEDING_ICD9)} |",
        "",
        "---",
        "",
        "## Variables NOT Available in MIMIC-III",
        "",
        "| Variable | Category | Reason |",
        "|---|---|---|",
        "| Fisher grade | Clinical Grades | Not coded in structured data |",
        "| WFNS (raw) | Clinical Grades | Not coded; derived from GCS instead |",
        "| IVH | Clinical Grades | Not reliably coded in ICD-9 |",
        "| ICH | Clinical Grades | Not reliably coded in ICD-9 |",
        "| Aneurysm size | Aneurysm | Not in structured data |",
        "| Aneurysm location | Aneurysm | Not in structured data |",
        "| Multiple aneurysms | Aneurysm | Not in structured data |",
        "| Time ictus-to-treatment | Treatment | Ictus time not recorded |",
        "| Pre-morbid mRS | Comorbidities | Not recorded |",
        "| CRP | Physiology | Not available in LABEVENTS |",
        "| mRS at discharge | Outcomes | Not recorded in MIMIC-III |",
        "| mRS at 1 year | Outcomes | Not recorded in MIMIC-III |",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VARIABLE EXTRACTION — MIMIC-III aSAH Study")
    print("=" * 80)

    # Load cohort
    print("\nLoading cohort...")
    cohort = load_cohort()

    # Build ID maps
    icustay_intime = dict(zip(cohort["ICUSTAY_ID"], cohort["INTIME"]))
    hadm_intime = dict(zip(cohort["HADM_ID"], cohort["INTIME"]))
    hadm_ids = set(cohort["HADM_ID"])
    icustay_ids = set(cohort["ICUSTAY_ID"])

    # Collect all needed ITEMIDs
    all_chart_itemids = _collect_all_chart_itemids()
    all_lab_itemids = _collect_all_lab_itemids()

    # --- Load large tables (single-pass chunked) ---
    print("\nLoading CHARTEVENTS (chunked)...")
    chartevents = load_filtered_events(
        os.path.join(DATA_DIR, "CHARTEVENTS.csv.gz"),
        id_col="ICUSTAY_ID",
        cohort_ids=icustay_ids,
        target_itemids=all_chart_itemids,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID",
                 "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM"],
    )

    print("\nLoading LABEVENTS (chunked)...")
    labevents = load_filtered_events(
        os.path.join(DATA_DIR, "LABEVENTS.csv.gz"),
        id_col="HADM_ID",
        cohort_ids=hadm_ids,
        target_itemids=all_lab_itemids,
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME",
                 "VALUE", "VALUENUM", "VALUEUOM"],
    )

    print("\nLoading INPUTEVENTS_MV...")
    inputevents_mv = load_inputevents_mv(hadm_ids)

    print("\nLoading INPUTEVENTS_CV...")
    inputevents_cv = load_inputevents_cv(hadm_ids)

    print("\nLoading PRESCRIPTIONS...")
    prescriptions = load_prescriptions(hadm_ids)

    # --- Extract variables ---
    print("\nExtracting vitals...")
    vitals = extract_vitals(chartevents, icustay_intime)
    print(f"  Vitals extracted: {vitals.notna().sum().to_dict()}")

    print("\nExtracting labs...")
    labs = extract_labs(labevents, hadm_intime)
    print(f"  Labs extracted: {labs.notna().sum().to_dict()}")

    print("\nExtracting GCS...")
    gcs = extract_gcs(chartevents, icustay_intime)
    print(f"  GCS extracted: {gcs.notna().sum().to_dict()}")

    print("\nExtracting comorbidities...")
    comorbidities = extract_comorbidities(
        cohort, os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv.gz")
    )
    print(f"  Comorbidities: {comorbidities.sum().to_dict()}")

    print("\nExtracting treatment...")
    treatment = extract_treatment(
        inputevents_mv, inputevents_cv, prescriptions, cohort
    )
    print(f"  Heparin: {treatment['heparin_ever'].sum()} patients")
    print(f"  Enoxaparin: {treatment['enoxaparin_ever'].sum()} patients")
    print(f"  Both: {treatment['received_both'].sum()} patients")

    print("\nExtracting procedures...")
    procedures = extract_procedures(
        cohort, os.path.join(DATA_DIR, "PROCEDURES_ICD.csv.gz")
    )
    print(f"  Procedures: {procedures.sum().to_dict()}")

    print("\nExtracting outcomes...")
    outcomes = extract_outcomes(
        cohort, os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv.gz")
    )
    print(f"  Hospital mortality: {outcomes['hospital_mortality'].sum()}")
    print(f"  DCI (ICD-9): {outcomes['dci_icd9'].sum()}")
    print(f"  Rebleeding (ICD-9): {outcomes['rebleeding_icd9'].sum()}")

    # --- Build patient table ---
    print("\nBuilding patient table...")
    pt = build_patient_table(
        cohort, vitals, labs, gcs, comorbidities,
        treatment, procedures, outcomes
    )
    print(f"  Patient table: {len(pt)} rows x {len(pt.columns)} columns")

    # Verification
    assert len(pt) == len(cohort), (
        f"Row count mismatch: {len(pt)} vs {len(cohort)}"
    )
    assert pt["SUBJECT_ID"].is_unique, "Duplicate SUBJECT_IDs!"
    if pt["gcs_admission"].notna().any():
        gcs_range = pt["gcs_admission"].dropna()
        assert gcs_range.min() >= 3 and gcs_range.max() <= 15, (
            f"GCS out of range: {gcs_range.min()}-{gcs_range.max()}"
        )
    if pt["hunt_hess_derived"].notna().any():
        hh_range = pt["hunt_hess_derived"].dropna()
        assert hh_range.min() >= 1 and hh_range.max() <= 5, (
            f"Hunt-Hess out of range: {hh_range.min()}-{hh_range.max()}"
        )

    # --- Compute missingness ---
    print("\nComputing missingness...")
    miss = compute_missingness(pt)

    # --- Save outputs ---
    pt_path = ARTIFACTS_DIR / "patient_table.csv"
    pt.to_csv(pt_path, index=False)
    print(f"\nSaved {pt_path} ({len(pt)} patients x {len(pt.columns)} cols)")

    miss_path = ARTIFACTS_DIR / "missingness_table.csv"
    miss.to_csv(miss_path, index=False)
    print(f"Saved {miss_path} ({len(miss)} variables)")

    dd_text = generate_data_dictionary(pt, miss)
    dd_path = ARTIFACTS_DIR / "data_dictionary.md"
    dd_path.write_text(dd_text)
    print(f"Saved {dd_path}")

    # --- Print summary ---
    print(f"\n{'=' * 80}")
    print(f"MISSINGNESS SUMMARY (N = {len(pt)} patients)")
    print(f"{'=' * 80}")
    for cat in miss["category"].unique():
        cat_df = miss[miss["category"] == cat]
        print(f"\n--- {cat} ---")
        for _, row in cat_df.iterrows():
            status = "AVAILABLE" if row["available_in_data"] else "NOT IN DATA"
            bar = f"{row['n_available']:>4}/{row['n_total']}"
            print(f"  {row['variable']:<35} {bar}  "
                  f"({row['pct_missing']:>5.1f}% missing)  [{status}]")

    # Key stats
    print(f"\n{'=' * 80}")
    print("KEY STATISTICS")
    print(f"  DCI rate: {outcomes['dci_icd9'].mean() * 100:.1f}%")
    print(f"  Rebleeding rate: {outcomes['rebleeding_icd9'].mean() * 100:.1f}%")
    print(f"  Hospital mortality: {outcomes['hospital_mortality'].mean() * 100:.1f}%")
    print(f"  Median ICU LOS: {outcomes['icu_los_days'].median():.1f} days")
    print(f"  Median hospital LOS: {outcomes['hospital_los_days'].median():.1f} days")
    if pt["gcs_admission"].notna().any():
        print(f"  Median GCS: {pt['gcs_admission'].median():.0f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
