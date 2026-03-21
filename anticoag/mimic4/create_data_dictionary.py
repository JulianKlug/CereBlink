"""
Step 2: MIMIC-IV variable extraction for aSAH anticoagulation study.

Extracts treatment exposure, baseline confounders (within 48h of ICU admission),
and outcomes from MIMIC-IV. Produces patient_table.csv, data_dictionary.md,
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
HOSP_DIR = "/mnt/hdd1/datasets/mimiciv_3.1/hosp"
ICU_DIR = "/mnt/hdd1/datasets/mimiciv_3.1/icu"
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
COHORT_PATH = ARTIFACTS_DIR / "cohort.csv"

BASELINE_WINDOW_HOURS = 48

# ---------------------------------------------------------------------------
# ITEMID maps (Metavision only — MIMIC-IV has no CareVue)
# ---------------------------------------------------------------------------

# Treatment ITEMIDs
HEPARIN_ITEMIDS = {225152, 225975}
ENOXAPARIN_ITEMIDS = {225906}

# Vitals (CHARTEVENTS — Metavision only)
VITALS_ITEMIDS = {
    "map": [220052, 220181],
    "sbp": [220050, 220179],
    "heart_rate": [220045],
    "temp_c": [223762],
    "temp_f": [223761],
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
}

# Labs (LABEVENTS — same ITEMIDs)
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

# ICD prefix maps for comorbidities (dual ICD-9/ICD-10)
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

COMORBIDITY_ICD10 = {
    "hypertension": ["I10", "I11", "I12", "I13", "I14", "I15", "I16"],
    "diabetes": ["E10", "E11", "E12", "E13", "E14"],
    "smoking_history": ["F17", "Z87891"],
    "renal_disease": ["N18", "Z940"],
    "liver_disease": ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"],
    "prior_anticoagulant": ["Z7901"],
    "prior_antiplatelet": ["Z7902", "Z7982"],
}

# Procedure ICD codes (dual)
PROCEDURE_ICD9 = {
    "treatment_clipping": ["3951"],
    "treatment_coiling": ["3972", "3975", "3976"],
    "evd_placement": ["0231", "231"],
}

PROCEDURE_ICD10_PREFIXES = {
    "treatment_clipping": ["03VG0", "03LG0"],
    "treatment_coiling": ["03VG3", "03LG3"],
    "evd_placement": ["009630Z", "00960"],
}

# Outcome ICD codes (dual)
DCI_ICD9 = ["43401", "43411", "43491", "4371"]
DCI_ICD10_PREFIXES = ["I63", "I6784"]

REBLEEDING_ICD9 = ["4320", "4321", "4329", "99811", "99812"]
REBLEEDING_ICD10_PREFIXES = ["I60", "I61"]

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
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="coerce")
    print(f"  Cohort: {len(cohort)} patients")
    return cohort


def _collect_all_chart_itemids():
    """Collect all CHARTEVENTS ITEMIDs needed in a single set."""
    ids = set()
    for itemids in VITALS_ITEMIDS.values():
        ids.update(itemids)
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

    Single-pass through potentially huge tables (chartevents ~330M rows).
    """
    cohort_set = set(cohort_ids)
    itemid_set = set(target_itemids)
    chunks = []

    for chunk in pd.read_csv(path, compression="gzip", chunksize=chunksize,
                             usecols=usecols):
        filtered = chunk[
            (chunk[id_col].isin(cohort_set)) &
            (chunk["itemid"].isin(itemid_set))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {path.split('/')[-1]}: {len(result):,} relevant rows")
    return result


def load_inputevents(hadm_ids):
    """Load inputevents for heparin/enoxaparin (single table in MIMIC-IV)."""
    path = os.path.join(ICU_DIR, "inputevents.csv.gz")
    target_ids = HEPARIN_ITEMIDS | ENOXAPARIN_ITEMIDS
    usecols = ["subject_id", "hadm_id", "stay_id", "starttime", "endtime",
               "itemid", "amount", "amountuom", "rate", "rateuom",
               "statusdescription", "ordercategoryname"]

    chunks = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=2_000_000,
                             usecols=usecols):
        filtered = chunk[
            (chunk["hadm_id"].isin(set(hadm_ids))) &
            (chunk["itemid"].isin(target_ids))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    # Exclude rewritten orders
    result = result[result["statusdescription"] != "Rewritten"]
    print(f"  inputevents (heparin/enoxaparin): {len(result):,} rows")
    return result


def load_prescriptions(hadm_ids):
    """Load prescriptions for heparin/enoxaparin."""
    path = os.path.join(HOSP_DIR, "prescriptions.csv.gz")
    usecols = ["subject_id", "hadm_id", "starttime", "stoptime",
               "drug", "dose_val_rx", "dose_unit_rx", "doses_per_24_hrs",
               "route"]

    chunks = []
    for chunk in pd.read_csv(path, compression="gzip", chunksize=2_000_000,
                             usecols=usecols):
        filtered = chunk[chunk["hadm_id"].isin(set(hadm_ids))]
        if len(filtered) > 0:
            drug_lower = filtered["drug"].fillna("").str.lower()

            is_heparin = (
                drug_lower.str.contains("heparin", na=False)
            ) & ~(
                drug_lower.str.contains("flush|lock|enoxaparin|dalteparin|"
                                        "tinzaparin|fondaparinux", na=False)
            )

            is_enoxaparin = (
                drug_lower.str.contains("enoxaparin|lovenox", na=False)
            )

            matched = filtered[is_heparin | is_enoxaparin].copy()
            matched["_is_heparin_rx"] = is_heparin[matched.index]
            matched["_is_enoxaparin_rx"] = is_enoxaparin[matched.index]
            if len(matched) > 0:
                chunks.append(matched)

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks, ignore_index=True)
    print(f"  prescriptions (heparin/enoxaparin): {len(result):,} rows")
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
    intime_map : {id: intime datetime}
    time_col : timestamp column name
    value_col : numeric value column name
    id_col : patient ID column (stay_id or hadm_id)
    itemids : list of ITEMIDs to include
    agg : 'first' or 'max'
    """
    if events_df.empty:
        return pd.Series(dtype=float)

    df = events_df[events_df["itemid"].isin(set(itemids))].copy()
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
    """Extract admission vitals from chartevents."""
    result = pd.DataFrame(index=pd.Index(sorted(intime_map.keys()),
                                         name="stay_id"))

    result["admission_map"] = extract_baseline_values(
        chartevents, intime_map, "charttime", "valuenum",
        "stay_id", VITALS_ITEMIDS["map"], agg="first"
    )

    result["admission_sbp"] = extract_baseline_values(
        chartevents, intime_map, "charttime", "valuenum",
        "stay_id", VITALS_ITEMIDS["sbp"], agg="first"
    )

    result["admission_heart_rate"] = extract_baseline_values(
        chartevents, intime_map, "charttime", "valuenum",
        "stay_id", VITALS_ITEMIDS["heart_rate"], agg="first"
    )

    # Temperature: get max, combine C and F (convert F->C)
    temp_c = extract_baseline_values(
        chartevents, intime_map, "charttime", "valuenum",
        "stay_id", VITALS_ITEMIDS["temp_c"], agg="max"
    )

    temp_f = extract_baseline_values(
        chartevents, intime_map, "charttime", "valuenum",
        "stay_id", VITALS_ITEMIDS["temp_f"], agg="max"
    )
    # Convert F to C
    temp_f_as_c = (temp_f - 32) * 5 / 9

    # Take max of both
    result["admission_temp"] = pd.concat([temp_c, temp_f_as_c], axis=1).max(axis=1)

    return result


def extract_labs(labevents, hadm_intime_map):
    """Extract admission labs from labevents."""
    result = pd.DataFrame(index=pd.Index(sorted(hadm_intime_map.keys()),
                                         name="hadm_id"))

    for var_name, itemids in LAB_ITEMIDS.items():
        result[f"admission_{var_name}"] = extract_baseline_values(
            labevents, hadm_intime_map, "charttime", "valuenum",
            "hadm_id", itemids, agg="first"
        )

    return result


def _parse_gcs_component(chartevents, intime_map, itemids, value_map):
    """Parse a GCS component from chartevents, mapping text to numeric."""
    if not itemids:
        return pd.Series(dtype=float)

    df = chartevents[chartevents["itemid"].isin(set(itemids))].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

    # Try valuenum first; fall back to text mapping
    df["_numeric"] = pd.to_numeric(df["valuenum"], errors="coerce")
    mask_missing = df["_numeric"].isna()
    if mask_missing.any():
        df.loc[mask_missing, "_numeric"] = (
            df.loc[mask_missing, "value"].map(value_map)
        )
    df = df.dropna(subset=["stay_id", "charttime", "_numeric"])

    # Filter to baseline window
    df["_intime"] = df["stay_id"].map(intime_map)
    df = df.dropna(subset=["_intime"])
    df["_hours"] = (df["charttime"] - df["_intime"]).dt.total_seconds() / 3600
    df = df[(df["_hours"] >= 0) & (df["_hours"] <= BASELINE_WINDOW_HOURS)]

    if df.empty:
        return pd.Series(dtype=float)

    return df.sort_values("charttime").groupby("stay_id")["_numeric"].first()


def extract_gcs(chartevents, intime_map):
    """Extract GCS and derive Hunt-Hess / WFNS grades."""
    result = pd.DataFrame(index=pd.Index(sorted(intime_map.keys()),
                                         name="stay_id"))

    # MV components (MIMIC-IV is Metavision only)
    eye = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_eye"], GCS_EYE_MAP
    )
    verbal = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_verbal"], GCS_VERBAL_MAP
    )
    motor = _parse_gcs_component(
        chartevents, intime_map,
        VITALS_ITEMIDS["gcs_motor"], GCS_MOTOR_MAP
    )

    # Total = sum of components (only valid if all 3 present)
    mv_total = (eye.add(verbal, fill_value=0)
                   .add(motor, fill_value=0))
    mv_total = mv_total.where(
        eye.notna() &
        verbal.reindex(eye.index).notna() &
        motor.reindex(eye.index).notna()
    )

    gcs = mv_total.reindex(result.index)
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
    result["wfns_derived"] = result["gcs_admission"].apply(gcs_to_hh)

    return result


def extract_comorbidities(cohort, diagnoses_path):
    """Extract comorbidity flags from diagnoses_icd (dual ICD-9/ICD-10)."""
    diag = pd.read_csv(diagnoses_path, compression="gzip")
    diag["icd_code"] = diag["icd_code"].astype(str).str.strip()

    hadm_ids = set(cohort["hadm_id"])
    diag = diag[diag["hadm_id"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="hadm_id"))

    for var_name in COMORBIDITY_ICD9:
        icd9_prefixes = COMORBIDITY_ICD9[var_name]
        icd10_prefixes = COMORBIDITY_ICD10[var_name]

        # ICD-9 matches
        icd9_mask = (diag["icd_version"] == 9) & diag["icd_code"].apply(
            lambda c: any(c.startswith(p) for p in icd9_prefixes)
        )
        # ICD-10 matches
        icd10_mask = (diag["icd_version"] == 10) & diag["icd_code"].apply(
            lambda c: any(c.startswith(p) for p in icd10_prefixes)
        )

        positive_hadm = set(diag.loc[icd9_mask | icd10_mask, "hadm_id"])
        result[var_name] = result.index.isin(positive_hadm).astype(int)

    return result


def extract_treatment(inputevents, prescriptions, cohort):
    """Extract heparin/enoxaparin treatment variables."""
    hadm_ids = sorted(cohort["hadm_id"].unique())
    result = pd.DataFrame(index=pd.Index(hadm_ids, name="hadm_id"))

    # --- Heparin from inputevents ---
    hep_iv = pd.DataFrame()
    if not inputevents.empty:
        hep_iv = inputevents[
            inputevents["itemid"].isin(HEPARIN_ITEMIDS)
        ].copy()
        # Exclude flushes
        if "ordercategoryname" in hep_iv.columns:
            hep_iv = hep_iv[
                ~hep_iv["ordercategoryname"].fillna("").str.lower()
                .str.contains("flush|lock")
            ]
        hep_iv["amount"] = pd.to_numeric(hep_iv["amount"], errors="coerce")
        hep_iv = hep_iv[hep_iv["amount"] >= 100]  # exclude flush doses

    # --- Enoxaparin from inputevents ---
    enox_iv = pd.DataFrame()
    if not inputevents.empty:
        enox_iv = inputevents[
            inputevents["itemid"].isin(ENOXAPARIN_ITEMIDS)
        ].copy()
        enox_iv["amount"] = pd.to_numeric(enox_iv["amount"], errors="coerce")

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
    if not hep_iv.empty:
        hep_hadm.update(hep_iv["hadm_id"].unique())
    if not hep_rx.empty:
        hep_hadm.update(hep_rx["hadm_id"].unique())

    # Combine enoxaparin sources
    enox_hadm = set()
    if not enox_iv.empty:
        enox_hadm.update(enox_iv["hadm_id"].unique())
    if not enox_rx.empty:
        enox_hadm.update(enox_rx["hadm_id"].unique())

    result["heparin_ever"] = result.index.isin(hep_hadm).astype(int)
    result["enoxaparin_ever"] = result.index.isin(enox_hadm).astype(int)
    result["received_both"] = (
        (result["heparin_ever"] == 1) & (result["enoxaparin_ever"] == 1)
    ).astype(int)

    # Max daily heparin dose (from inputevents + prescriptions)
    hep_dose_parts = []
    if not hep_iv.empty:
        hep_dose_parts.append(
            hep_iv[["hadm_id", "amount"]].rename(columns={"amount": "dose"})
        )
    if not hep_rx.empty:
        rx_dose = hep_rx[["hadm_id", "dose_val_rx"]].copy()
        rx_dose["dose"] = pd.to_numeric(rx_dose["dose_val_rx"], errors="coerce")
        hep_dose_parts.append(rx_dose[["hadm_id", "dose"]])

    if hep_dose_parts:
        hep_all = pd.concat(hep_dose_parts, ignore_index=True)
        hep_all["dose"] = pd.to_numeric(hep_all["dose"], errors="coerce")
        hep_max = hep_all.groupby("hadm_id")["dose"].max()
        result["heparin_max_daily_dose"] = hep_max
    else:
        result["heparin_max_daily_dose"] = np.nan

    # Max daily enoxaparin dose
    enox_dose_parts = []
    if not enox_iv.empty:
        enox_dose_parts.append(
            enox_iv[["hadm_id", "amount"]].rename(columns={"amount": "dose"})
        )
    if not enox_rx.empty:
        enox_rx_dose = enox_rx[["hadm_id", "dose_val_rx"]].copy()
        enox_rx_dose["dose"] = pd.to_numeric(enox_rx_dose["dose_val_rx"],
                                             errors="coerce")
        enox_dose_parts.append(enox_rx_dose[["hadm_id", "dose"]])

    if enox_dose_parts:
        enox_all = pd.concat(enox_dose_parts, ignore_index=True)
        enox_all["dose"] = pd.to_numeric(enox_all["dose"], errors="coerce")
        enox_max = enox_all.groupby("hadm_id")["dose"].max()
        result["enoxaparin_max_daily_dose"] = enox_max
    else:
        result["enoxaparin_max_daily_dose"] = np.nan

    # First dose timestamps
    def _first_timestamp(df, time_col, hadm_col="hadm_id"):
        if df.empty:
            return pd.Series(dtype="datetime64[ns]")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        return df.dropna(subset=[time_col]).sort_values(time_col).groupby(hadm_col)[time_col].first()

    hep_times = []
    if not hep_iv.empty:
        hep_times.append(_first_timestamp(hep_iv, "starttime"))
    if hep_times:
        hep_first = pd.concat(hep_times).groupby(level=0).min()
        result["heparin_first_dose_time"] = hep_first
    else:
        result["heparin_first_dose_time"] = pd.NaT

    enox_times = []
    if not enox_iv.empty:
        enox_times.append(_first_timestamp(enox_iv, "starttime"))
    if enox_times:
        enox_first = pd.concat(enox_times).groupby(level=0).min()
        result["enoxaparin_first_dose_time"] = enox_first
    else:
        result["enoxaparin_first_dose_time"] = pd.NaT

    return result


def extract_procedures(cohort, procedures_path):
    """Extract procedure flags from procedures_icd (dual ICD-9/ICD-10)."""
    proc = pd.read_csv(procedures_path, compression="gzip")
    proc["icd_code"] = proc["icd_code"].astype(str).str.strip()

    hadm_ids = set(cohort["hadm_id"])
    proc = proc[proc["hadm_id"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="hadm_id"))

    for var_name, icd9_codes in PROCEDURE_ICD9.items():
        icd10_prefixes = PROCEDURE_ICD10_PREFIXES[var_name]

        # ICD-9 exact match
        icd9_mask = (proc["icd_version"] == 9) & proc["icd_code"].isin(icd9_codes)
        # ICD-10 prefix match
        icd10_mask = (proc["icd_version"] == 10) & proc["icd_code"].apply(
            lambda c: any(c.startswith(p) for p in icd10_prefixes)
        )

        positive_hadm = set(proc.loc[icd9_mask | icd10_mask, "hadm_id"])
        result[var_name] = result.index.isin(positive_hadm).astype(int)

    return result


def extract_outcomes(cohort, diagnoses_path):
    """Extract outcome variables."""
    diag = pd.read_csv(diagnoses_path, compression="gzip")
    diag["icd_code"] = diag["icd_code"].astype(str).str.strip()

    hadm_ids = set(cohort["hadm_id"])
    diag = diag[diag["hadm_id"].isin(hadm_ids)]

    result = pd.DataFrame(index=pd.Index(sorted(hadm_ids), name="hadm_id"))

    # Hospital mortality (from cohort)
    mort_map = cohort.set_index("hadm_id")["hospital_expire_flag"]
    result["hospital_mortality"] = mort_map

    # DCI (ICD proxy — dual version)
    dci_icd9_mask = (diag["icd_version"] == 9) & diag["icd_code"].isin(DCI_ICD9)
    dci_icd10_mask = (diag["icd_version"] == 10) & diag["icd_code"].apply(
        lambda c: any(c.startswith(p) for p in DCI_ICD10_PREFIXES)
    )
    dci_hadm = set(diag.loc[dci_icd9_mask | dci_icd10_mask, "hadm_id"])
    result["dci_icd"] = result.index.isin(dci_hadm).astype(int)

    # Rebleeding (ICD proxy — dual version)
    # For rebleeding, only count secondary diagnoses (seq_num > 1) for I60
    # to avoid counting the primary SAH as rebleeding
    rebleed_icd9_mask = (diag["icd_version"] == 9) & diag["icd_code"].isin(REBLEEDING_ICD9)
    rebleed_icd10_mask = (diag["icd_version"] == 10) & diag["icd_code"].apply(
        lambda c: any(c.startswith(p) for p in REBLEEDING_ICD10_PREFIXES)
    )
    # For I60 codes, only count if seq_num > 1 (secondary diagnosis)
    if "seq_num" in diag.columns:
        i60_mask = (diag["icd_version"] == 10) & diag["icd_code"].str.startswith("I60")
        rebleed_icd10_mask = rebleed_icd10_mask & (
            ~i60_mask | (diag["seq_num"] > 1)
        )
    rebleed_hadm = set(diag.loc[rebleed_icd9_mask | rebleed_icd10_mask, "hadm_id"])
    result["rebleeding_icd"] = result.index.isin(rebleed_hadm).astype(int)

    # ICU LOS
    cohort_idx = cohort.set_index("hadm_id")
    result["icu_los_days"] = (
        (cohort_idx["outtime"] - cohort_idx["intime"])
        .dt.total_seconds() / 86400
    )

    # Hospital LOS
    result["hospital_los_days"] = (
        (cohort_idx["dischtime"] - cohort_idx["admittime"])
        .dt.total_seconds() / 86400
    )

    return result


# ---------------------------------------------------------------------------
# Build patient table
# ---------------------------------------------------------------------------
def build_patient_table(cohort, vitals, labs, gcs, comorbidities,
                        treatment, procedures, outcomes):
    """Combine all extractions into one-row-per-patient table."""
    pt = cohort[["subject_id", "hadm_id", "stay_id", "age", "gender",
                 "race"]].copy()
    pt = pt.set_index("hadm_id")

    # Demographics
    pt["sex"] = pt["gender"].str.upper().map({"M": 0, "F": 1})
    pt.drop(columns=["gender"], inplace=True)

    # Vitals (indexed by stay_id -> map to hadm_id)
    icu_to_hadm = cohort.set_index("stay_id")["hadm_id"].to_dict()
    vitals_reindexed = vitals.copy()
    vitals_reindexed.index = vitals_reindexed.index.map(
        lambda x: icu_to_hadm.get(x, x)
    )
    for col in vitals_reindexed.columns:
        pt[col] = vitals_reindexed[col]

    # Labs (indexed by hadm_id)
    for col in labs.columns:
        pt[col] = labs[col]

    # GCS (indexed by stay_id -> map to hadm_id)
    gcs_reindexed = gcs.copy()
    gcs_reindexed.index = gcs_reindexed.index.map(
        lambda x: icu_to_hadm.get(x, x)
    )
    for col in gcs_reindexed.columns:
        pt[col] = gcs_reindexed[col]

    # Comorbidities (indexed by hadm_id)
    for col in comorbidities.columns:
        pt[col] = comorbidities[col]

    # Treatment (indexed by hadm_id)
    for col in treatment.columns:
        pt[col] = treatment[col]

    # Procedures (indexed by hadm_id)
    for col in procedures.columns:
        pt[col] = procedures[col]

    # Outcomes (indexed by hadm_id)
    for col in outcomes.columns:
        pt[col] = outcomes[col]

    # Unavailable variables -- NaN columns
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
    ("race", "Race/Ethnicity", "Demographics", True),
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
    ("dci_icd", "DCI (ICD proxy)", "Outcomes", True),
    ("rebleeding_icd", "Rebleeding (ICD proxy)", "Outcomes", True),
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
        "# Data Dictionary: MIMIC-IV aSAH Anticoagulation Study",
        "",
        "Maps each propensity covariate and outcome to its MIMIC-IV source, "
        "extraction logic, and availability.",
        "",
        "## Data Sources",
        "",
        "| Abbreviation | MIMIC-IV Table | Notes |",
        "|---|---|---|",
        "| chartevents | `icu/chartevents.csv.gz` | Vitals, GCS (Metavision only) |",
        "| labevents | `hosp/labevents.csv.gz` | Laboratory results |",
        "| inputevents | `icu/inputevents.csv.gz` | Medication inputs (Metavision) |",
        "| prescriptions | `hosp/prescriptions.csv.gz` | Prescription orders |",
        "| diagnoses_icd | `hosp/diagnoses_icd.csv.gz` | ICD-9/ICD-10 diagnosis codes |",
        "| procedures_icd | `hosp/procedures_icd.csv.gz` | ICD-9/ICD-10 procedure codes |",
        "",
        f"Baseline window: first {BASELINE_WINDOW_HOURS} hours after ICU admission (intime).",
        "",
        "---",
        "",
    ]

    # Group by category
    categories = {}
    for col, name, category, available in VARIABLE_META:
        if category not in categories:
            categories[category] = []
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
        "### Vitals (chartevents — Metavision only)",
        "",
        "| Variable | ITEMIDs | Aggregation |",
        "|---|---|---|",
    ])
    vital_aggs = {"map": "first", "sbp": "first", "heart_rate": "first",
                  "temp_c": "max", "temp_f": "max (->C)"}
    for var, ids in VITALS_ITEMIDS.items():
        if var.startswith("gcs"):
            continue
        ids_str = ", ".join(str(i) for i in ids)
        agg = vital_aggs.get(var, "first")
        lines.append(f"| {var} | {ids_str} | {agg} |")

    lines.extend([
        "",
        "### GCS (chartevents)",
        "",
        "| Component | ITEMIDs |",
        "|---|---|",
        "| Eye | 220739 |",
        "| Verbal | 223900 |",
        "| Motor | 223901 |",
        "",
        "### Labs (labevents)",
        "",
        "| Variable | ITEMIDs |",
        "|---|---|",
    ])
    for var, ids in LAB_ITEMIDS.items():
        ids_str = ", ".join(str(i) for i in ids)
        lines.append(f"| {var} | {ids_str} |")

    lines.extend([
        "",
        "### Treatment (inputevents / prescriptions)",
        "",
        "| Drug | ITEMIDs | PRESCRIPTIONS filter |",
        "|---|---|---|",
        "| Heparin | 225152, 225975 | drug ilike '%heparin%' "
        "(excl. flush/lock/LMWH) |",
        "| Enoxaparin | 225906 | drug ilike '%enoxaparin%' or '%lovenox%' |",
        "",
        "### Comorbidities (diagnoses_icd, prefix matching)",
        "",
        "| Variable | ICD-9 prefixes | ICD-10 prefixes |",
        "|---|---|---|",
    ])
    for var in COMORBIDITY_ICD9:
        icd9_str = ", ".join(COMORBIDITY_ICD9[var])
        icd10_str = ", ".join(COMORBIDITY_ICD10[var])
        lines.append(f"| {var} | {icd9_str} | {icd10_str} |")

    lines.extend([
        "",
        "### Procedures (procedures_icd)",
        "",
        "| Variable | ICD-9 codes | ICD-10 prefixes |",
        "|---|---|---|",
    ])
    for var, codes in PROCEDURE_ICD9.items():
        icd9_str = ", ".join(codes)
        icd10_str = ", ".join(PROCEDURE_ICD10_PREFIXES[var])
        lines.append(f"| {var} | {icd9_str} | {icd10_str} |")

    lines.extend([
        "",
        "### Outcomes (diagnoses_icd)",
        "",
        "| Variable | ICD-9 codes | ICD-10 prefixes |",
        "|---|---|---|",
        f"| DCI (proxy) | {', '.join(DCI_ICD9)} | {', '.join(DCI_ICD10_PREFIXES)} |",
        f"| Rebleeding | {', '.join(REBLEEDING_ICD9)} | {', '.join(REBLEEDING_ICD10_PREFIXES)} |",
        "",
        "---",
        "",
        "## Variables NOT Available in MIMIC-IV",
        "",
        "| Variable | Category | Reason |",
        "|---|---|---|",
        "| Fisher grade | Clinical Grades | Not coded in structured data |",
        "| WFNS (raw) | Clinical Grades | Not coded; derived from GCS instead |",
        "| IVH | Clinical Grades | Not reliably coded in ICD |",
        "| ICH | Clinical Grades | Not reliably coded in ICD |",
        "| Aneurysm size | Aneurysm | Not in structured data |",
        "| Aneurysm location | Aneurysm | Not in structured data |",
        "| Multiple aneurysms | Aneurysm | Not in structured data |",
        "| Time ictus-to-treatment | Treatment | Ictus time not recorded |",
        "| Pre-morbid mRS | Comorbidities | Not recorded |",
        "| CRP | Physiology | Not available in labevents |",
        "| mRS at discharge | Outcomes | Not recorded in MIMIC-IV |",
        "| mRS at 1 year | Outcomes | Not recorded in MIMIC-IV |",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VARIABLE EXTRACTION — MIMIC-IV aSAH Study")
    print("=" * 80)

    # Load cohort
    print("\nLoading cohort...")
    cohort = load_cohort()

    # Build ID maps
    stay_intime = dict(zip(cohort["stay_id"], cohort["intime"]))
    hadm_intime = dict(zip(cohort["hadm_id"], cohort["intime"]))
    hadm_ids = set(cohort["hadm_id"])
    stay_ids = set(cohort["stay_id"])

    # Collect all needed ITEMIDs
    all_chart_itemids = _collect_all_chart_itemids()
    all_lab_itemids = _collect_all_lab_itemids()

    # --- Load large tables (single-pass chunked) ---
    print("\nLoading chartevents (chunked)...")
    chartevents = load_filtered_events(
        os.path.join(ICU_DIR, "chartevents.csv.gz"),
        id_col="stay_id",
        cohort_ids=stay_ids,
        target_itemids=all_chart_itemids,
        usecols=["subject_id", "hadm_id", "stay_id", "itemid",
                 "charttime", "value", "valuenum", "valueuom"],
    )

    print("\nLoading labevents (chunked)...")
    labevents = load_filtered_events(
        os.path.join(HOSP_DIR, "labevents.csv.gz"),
        id_col="hadm_id",
        cohort_ids=hadm_ids,
        target_itemids=all_lab_itemids,
        usecols=["subject_id", "hadm_id", "itemid", "charttime",
                 "value", "valuenum", "valueuom"],
    )

    print("\nLoading inputevents...")
    inputevents = load_inputevents(hadm_ids)

    print("\nLoading prescriptions...")
    prescriptions = load_prescriptions(hadm_ids)

    # --- Extract variables ---
    print("\nExtracting vitals...")
    vitals = extract_vitals(chartevents, stay_intime)
    print(f"  Vitals extracted: {vitals.notna().sum().to_dict()}")

    print("\nExtracting labs...")
    labs = extract_labs(labevents, hadm_intime)
    print(f"  Labs extracted: {labs.notna().sum().to_dict()}")

    print("\nExtracting GCS...")
    gcs = extract_gcs(chartevents, stay_intime)
    print(f"  GCS extracted: {gcs.notna().sum().to_dict()}")

    print("\nExtracting comorbidities...")
    comorbidities = extract_comorbidities(
        cohort, os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz")
    )
    print(f"  Comorbidities: {comorbidities.sum().to_dict()}")

    print("\nExtracting treatment...")
    treatment = extract_treatment(inputevents, prescriptions, cohort)
    print(f"  Heparin: {treatment['heparin_ever'].sum()} patients")
    print(f"  Enoxaparin: {treatment['enoxaparin_ever'].sum()} patients")
    print(f"  Both: {treatment['received_both'].sum()} patients")

    print("\nExtracting procedures...")
    procedures = extract_procedures(
        cohort, os.path.join(HOSP_DIR, "procedures_icd.csv.gz")
    )
    print(f"  Procedures: {procedures.sum().to_dict()}")

    print("\nExtracting outcomes...")
    outcomes = extract_outcomes(
        cohort, os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz")
    )
    print(f"  Hospital mortality: {outcomes['hospital_mortality'].sum()}")
    print(f"  DCI (ICD): {outcomes['dci_icd'].sum()}")
    print(f"  Rebleeding (ICD): {outcomes['rebleeding_icd'].sum()}")

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
    assert pt["subject_id"].is_unique, "Duplicate subject_ids!"
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
    print(f"  DCI rate: {outcomes['dci_icd'].mean() * 100:.1f}%")
    print(f"  Rebleeding rate: {outcomes['rebleeding_icd'].mean() * 100:.1f}%")
    print(f"  Hospital mortality: {outcomes['hospital_mortality'].mean() * 100:.1f}%")
    print(f"  Median ICU LOS: {outcomes['icu_los_days'].median():.1f} days")
    print(f"  Median hospital LOS: {outcomes['hospital_los_days'].median():.1f} days")
    if pt["gcs_admission"].notna().any():
        print(f"  Median GCS: {pt['gcs_admission'].median():.0f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
