# MIMIC-III aSAH Cohort Identification

## Overview

This document describes how the aneurysmal subarachnoid hemorrhage (aSAH) cohort was identified from the MIMIC-III v1.4 database for the heparin vs enoxaparin thromboprophylaxis comparison.

## Data Source

- **Database:** MIMIC-III v1.4 (Medical Information Mart for Intensive Care)
- **Location:** `/mnt/hdd1/datasets/mimiciii_1.4`
- **Format:** Gzip-compressed CSV files
- **Tables used:** DIAGNOSES_ICD, PROCEDURES_ICD, PATIENTS, ADMISSIONS, ICUSTAYS

## Inclusion Criteria

### SAH Diagnosis (ICD-9 430)

All hospital admissions with ICD-9 diagnosis code **430** ("Subarachnoid hemorrhage") in the DIAGNOSES_ICD table were identified as the initial candidate pool. This code covers all non-traumatic SAH regardless of etiology (aneurysmal, perimesencephalic, AVM-related, etc.).

**Result:** 658 admissions across 647 unique subjects.

## Exclusion Criteria

Exclusions were applied sequentially:

### 1. Traumatic SAH

Admissions with concurrent ICD-9 codes for skull fractures (800–804) or intracranial injury (850–854) were excluded, as these represent traumatic rather than spontaneous SAH.

**Excluded:** 7 admissions.

### 2. Pediatric Patients (Age < 18)

Patients under 18 years of age at admission were excluded.

**Note on age calculation:** MIMIC-III shifts the date of birth for patients older than 89 years to approximately 300 years before admission, for de-identification. These patients were assigned an age of 89. Age was computed as the year difference between admission date and DOB, adjusted for month/day to avoid integer overflow from the shifted dates.

**Excluded:** 0 admissions (no pediatric SAH patients in MIMIC-III).

### 3. No ICU Stay

Patients without a corresponding record in the ICUSTAYS table were excluded, as ICU-level monitoring data is required for the anticoagulation analysis.

**Excluded:** 4 admissions.

### 4. Multiple Admissions

For patients with more than one SAH admission, only the chronologically first admission was retained. This avoids double-counting and ensures independence of observations.

**Excluded:** 8 admissions (subsequent readmissions).

## Aneurysm Confirmation

ICD-9 code 430 does not distinguish aneurysmal SAH from other etiologies. To enable sensitivity analyses restricted to confirmed aSAH, each patient was flagged based on procedural and diagnostic evidence:

### Confirmed aneurysmal (`aneurysm_confirmed = 1`)

A patient is flagged as confirmed if **any** of the following ICD-9 procedure or diagnosis codes are present during the admission:

| Code | Description | Type |
|------|-------------|------|
| 3951 | Clipping of cerebral aneurysm | Procedure |
| 3972 | Endovascular repair of head/neck vessels | Procedure |
| 3975 | Endovascular embolization of head/neck vessels | Procedure |
| 3976 | Endovascular embolization of head/neck w/ coils | Procedure |
| 4373 | Cerebral aneurysm, nonruptured | Diagnosis |

**Result:** 284 patients (44.4%) confirmed aneurysmal.

### Cerebral angiography flag

Separately, patients with ICD-9 procedure code **8841** (cerebral angiography) are flagged, as this is the standard diagnostic workup for suspected aSAH.

**Result:** 404 patients (63.2%) had angiography.

### Rationale for not excluding unconfirmed patients

Patients without aneurysm treatment codes were **retained** rather than excluded because:

1. Some patients may have died before intervention could be performed
2. Conservative management may have been chosen
3. Angiography-negative SAH (perimesencephalic) is clinically similar in the acute phase and may still receive thromboprophylaxis
4. Excluding unconfirmed patients would introduce survival/treatment bias

The `aneurysm_confirmed` flag allows subsetting for sensitivity analyses restricted to definite aSAH.

## Final Cohort Summary

| Metric | Value |
|--------|-------|
| Total patients | 639 |
| Aneurysm confirmed | 284 (44.4%) |
| Aneurysm not confirmed | 355 (55.6%) |
| Female | 363 (56.8%) |
| Male | 276 (43.2%) |
| Age range | 19–89 years |

## ICU Stay Selection

For patients with multiple ICU stays within the same admission, the **first** ICU stay (by INTIME) was selected. The ICUSTAY_ID, admission/discharge times, length of stay, and first care unit are included in the output.

## Output Files

| File | Description |
|------|-------------|
| `artifacts/cohort.csv` | One row per patient with demographics, ICU stay info, and aneurysm flags |
| `artifacts/cohort_flow.csv` | Step-by-step counts for each inclusion/exclusion criterion |
| `artifacts/cohort_flowchart.png` | CONSORT-style flow diagram |

## Reproducibility

The cohort is fully reproducible by running:

```bash
python mimic3/cohort_selection.py
```

All parameters (ICD codes, age threshold, exclusion logic) are defined as constants at the top of the script.
