# Data Dictionary: MIMIC-IV aSAH Anticoagulation Study

Maps each propensity covariate and outcome to its MIMIC-IV source, extraction logic, and availability.

## Data Sources

| Abbreviation | MIMIC-IV Table | Notes |
|---|---|---|
| chartevents | `icu/chartevents.csv.gz` | Vitals, GCS (Metavision only) |
| labevents | `hosp/labevents.csv.gz` | Laboratory results |
| inputevents | `icu/inputevents.csv.gz` | Medication inputs (Metavision) |
| prescriptions | `hosp/prescriptions.csv.gz` | Prescription orders |
| diagnoses_icd | `hosp/diagnoses_icd.csv.gz` | ICD-9/ICD-10 diagnosis codes |
| procedures_icd | `hosp/procedures_icd.csv.gz` | ICD-9/ICD-10 procedure codes |

Baseline window: first 48 hours after ICU admission (intime).

---

## Demographics

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Age | `age` | Yes | 0.0% |
| Sex | `sex` | Yes | 0.0% |
| Race/Ethnicity | `race` | Yes | 0.0% |

## Comorbidities

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Hypertension | `hypertension` | Yes | 0.0% |
| Diabetes mellitus | `diabetes` | Yes | 0.0% |
| Smoking history | `smoking_history` | Yes | 0.0% |
| Renal disease | `renal_disease` | Yes | 0.0% |
| Liver disease | `liver_disease` | Yes | 0.0% |
| Prior anticoagulant use | `prior_anticoagulant` | Yes | 0.0% |
| Prior antiplatelet use | `prior_antiplatelet` | Yes | 0.0% |

## Clinical Grades

| Variable | Column | Available | Missingness |
|---|---|---|---|
| GCS on admission | `gcs_admission` | Yes | 0.5% |
| Hunt-Hess (derived from GCS) | `hunt_hess_derived` | Yes | 0.5% |
| WFNS (derived from GCS) | `wfns_derived` | Yes | 0.5% |
| Fisher grade | `fisher_grade` | No | 100.0% |
| IVH | `ivh` | No | 100.0% |
| ICH | `ich` | No | 100.0% |

## Physiology

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Admission MAP | `admission_map` | Yes | 0.5% |
| Admission SBP | `admission_sbp` | Yes | 0.5% |
| Admission heart rate | `admission_heart_rate` | Yes | 0.2% |
| Admission temperature | `admission_temp` | Yes | 1.0% |
| Admission WBC | `admission_wbc` | Yes | 4.5% |
| Admission hemoglobin | `admission_hemoglobin` | Yes | 4.5% |
| Admission creatinine | `admission_creatinine` | Yes | 4.4% |
| Admission glucose | `admission_glucose` | Yes | 4.4% |
| Admission sodium | `admission_sodium` | Yes | 4.3% |
| Admission potassium | `admission_potassium` | Yes | 4.4% |
| Admission platelets | `admission_platelets` | Yes | 4.5% |
| Admission INR | `admission_inr` | Yes | 9.8% |
| Admission PTT | `admission_ptt` | Yes | 10.2% |
| Admission lactate | `admission_lactate` | Yes | 59.1% |
| Admission albumin | `admission_albumin` | Yes | 73.9% |
| Admission bilirubin | `admission_bilirubin` | Yes | 68.3% |
| CRP | `crp` | No | 100.0% |

## Treatment

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Heparin exposure | `heparin_ever` | Yes | 0.0% |
| Enoxaparin exposure | `enoxaparin_ever` | Yes | 0.0% |
| Received both agents | `received_both` | Yes | 0.0% |
| Heparin max daily dose | `heparin_max_daily_dose` | Yes | 22.9% |
| Enoxaparin max daily dose | `enoxaparin_max_daily_dose` | Yes | 97.7% |

## Procedures

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Clipping | `treatment_clipping` | Yes | 0.0% |
| Coiling | `treatment_coiling` | Yes | 0.0% |
| EVD placement | `evd_placement` | Yes | 0.0% |

## Aneurysm

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Aneurysm size | `aneurysm_size` | No | 100.0% |
| Aneurysm location | `aneurysm_location` | No | 100.0% |
| Multiple aneurysms | `aneurysm_multiplicity` | No | 100.0% |
| Time ictus-to-treatment | `time_ictus_to_treatment` | No | 100.0% |
| Pre-morbid mRS | `premorbid_mrs` | No | 100.0% |

## Outcomes

| Variable | Column | Available | Missingness |
|---|---|---|---|
| Hospital mortality | `hospital_mortality` | Yes | 0.0% |
| DCI (ICD proxy) | `dci_icd` | Yes | 0.0% |
| Rebleeding (ICD proxy) | `rebleeding_icd` | Yes | 0.0% |
| ICU length of stay | `icu_los_days` | Yes | 0.1% |
| Hospital length of stay | `hospital_los_days` | Yes | 0.0% |
| mRS at discharge | `mrs_discharge` | No | 100.0% |
| mRS at 1 year | `mrs_1y` | No | 100.0% |

---

## ITEMID Reference

### Vitals (chartevents — Metavision only)

| Variable | ITEMIDs | Aggregation |
|---|---|---|
| map | 220052, 220181 | first |
| sbp | 220050, 220179 | first |
| heart_rate | 220045 | first |
| temp_c | 223762 | max |
| temp_f | 223761 | max (->C) |

### GCS (chartevents)

| Component | ITEMIDs |
|---|---|
| Eye | 220739 |
| Verbal | 223900 |
| Motor | 223901 |

### Labs (labevents)

| Variable | ITEMIDs |
|---|---|
| wbc | 51300, 51301 |
| hemoglobin | 51222, 50811 |
| creatinine | 50912 |
| glucose | 50931, 50809 |
| sodium | 50983, 50824 |
| potassium | 50971, 50822 |
| platelets | 51265 |
| inr | 51237 |
| ptt | 51275 |
| lactate | 50813 |
| albumin | 50862 |
| bilirubin | 50885 |

### Treatment (inputevents / prescriptions)

| Drug | ITEMIDs | PRESCRIPTIONS filter |
|---|---|---|
| Heparin | 225152, 225975 | drug ilike '%heparin%' (excl. flush/lock/LMWH) |
| Enoxaparin | 225906 | drug ilike '%enoxaparin%' or '%lovenox%' |

### Comorbidities (diagnoses_icd, prefix matching)

| Variable | ICD-9 prefixes | ICD-10 prefixes |
|---|---|---|
| hypertension | 401, 402, 403, 404, 405 | I10, I11, I12, I13, I14, I15, I16 |
| diabetes | 250 | E10, E11, E12, E13, E14 |
| smoking_history | V1582, 3051 | F17, Z87891 |
| renal_disease | 585, 586, V420, V451 | N18, Z940 |
| liver_disease | 5712, 5713, 5714, 5715, 5716, 5717, 5718, 5719, 456, 572 | K70, K71, K72, K73, K74, K75, K76, K77 |
| prior_anticoagulant | V5861 | Z7901 |
| prior_antiplatelet | V5863 | Z7902, Z7982 |

### Procedures (procedures_icd)

| Variable | ICD-9 codes | ICD-10 prefixes |
|---|---|---|
| treatment_clipping | 3951 | 03VG0, 03LG0 |
| treatment_coiling | 3972, 3975, 3976 | 03VG3, 03LG3 |
| evd_placement | 0231, 231 | 009630Z, 00960 |

### Outcomes (diagnoses_icd)

| Variable | ICD-9 codes | ICD-10 prefixes |
|---|---|---|
| DCI (proxy) | 43401, 43411, 43491, 4371 | I63, I6784 |
| Rebleeding | 4320, 4321, 4329, 99811, 99812 | I60, I61 |

---

## Variables NOT Available in MIMIC-IV

| Variable | Category | Reason |
|---|---|---|
| Fisher grade | Clinical Grades | Not coded in structured data |
| WFNS (raw) | Clinical Grades | Not coded; derived from GCS instead |
| IVH | Clinical Grades | Not reliably coded in ICD |
| ICH | Clinical Grades | Not reliably coded in ICD |
| Aneurysm size | Aneurysm | Not in structured data |
| Aneurysm location | Aneurysm | Not in structured data |
| Multiple aneurysms | Aneurysm | Not in structured data |
| Time ictus-to-treatment | Treatment | Ictus time not recorded |
| Pre-morbid mRS | Comorbidities | Not recorded |
| CRP | Physiology | Not available in labevents |
| mRS at discharge | Outcomes | Not recorded in MIMIC-IV |
| mRS at 1 year | Outcomes | Not recorded in MIMIC-IV |
