# Harmonization Report: Pooled KSSG + MIMIC-IV Cohort

## Cohort Summary

| Site | Total | Heparin | Enoxaparin |
|------|-------|---------|------------|
| KSSG | 179 | 7 | 172 |
| MIMIC-IV | 870 | 868 | 2 |
| **Pooled** | **1049** | **875** | **174** |

## MIMIC-IV Exclusion Flow

- Starting patients: 1280
- Excluded (received both): 28
- Excluded (therapeutic heparin >10,000 UI/24h): 56
- Excluded (therapeutic enoxaparin >40 mg/d): 0
- Excluded (no anticoagulant): 285
- Excluded (renal failure, creatinine >1.70 mg/dL): 41
- **After exclusions: 870**

## Variable Harmonization

### Shared variables (propensity model)

| Variable | N available | % missing | KSSG avail | MIMIC avail |
|----------|------------|-----------|------------|-------------|
| `age` | 1047 | 0.2% | 177 | 870 |
| `sex` | 1049 | 0.0% | 179 | 870 |
| `hypertension` | 1038 | 1.0% | 168 | 870 |
| `diabetes` | 1038 | 1.0% | 168 | 870 |
| `smoking` | 1040 | 0.9% | 170 | 870 |
| `prior_anticoagulant` | 1037 | 1.1% | 167 | 870 |
| `prior_antiplatelet` | 1037 | 1.1% | 167 | 870 |
| `gcs_admission` | 1046 | 0.3% | 179 | 867 |
| `hunt_hess` | 1035 | 1.3% | 168 | 867 |
| `admission_map` | 1043 | 0.6% | 178 | 865 |
| `admission_temp` | 1045 | 0.4% | 179 | 866 |
| `admission_wbc` | 1008 | 3.9% | 144 | 864 |
| `admission_hemoglobin` | 1040 | 0.9% | 176 | 864 |
| `admission_glucose` | 1040 | 0.9% | 176 | 864 |
| `admission_sodium` | 1040 | 0.9% | 176 | 864 |
| `treatment_clipping` | 1049 | 0.0% | 179 | 870 |
| `treatment_coiling` | 1046 | 0.3% | 176 | 870 |
| `evd` | 1033 | 1.5% | 163 | 870 |

### Outcome variables

| Variable | N available | % missing | Notes |
|----------|------------|-----------|-------|
| `dci` | 1049 | 0.0% | KSSG: clinician-verified; MIMIC: ICD proxy |
| `rebleeding` | 1035 | 1.3% | KSSG: clinician-verified; MIMIC: ICD proxy |
| `mortality` | 1035 | 1.3% | KSSG: death; MIMIC: hospital_mortality |
| `icu_los_days` | 869 | 17.2% | MIMIC only |
| `hospital_los_days` | 870 | 17.1% | MIMIC only |

### Site-specific variables

| Variable | Source | N available |
|----------|--------|------------|
| `admission_sbp` | KSSG | 178 |
| `fisher_score` | KSSG | 166 |
| `wfns` | KSSG | 168 |
| `ivh` | KSSG | 172 |
| `ich` | KSSG | 166 |
| `intubated_admission` | KSSG | 179 |
| `aneurysm_anterior` | KSSG | 155 |
| `aneurysm_size` | KSSG | 159 |
| `multiple_aneurysms` | KSSG | 164 |
| `time_ictus_to_treatment_days` | KSSG | 160 |
| `premorbid_mrs` | KSSG | 169 |
| `mrs_discharge` | KSSG | 171 |
| `mrs_1y` | KSSG | 142 |
| `admission_crp` | KSSG | 92 |
| `admission_hematocrit` | KSSG | 144 |
| `race` | MIMIC-IV | 870 |
| `renal_disease` | MIMIC-IV | 870 |
| `liver_disease` | MIMIC-IV | 870 |
| `admission_creatinine` | MIMIC-IV | 864 |
| `admission_potassium` | MIMIC-IV | 864 |
| `admission_platelets` | MIMIC-IV | 863 |
| `admission_inr` | MIMIC-IV | 817 |
| `admission_ptt` | MIMIC-IV | 815 |
| `admission_lactate` | MIMIC-IV | 344 |
| `admission_heart_rate` | MIMIC-IV | 868 |

## Binary Variable Validation

| Variable | Unique values | Valid (0/1/NaN only) |
|----------|--------------|---------------------|
| `sex` | [0, 1] | Yes |
| `hypertension` | [0.0, 1.0] | Yes |
| `diabetes` | [0.0, 1.0] | Yes |
| `smoking` | [0.0, 1.0] | Yes |
| `prior_anticoagulant` | [0.0, 1.0] | Yes |
| `prior_antiplatelet` | [0.0, 1.0] | Yes |
| `treatment_clipping` | [0.0, 1.0] | Yes |
| `treatment_coiling` | [0.0, 1.0] | Yes |
| `evd` | [0.0, 1.0] | Yes |
| `dci` | [0.0, 1.0] | Yes |
| `rebleeding` | [0.0, 1.0] | Yes |
| `mortality` | [0.0, 1.0] | Yes |

## Propensity Model Feasibility

- Smaller group (enoxaparin): 174
- Number of covariates: 18
- Ratio (events per covariate): 9.7
- Feasible (≥10 per covariate): No
