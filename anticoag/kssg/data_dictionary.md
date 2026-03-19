# Data Dictionary: Heparin vs Enoxaparin after aSAH

Maps each propensity covariate (from literature review §4) and outcome to its data source, column, derivation logic, type, and availability.

## Data Sources

| Abbreviation | File | Format |
|---|---|---|
| Registry | `post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx` | Encrypted Excel (password line 2 of `.secrets`) |
| Outcomes | `outcomes_aSAH_DATA_2009_2024_18122024.xlsx` | Encrypted Excel (password line 1 of `.secrets`) |
| Correspondence | `registry_pdms_correspondence.csv` | CSV (comma-sep) |
| BGA | `extracted_data/20240116_SAH_SOS_BGA.csv` | CSV (semicolon-sep, UTF-8 BOM) |
| Labor | `extracted_data/20240116_SAH_SOS_Labor.csv` | CSV (semicolon-sep, UTF-8 BOM) |
| Blutdruecke | `extracted_data/20240116_SAH_SOS_Blutdruecke.csv` | CSV (semicolon-sep, UTF-8 BOM) |
| Temperatur | `extracted_data/20240116_SAH_SOS_Temperatur.csv` | CSV (semicolon-sep, UTF-8 BOM) |
| GCS | `extracted_data/20240117_SAH_SOS_GCS.csv` | CSV (semicolon-sep) |

All PDMS files are in `/mnt/data1/klug/datasets/kssg/SAH/`. Registry ↔ PDMS linkage via Correspondence file on `SOS-CENTER-YEAR-NO.` + `Name` + `Date_birth` → `pNr`.

---

## Available Variables (35)

### Demographics

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Age | Registry | `Age` | Direct | Continuous |
| Sex | Registry | `Sex` | Map M→0, F/W→1 | Binary |

### Comorbidities

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Hypertension | Registry | `HTN` | Direct | Binary |
| Diabetes mellitus | Registry | `DM` | Direct | Binary |
| Smoking history | Registry | `Smoker_0no_1yes_2ex` | Direct (0=no, 1=yes, 2=ex) | Categorical |
| Prior antiplatelet (ASS) | Registry | `ASS` | Direct | Binary |
| Prior antiplatelet (Clopidogrel) | Registry | `Clopidogrel` | Direct | Binary |
| Prior oral anticoagulation | Registry | `OAC` | Direct | Binary |
| Pre-morbid mRS | Registry | `mRS_before_ictus` | Direct | Ordinal 0–5 |

### Clinical Grades (admission)

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Hunt-Hess grade | Registry | `HH` | Direct | Ordinal 1–5 |
| WFNS grade | Registry | `WFNS` | Direct | Ordinal 1–5 |
| GCS on admission | Registry + GCS CSV | `GCS_admission` / `eyes+movement+verbal` | Registry primary; PDMS fallback = first value by `timeGCS` | Ordinal 3–15 |
| Modified Fisher grade | Registry | `Fisher_Score` | Direct | Ordinal 1–4 |
| Intubated on admission | Registry + GCS CSV | `Intubated_on_admission_YN` / `intubated` | Registry primary; PDMS fallback | Binary |
| IVH | Registry | `IVH` | Direct | Binary |
| ICH | Registry | `ICH` | Direct | Binary |

### Admission Physiology (baseline window: day 0–2)

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Admission MAP | Blutdruecke CSV | `mitteldruck`, `timeBd` | First value in baseline window | Continuous (mmHg) |
| Admission SBP | Blutdruecke CSV | `systole`, `timeBd` | First value in baseline window | Continuous (mmHg) |
| Admission temperature | Temperatur CSV | `temperatur`, `timeTemp` | Max (worst) in baseline window | Continuous (°C) |
| Admission WBC | Labor CSV | `Wert` where `Labor=='Leukozyten'` | First value in baseline window | Continuous (G/L) |
| Admission hematocrit | Labor CSV | `Wert` where `Labor=='HKT'` | First value in baseline window | Continuous (ratio) |
| Admission CRP | Labor CSV | `Wert` where `Labor=='CRP'` | First value in baseline window | Continuous (mg/L) |
| Admission glucose | BGA CSV | `glc`, `timeBGA` | First value in baseline window | Continuous (mmol/L) |
| Admission sodium | BGA CSV | `na`, `timeBGA` | First value in baseline window | Continuous (mmol/L) |
| Admission hemoglobin | BGA CSV | `hb`, `timeBGA` | First value in baseline window | Continuous (g/dL) |
| Admission PaO2 | BGA CSV | `pO2`, `timeBGA` | First arterial (`bgaOrt=='arteriell'`) in baseline window | Continuous (kPa) |

### Aneurysm Characteristics

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Aneurysm location | Registry | `Aneurysm_Artery_Code` | Map to anterior/posterior (see code mapping below) | Binary |
| Aneurysm size | Registry | `Aneurysm_diameter` | Direct | Continuous (mm) |
| Multiple aneurysms | Registry | `Multiple_Aneurysms_2unk` | Direct (0=no, 1=yes, 2=unknown) | Categorical |

### Treatment (day 0–1, before anticoag decision)

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| Treatment modality | Registry | `Coiling`, `Clipping`, `Stenting` | Derive clip/coil/stent/conservative | Categorical |
| Time ictus-to-treatment | Registry | `Date_Ictus`, `Date_First_Th` | Compute delta (days) | Continuous |
| EVD placement | Registry | `EVD_YN` | Direct | Binary |

### Outcomes

| Variable | Source | Column(s) | Derivation | Type |
|---|---|---|---|---|
| DCI | Registry | `DCI_YN_verified` (fallback `DCI_YN`) | Direct | Binary |
| Rebleeding | Registry | `Rebleeding_YN` | Direct | Binary |
| Death | Registry | `Death` | Direct | Binary |
| mRS at discharge | Outcomes | `mRS_discharge` | Direct | Ordinal 0–6 |
| mRS at 1 year | Outcomes | `mRS_FU_1y` | Direct | Ordinal 0–6 |

---

## NOT Available in Data (7 variables)

| Variable | Category | Reason |
|---|---|---|
| BMI / obesity | Demographics | No height/weight in any source |
| Cardiovascular disease | Comorbidities | Not specifically recorded in registry |
| Thick SAH burden | Clinical Grades | Only Fisher score (partially captures this) |
| Heart rate | Physiology | Not in any PDMS file |
| Respiratory rate | Physiology | Not in any PDMS file |
| Potassium | Physiology | Not in BGA (only pO2, pCO2, glc, na, hb) |
| pH / HCO3 | Physiology | Not in BGA |

---

## Aneurysm Location Code Mapping

| Group | Codes | Classification |
|---|---|---|
| ACoA | 8 | Anterior |
| ACA | 9, 22, 24 | Anterior |
| MCA | 7, 20, 21 | Anterior |
| ICA | 1–6, 18, 19, 25–27, 29, 31 | Anterior |
| Vertebrobasilar | 10–17 | Posterior |
| PCA | 23, 28 | Posterior |
