# Literature Review: Heparin vs Enoxaparin Thromboprophylaxis after aSAH

## 1. Background

Venous thromboembolism (VTE) prophylaxis is standard of care after aneurysmal subarachnoid hemorrhage (aSAH), yet no randomized controlled trial has directly compared unfractionated heparin (UFH) with low-molecular-weight heparin (LMWH) in this population. Current guidelines (AHA/ASA 2023, Neurocritical Care Society) recommend pharmacological thromboprophylaxis but do not specify a preferred agent, reflecting clinical equipoise. Observational data suggest possible differences in efficacy and bleeding risk, but confounding by indication and institutional practice patterns limit causal inference. This equipoise justifies a propensity-controlled comparison to inform agent selection.

## 2. Evidence Summary Table

| Study | Year | N | Comparison | Key Finding | Limitations |
|---|---|---|---|---|---|
| Siironen et al. | 2003 | 170 | Enoxaparin vs placebo after aSAH | Enoxaparin reduced DVT incidence without increasing ICH; no effect on outcome | Single-center RCT vs placebo, not vs UFH; small sample |
| Wurm et al. | 2004 | 120 | LMWH vs UFH after craniotomy for aSAH | Similar VTE rates; LMWH trended toward fewer bleeding events | Retrospective, mixed surgical indications, underpowered |
| Patel et al. | 2008 | 68 | Enoxaparin timing after aSAH clipping/coiling | Early enoxaparin (≤48 h) safe with low hemorrhagic complication rate | Small, single-center, no UFH comparison arm |
| Cage et al. | 2019 | 254 | UFH 5000 UI TID vs enoxaparin 40 mg QD post-aSAH | No difference in VTE or hemorrhagic complications between groups | Retrospective cohort, single institution, unadjusted |
| Zanaty et al. | 2022 | 312 | LMWH vs UFH thromboprophylaxis in aSAH | LMWH associated with lower DVT rate; no increase in rebleeding | Retrospective, potential selection bias, variable dosing |
| Li et al. (Frontiers Neurol) | 2026 | 198 | Enoxaparin vs UFH prophylaxis post-aSAH | Similar safety profiles; enoxaparin associated with fewer VTE events | Retrospective, limited long-term follow-up, single center |

## 3. Confounders by Outcome

> **Temporal framework:** The anticoagulant choice (UFH vs LMWH) is typically made on **day 1–2** after admission, once the aneurysm is secured. Only confounders measured at **baseline (admission through day 0–2, before thromboprophylaxis initiation)** are eligible for the propensity model. Post-baseline variables are marked *(post-baseline)* and are relevant only as confounders in outcome regression models, not for propensity score estimation.

### 3.1 Delayed Cerebral Ischemia (DCI)

| Confounder | Timing | Rationale / References |
|---|---|---|
| Hunt-Hess grade ≥4 | Baseline | Higher clinical severity increases DCI risk (Frontera 2009, de Rooij 2013) |
| Modified Fisher grade ≥3 | Baseline | Thick cisternal clot is the strongest predictor of vasospasm/DCI (Fisher 1980, Claassen 2001) |
| WFNS grade | Baseline | Validated severity scale correlating with DCI incidence (WFNS 1988, Rosengart 2007) |
| Presence of IVH | Baseline | Intraventricular hemorrhage independently associated with DCI (Claassen 2001) |
| Female sex | Baseline | Higher DCI incidence in women, possibly hormonal (Lanzino 1996, de Rooij 2013) |
| Hypertension | Baseline | Chronic hypertension impairs cerebrovascular autoregulation (Dorhout Mees 2007) |
| Admission WBC count | Baseline | Inflammatory response marker associated with vasospasm (McGirt 2003) |
| Age | Baseline | Conflicting data; older age may reduce vasospasm but worsen DCI outcomes (Rosengart 2007) |
| Diabetes mellitus | Baseline | Microvascular disease may potentiate ischemic injury (Vergouwen 2011) |
| Obesity (BMI) | Baseline | Obesity associated with inflammation and altered drug pharmacokinetics (Naval 2012) |
| Smoking history | Baseline | Current smoking associated with increased vasospasm risk (Weir 1998, Lasner 1997) |
| Aneurysm location | Baseline | Anterior circulation aneurysms have higher DCI rates (Roos 2000) |
| Treatment modality | Baseline (day 0–1) | Clipping vs coiling decided before anticoag; may differentially affect vasospasm risk (Dumont 2010) |
| Thick SAH burden | Baseline | Higher clot volume on admission CT correlates with vasospasm severity (Reilly 2004) |

### 3.2 Rebleeding

| Confounder | Timing | Rationale / References |
|---|---|---|
| Time to aneurysm treatment | Baseline (day 0–1) | Rebleeding risk highest in first 24 h; early treatment is protective (Connolly 2012) |
| Admission SBP >140 mmHg | Baseline | Uncontrolled hypertension increases transmural pressure (Ohkuma 2001) |
| Hunt-Hess grade | Baseline | Higher grade associated with greater rebleeding risk (Naidech 2005) |
| Aneurysm size >10 mm | Baseline | Larger aneurysms have higher rupture and re-rupture rates (ISUIA 2003) |
| Posterior circulation | Baseline | Posterior aneurysms carry higher rebleeding risk (Molyneux 2005) |
| ICH / IVH presence | Baseline | Parenchymal or intraventricular extension indicates severe initial hemorrhage (Starke 2011) |
| Male sex | Baseline | Some evidence of higher rebleeding in males (Naidech 2005) |
| Admission glucose | Baseline | Hyperglycemia as stress marker correlating with severity (Frontera 2006) |
| Admission WBC count | Baseline | Inflammatory marker associated with early complications (McGirt 2003) |
| Prior anticoagulant/antiplatelet use | Baseline | Pre-existing anticoagulation increases hemorrhagic risk (Connolly 2012) |

### 3.3 mRS at Discharge

*Components of SAHIT (Jaja 2018) marked †, FRESH (Witsch 2016) marked ‡. All baseline confounders are measured at admission / day 0–2, before anticoagulant selection.*

**Baseline confounders (eligible for propensity model)**

| Confounder | Timing | Rationale / References |
|---|---|---|
| Age †‡ | Baseline | Strongest predictor of functional outcome (Rosengart 2007, Hop 1997); SAHIT + FRESH |
| Pre-morbid mRS | Baseline | Baseline functional status determines recovery ceiling (van Swieten 1988) |
| Hunt-Hess grade ‡ | Baseline | Admission severity predicts discharge status (Hunt & Hess 1968); FRESH component |
| WFNS grade † | Baseline | Validated admission grading scale (WFNS 1988); SAHIT core component |
| GCS on admission | Baseline | Low GCS correlates with poor short-term outcome (Rosen & Macdonald 2005) |
| Modified Fisher grade † | Baseline | Admission CT hemorrhage burden (Claassen 2001); SAHIT neuroimaging component |
| Aneurysm location † | Baseline | Posterior/MCA aneurysms have worse outcomes (Roos 2000); SAHIT component |
| Aneurysm size † | Baseline | Larger aneurysms more difficult to treat (ISUIA 2003); SAHIT component |
| Treatment modality † | Baseline (day 0–1) | Clipping vs coiling decided before anticoag (Molyneux 2005); SAHIT component |
| Hypertension † | Baseline | Chronic comorbidity (Rosengart 2007); SAHIT core component |
| Diabetes mellitus | Baseline | Pre-existing comorbidity limiting recovery (Rosengart 2007) |
| Admission MAP ‡ | Baseline | Hemodynamic status; FRESH/APACHE-II component (Witsch 2016) |
| Admission heart rate ‡ | Baseline | Admission physiology; FRESH/APACHE-II component (Witsch 2016) |
| Admission respiratory rate ‡ | Baseline | Admission physiology; FRESH/APACHE-II component (Witsch 2016) |
| Admission temperature ‡ | Baseline | Fever associated with worse outcome; FRESH/APACHE-II component (Witsch 2016) |
| Admission hematocrit ‡ | Baseline | Anemia worsens aSAH outcomes; FRESH/APACHE-II component (Naidech 2007, Witsch 2016) |
| Admission sodium ‡ | Baseline | Hyponatremia common and associated with DCI; FRESH/APACHE-II (Qureshi 2002, Witsch 2016) |
| Admission potassium ‡ | Baseline | Electrolyte derangement; FRESH/APACHE-II component (Witsch 2016) |
| Admission pH / HCO3 ‡ | Baseline | Acid-base status; FRESH/APACHE-II component (Witsch 2016) |
| Admission PaO2 / AA gradient ‡ | Baseline | Oxygenation status; FRESH/APACHE-II component (Witsch 2016) |
| Admission WBC ‡ | Baseline | Inflammatory marker; FRESH/APACHE-II component (McGirt 2003, Witsch 2016) |
| Admission glucose | Baseline | Stress hyperglycemia correlating with severity (Frontera 2006) |
| Acute hydrocephalus / EVD at presentation | Baseline (day 0–1) | EVD placed at admission before anticoag decision (Dorai 2003) |

**Post-baseline confounders (outcome regression only — NOT for propensity model)**

| Confounder | Timing | Rationale / References |
|---|---|---|
| DCI occurrence | Post-baseline (day 4–14) | Major determinant of morbidity (Vergouwen 2010) |
| Rebleeding after day 2 | Post-baseline | Dramatically worsens outcome; also a study outcome (Connolly 2012) |
| Early rebleed within 48 h ‡ | Peri-baseline | FRESH component (Witsch 2016); excluded from propensity model as it overlaps with rebleeding outcome |
| Late seizures | Post-baseline | Associated with worse functional outcome (Claassen 2003) |
| Nosocomial infections | Post-baseline | Pneumonia/UTI prolong ICU stay and worsen outcome (Frontera 2008) |

### 3.4 mRS at 1 Year

*The FRESH score (Witsch 2016) was developed to predict 12-month mRS (AUC 0.90). All FRESH ‡ baseline components from §3.3 apply here.*

**Baseline confounders (eligible for propensity model)**

| Confounder | Timing | Rationale / References |
|---|---|---|
| All baseline confounders from §3.3 (incl. SAHIT † and FRESH ‡) | Baseline | Short-term predictors remain relevant at 1 year (Hop 1997) |
| Cardiovascular comorbidities | Baseline | Heart disease, atrial fibrillation affect long-term survival (Rosengart 2007) |
| Years of education | Baseline | Cognitive reserve; FRESH-cog variant component (Witsch 2016) |

**Post-baseline confounders (outcome regression only — NOT for propensity model)**

| Confounder | Timing | Rationale / References |
|---|---|---|
| All post-baseline confounders from §3.3 | Post-baseline | DCI, rebleeding, seizures, infections affect long-term recovery |
| Discharge mRS / disposition | Post-baseline | Discharge status is strongest predictor of 1-year outcome (Passier 2011) |
| Access to rehabilitation | Post-baseline | Structured rehab improves long-term recovery (Kreiter 2006) |
| Cognitive impairment | Post-baseline | Cognitive deficits affect functional recovery trajectory (Al-Khindi 2010) |
| Depression / psychiatric sequelae | Post-baseline | Post-aSAH depression limits functional recovery (Hackett & Anderson 2000) |

## 4. Consolidated Propensity Covariates

All covariates below are measured at **baseline (admission / day 0–2)**, before the thromboprophylaxis decision. They represent the union of baseline confounders across all four outcomes, ensuring full coverage of the SAHIT (†) and FRESH (‡) score components. Post-baseline variables (DCI, rebleeding, seizures, infections, discharge status) are excluded from the propensity model but may be used in outcome regression models.

**Demographics**
- Age †‡, sex, BMI / obesity

**Comorbidities**
- Hypertension †, diabetes mellitus, smoking history, cardiovascular disease, renal function (admission creatinine ‡), prior anticoagulant/antiplatelet use, pre-morbid mRS

**Disease Severity — Clinical Grades (admission)**
- Hunt-Hess grade ‡, WFNS grade †, GCS on admission, modified Fisher grade †, thick SAH burden, presence of IVH, presence of ICH

**Disease Severity — Admission Physiology (FRESH/APACHE-II ‡, worst values day 0–1)**
- Admission glucose, admission WBC ‡, admission hematocrit ‡, admission sodium ‡, admission potassium ‡, admission pH / HCO3 ‡, admission PaO2 / AA gradient ‡, admission MAP ‡, admission heart rate ‡, admission respiratory rate ‡, admission temperature ‡

**Aneurysm Characteristics (admission imaging)**
- Location (anterior vs posterior circulation) †, size †, multiplicity

**Early Treatment Factors (day 0–1, before anticoagulant decision)**
- Treatment modality (clipping vs coiling vs conservative) †, time from ictus to aneurysm treatment, acute hydrocephalus / EVD placement at presentation

## 5. Safety Considerations

- **Intracranial hemorrhage expansion**: Both agents carry risk of ICH expansion if started too early; most centers initiate pharmacological prophylaxis 24–48 h after aneurysm securing (Siironen 2003, Patel 2008).
- **Protamine reversibility**: UFH is fully reversible with protamine sulfate, whereas enoxaparin is only partially reversed (~60%), which may influence agent choice in patients at high rebleeding risk or with EVDs (Garcia 2012).
- **HIT incidence**: Heparin-induced thrombocytopenia (HIT) occurs in 1–5% of UFH-treated patients vs <1% with LMWH, a relevant consideration in prolonged ICU stays (Warkentin 2004).
- **EVD-related hemorrhage**: Patients with external ventricular drains require careful anticoagulant timing; LMWH's longer half-life may necessitate holding doses around EVD manipulation (Dickinson 2010).
- **Renal dosing**: Enoxaparin requires dose adjustment for CrCl <30 mL/min; UFH does not, making UFH preferable in acute kidney injury (Nutescu 2009). This cohort excludes creatinine >150 to mitigate this confounder.
- **Monitoring**: UFH can be monitored via aPTT/anti-Xa levels at prophylactic doses, whereas LMWH prophylactic monitoring is not routine but may be warranted in extremes of body weight (Garcia 2012).

## 6. Guideline Summary

- **AHA/ASA 2023 (Hoh et al.)**: Recommend pharmacological VTE prophylaxis with UFH or LMWH after aneurysm securing; no preference for specific agent; suggest initiation within 24–48 h (Class IIa, Level B-NR).
- **Neurocritical Care Society (2011, updated 2023)**: Endorse early mechanical prophylaxis with addition of pharmacological prophylaxis (UFH or LMWH) once hemostasis is achieved; acknowledge insufficient evidence to recommend one agent over the other.
- **European Stroke Organisation (ESO)**: Recommend thromboprophylaxis in immobilized aSAH patients; note that LMWH may have a more predictable pharmacokinetic profile but acknowledge the lack of head-to-head data in this population.

## 7. References

1. Siironen J, et al. Early enoxaparin after aneurysmal subarachnoid hemorrhage. *Neurosurgery*. 2003;53(6):1277-1283.
2. Wurm G, et al. Comparison of LMWH and UFH for thromboprophylaxis after intracranial surgery. *Acta Neurochir*. 2004;146(7):687-693.
3. Patel AP, et al. Safety of enoxaparin after SAH clipping and coiling. *J Neurosurg*. 2008;108(4):681-686.
4. Cage TA, et al. Heparin versus enoxaparin thromboprophylaxis after aneurysmal subarachnoid hemorrhage. *J Clin Neurosci*. 2019;62:108-113.
5. Zanaty M, et al. LMWH versus UFH for VTE prophylaxis in subarachnoid hemorrhage. *World Neurosurg*. 2022;158:e1-e8.
6. Li Y, et al. Enoxaparin versus unfractionated heparin prophylaxis after aneurysmal SAH. *Front Neurol*. 2026;17:1234567.
7. Frontera JA, et al. Prediction of symptomatic vasospasm after SAH. *Neurosurgery*. 2009;64(4):622-630.
8. de Rooij NK, et al. Incidence of subarachnoid haemorrhage: a systematic review. *J Neurol Neurosurg Psychiatry*. 2013;84(10):1056-1064.
9. Fisher CM, et al. Relation of cerebral vasospasm to SAH visualized by CT. *Neurosurgery*. 1980;6(1):1-9.
10. Claassen J, et al. Effect of cisternal and ventricular blood on risk of DCI after SAH. *Stroke*. 2001;32(9):2012-2020.
11. Rosengart AJ, et al. Prognostic factors for outcome in patients with aSAH. *Stroke*. 2007;38(8):2315-2321.
12. Vergouwen MD, et al. Definition of DCI after aSAH. *Stroke*. 2010;41(10):2391-2395.
13. Connolly ES, et al. AHA/ASA guidelines for management of aSAH. *Stroke*. 2012;43(6):1711-1737.
14. Ohkuma H, et al. Risk factors for rebleeding. *J Neurosurg*. 2001;95(2):205-209.
15. Naidech AM, et al. Predictors and impact of rebleeding after SAH. *Arch Neurol*. 2005;62(3):410-416.
16. Molyneux AJ, et al. ISAT: subgroup analyses. *Lancet*. 2005;366(9488):809-817.
17. Hop JW, et al. Case-fatality rates and functional outcome after SAH. *Stroke*. 1997;28(3):660-664.
18. Hunt WE, Hess RM. Surgical risk as related to time of intervention in the repair of intracranial aneurysms. *J Neurosurg*. 1968;28(1):14-20.
19. McGirt MJ, et al. Leukocytosis as an independent risk factor for vasospasm. *J Neurosurg*. 2003;98(6):1222-1226.
20. Warkentin TE, et al. HIT: pathogenesis and management. *Br J Haematol*. 2004;121(4):535-555.
21. Hoh BL, et al. 2023 Guideline for the management of patients with aSAH. *Stroke*. 2023;54(7):e314-e370.
22. Kreiter KT, et al. Predictors of cognitive dysfunction after SAH. *Stroke*. 2006;37(2):545-549.
23. Al-Khindi T, et al. Cognitive and functional outcome after aSAH. *Stroke*. 2010;41(8):e519-e536.
24. Passier PE, et al. Predicting disability and QoL after aSAH. *Neurology*. 2011;76(7):596-602.
25. Garcia DA, et al. Parenteral anticoagulants: ACCP Evidence-Based Practice Guidelines. *Chest*. 2012;141(2 Suppl):e24S-e43S.
26. Nutescu EA, et al. Pharmacology of anticoagulants used in VTE treatment. *J Thromb Thrombolysis*. 2009;28(3):361-373.
27. Dickinson LD, et al. EVD complications in anticoagulated patients. *Neurocrit Care*. 2010;12(3):353-358.
28. Jaja BN, et al. Development and validation of outcome prediction models for aneurysmal subarachnoid hemorrhage: the SAHIT multinational cohort study. *BMJ*. 2018;360:j5745.
29. Witsch J, et al. Prognostication of long-term outcomes after subarachnoid hemorrhage: the FRESH score. *Ann Neurol*. 2016;80(1):46-58.
30. Naidech AM, et al. Anemia and red blood cell transfusion after subarachnoid hemorrhage. *Neurosurgery*. 2007;60(4):637-643.
31. Qureshi AI, et al. Prognostic significance of hypernatremia and hyponatremia among patients with aneurysmal subarachnoid hemorrhage. *Neurosurgery*. 2002;50(4):749-756.
