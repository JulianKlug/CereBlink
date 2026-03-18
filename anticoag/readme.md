# Comparison of heparine vs enoxaparine for thromboprophylaxis after aSAH

Gist: propensity controlled comparison of enoxaparine vs heparine at thromboprophylactic doses after aSAH

outcomes: DCI, rebleeding, mRs at discharge, mrs at 1 year

Data path: /mnt/data1/klug/datasets/kssg/SAH  

Steps:
1. research litterature to identify all relevant confounders - save in a concise summary
2. identify selected confounders and outcomes in data - create a data dictionary and a missigness table 
3. identify patient cohort: exclude patients with renal failure (creatinine > 150), exclude patients with therapeutic anticoagulaiton (heparine > 10000 UI/24h at any timepoint, enoxaparin > 40mg/d on any day), exclude patients that received both heparine and enoxaparine, exclude duplicates
-- create script to ensure reproducibility
-- create flow chart 
4. identify groups: group hpearin, group enoxaprine - ensure there is no overlap
5. do propensity coontrolled comparison between groups for all outcomes
6. create table1, results tables and figures (with multiple otions of each figure)
7. create summary pdf report 
