# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Propensity-controlled comparison of **heparin vs enoxaparin** at thromboprophylactic doses after aneurysmal subarachnoid hemorrhage (aSAH).

**Outcomes:** DCI, rebleeding, mRs at discharge, mRs at 1 year

## Data

- Source: `/mnt/data1/klug/datasets/kssg/SAH`

## Analysis Plan

1. Literature review of relevant confounders → concise summary
2. Map confounders and outcomes to data → data dictionary and a missigness table 
3. Define patient cohort with exclusions:
   - Renal failure (creatinine > 150)
   - Therapeutic anticoagulation (heparin > 10,000 UI/24h at any timepoint, enoxaparin > 40mg/d on any day)
   - Patients who received both heparin and enoxaparin
   - Duplicates
   - Produce reproducible cohort selection script and flow chart
4. Assign groups (heparin vs enoxaparin) with no overlap
5. Propensity-controlled comparison between groups for all outcomes
6. Table 1, results tables, and figures (with multiple options per figure)
7. Summary PDF report

## Setup

```bash
pip install -r ../requirements.txt
```

## Conventions

- Scripts should ensure reproducibility of cohort selection and analyses
- Parent repo dependencies are in `../requirements.txt`
