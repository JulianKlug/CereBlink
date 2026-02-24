# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CereBlink is a clinical research project for **Delayed Cerebral Ischemia (DCI) detection in Subarachnoid Hemorrhage (SAH) patients**. This subdirectory (`bp/bp_burden`) analyzes the burden of blood pressure events and their correlation with clinical outcomes.

Related paper: Klug et al., Critical Care Explorations, August 2024.

## Data Paths
output data: /mnt/data1/klug/output/cereblink_bp

## Running the Analysis

The pipeline runs from the repository root (`CereBlink/`), not from `bp/bp_burden/`:

```bash
# From CereBlink/ root (required for utils imports to resolve)
python -m bp.bp_burden.analysis_pipeline \
  -r /path/to/registry.xlsx \
  -b /path/to/bp_data.csv \
  -c /path/to/correspondence.csv \
  -o /path/to/outcome.xlsx \
  -d /output/directory \
  --bp_parameter systole \
  --outcomes mrs_1y DCI_YN_verified \
  --intensity_threshold_range 140 220 \
  --intensity_threshold_step 10 \
  --duration_range 0 180 \
  --duration_step 10 \
  -v
```

Optional flags: `--filter_noradrenaline`, `--restrict_to_DCI`, `--use_average_event_counts`.

Install dependencies: `pip install -r requirements.txt` (from repo root). Also requires `msoffcrypto` for encrypted Excel files.

## Architecture

### Data Flow

```
Encrypted Registry (.xlsx) + BP Data (.csv) + Correspondence (.csv) + Encrypted Outcome (.xlsx)
    → Data merging & preprocessing (analysis_pipeline.py)
    → Event detection across intensity thresholds (analysis_utils.py)
    → Duration filtering across duration thresholds (analysis_utils.py)
    → Event counting per patient/threshold combo
    → Correlation with outcomes (Pearson for mRS, logistic regression for DCI)
    → Heatmap visualization + Decision boundary analysis (uni & multivariable)
    → CSV results + PNG plots
```

### Key Files

- **`analysis_pipeline.py`** — Orchestration. `bp_events_analysis_pipeline()` is the core function; it loads data, merges registries, computes timing fields, then runs `event_burden_analysis()` for four time periods (first 24h, after 24h, before/after aneurysm treatment). `all_outcomes_bp_events_analysis_pipeline()` wraps it to iterate over outcomes.
- **`analysis_utils.py`** — All analytical logic: event detection (`define_events_over_intensity_thresholds`), duration filtering, event counting, correlation functions, decision boundary analysis (ordinal/logistic regression with covariates), heatmap plotting, CSV export.
- **`utils/utils.py`** (repo root) — Shared utilities: `load_encrypted_xlsx()` for password-protected Excel, `ensure_dir()` for directory creation.

### Outcome-Dependent Logic

The pipeline branches based on the outcome variable:
- **`mrs_1y`** (Modified Rankin Scale at 1 year): Uses Pearson correlation for sweep analysis, ordinal regression (proportional odds) for decision boundary analysis.
- **`DCI_YN_verified`** (DCI presence): Uses logistic regression for both sweep and decision boundary analysis.

Multivariable models include covariates: Age, WFNS, Fisher_Score, Coiling, Clipping.

### BP Event Definition

An "event" is a contiguous period where a BP parameter (systole/diastole/mitteldruck) exceeds an intensity threshold. The analysis sweeps across a grid of intensity thresholds × duration thresholds, producing heatmaps of correlation coefficients. The decision boundary analysis then computes the proportion of monitoring time spent in positively vs. negatively correlated event zones.

## Conventions

- Patient identifier column: `pNr`
- BP parameters: `systole`, `diastole`, `mitteldruck`
- Time is tracked as `relative_time` in minutes from first measurement (`T0`)
- All imports assume the working directory is the CereBlink repo root (e.g., `from utils.utils import ...`, `from bp.bp_burden.analysis_utils import ...`)
- Registry and outcome data are password-protected Excel files decrypted at runtime via `msoffcrypto`
- Analysis arguments are logged to `analysis_arguments.json` in the output directory
