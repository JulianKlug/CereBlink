"""
Batch runner for all BP burden analysis configurations.

Loads data files once, then iterates over all configuration combinations
calling bp_events_analysis_pipeline() for each.

Usage (from CereBlink root):
    conda run -n cereblink python -m bp.bp_burden.run_all_analyses
"""

import os
import json
import time

import pandas as pd

from utils.utils import load_encrypted_xlsx, ensure_dir
from bp.bp_burden.analysis_pipeline import bp_events_analysis_pipeline

# ── Data paths ────────────────────────────────────────────────────────────────
# registry = post_hoc file (has DCI_YN_verified, Fisher_Score, covariates)
# outcome  = aSAH file (has mRS_FU_1y)
REGISTRY_PATH = "/mnt/data1/klug/datasets/kssg/SAH/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx"
BP_DATA_PATH = "/mnt/data1/klug/datasets/kssg/SAH/20240116_SAH_SOS_Blutdruecke_nor_annotated.csv"
CORRESPONDANCE_PATH = "/mnt/data1/klug/datasets/kssg/SAH/registry_pdms_correspondence.csv"
OUTCOME_PATH = "/mnt/data1/klug/datasets/kssg/SAH/aSAH_DATA_2009_2024_18122024.xlsx"

SECRET_FILE = os.path.join(os.path.dirname(__file__), ".secret")

OUTPUT_ROOT = "/mnt/data1/klug/output/cereblink_bp"

# ── Threshold parameters ─────────────────────────────────────────────────────
INTENSITY_THRESHOLD_RANGE = (0, 200)
INTENSITY_THRESHOLD_STEP = 10
DURATION_RANGE = (0, 180)
DURATION_STEP = 10

# ── BP parameters and directions ─────────────────────────────────────────────
BP_PARAMS = ["systole", "diastole", "mitteldruck"]
DIRECTIONS = ["hypertension", "hypotension"]

# ── Configuration table ──────────────────────────────────────────────────────
# Each entry: (outcome, na_filtered, restrict_to_DCI, restrict_to_non_DCI, output_subdir)
CONFIGS = [
    ("DCI_YN_verified", False, False, False, "dci"),
    ("DCI_YN_verified", True,  False, False, "na_filtered_dci"),
    ("mrs_1y",          False, False, False, "mrs_1y"),
    ("mrs_1y",          True,  False, False, "na_filtered_mrs_1y"),
    ("mrs_1y",          False, True,  False, "subgroup_dci"),
    ("mrs_1y",          True,  True,  False, "na_filtered_subgroup_dci"),
    ("mrs_1y",          False, False, True,  "subgroup_non_dci"),
    ("mrs_1y",          True,  False, True,  "na_filtered_subgroup_non_dci"),
]


def load_passwords():
    """Load passwords from .secret file."""
    passwords = {}
    with open(SECRET_FILE) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, val = line.split("=", 1)
                passwords[key.strip()] = val.strip()
    return passwords["registry_password"], passwords["outcome_password"]


def main():
    registry_password, outcome_password = load_passwords()

    # Load data files once
    print("Loading data files...", flush=True)
    t0 = time.time()
    registry_df = load_encrypted_xlsx(REGISTRY_PATH, password=registry_password)
    outcome_df = load_encrypted_xlsx(OUTCOME_PATH, password=outcome_password)
    bp_df = pd.read_csv(BP_DATA_PATH)
    correspondance_df = pd.read_csv(CORRESPONDANCE_PATH)
    print(f"Data loaded in {time.time() - t0:.1f}s", flush=True)

    total_runs = len(CONFIGS) * len(BP_PARAMS) * len(DIRECTIONS)
    run_number = 0
    failed_runs = []

    for outcome, na_filtered, restrict_dci, restrict_non_dci, output_subdir in CONFIGS:
        for bp_param in BP_PARAMS:
            for direction in DIRECTIONS:
                run_number += 1
                output_dir = os.path.join(OUTPUT_ROOT, output_subdir)
                ensure_dir(output_dir)

                print(f"\n{'='*70}", flush=True)
                print(f"Run {run_number}/{total_runs}: {output_subdir} | {bp_param} | {direction} | {outcome}", flush=True)
                print(f"{'='*70}", flush=True)

                t_start = time.time()

                try:
                    bp_events_analysis_pipeline(
                        registry_data_path=REGISTRY_PATH,
                        nor_annotated_bp_data_path=BP_DATA_PATH,
                        correspondance_data_path=CORRESPONDANCE_PATH,
                        outcome_data_path=OUTCOME_PATH,
                        output_dir=output_dir,
                        intensity_threshold_range=INTENSITY_THRESHOLD_RANGE,
                        intensity_threshold_step=INTENSITY_THRESHOLD_STEP,
                        duration_range=DURATION_RANGE,
                        duration_step=DURATION_STEP,
                        bp_parameter=bp_param,
                        bp_focus=direction,
                        filter_noradrenaline=na_filtered,
                        restrict_to_DCI=restrict_dci,
                        restrict_to_non_DCI=restrict_non_dci,
                        outcome=outcome,
                        registry_password=registry_password,
                        outcome_password=outcome_password,
                        verbose=True,
                        preloaded_registry_df=registry_df,
                        preloaded_outcome_df=outcome_df,
                        preloaded_bp_df=bp_df,
                        preloaded_correspondance_df=correspondance_df,
                    )
                except Exception as e:
                    print(f"ERROR in run {run_number}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    failed_runs.append((run_number, output_subdir, bp_param, direction, str(e)))
                    continue

                elapsed = time.time() - t_start
                print(f"Completed in {elapsed:.1f}s", flush=True)

                # Save run config to output directory
                run_config = {
                    "outcome": outcome,
                    "bp_parameter": bp_param,
                    "bp_focus": direction,
                    "filter_noradrenaline": na_filtered,
                    "restrict_to_DCI": restrict_dci,
                    "restrict_to_non_DCI": restrict_non_dci,
                    "intensity_threshold_range": list(INTENSITY_THRESHOLD_RANGE),
                    "intensity_threshold_step": INTENSITY_THRESHOLD_STEP,
                    "duration_range": list(DURATION_RANGE),
                    "duration_step": DURATION_STEP,
                }
                config_path = os.path.join(output_dir, f"run_config_{bp_param}_{direction}.json")
                with open(config_path, "w") as f:
                    json.dump(run_config, f, indent=4)

    print(f"\n{'='*70}", flush=True)
    print(f"All {total_runs} runs completed.", flush=True)
    if failed_runs:
        print(f"\n{len(failed_runs)} FAILED runs:", flush=True)
        for r in failed_runs:
            print(f"  Run {r[0]}: {r[1]} | {r[2]} | {r[3]} — {r[4]}", flush=True)
    else:
        print("No failures.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
