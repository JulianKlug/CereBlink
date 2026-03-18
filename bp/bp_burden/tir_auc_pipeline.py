"""
Orchestration pipeline for Time-in-Range (TIR) and AUC-based BP burden analysis.

Connects data loading (via prepare_bp_data) → time period filtering →
TIR and AUC analysis → CSV/plot output.

Usage (from CereBlink root):
    python -m bp.bp_burden.tir_auc_pipeline --help
"""

import os
import json

import pandas as pd

from utils.utils import ensure_dir
from bp.bp_burden.analysis_pipeline import prepare_bp_data
from bp.bp_burden.tir_auc_analysis import (
    compute_tir_proportions,
    run_tir_regression,
    run_auc_regression,
    plot_tir_forest,
    plot_tir_stacked_bar,
    plot_auc_coefficient_curve,
    plot_auc_scatter_best,
    AUC_THRESHOLDS,
)


def _get_time_periods(working_df, restrict_to_DCI):
    """Return dict of {period_name: filtered_df} for the requested time periods."""
    periods = {}

    # Before aneurysm secured
    periods['before_aneurysm_secured'] = working_df[
        working_df['relative_time'] < (working_df['first_Th_relative_date'] + 24 * 60)
    ]

    # After aneurysm secured
    periods['after_aneurysm_secured'] = working_df[
        working_df['relative_time'] >= (working_df['first_Th_relative_date'] + 24 * 60)
    ]

    # Before/after DCI (only for DCI-restricted analyses)
    if restrict_to_DCI:
        periods['before_dci'] = working_df[
            working_df['relative_time'] <= working_df['relative_dci_time']
        ]
        periods['after_dci'] = working_df[
            working_df['relative_time'] > working_df['relative_dci_time']
        ]

    return periods


def _find_best_auc_threshold(auc_results):
    """Find the threshold with the lowest univariable p-value."""
    best_threshold = None
    best_pval = 1.0
    for entry in auc_results.get('univariable', []):
        result = entry['result']
        try:
            # Get p-value for the burden predictor
            if 'burden' in result.pvalues.index:
                pval = result.pvalues['burden']
            else:
                pval = result.pvalues.iloc[0]
            if pval < best_pval:
                best_pval = pval
                best_threshold = entry['threshold']
        except Exception:
            continue
    return best_threshold


def tir_auc_analysis_pipeline(
        registry_data_path: str,
        nor_annotated_bp_data_path: str,
        correspondance_data_path: str,
        outcome_data_path: str,
        output_dir: str,
        bp_parameter: str,
        outcome: str = 'mrs_1y',
        restrict_to_DCI: bool = False,
        restrict_to_non_DCI: bool = False,
        registry_password: str = None,
        outcome_password: str = None,
        verbose: bool = False,
        preloaded_registry_df: pd.DataFrame = None,
        preloaded_outcome_df: pd.DataFrame = None,
        preloaded_bp_df: pd.DataFrame = None,
        preloaded_correspondance_df: pd.DataFrame = None,
):
    """Run the full TIR + AUC burden analysis pipeline.

    Always uses noradrenaline-filtered data.
    """
    if outcome not in ['mrs_1y', 'DCI_YN_verified']:
        raise ValueError(f'Invalid outcome: {outcome}')
    if bp_parameter not in ['systole', 'diastole', 'mitteldruck']:
        raise ValueError(f'Invalid bp_parameter: {bp_parameter}')

    ensure_dir(output_dir)

    # ── Data preparation (always NA-filtered) ──
    main_df, monitoring_duration_df = prepare_bp_data(
        registry_data_path=registry_data_path,
        nor_annotated_bp_data_path=nor_annotated_bp_data_path,
        correspondance_data_path=correspondance_data_path,
        outcome_data_path=outcome_data_path,
        filter_noradrenaline=True,
        restrict_to_DCI=restrict_to_DCI,
        restrict_to_non_DCI=restrict_to_non_DCI,
        registry_password=registry_password,
        outcome_password=outcome_password,
        verbose=verbose,
        preloaded_registry_df=preloaded_registry_df,
        preloaded_outcome_df=preloaded_outcome_df,
        preloaded_bp_df=preloaded_bp_df,
        preloaded_correspondance_df=preloaded_correspondance_df,
    )

    working_df = main_df[['relative_time', 'pNr', 'delta_time',
                           'systole', 'diastole', 'mitteldruck',
                           'DCI_YN_verified', 'mrs_1y',
                           'first_Th_relative_date', 'relative_dci_time']].copy()

    # Covariate df for multivariable models
    covariate_df = main_df[['pNr', 'Age', 'WFNS', 'Fisher_Score',
                             'Coiling', 'Clipping']].drop_duplicates(subset='pNr')

    # ── Time period filtering ──
    periods = _get_time_periods(working_df, restrict_to_DCI)

    if verbose:
        print(f"\nPatients: {working_df['pNr'].nunique()}")
        print(f"BP parameter: {bp_parameter}, Outcome: {outcome}")
        print(f"Time periods: {list(periods.keys())}\n")

    # ── Run analyses per time period ──
    for period_name, period_df in periods.items():
        n_patients = period_df['pNr'].nunique()
        if verbose:
            print(f"\n{'='*60}")
            print(f"Period: {period_name} ({n_patients} patients)")
            print(f"{'='*60}")

        if n_patients < 10:
            if verbose:
                print(f"Skipping {period_name}: only {n_patients} patients")
            continue

        period_dir = output_dir  # all outputs in same dir, differentiated by filename

        # ── TIR analysis ──
        if verbose:
            print(f"\n--- TIR Analysis ---")

        tir_df = compute_tir_proportions(period_df, bp_parameter, outcome)

        # Save per-patient TIR data
        tir_df.to_csv(os.path.join(period_dir,
                                    f'{bp_parameter}_tir_proportions_{outcome}_{period_name}.csv'),
                       index=False)

        tir_results = run_tir_regression(
            tir_df, bp_parameter, outcome, covariate_df=covariate_df,
            output_dir=period_dir, period_name=period_name, verbose=verbose
        )

        # TIR plots
        plot_tir_forest(tir_results, bp_parameter, outcome, period_name,
                        output_dir=period_dir)
        plot_tir_stacked_bar(tir_df, bp_parameter, outcome, period_name,
                             output_dir=period_dir)

        # ── AUC analysis (both directions) ──
        for direction in ['above', 'below']:
            thresholds = AUC_THRESHOLDS[bp_parameter][direction]
            if not thresholds:
                continue

            if verbose:
                print(f"\n--- AUC Analysis ({direction}) ---")

            auc_results = run_auc_regression(
                period_df, bp_parameter, outcome, direction,
                monitoring_duration_df, covariate_df=covariate_df,
                output_dir=period_dir, period_name=period_name, verbose=verbose
            )

            # Save per-patient burden data for each threshold
            for threshold in thresholds:
                from bp.bp_burden.tir_auc_analysis import compute_auc_burden
                burden_df = compute_auc_burden(
                    period_df, bp_parameter, threshold, direction, outcome,
                    monitoring_duration_df
                )
                burden_df.to_csv(os.path.join(
                    period_dir,
                    f'{bp_parameter}_auc_{direction}_{outcome}_{period_name}_t{threshold}_burden.csv'
                ), index=False)

            # AUC plots
            plot_auc_coefficient_curve(
                auc_results, bp_parameter, outcome, direction,
                period_name, output_dir=period_dir
            )

            best_threshold = _find_best_auc_threshold(auc_results)
            if best_threshold is not None:
                plot_auc_scatter_best(
                    period_df, bp_parameter, outcome, direction,
                    monitoring_duration_df, best_threshold,
                    period_name, output_dir=period_dir
                )

    if verbose:
        print(f"\nAll results saved to {output_dir}")
