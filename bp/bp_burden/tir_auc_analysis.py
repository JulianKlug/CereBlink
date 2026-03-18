"""
Core computation functions for Time-in-Range (TIR) and AUC-based BP burden analysis.

TIR: Bin BP values into clinically relevant ranges, compute proportion of monitoring
time per bin per patient, and use as regression predictors.

AUC Burden: Compute area above/below threshold (time x deviation magnitude),
normalized by monitoring duration.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from bp.bp_burden.analysis_utils import save_regression_analysis_results_to_csv

# ── TIR bin definitions ────────────────────────────────────────────────────
TIR_BINS = {
    'systole':      [0, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 500],
    'diastole':     [0, 40, 50, 60, 70, 80, 90, 100, 110, 120, 500],
    'mitteldruck':  [0, 50, 60, 70, 80, 90, 100, 110, 120, 130, 500],
}

# Reference bins (dropped to avoid collinearity — proportions sum to 1)
TIR_REFERENCE_BINS = {
    'systole':      '(120, 130]',
    'diastole':     '(70, 80]',
    'mitteldruck':  '(80, 90]',
}

# ── AUC threshold ranges ──────────────────────────────────────────────────
AUC_THRESHOLDS = {
    'systole':      {'above': range(120, 200, 10), 'below': range(70, 120, 10)},
    'diastole':     {'above': range(80, 130, 10),  'below': range(40, 70, 10)},
    'mitteldruck':  {'above': range(90, 140, 10),  'below': range(50, 80, 10)},
}

COVARIATES = ['Age', 'WFNS', 'Fisher_Score', 'Coiling', 'Clipping']


# ═══════════════════════════════════════════════════════════════════════════
# TIR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_tir_proportions(working_df, bp_parameter, outcome):
    """Compute per-patient time-in-range proportions.

    Args:
        working_df: DataFrame with columns [pNr, bp_parameter, delta_time, outcome].
        bp_parameter: One of 'systole', 'diastole', 'mitteldruck'.
        outcome: Outcome column name ('mrs_1y' or 'DCI_YN_verified').

    Returns:
        DataFrame with one row per patient, columns = bin labels (proportions)
        plus pNr and outcome.
    """
    bins = TIR_BINS[bp_parameter]
    df = working_df[['pNr', bp_parameter, 'delta_time', outcome]].copy()
    df = df.dropna(subset=[bp_parameter, 'delta_time'])
    df = df[df['delta_time'] >= 0]

    df['bin'] = pd.cut(df[bp_parameter], bins=bins, include_lowest=True)

    # Weighted time per bin per patient
    weighted = df.groupby(['pNr', 'bin'], observed=False).agg(
        bin_time=('delta_time', 'sum'),
    ).reset_index()

    # Total monitoring time per patient (from this working_df subset)
    total_time = df.groupby('pNr')['delta_time'].sum().rename('total_time')
    weighted = weighted.merge(total_time, on='pNr', how='left')
    weighted['proportion'] = weighted['bin_time'] / weighted['total_time']
    weighted['proportion'] = weighted['proportion'].fillna(0)

    # Pivot to wide format: one column per bin
    pivot = weighted.pivot_table(
        index='pNr', columns='bin', values='proportion', fill_value=0
    )
    pivot.columns = [str(c) for c in pivot.columns]

    # Add outcome
    outcome_per_patient = df.groupby('pNr')[outcome].first()
    pivot = pivot.merge(outcome_per_patient, on='pNr', how='left')

    return pivot.reset_index()


# ═══════════════════════════════════════════════════════════════════════════
# AUC BURDEN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_auc_burden(working_df, bp_parameter, threshold, direction, outcome,
                       monitoring_duration_df):
    """Compute AUC-based burden for a single threshold.

    Args:
        working_df: DataFrame with [pNr, bp_parameter, delta_time, outcome].
        bp_parameter: BP column name.
        threshold: Numeric threshold value.
        direction: 'above' or 'below'.
        outcome: Outcome column name.
        monitoring_duration_df: DataFrame with [pNr, monitoring_duration].

    Returns:
        DataFrame with [pNr, burden, outcome] — one row per patient.
    """
    df = working_df[['pNr', bp_parameter, 'delta_time', outcome]].copy()
    df = df.dropna(subset=[bp_parameter, 'delta_time'])
    df = df[df['delta_time'] >= 0]

    if direction == 'above':
        df['deviation'] = (df[bp_parameter] - threshold).clip(lower=0)
    else:
        df['deviation'] = (threshold - df[bp_parameter]).clip(lower=0)

    df['area'] = df['deviation'] * df['delta_time']

    patient_auc = df.groupby('pNr').agg(
        auc=('area', 'sum'),
        **{outcome: (outcome, 'first')},
    ).reset_index()

    # Normalize by monitoring duration
    patient_auc = patient_auc.merge(
        monitoring_duration_df[['pNr', 'monitoring_duration']],
        on='pNr', how='left'
    )
    patient_auc['burden'] = patient_auc['auc'] / patient_auc['monitoring_duration']
    patient_auc['burden'] = patient_auc['burden'].fillna(0)

    return patient_auc[['pNr', 'burden', outcome]]


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════

def _fit_model(endog, exog, outcome, reg_type):
    """Fit ordinal or logistic regression model.

    Returns the fitted result or None on failure.
    """
    try:
        if reg_type == 'ordinal':
            model = OrderedModel(endog, exog, distr='logit')
            return model.fit(method='bfgs', disp=0)
        else:
            model = sm.Logit(endog, sm.add_constant(exog))
            return model.fit(disp=0, method='bfgs')
    except Exception as e:
        warnings.warn(f"Model fitting failed: {e}")
        return None


def run_tir_regression(tir_df, bp_parameter, outcome, covariate_df=None,
                       output_dir=None, period_name='', verbose=False):
    """Run TIR univariable and multivariable regressions.

    Args:
        tir_df: Wide-format TIR proportions DataFrame from compute_tir_proportions().
        bp_parameter: BP parameter name.
        outcome: Outcome column name.
        covariate_df: DataFrame with pNr + covariate columns for multivariable.
        output_dir: Directory to save results CSVs.
        period_name: Time period label for filenames.
        verbose: Print model summaries.

    Returns:
        dict with keys 'univariable' and 'multivariable', each containing
        lists of {bin, result} dicts.
    """
    reg_type = 'ordinal' if outcome == 'mrs_1y' else 'log'
    reference_bin = TIR_REFERENCE_BINS[bp_parameter]

    # Identify bin columns (exclude pNr and outcome)
    bin_cols = [c for c in tir_df.columns if c not in ['pNr', outcome]]

    # Remove reference bin
    predictor_bins = [c for c in bin_cols if c != reference_bin]

    results = {'univariable': [], 'multivariable': []}

    # ── Univariable: one model per bin ──
    for bin_col in predictor_bins:
        temp = tir_df[['pNr', bin_col, outcome]].dropna(subset=[bin_col, outcome])
        if temp[bin_col].std() == 0 or len(temp) < 10:
            continue

        result = _fit_model(temp[outcome], temp[[bin_col]], outcome, reg_type)
        if result is not None:
            results['univariable'].append({'bin': bin_col, 'result': result})
            if output_dir:
                save_regression_analysis_results_to_csv(
                    result, output_dir,
                    f'{bp_parameter}_tir_univariable_{outcome}_{period_name}_bin_{bin_col}'
                )
            if verbose:
                print(f"TIR univariable {bin_col}:")
                print(result.summary())

    # ── Multivariable: all bins + covariates ──
    if covariate_df is not None:
        temp = tir_df.merge(
            covariate_df[['pNr'] + COVARIATES].drop_duplicates(subset='pNr'),
            on='pNr', how='left'
        )
        all_predictors = predictor_bins + COVARIATES
        temp = temp[all_predictors + [outcome]].dropna()

        for col in COVARIATES:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        temp = temp.dropna()

        if len(temp) >= 10:
            result = _fit_model(temp[outcome], temp[all_predictors], outcome, reg_type)
            if result is not None:
                results['multivariable'].append({'bins': predictor_bins, 'result': result})
                if output_dir:
                    save_regression_analysis_results_to_csv(
                        result, output_dir,
                        f'{bp_parameter}_tir_multivariable_{outcome}_{period_name}'
                    )
                if verbose:
                    print(f"TIR multivariable model:")
                    print(result.summary())

    return results


def run_auc_regression(working_df, bp_parameter, outcome, direction,
                       monitoring_duration_df, covariate_df=None,
                       output_dir=None, period_name='', verbose=False):
    """Run AUC burden regression across all thresholds for a direction.

    Args:
        working_df: BP data with [pNr, bp_parameter, delta_time, outcome].
        bp_parameter: BP parameter name.
        outcome: Outcome column name.
        direction: 'above' or 'below'.
        monitoring_duration_df: Per-patient monitoring durations.
        covariate_df: For multivariable models.
        output_dir: Save directory.
        period_name: Time period label.
        verbose: Print summaries.

    Returns:
        dict with 'univariable' and 'multivariable' lists of
        {threshold, result} dicts.
    """
    reg_type = 'ordinal' if outcome == 'mrs_1y' else 'log'
    thresholds = AUC_THRESHOLDS[bp_parameter][direction]

    results = {'univariable': [], 'multivariable': []}

    for threshold in thresholds:
        burden_df = compute_auc_burden(
            working_df, bp_parameter, threshold, direction, outcome,
            monitoring_duration_df
        )

        # ── Univariable ──
        temp = burden_df[['burden', outcome]].dropna()
        if temp['burden'].std() == 0 or len(temp) < 10:
            continue

        uni_result = _fit_model(temp[outcome], temp[['burden']], outcome, reg_type)
        if uni_result is not None:
            results['univariable'].append({
                'threshold': threshold, 'result': uni_result
            })
            if output_dir:
                save_regression_analysis_results_to_csv(
                    uni_result, output_dir,
                    f'{bp_parameter}_auc_{direction}_{outcome}_{period_name}_t{threshold}_univariable'
                )

        # ── Multivariable ──
        if covariate_df is not None:
            temp_multi = burden_df.merge(
                covariate_df[['pNr'] + COVARIATES].drop_duplicates(subset='pNr'),
                on='pNr', how='left'
            )
            predictors = ['burden'] + COVARIATES
            temp_multi = temp_multi[predictors + [outcome]].dropna()
            for col in COVARIATES:
                temp_multi[col] = pd.to_numeric(temp_multi[col], errors='coerce')
            temp_multi = temp_multi.dropna()

            if len(temp_multi) >= 10 and temp_multi['burden'].std() > 0:
                multi_result = _fit_model(
                    temp_multi[outcome], temp_multi[predictors], outcome, reg_type
                )
                if multi_result is not None:
                    results['multivariable'].append({
                        'threshold': threshold, 'result': multi_result
                    })
                    if output_dir:
                        save_regression_analysis_results_to_csv(
                            multi_result, output_dir,
                            f'{bp_parameter}_auc_{direction}_{outcome}_{period_name}_t{threshold}_multivariable'
                        )

        if verbose and uni_result is not None:
            print(f"AUC {direction} threshold={threshold}:")
            print(uni_result.summary())

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_tir_forest(tir_results, bp_parameter, outcome, period_name,
                    output_dir=None):
    """Forest plot of per-bin coefficients with 95% CIs from univariable models."""
    uni_results = tir_results['univariable']
    if not uni_results:
        return None

    bins_list = []
    coefs = []
    ci_lowers = []
    ci_uppers = []

    for entry in uni_results:
        bin_label = entry['bin']
        result = entry['result']
        try:
            # For ordinal models the first param is the bin; for logit the bin is
            # the second param (after const)
            if outcome == 'mrs_1y':
                idx = 0  # ordinal: first param is the predictor
            else:
                idx = result.params.index.get_loc(bin_label) if bin_label in result.params.index else 1
            coef = result.params.iloc[idx]
            ci = result.conf_int().iloc[idx]
            bins_list.append(bin_label)
            coefs.append(coef)
            ci_lowers.append(ci[0])
            ci_uppers.append(ci[1])
        except Exception:
            continue

    if not bins_list:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(bins_list) * 0.5)))
    y_pos = range(len(bins_list))
    errors = [[c - l for c, l in zip(coefs, ci_lowers)],
              [u - c for c, u in zip(coefs, ci_uppers)]]

    ax.errorbar(coefs, y_pos, xerr=errors, fmt='o', color='steelblue',
                capsize=3, linewidth=1.5, markersize=6)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bins_list)
    ax.set_xlabel('Coefficient (95% CI)')
    ax.set_title(f'TIR Univariable — {bp_parameter} — {outcome} — {period_name}')
    ax.invert_yaxis()
    fig.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 f'{bp_parameter}_tir_forest_{outcome}_{period_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_tir_stacked_bar(tir_df, bp_parameter, outcome, period_name,
                         output_dir=None):
    """Stacked bar of TIR distributions stratified by outcome."""
    reference_bin = TIR_REFERENCE_BINS[bp_parameter]
    bin_cols = [c for c in tir_df.columns if c not in ['pNr', outcome]]

    if outcome == 'DCI_YN_verified':
        groups = {0: 'No DCI', 1: 'DCI'}
    else:
        groups = None  # use unique mRS values

    fig, ax = plt.subplots(figsize=(12, 6))

    if groups is not None:
        group_means = {}
        for val, label in groups.items():
            subset = tir_df[tir_df[outcome] == val]
            if len(subset) > 0:
                group_means[label] = subset[bin_cols].mean()
        if not group_means:
            plt.close(fig)
            return None
        means_df = pd.DataFrame(group_means).T
    else:
        tir_df_copy = tir_df.dropna(subset=[outcome]).copy()
        tir_df_copy[outcome] = tir_df_copy[outcome].astype(int)
        if tir_df_copy.empty:
            plt.close(fig)
            return None
        means_df = tir_df_copy.groupby(outcome)[bin_cols].mean()

    means_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel('Mean proportion of monitoring time')
    ax.set_xlabel(outcome)
    ax.set_title(f'TIR Distribution — {bp_parameter} — {period_name}')
    ax.legend(title='BP Range', bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize='small')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 f'{bp_parameter}_tir_stacked_bar_{outcome}_{period_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_auc_coefficient_curve(auc_results, bp_parameter, outcome, direction,
                               period_name, output_dir=None):
    """Line plot of coefficient vs threshold with CI bands and p-value shading."""
    uni_results = auc_results['univariable']
    if not uni_results:
        return None

    thresholds = []
    coefs = []
    ci_lowers = []
    ci_uppers = []
    pvalues = []

    for entry in uni_results:
        result = entry['result']
        try:
            if outcome == 'mrs_1y':
                idx = 0
            else:
                # logit: burden is first non-const predictor
                idx = result.params.index.get_loc('burden') if 'burden' in result.params.index else 1
            coef = result.params.iloc[idx]
            ci = result.conf_int().iloc[idx]
            pval = result.pvalues.iloc[idx]
            thresholds.append(entry['threshold'])
            coefs.append(coef)
            ci_lowers.append(ci[0])
            ci_uppers.append(ci[1])
            pvalues.append(pval)
        except Exception:
            continue

    if not thresholds:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, coefs, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.fill_between(thresholds, ci_lowers, ci_uppers, alpha=0.2, color='steelblue')
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    # Shade significant thresholds
    for i, (t, p) in enumerate(zip(thresholds, pvalues)):
        if p < 0.05:
            ax.axvspan(t - 4, t + 4, alpha=0.1, color='green')

    ax.set_xlabel(f'{bp_parameter} threshold (mmHg)')
    ax.set_ylabel('Coefficient (95% CI)')
    ax.set_title(f'AUC Burden ({direction}) — {bp_parameter} — {outcome} — {period_name}')
    fig.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 f'{bp_parameter}_auc_{direction}_coef_curve_{outcome}_{period_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_auc_scatter_best(working_df, bp_parameter, outcome, direction,
                          monitoring_duration_df, best_threshold,
                          period_name, output_dir=None):
    """Scatter plot of AUC burden vs outcome at the optimal threshold."""
    burden_df = compute_auc_burden(
        working_df, bp_parameter, best_threshold, direction, outcome,
        monitoring_duration_df
    )
    burden_df = burden_df.dropna(subset=['burden', outcome])

    fig, ax = plt.subplots(figsize=(8, 6))

    if outcome == 'DCI_YN_verified':
        # Jittered strip plot
        jitter = np.random.normal(0, 0.05, size=len(burden_df))
        ax.scatter(burden_df[outcome] + jitter, burden_df['burden'],
                   alpha=0.5, s=30, color='steelblue')
        ax.set_xlabel(outcome)
    else:
        ax.scatter(burden_df[outcome], burden_df['burden'],
                   alpha=0.5, s=30, color='steelblue')
        ax.set_xlabel(f'{outcome}')

    ax.set_ylabel(f'AUC Burden ({direction}, threshold={best_threshold})')
    ax.set_title(f'AUC Burden vs {outcome} — {bp_parameter} — {period_name}')
    fig.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir,
                                 f'{bp_parameter}_auc_{direction}_scatter_t{best_threshold}_{outcome}_{period_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig
