import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import statsmodels.stats.multitest
from multipy.fdr import qvalue
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
from pymer4.models import Lmer
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro


def timebin_analysis(timebin_metrics_df, metrics_over_time=['median', 'min', 'max'],
                     bp_metrics=['systole', 'diastole', 'mitteldruck'], use_R=False):
    # combination of bp_metrics and metrics_over_time
    timebin_metrics = [f'{bp_metric}_{metric_over_time}' for bp_metric in bp_metrics for metric_over_time in metrics_over_time]

    # directly call R
    if use_R:
        stats = importr('stats')
        lme4 = importr('lme4')
        base = importr('base')
        lmerT = importr('lmerTest')

        r_pvals_per_metric = {}
        r_model_warnings_df = pd.DataFrame(columns=['metric', 'warning'])

        for metric in tqdm(timebin_metrics, total=len(timebin_metrics)):
            metric_df = timebin_metrics_df[[metric, 'label', 'pNr']]
            n_inf = (metric_df[metric] == np.inf).sum() + (metric_df[metric] == -np.inf).sum()
            n_nan = metric_df[metric].isna().sum()
            # drop inf and nan values
            metric_df.dropna(subset=[metric], inplace=True)
            metric_df = metric_df[~metric_df[metric].isin([np.inf, -np.inf])]

            if n_inf > 0:
                print(f'Warning: {n_inf} inf values in metric {metric}')
            if n_nan > 0:
                print(f'Warning: {n_nan} nan values in metric {metric}')

            with (ro.default_converter + pandas2ri.converter).context():
                metric_r_df = ro.conversion.get_conversion().py2rpy(metric_df)

            # lmc = ro.r(f'lmerControl({"optCtrl = list(ftol_abs=1e-15, xtol_abs=1e-15)"})')
            try:
                model = lmerT.lmer(f"label  ~ {metric}  + (1|pNr)",
                               data=metric_r_df)
                coeffs = base.summary(model).rx2('coefficients')
                indices = np.asarray(list(coeffs.names)[0])
                column_names = np.asarray(list(coeffs.names)[1])

                with (ro.default_converter + pandas2ri.converter).context():
                    coeffs_df = pd.DataFrame(ro.conversion.get_conversion().rpy2py(coeffs),
                                             index=indices, columns=column_names)

                r_pvals_per_metric[metric] = coeffs_df.loc[metric, 'Pr(>|t|)']

                warnings = base.summary(model).rx2('warnings')
                # check if warnings is null
                if warnings != ro.rinterface.NULL:
                    r_model_warnings_df = pd.concat(
                        [r_model_warnings_df, pd.DataFrame({'metric': [metric], 'warning': [warnings]})])
            # Catch non definite VtV (happens when there is not enough variance in the random effect, ie single sample in pos. class
            except Exception as e:
                accepted_errors = ['Erreur dans eval_f(x, ...) : Downdated VtV is not positive definite\n',
                                     'Erreur dans asMethod(object) : not a positive definite matrix\n',
                                   'Erreur dans devfun(theta) : Downdated VtV is not positive definite\n']
                if (isinstance(e, RRuntimeError) and e.args[0] in accepted_errors):
                    print(f'Error in metric: {metric}')
                    r_pvals_per_metric[metric] = 1
                    r_model_warnings_df = pd.concat(
                        [r_model_warnings_df, pd.DataFrame({'metric': [metric],
                                                            'warning': [str(e) + ', n_pos=' + str(metric_df.label.sum())]})])
                else:
                    raise e

        pvals_per_metric = r_pvals_per_metric
        model_warnings_df = r_model_warnings_df

    # Use python
    else:
        # CAVE: convergence errors
        pvals_per_metric = {}
        model_warnings_df = pd.DataFrame(columns=['metric', 'warning'])
        for metric in tqdm(timebin_metrics, total=len(timebin_metrics)):
            metric_df = timebin_metrics_df[[metric, 'label', 'pNr']]
            metric_df.dropna(subset=[metric], inplace=True)
            model = Lmer(f"label  ~ {metric}  + (1|pNr)",
                         data=metric_df, family='binomial')
            model.fit(control="optimizer='Nelder_Mead'")

            # singular fit is allowed (pNr may not always have enough variance to have an effect, we want to include it anyway)
            allowed_warning = 'boundary (singular) fit: see ?isSingular'
            # do not allow any other warnings
            if len(model.warnings) > 0:
                # assert all([allowed_warning == warning for warning in model.warnings])
                if not all([allowed_warning == warning for warning in model.warnings]):
                    model_warnings_df = pd.concat(
                        [model_warnings_df, pd.DataFrame({'metric': [metric], 'warning': [model.warnings]})])
            pvals_per_metric[metric] = model.coefs['P-val'].to_dict()[metric]


    pvals_per_metric_df = pd.DataFrame.from_dict(pvals_per_metric, orient='index', columns=['pval'])
    pvals_per_metric_df = pvals_per_metric_df.merge(model_warnings_df.set_index('metric'), left_index=True, right_index=True, how='left')

    ## Correct for multiple comparisons
    # correct for with Reiner et al 2003 (independence of measures not needed)
    sign_flags, adj_pvals, alpha_sidak, alphacBonf = statsmodels.stats.multitest.multipletests(
        pvals_per_metric_df['pval'].values, alpha=0.05, method='fdr_by')
    pvals_per_metric_df['adjusted_pval'] = adj_pvals
    pvals_per_metric_df['reiner_significance'] = sign_flags

    # correct using Storey 2003 (qvalue)
    significance_flags, qvals = qvalue(pvals_per_metric_df['pval'].values)
    pvals_per_metric_df['qval'] = qvals
    pvals_per_metric_df['storey_significance'] = significance_flags

    return pvals_per_metric_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-r', '--use_R', action='store_true')
    parser.add_argument('-N', '--normalisation', action='store_true')

    args = parser.parse_args()

    timebin_metrics_df = pd.read_csv(args.input_file)

    bp_metrics = ['systole', 'diastole', 'mitteldruck']
    if args.normalisation:
        bp_metrics = [f'{bp_metric}_normalised' for bp_metric in bp_metrics]
    metrics_over_time = [col.split('_')[-1] for col in
                         timebin_metrics_df.columns[timebin_metrics_df.columns.str.startswith(bp_metrics[0])]]

    timebin_pvals_per_metric_df = timebin_analysis(timebin_metrics_df, bp_metrics=bp_metrics, metrics_over_time=metrics_over_time,
                                                   use_R=args.use_R)

    timebin_pvals_per_metric_df.to_csv(args.input_file.replace('.csv', '_pvals.csv'), index=False)