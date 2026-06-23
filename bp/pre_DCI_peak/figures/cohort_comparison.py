import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator


def _normalise_noradrenaline_handling(noradrenaline_handling):
    if isinstance(noradrenaline_handling, (str, type(None))):
        return [noradrenaline_handling, noradrenaline_handling]

    if len(noradrenaline_handling) != 2:
        raise ValueError('noradrenaline_handling must be a string, None, or a sequence with two entries')

    return list(noradrenaline_handling)


def _load_cohort_timebin_df(input_folder, metric_over_time, bp_metrics, noradrenaline_handling, normalisation):
    bp_df = pd.DataFrame()

    timebin_folders = sorted(
        (folder for folder in os.listdir(input_folder) if folder.startswith('bp_timebin_')),
        key=lambda folder: int(folder.split('_')[-1][:-1]),
    )

    for timebin_folder in timebin_folders:
        timebin_folder_path = os.path.join(input_folder, timebin_folder)
        timebin_size = int(timebin_folder.split('_')[-1][:-1])

        target_file_start = f'bp_timebins_{timebin_size}h'
        if noradrenaline_handling == 'filter':
            target_file_start = f'bp_timebins_{timebin_size}h_nor_filtered'
        elif noradrenaline_handling in [None, 'none']:
            target_file_start = f'bp_timebins_{timebin_size}h'
        else:
            raise NotImplementedError(f'Noradrenaline handling {noradrenaline_handling} not implemented')

        target_file_ending = 'metrics.csv'
        if normalisation:
            target_file_ending = target_file_ending.replace('.csv', '_normalised.csv')

        timebin_metrics_path = next(
            (
                os.path.join(timebin_folder_path, file_name)
                for file_name in os.listdir(timebin_folder_path)
                if file_name.endswith(target_file_ending) and file_name.startswith(target_file_start)
            ),
            None,
        )

        if timebin_metrics_path is None:
            raise FileNotFoundError(
                f'Could not find a metrics file for {timebin_folder_path} '
                f'with prefix {target_file_start} and suffix {target_file_ending}'
            )

        timebin_metrics_df = pd.read_csv(timebin_metrics_path)
        timebin_metrics_df['timebin_size'] = timebin_size
        bp_df = pd.concat([bp_df, timebin_metrics_df], axis=0, ignore_index=True)

    expected_columns = ['label', 'timebin_size'] + [
        f'{bp_metric}{"_normalised" if normalisation else ""}_{metric_over_time}'
        for bp_metric in bp_metrics
    ]
    missing_columns = [column_name for column_name in expected_columns if column_name not in bp_df.columns]
    if missing_columns:
        raise KeyError(f'Missing expected columns in loaded cohort data: {missing_columns}')

    return bp_df


def _load_cohort_pval_df(input_folder, noradrenaline_handling, normalisation):
    pval_data_path = os.path.join(
        input_folder,
        f'overall_pvals_nor_{noradrenaline_handling}{"_normalised" if normalisation else ""}.csv',
    )
    if not os.path.exists(pval_data_path):
        raise FileNotFoundError(f'Could not find p-value file: {pval_data_path}')
    return pd.read_csv(pval_data_path)


def boxplots_cohort_comparison(input_folder_kssg, input_folder_foch, metric_over_time='median',
                               bp_metrics=['systole', 'diastole', 'mitteldruck'], noradrenaline_handling=['filter', 'none'],
                               normalisation=False, use_qvalues=True,
                               no_annotation=False, output_path=None):
    # Plot boxplots for one metric over time with KSSG and FOCH in separate columns.
    assert int(pd.__version__[0]) < 2, 'Please < 2 required for statannotations'

    noradrenaline_handling = _normalise_noradrenaline_handling(noradrenaline_handling)
    cohort_specs = [
        ('Derivation cohort', input_folder_kssg, noradrenaline_handling[0]),
        ('External validation cohort', input_folder_foch, noradrenaline_handling[1]),
    ]

    cohort_dfs = {
        cohort_name: _load_cohort_timebin_df(
            cohort_folder,
            metric_over_time=metric_over_time,
            bp_metrics=bp_metrics,
            noradrenaline_handling=cohort_noradrenaline,
            normalisation=normalisation,
        )
        for cohort_name, cohort_folder, cohort_noradrenaline in cohort_specs
    }

    pval_method = 'qval' if use_qvalues else 'adjusted_pval'
    cohort_pvals = {}
    if not no_annotation:
        cohort_pvals = {
            cohort_name: _load_cohort_pval_df(
                cohort_folder,
                noradrenaline_handling=cohort_noradrenaline,
                normalisation=normalisation,
            )
            for cohort_name, cohort_folder, cohort_noradrenaline in cohort_specs
        }

    n_rows = len(bp_metrics)
    n_columns = len(cohort_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(n_columns * 10, n_rows * 4),
        sharex=True,
        sharey='row',
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    custom_palette = {0: '#FFA987', 1: '#049b9a'}
    y_metric_names = [
        f'{bp_metric}{"_normalised" if normalisation else ""}_{metric_over_time}'
        for bp_metric in bp_metrics
    ]

    for row_idx, (bp_metric, y_metric) in enumerate(zip(bp_metrics, y_metric_names)):
        for col_idx, (cohort_name, _, _) in enumerate(cohort_specs):
            ax = axes[row_idx, col_idx]
            cohort_df = cohort_dfs[cohort_name]

            plot_params = {
                'data': cohort_df,
                'x': 'timebin_size',
                'y': y_metric,
                'hue': 'label',
                'palette': custom_palette,
                'showfliers': False,
            }
            sns.boxplot(**plot_params, ax=ax)

            if bp_metric == 'mitteldruck':
                ax.set_ylabel('Mean arterial pressure', fontsize=12)
            if bp_metric == 'systole':
                ax.set_ylabel('Systolic blood pressure', fontsize=12)
            if bp_metric == 'diastole':
                ax.set_ylabel('Diastolic blood pressure', fontsize=12)

            ax.set_xlabel('Timebin size (hours)' if row_idx == n_rows - 1 else '', fontsize=12)
            ax.tick_params('x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11, labelleft=True)
            ax.set_title(cohort_name if row_idx == 0 else '', fontsize=14)
            sns.despine(ax=ax, top=True, right=True)

            style = {
                0: {
                    'alpha': 0.5,
                    'linewidth': 1.2,
                    'hatch': None,
                },
                1: {
                    'alpha': 0.25,
                    'linewidth': 1.8,
                    'hatch': '//',
                },
            }[col_idx]

            for patch in ax.patches:
                r, g, b, _ = patch.get_facecolor()
                patch.set_facecolor((r, g, b, style['alpha']))
                patch.set_edgecolor((r, g, b, 1.0))
                patch.set_linewidth(style['linewidth'])

                if style['hatch'] is not None:
                    patch.set_hatch(style['hatch'])


            if not no_annotation:
                pvals_metric = cohort_pvals[cohort_name][cohort_pvals[cohort_name]['metric'] == y_metric]
                pvals_metric = pvals_metric.sort_values(by='timebin_size')
                pvals_metric = pvals_metric.dropna(subset=[pval_method])

                if not pvals_metric.empty:
                    timebin_values = pvals_metric['timebin_size'].unique()
                    pairs = tuple([[(tbx, 0), (tbx, 1)] for tbx in timebin_values])
                    if pairs:
                        annotator = Annotator(ax, pairs, **plot_params, verbose=False)
                        annotator.set_pvalues(pvals_metric[pval_method].values)
                        annotator.annotate()

            legend = ax.get_legend()
            if row_idx == 0:
                handles, _ = ax.get_legend_handles_labels()
                for handle in handles:
                    handle.set_alpha(0.5)
                ax.legend(
                    handles,
                    ['No DCI', 'DCI'],
                    title='',
                    loc='upper right',
                    facecolor='white',
                    framealpha=0.9,
                    fontsize=11,
                    title_fontsize=13,
                )
            elif legend is not None:
                legend.remove()

    # decrease width between subplots
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, axes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--input_folder_kssg', type=str, required=True)
    parser.add_argument('-f', '--input_folder_foch', type=str, required=True)
    parser.add_argument('-m', '--metric_over_time', type=str, default='median')
    parser.add_argument('-N', '--normalisation', action='store_true', help='Whether to plot normalised data or not')
    parser.add_argument('-q', '--use_qvalues', action='store_true', help='Whether to use qvalues instead of adjusted pvalues')
    parser.add_argument('-no_annotation', '--no_annotation', action='store_true')
    parser.add_argument('-nor_kssg', '--noradrenaline_handling_kssg', type=str, default='filter')
    parser.add_argument('-nor_foch', '--noradrenaline_handling_foch', type=str, default='none')
    parser.add_argument('-o', '--output_path', type=str, default=None)
    args = parser.parse_args()

    noradrenaline_handling = [args.noradrenaline_handling_kssg, args.noradrenaline_handling_foch]
    if noradrenaline_handling[0] in ['None', 'none', '0']:
        noradrenaline_handling[0] = None
    if noradrenaline_handling[1] in ['None', 'none', '0']:
        noradrenaline_handling[1] = None

    boxplots_cohort_comparison(
        args.input_folder_kssg,
        args.input_folder_foch,
        metric_over_time=args.metric_over_time,
        noradrenaline_handling=noradrenaline_handling,
        normalisation=args.normalisation,
        use_qvalues=args.use_qvalues,
        no_annotation=args.no_annotation,
        output_path=args.output_path,
    )
