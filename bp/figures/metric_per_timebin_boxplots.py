import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.plotting_utils import plot_metric_distributions_over_timebins


def plot_metric_per_timebin_boxplots(input_folder, metrics_over_time=['median', 'min', 'max'],
                                    bp_metrics=['systole', 'diastole', 'mitteldruck'], noradrenaline_handling='filter',
                                     normalisation=False):
    assert int(pd.__version__[0]) < 2, 'Please < 2 required for statannotations'

    timebin_metrics = [f'{bp_metric}_{metric_over_time}' for bp_metric in bp_metrics for metric_over_time in metrics_over_time]

    # Load timebin data for every timebin size
    bp_df = pd.DataFrame()
    for timebin_folder in os.listdir(input_folder):
        if not timebin_folder.startswith('bp_timebin_'):
            continue
        timebin_folder_path = os.path.join(input_folder, timebin_folder)
        timebin_size = int(timebin_folder.split('_')[-1][:-1])

        if noradrenaline_handling == 'filter':
            timebin_metrics_path = os.path.join(timebin_folder_path, f'bp_timebins_{timebin_size}h_nor_filtered_metrics.csv')
        else:
            raise NotImplementedError(f'Noradrenaline handling {noradrenaline_handling} not implemented')

        if normalisation:
            raise NotImplementedError('Normalisation not implemented')

        timebin_metrics_df = pd.read_csv(timebin_metrics_path)
        timebin_metrics_df['timebin_size'] = int(timebin_size)

        bp_df = pd.concat([bp_df, timebin_metrics_df], axis=0)

    # load pval data
    pval_data_path = os.path.join(input_folder, f'overall_pvals_nor_{noradrenaline_handling}.csv')
    pval_df = pd.read_csv(pval_data_path)

    fig, axes = plot_metric_distributions_over_timebins(
        bp_df, metrics_over_time, timebin_metrics, plot_type='box',
        pvals=pval_df, pval_method='adjusted_pval', alpha=0.5,
        plot_legend=True, tick_label_size=11,
        label_font_size=13, fig=None)

    fig.suptitle(f'DCI ischemia: BP metrics over timebins (not normalized/noradrenaline filtered)', fontsize=16, y=0.9)
    fig.savefig(os.path.join(input_folder, f'DCI_ischemia_bp_metrics_over_timebins_nor_filter.png'), format='png', dpi=300)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    args = parser.parse_args()
    plot_metric_per_timebin_boxplots(args.input_folder)
