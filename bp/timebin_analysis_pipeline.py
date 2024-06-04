import os
import pandas as pd

from bp.bp_timebin_metrics import bp_timebin_metrics
from bp.bp_timebin_stats import timebin_analysis

assert int(pd.__version__[0]) >= 2, 'Ensure CereBlink env is used (and not annotations)'



def multi_timebin_analysis(input_folder:str, verbose=False, noradrenaline_handling='filter', normalisation=False,
                            bp_metrics=['systole', 'diastole', 'mitteldruck'],
                           use_R=True):
    overall_stats = pd.DataFrame()

    if normalisation:
        bp_metrics = [f'{bp_metric}_normalised' for bp_metric in bp_metrics]

    for timebin_folder in os.listdir(input_folder):
        if not timebin_folder.startswith('bp_timebin_'):
            continue
        timebin_folder_path = os.path.join(input_folder, timebin_folder)
        timebin_size = int(timebin_folder.split('_')[-1][:-1])

        if verbose:
            print(f'Analyzing timebin {timebin_size}h')

        if noradrenaline_handling == 'filter':
            timebin_bp_path = os.path.join(timebin_folder_path, f'bp_timebins_{timebin_size}h_nor_filtered.csv')
        else:
            raise NotImplementedError(f'Noradrenaline handling {noradrenaline_handling} not implemented')

        if normalisation:
            timebin_bp_path = timebin_bp_path.replace('.csv', '_normalised.csv')

        try:
            timebin_bp_df = pd.read_csv(timebin_bp_path)
        except:
            print(f'Error reading {timebin_bp_path}')
            continue

        timebin_metrics_df, timebin_creation_log_df = bp_timebin_metrics(timebin_bp_df, normalisation=normalisation, verbose=verbose)
        metrics_over_time = [col.split('_')[-1] for col in timebin_metrics_df.columns[timebin_metrics_df.columns.str.startswith(bp_metrics[0])]]
        timebin_metrics_df['timebin_size'] = int(timebin_size)

        timebin_metrics_path = timebin_bp_path.replace('.csv', '_metrics.csv')
        if normalisation:
            timebin_metrics_path = timebin_metrics_path.replace('.csv', '_normalised.csv')

        timebin_metrics_df.to_csv(timebin_metrics_path, index=False)
        timebin_creation_log_df.to_csv(os.path.join(timebin_folder_path, 'logs',
                                                    f'timebin_creation_log_{timebin_size}h_nor_{noradrenaline_handling}{"_normalised" if normalisation else ""}.csv'),
                                       index=False)

        timebin_pvals_per_metric_df = timebin_analysis(timebin_metrics_df,
                                                       metrics_over_time=metrics_over_time, bp_metrics=bp_metrics,
                                                        use_R=use_R)
        timebin_pvals_per_metric_df['timebin_size'] = int(timebin_size)
        timebin_pvals_per_metric_df['noradrenaline_handling'] = noradrenaline_handling
        timebin_pvals_per_metric_df['normalisation'] = normalisation
        timebin_pvals_per_metric_df.reset_index(inplace=True)
        timebin_pvals_per_metric_df.rename(columns={'index': 'metric'}, inplace=True)
        timebin_pvals_path = os.path.join(timebin_folder_path, f'timebin_pvals_{timebin_size}h_nor_{noradrenaline_handling}.csv')
        if normalisation:
            timebin_pvals_path = timebin_pvals_path.replace('.csv', '_normalised.csv')
        timebin_pvals_per_metric_df.to_csv(timebin_pvals_path, index=False)

        overall_stats = pd.concat([overall_stats, timebin_pvals_per_metric_df])

    overall_stats_path = os.path.join(input_folder, f'overall_pvals_nor_{noradrenaline_handling}.csv')
    if normalisation:
        overall_stats_path = overall_stats_path.replace('.csv', '_normalised.csv')

    overall_stats.to_csv(overall_stats_path, index=False)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-nor', '--noradrenaline_handling', type=str, default='filter')
    parser.add_argument('-N', '--normalisation', action='store_true')
    parser.add_argument('-r', '--use_R', action='store_true')


    args = parser.parse_args()

    multi_timebin_analysis(args.input_folder, verbose=args.verbose, noradrenaline_handling=args.noradrenaline_handling,
                           normalisation=args.normalisation,
                           use_R=args.use_R)