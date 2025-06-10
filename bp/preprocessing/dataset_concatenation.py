import pandas as pd
import os

from utils.utils import ensure_dir


def join_multicenter_df(kssg_df_path, foch_df_path):
    kssg_df = pd.read_csv(kssg_df_path)
    foch_df = pd.read_csv(foch_df_path)

    # Add a column to distinguish the datasets
    kssg_df['dataset'] = 'kssg'
    foch_df['dataset'] = 'foch'

    foch_df.rename(columns={'PatientID': 'pNr'}, inplace=True)

    # Concatenate the datasets
    joined_df = pd.concat([kssg_df, foch_df], axis=0)

    joined_df['pNr'] = joined_df['dataset'] + '_' + joined_df['pNr'].astype(str)

    # return the concatenated dataset
    return joined_df

def join_multicenter_dataset(kssg_dir, foch_dir, output_dir,
                             normalisation=True, verbose=False):
    # TODO add Nor handling option

    for timebin_folder in os.listdir(kssg_dir):
        if not timebin_folder.startswith('bp_timebin_'):
            continue
        timebin_size = int(timebin_folder.split('_')[-1][:-1])

        if verbose:
            print(f'Joining timebin {timebin_size}h')

        kssg_folder_path = os.path.join(kssg_dir, timebin_folder)
        foch_folder_path = os.path.join(foch_dir, timebin_folder)

        target_file_ending = 'metrics.csv'
        if normalisation:
            target_file_ending = 'metrics_normalised.csv'
        # search for the file ending with metrics_normalised.csv
        kssg_metrics_file = next((os.path.join(kssg_folder_path, f) for f in os.listdir(kssg_folder_path) if
                             f.endswith(target_file_ending)), None)
        foch_metrics_file = next((os.path.join(foch_folder_path, f) for f in os.listdir(foch_folder_path) if
                                f.endswith(target_file_ending)), None)

        if kssg_metrics_file is None or foch_metrics_file is None:
            print(f'No metrics file found for timebin {timebin_size}h')
            continue

        joined_df = join_multicenter_df(kssg_metrics_file, foch_metrics_file)

        output_timebin_folder = os.path.join(output_dir, timebin_folder)
        ensure_dir(output_timebin_folder)
        output_file = os.path.join(output_timebin_folder, f'bp_timebins_{timebin_size}h_metrics.csv')
        if normalisation:
            output_file = output_file.replace('.csv', '_normalised.csv')

        joined_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kssg_dir', type=str, required=True)
    parser.add_argument('-f', '--foch_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--normalisation', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    join_multicenter_dataset(args.kssg_dir, args.foch_dir, args.output_dir,
                             normalisation=args.normalisation, verbose=args.verbose)
