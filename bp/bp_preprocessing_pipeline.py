import os
import getpass

import pandas as pd
from bp.bp_timeseries_decomposition import bp_timeseries_decomposition
from bp.noradrenaline_preprocessing import annotate_concomitant_noradrenaline, filter_out_concomitant_noradrenaline
from utils.utils import ensure_dir


def bp_preprocessing_pipeline(registry_data_path, bp_data_path, registry_pdms_correspondence_path, nor_df_path,
                              output_dir,
                              timebin_hours_list, target='DCI_ischemia', censor_data_after_first_event=True,
                              censor_before=None, verbose=False):
    """
    Preprocesses the blood pressure data for the registry data
    # 1- timeseries decomposition
    # 2- annotate concomitant noradrenaline
    # 3- filter out concomitant noradrenaline (ToDo: or correct for noradrenaline)
    # (ToDo: 4- normalise)

    :param registry_data_path:
    :param bp_data_path:
    :param registry_pdms_correspondence_path:
    :param nor_df_path:
    :param output_dir:
    :param timebin_hours_list:
    :param target:
    :param censor_data_after_first_event:
    :param censor_before:
    :param verbose:
    :return:
    """
    nor_df = pd.read_csv(nor_df_path, sep=';', decimal='.')
    password = getpass.getpass()

    for timebin_hours in timebin_hours_list:
        if verbose:
            print(f'Preprocessing blood pressure data for timebin {timebin_hours}h')
        # 1. timeseries decomposition
        timebin_bp_df, log_df, missing_outcomes_df, missing_bp_data_df = bp_timeseries_decomposition(
            registry_data_path,
            bp_data_path,
            registry_pdms_correspondence_path,
            timebin_hours=timebin_hours,
            target=target,
            censor_data_after_first_event=censor_data_after_first_event,
            password=password,
            censor_before=censor_before,
            verbose=verbose
        )

        local_args = {'registry_data_path': registry_data_path,
                'bp_data_path': bp_data_path,
                'registry_correspondence_path': registry_pdms_correspondence_path,
                'timebin_hours': timebin_hours,
                'target': target,
                'censor_data_after_first_event': censor_data_after_first_event,
                'censor_before': censor_before,
                }

        timebin_folder_name = f'bp_timebin_{timebin_hours}h'
        ensure_dir(os.path.join(output_dir, timebin_folder_name))
        ensure_dir(os.path.join(output_dir, timebin_folder_name, 'logs'))
        timebin_bp_df.to_csv(os.path.join(output_dir, timebin_folder_name, f'bp_timebins_{timebin_hours}h.csv'), index=False)
        log_df.to_csv(os.path.join(output_dir, timebin_folder_name, 'logs', f'timeseries_decomposition_log_{timebin_hours}h.csv'), index=False)
        missing_outcomes_df.to_csv(
            os.path.join(output_dir, timebin_folder_name, 'logs', f'missing_outcomes_{timebin_hours}h.csv'), index=False)
        missing_bp_data_df.to_csv(
            os.path.join(output_dir, timebin_folder_name, 'logs', f'missing_bp_data_{timebin_hours}h.csv'), index=False)
        with open(os.path.join(output_dir, timebin_folder_name, 'logs', f'args_{timebin_hours}h.txt'), 'w') as f:
            f.write(str(local_args))

        # 2. annotate concomitant noradrenaline
        annotated_timebin_bp_df = annotate_concomitant_noradrenaline(timebin_bp_df, nor_df)
        annotated_timebin_bp_df.to_csv(os.path.join(output_dir, timebin_folder_name, f'bp_timebins_{timebin_hours}h_nor_annotated.csv'), index=False)

        # 3. filter out concomitant noradrenaline
        filtered_annotated_timebin_bp_df, filtering_logs_df = filter_out_concomitant_noradrenaline(annotated_timebin_bp_df, verbose=verbose)
        filtered_annotated_timebin_bp_df['noradrenaline_handling'] = 'filter'
        filtered_annotated_timebin_bp_df.to_csv(os.path.join(output_dir, timebin_folder_name, f'bp_timebins_{timebin_hours}h_nor_filtered.csv'), index=False)
        filtering_logs_df.to_csv(os.path.join(output_dir, timebin_folder_name, 'logs', f'filtering_logs_{timebin_hours}h.csv'), index=False)

        # 4. normalise
        # ToDo

    if verbose:
        print('Preprocessing done.')

    return filtered_annotated_timebin_bp_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--registry_data_path', type=str, required=True)
    parser.add_argument('-bp', '--bp_data_path', type=str, required=True)
    parser.add_argument('-c', '--registry_pdms_correspondence_path', type=str, required=True)
    parser.add_argument('-nor', '--nor_df_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-t', '--timebin_hours', type=int, required=True, nargs='+', help='List of timebin hours')
    parser.add_argument('-cen', '--censor_before', type=str, required=True)
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    args = parser.parse_args()

    bp_preprocessing_pipeline(
        registry_data_path=args.registry_data_path,
        bp_data_path=args.bp_data_path,
        registry_pdms_correspondence_path=args.registry_pdms_correspondence_path,
        nor_df_path=args.nor_df_path,
        output_dir=args.output_dir,
        timebin_hours_list=args.timebin_hours,
        censor_before=args.censor_before,
        verbose=args.verbose
    )


