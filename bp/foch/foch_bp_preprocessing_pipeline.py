import os
import pandas as pd

from bp.foch.foch_bp_timeseries_decomposition import foch_bp_timeseries_decomposition
from bp.normalisation import parallel_normalise
from utils.utils import ensure_dir


def foch_bp_preprocessing_pipeline(registry_data_path, bp_non_invasive_data_path, bp_invasive_data_path,
                                   output_dir,
                                   timebin_hours_list, apply_strict_DCI_definition=False,
                                   censor_data_after_first_event=True,
                                   censor_before=None, verbose=False):
    """
    Preprocesses the blood pressure data for the registry data
    # 1- timeseries decomposition
    # 4- normalise

    #  TODO: add the following steps
      # 2- annotate concomitant noradrenaline
      # 3- filter out concomitant noradrenaline (or correct for noradrenaline)

    Parameters
    :param registry_data_path: str, path to the registry data
    :param bp_non_invasive_data_path: str, path to the non-invasive blood pressure data
    :param bp_invasive_data_path: str, path to the invasive blood pressure data
    :param output_dir: str, path to the output directory
    :param timebin_hours_list: list, list of timebins in hours
    :param apply_strict_DCI_definition: bool, whether to apply the strict definition of DCI
    :param censor_data_after_first_event: bool, whether to censor data after the first event
    :param censor_before: str, date to censor data before
    :param verbose: bool, whether to print progress

    """

    for timebin_hours in timebin_hours_list:
       if verbose:
           print(f'Preprocessing blood pressure data for timebin {timebin_hours}h')

       # 1. timeseries decomposition
       timebin_bp_df, log_df, missing_outcomes_df, missing_bp_data_df = foch_bp_timeseries_decomposition(
                registry_data_path=registry_data_path,
                bp_non_invasive_data_path=bp_non_invasive_data_path,
                bp_invasive_data_path=bp_invasive_data_path,
                timebin_hours=timebin_hours,
                apply_strict_DCI_definition=apply_strict_DCI_definition,
                censor_data_after_first_event=censor_data_after_first_event,
                censor_before=censor_before,
                verbose=verbose
            )

       local_args = {'registry_data_path': registry_data_path,
                     'bp_non_invasive_data_path': bp_non_invasive_data_path,
                     'bp_invasive_data_path': bp_invasive_data_path,
                     'timebin_hours': timebin_hours,
                     'apply_strict_DCI_definition': apply_strict_DCI_definition,
                     'censor_data_after_first_event': censor_data_after_first_event,
                     'censor_before': censor_before
                     }

       timebin_folder_name = f'bp_timebin_{timebin_hours}h'
       ensure_dir(os.path.join(output_dir, timebin_folder_name))
       ensure_dir(os.path.join(output_dir, timebin_folder_name, 'logs'))

       log_df.to_csv(
           os.path.join(output_dir, timebin_folder_name, 'logs', f'timeseries_decomposition_log_{timebin_hours}h.csv'),
           index=False)
       missing_outcomes_df.to_csv(
           os.path.join(output_dir, timebin_folder_name, 'logs', f'missing_outcomes_{timebin_hours}h.csv'), index=False)
       missing_bp_data_df.to_csv(
           os.path.join(output_dir, timebin_folder_name, 'logs', f'missing_bp_data_{timebin_hours}h.csv'), index=False)
       with open(os.path.join(output_dir, timebin_folder_name, 'logs', f'args_{timebin_hours}h.txt'), 'w') as f:
           f.write(str(local_args))

       # Reorganise into common data structure
       systole_df = timebin_bp_df[timebin_bp_df['variable'] == 'systole']
       diastole_df = timebin_bp_df[timebin_bp_df['variable'] == 'diastole']
       mitteldruck_df = timebin_bp_df[timebin_bp_df['variable'] == 'mitteldruck']

       systole_df = systole_df.rename(columns={'Value': 'systole'})
       diastole_df = diastole_df.rename(columns={'Value': 'diastole'})
       mitteldruck_df = mitteldruck_df.rename(columns={'Value': 'mitteldruck'})

       timebin_bp_df = timebin_bp_df.merge(systole_df[['PatientID', 'datetime', 'systole']], on=['PatientID', 'datetime'], how='left')
       timebin_bp_df = timebin_bp_df.merge(diastole_df[['PatientID', 'datetime', 'diastole']], on=['PatientID', 'datetime'], how='left')
       timebin_bp_df = timebin_bp_df.merge(mitteldruck_df[['PatientID', 'datetime', 'mitteldruck']], on=['PatientID', 'datetime'],
                           how='left')

       timebin_bp_df = timebin_bp_df.drop(columns=['variable', 'Value', 'VariableID', 'status'])
       timebin_bp_df = timebin_bp_df.drop_duplicates()

       timebin_bp_df.to_csv(os.path.join(output_dir, timebin_folder_name, f'bp_timebins_{timebin_hours}h.csv'),
                            index=False)

       # TODO:
       # 2. annotate concomitant noradrenaline
       # 3. filter out concomitant noradrenaline (or correct for noradrenaline)
       normalised_timebin_bp_df = parallel_normalise(timebin_bp_df,
                                                        bp_metrics=['systole', 'diastole', 'mitteldruck'],
                                                        pid_column='PatientID', datetime_column='datetime')
       normalised_timebin_bp_df.to_csv(
           os.path.join(output_dir, timebin_folder_name, f'bp_timebins_{timebin_hours}h_normalised.csv'),
           index=False)

    if verbose:
        print('Preprocessing done.')

    return normalised_timebin_bp_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--registry_data_path', type=str, required=True)
    parser.add_argument('-ibp', '--invasive_bp_data_path', type=str, required=True)
    parser.add_argument('-nibp', '--non_invasive_bp_data_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-t', '--timebin_hours', type=int, required=True, nargs='+', help='List of timebin hours')
    parser.add_argument('-cen', '--censor_before', type=str, required=False, default=None)
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    parser.add_argument('-s', '--apply_strict_DCI_definition', default=False, action="store_true")
    args = parser.parse_args()

    foch_bp_preprocessing_pipeline(
        registry_data_path=args.registry_data_path,
        bp_non_invasive_data_path=args.non_invasive_bp_data_path,
        bp_invasive_data_path=args.invasive_bp_data_path,
        output_dir=args.output_dir,
        timebin_hours_list=args.timebin_hours,
        censor_before=args.censor_before,
        apply_strict_DCI_definition=args.apply_strict_DCI_definition,
        verbose=args.verbose
    )