from smtplib import bCRLF

import pandas as pd
import numpy as np
import math, os
from tqdm import tqdm
from utils.utils import safe_conversion_to_datetime, ensure_dir


def foch_bp_timeseries_decomposition(registry_data_path, bp_non_invasive_data_path, bp_invasive_data_path,
                                     timebin_hours, apply_strict_DCI_definition=False,
                                     censor_data_after_first_event=True, censor_before=None,
                                     verbose=False):
    """
    Decompose the blood pressure time series into timebins

    Aceppted ranges for blood pressure values:
    - systole: 1-300
    - diastole: 1-200
    - mean pressure: 1-250

    VariableIds
    - 100/110/120 : PA invasive s/m/d
    - 600/610/620 : PNI s/m/d

    :param registry_data_path: str, path to the registry data
    :param bp_non_invasive_data_path: str, path to the non-invasive blood pressure data
    :param bp_invasive_data_path: str, path to the invasive blood pressure data
    :param timebin_hours: int, timebin in hours
    :param censor_data_after_first_event: bool, censor data after first event
    :param censor_before: str, censor data before date
    :param verbose: bool, verbose
    :return:
    """

    assert int(pd.__version__[0]) >= 2, 'Please update pandas to version 2 or higher'

    # Load the registry data
    registry_df = pd.read_excel(registry_data_path)
    if verbose:
        print(f'Excluding {registry_df["exclude_yn"].sum()} patients because not aSAH')
    registry_df = registry_df[registry_df['exclude_yn'] == 0]

    # Load the blood pressure data
    invasive_bp_df = pd.read_csv(bp_invasive_data_path)
    invasive_bp_df['method'] = 'invasive'
    non_invasive_bp_df = pd.read_csv(bp_non_invasive_data_path)
    non_invasive_bp_df['method'] = 'non-invasive'

    bp_df = pd.concat([invasive_bp_df, non_invasive_bp_df], axis=0)

    # add admission time to bp data from registry data
    bp_df = bp_df.merge(registry_df[['PatientID', 'AdmissionTime']], on='PatientID', how='left')

    n_patients_before_filtering_year = registry_df['PatientID'].nunique()
    if censor_before is not None:
        # check if censor_before can be converted to datetime
        try:
            censor_before = pd.to_datetime(censor_before)
        except:
            raise ValueError('censor_before should be in format YYYY')
        # censor data before censor_before
        registry_df = registry_df[registry_df['AdmissionTime'] >= pd.to_datetime(censor_before, format='%Y')]
    n_patients_after_filtering_year = registry_df['PatientID'].nunique()
    if verbose:
        print(f'Number of patients before filtering: {n_patients_before_filtering_year}')
        print(f'Number of patients after filtering: {n_patients_after_filtering_year}')

    # check that all patients in the registry data are in the bp data (but not necessarily the other way around)
    assert set(registry_df['PatientID'].unique()).issubset(set(bp_df['PatientID'].unique())), 'Not all patients in the registry data are in the bp data'

    # only keep patients in registry data
    bp_df = bp_df[bp_df['PatientID'].isin(registry_df['PatientID'].unique())]

    if verbose:
        print(f'Retained {bp_df["PatientID"].nunique()} patients in the blood pressure data')

    # Preprocess the blood pressure data
    # add variable column to the bp data
    bp_df['variable'] = bp_df['VariableID'].map({100: 'systole', 110: 'diastole', 120: 'mitteldruck',
                                                 600: 'systole', 610: 'diastole', 620: 'mitteldruck'})

    # check that all values are numeric
    assert bp_df['Value'].apply(lambda x: isinstance(x, (int, float))).all(), 'Not all values are numeric'

    # Restrict to accepted ranges
    n_sys_out_of_range = bp_df[(bp_df['variable'] == 'systole') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 300))].shape[0]
    n_dia_out_of_range = bp_df[(bp_df['variable'] == 'diastole') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 200))].shape[0]
    n_mitt_out_of_range = bp_df[(bp_df['variable'] == 'mitteldruck') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 250))].shape[0]

    # remove rows in which systole == diastole == mitteldruck
    # ie remove rows with equal timedate, patientID and value but different variable
    n_patients_with_equal_values = bp_df[bp_df.duplicated(subset=['datetime', 'PatientID', 'Value'], keep=False)].PatientID.nunique()
    bp_df = bp_df[~bp_df.duplicated(subset=['datetime', 'PatientID', 'Value'], keep=False)]

    if verbose:
        print(f'Excluding {n_sys_out_of_range} systole values out of range')
        print(f'Excluding {n_dia_out_of_range} diastole values out of range')
        print(f'Excluding {n_mitt_out_of_range} mean pressure values out of range')

    bp_df = bp_df[(
                ~((bp_df['variable'] == 'systole') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 300)))
                & ~((bp_df['variable'] == 'diastole') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 200)))
                & ~((bp_df['variable'] == 'mitteldruck') & ((bp_df['Value'] < 1) | (bp_df['Value'] > 250)))
    )]

    # convert to datetime
    bp_df['datetime'] = pd.to_datetime(bp_df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')


    # Preprocess the outcome data
    patients_with_bp_but_no_outcome = bp_df[~bp_df['PatientID'].isin(registry_df['PatientID'])].PatientID.drop_duplicates()
    patients_with_outcome_but_no_bp = registry_df[
        ~((registry_df['PatientID'].isin(bp_df['PatientID'])) & (~registry_df['PatientID'].isnull()))]
    n_patients_with_bp_but_no_outcome = patients_with_bp_but_no_outcome.shape[0]
    n_patients_with_outcome_but_no_bp = patients_with_outcome_but_no_bp.shape[0]
    if verbose:
        print(f'Number of patients with BP but no outcome: {n_patients_with_bp_but_no_outcome}')
        print(f'Number of patients with outcome but no BP: {n_patients_with_outcome_but_no_bp}')


    # convert to datetime
    registry_df['DCI_exact_date'] = registry_df.DCI_exact_date.str.replace(':', 'h')
    registry_df['DCI_exact_date'] = pd.to_datetime(registry_df['DCI_exact_date'], format='%d/%m/%Y à %Hh%M')

    # Construct target events df
    assert registry_df['DCI_YN'].dropna().apply(lambda x: x in [0, 1]).all()
    registry_df['DCI_YN'] = registry_df['DCI_YN'].astype(int, errors='ignore')
    target_events_df = registry_df[registry_df['DCI_YN'] == 1]

    if apply_strict_DCI_definition:
        # apply strict DCI definition
        if verbose:
            print(f'Excluding {target_events_df["exclude_per_strict_def"].sum()} patients per strict DCI definition')
        target_events_df = target_events_df[target_events_df['exclude_per_strict_def'] != 1]

    if verbose:
        print(f'Number of patients with DCI: {target_events_df["PatientID"].nunique()}')

    # Label positive timebins
    # loop through all events and label bp data with event
    for index, row in target_events_df.iterrows():
        # verify that patient is in bp data
        if not row['PatientID'] in bp_df['PatientID'].values:
            if verbose:
                print(f'Patient {row["Name"]} not in bp data')
            continue

        timebin_begin = pd.to_datetime(row['DCI_exact_date']) - pd.Timedelta(hours=timebin_hours, unit='h')
        timebin_end = pd.to_datetime(row['DCI_exact_date'])

        bp_df.loc[(bp_df['PatientID'] == row['PatientID'])
                  & (bp_df['datetime'] >= timebin_begin)
                  & (bp_df['datetime'] <= timebin_end),
        'within_event_timebin'] = 1
        bp_df.loc[(bp_df['PatientID'] == row['PatientID'])
                  & (bp_df['datetime'] >= timebin_begin)
                  & (bp_df['datetime'] <= timebin_end),
        'associated_event_time'] = row['DCI_exact_date']

        # if no bp data within timebin, print warning
        if bp_df.loc[(bp_df['PatientID'] == row['PatientID'])
                     & (bp_df['datetime'] >= timebin_begin)
                     & (bp_df['datetime'] <= timebin_end)].shape[0] == 0:
            if verbose:
                print(f'No BP data within timebin for patient {row["PatientID"]}')

        # Censor data after first event
        if censor_data_after_first_event:
            # drop rows for PatientID with timeBd > timebin_end
            bp_df = bp_df[~((bp_df['PatientID'] == row['PatientID']) & (bp_df['datetime'] > timebin_end))]


    bp_df['within_event_timebin'] = bp_df['within_event_timebin'].fillna(0).astype(int)

    n_patients_with_event_and_bp_data = target_events_df[target_events_df['PatientID'].isin(bp_df['PatientID'])].PatientID.nunique()
    n_patients_with_event_and_bp_in_timebin = bp_df[bp_df['within_event_timebin'] == 1].PatientID.nunique()
    if verbose:
        print(f'Number of patients with event and BP data: {n_patients_with_event_and_bp_data}')
        print(f'Number of patients with event and BP data in timebin: {n_patients_with_event_and_bp_in_timebin}')

    # Label negative timebins
    # for every patient in bp_df, add a column with the index of the timebin (starting at admission, timebin_hours apart, ending at last measurement or at start of positive timebin)
    bp_df['negative_timebin'] = np.nan

    n_negative_timebins = 0
    for patient in tqdm(bp_df['PatientID'].unique()):
        patient_df = bp_df[bp_df['PatientID'] == patient]
        patient_last_measurement = patient_df['datetime'].max()
        patient_first_measurement = patient_df['datetime'].min()

        patient_start_positive_timebin = patient_df[patient_df['within_event_timebin'] == 1][
                                             'associated_event_time'].min() - pd.Timedelta(hours=timebin_hours,
                                                                                           unit='h')
        # positive and negative timebins should not overlap
        patient_start_positive_timebin = patient_start_positive_timebin - pd.Timedelta(minutes=1)
        patient_end_negative_timebins = patient_start_positive_timebin if not pd.isnull(
            patient_start_positive_timebin) else patient_last_measurement
        n_patient_negative_timebins = math.ceil(
            (patient_end_negative_timebins - patient_first_measurement).total_seconds() / (timebin_hours * 3600))

        timebin_end = patient_end_negative_timebins
        for i in range(n_patient_negative_timebins):
            timebin_begin = timebin_end - pd.Timedelta(hours=timebin_hours, unit='h')
            bp_df.loc[(bp_df['PatientID'] == patient)
                      & (bp_df['datetime'] > timebin_begin)
                      & (bp_df['datetime'] <= timebin_end),
                        'negative_timebin'] = n_patient_negative_timebins - i
            timebin_end = timebin_begin

        n_negative_timebins += n_patient_negative_timebins

    n_positive_timebins = bp_df[bp_df['within_event_timebin'] == 1].PatientID.nunique()

    if verbose:
        print(f'Number of negative timebins: {n_negative_timebins}')
        print(f'Number of positive timebins: {n_positive_timebins}')

    n_patients = bp_df['PatientID'].nunique()

    log_df = pd.DataFrame({'n_sys_out_of_range': [n_sys_out_of_range],
                           'n_dia_out_of_range': [n_dia_out_of_range],
                           'n_mitt_out_of_range': [n_mitt_out_of_range],
                           'n_patients_with_equal_values': [n_patients_with_equal_values],
                           'n_patients_before_filtering_year': [n_patients_before_filtering_year],
                           'n_patients_after_filtering_year': [n_patients_after_filtering_year],
                           'n_patients_with_bp_but_no_outcome': [n_patients_with_bp_but_no_outcome],
                           'n_patients_with_outcome_but_no_bp': [n_patients_with_outcome_but_no_bp],
                           'n_patients_with_event_and_bp_data': [n_patients_with_event_and_bp_data],
                           'n_patients_with_event_and_bp_in_timebin': [n_patients_with_event_and_bp_in_timebin],
                           'n_patients': [n_patients],
                           'n_negative_timebins': [n_negative_timebins],
                           'n_positive_timebins': [n_positive_timebins],
                           'apply_strict_DCI_definition': [apply_strict_DCI_definition],
                           'timebin_hours': [timebin_hours],
                            'censor_data_after_first_event': [censor_data_after_first_event],
                            'censor_before': [censor_before]
                           })


    return bp_df, log_df, patients_with_bp_but_no_outcome, patients_with_outcome_but_no_bp



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--registry_data_path', type=str, required=True)
    parser.add_argument('-ibp', '--invasive_bp_data_path', type=str, required=True)
    parser.add_argument('-nibp', '--non_invasive_bp_data_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-t', '--timebin_hours', type=int, required=True)
    parser.add_argument('-cen', '--censor_before', type=str, required=False, default=None)
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    parser.add_argument('-s', '--apply_strict_DCI_definition', default=False, action="store_true")
    args = parser.parse_args()

    bp_df, log_df, missing_outcomes_df, missing_bp_data_df = foch_bp_timeseries_decomposition(
        registry_data_path=args.registry_data_path,
        bp_non_invasive_data_path=args.non_invasive_bp_data_path,
        bp_invasive_data_path=args.invasive_bp_data_path,
        timebin_hours=args.timebin_hours,
        apply_strict_DCI_definition=args.apply_strict_DCI_definition,
        censor_data_after_first_event=True,
        censor_before=args.censor_before,
        verbose=args.verbose
    )

    local_args = {'registry_data_path': args.registry_data_path,
                    'bp_non_invasive_data_path': args.non_invasive_bp_data_path,
                    'bp_invasive_data_path': args.invasive_bp_data_path,
                    'timebin_hours': args.timebin_hours,
                    'apply_strict_DCI_definition': args.apply_strict_DCI_definition,
                    'censor_data_after_first_event': True,
                    'censor_before': args.censor_before
                    }

    folder_name = f'bp_timebin_{args.timebin_hours}h'
    ensure_dir(os.path.join(args.output_dir, folder_name))
    ensure_dir(os.path.join(args.output_dir, folder_name, 'logs'))

    bp_df.to_csv(os.path.join(args.output_dir, folder_name, f'bp_timebins_{args.timebin_hours}h.csv'), index=False)
    log_df.to_csv(
        os.path.join(args.output_dir, folder_name, 'logs', f'timeseries_decomposition_log_{args.timebin_hours}h.csv'),
        index=False)
    missing_outcomes_df.to_csv(
        os.path.join(args.output_dir, folder_name, 'logs', f'missing_outcomes_{args.timebin_hours}h.csv'), index=False)
    missing_bp_data_df.to_csv(
        os.path.join(args.output_dir, folder_name, 'logs', f'missing_bp_data_{args.timebin_hours}h.csv'), index=False)
    with open(os.path.join(args.output_dir, folder_name, 'logs', f'args_{args.timebin_hours}h.txt'), 'w') as f:
        f.write(str(local_args))
