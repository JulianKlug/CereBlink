import pandas as pd
import numpy as np
import math, os
from tqdm import tqdm
from utils import load_encrypted_xlsx, safe_conversion_to_datetime, ensure_dir


# ToDo:
# - filter from a certain date
# - count patients at every step
# - filter out errors
# - only parse into timebins - no analysis?

def bp_timeseries_decomposition(registry_data_path, bp_data_path, registry_correspondence_path,
                                timebin_hours, target,
                                censor_data_after_first_event=True, password=None,
                                censor_before = None,
                                verbose=False):
    """
    Decompose the blood pressure time series into timebins

    Aceppted ranges for blood pressure values:
    - systole: 0-300
    - diastole: 0-200
    - mean pressure: 0-250

    :param registry_data_path:
    :param bp_data_path:
    :param registry_correspondence_path:
    :param timebin_hours:
    :param target: used definition of DCI ['DCI_ischemia', 'DCI_infarct']
    :param censor_data_after_first_event:
    :param password:
    :param censor_before: Censor data before this date, format YYYY
    :param verbose:
    :return:
    """
    assert int(pd.__version__[0]) >= 2, 'Please update pandas to version 2 or higher'

    # Load data
    registry_df = load_encrypted_xlsx(registry_data_path, password=password)
    bp_df = pd.read_csv(bp_data_path, sep=';', decimal='.')
    registry_pdms_correspondence_df = pd.read_csv(registry_correspondence_path)
    registry_pdms_correspondence_df['Date_birth'] = pd.to_datetime(registry_pdms_correspondence_df['Date_birth'],
                                                                   format='%Y-%m-%d')
    registry_pdms_correspondence_df.rename(columns={'JoinedName': 'Name'}, inplace=True)
    registry_df = registry_df.merge(registry_pdms_correspondence_df, on=['SOS-CENTER-YEAR-NO.', 'Name', 'Date_birth'],
                                    how='left')
    bp_df = bp_df.merge(registry_df[['pNr', 'Date_admission']], how='left', on='pNr')

    n_patients_before_filtering_year = registry_df[registry_df.pNr.isin(bp_df['pNr'])].shape[0]
    # Filter out patients before certain date
    if censor_before is not None:
        # check if censor_before can be converted to datetime
        try:
            censor_before = pd.to_datetime(censor_before)
        except:
            raise ValueError('censor_before should be in format YYYY')
        bp_df = bp_df[bp_df['Date_admission'] >= pd.to_datetime(censor_before, format='%Y')]
        registry_df = registry_df[registry_df['Date_admission'] >= pd.to_datetime(censor_before, format='%Y')]

    n_patients_after_filtering_year = registry_df[registry_df.pNr.isin(bp_df['pNr'])].shape[0]

    if verbose:
        print(f'Number of patients before censoring before {censor_before}: {n_patients_before_filtering_year}')
        print(f'Number of patients after censoring before {censor_before}: {n_patients_after_filtering_year}')

    # check if all values in systole, diastole, mitteldruck are numeric
    assert bp_df['systole'].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().sum() == 0
    assert bp_df['diastole'].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().sum() == 0
    assert bp_df['mitteldruck'].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().sum() == 0

    # Restrict to accepted ranges
    n_sys_out_of_range = bp_df[(bp_df['systole'] < 0) | (bp_df['systole'] > 300)].shape[0]
    n_dia_out_of_range = bp_df[(bp_df['diastole'] < 0) | (bp_df['diastole'] > 200)].shape[0]
    n_mitt_out_of_range = bp_df[(bp_df['mitteldruck'] < 0) | (bp_df['mitteldruck'] > 250)].shape[0]
    # accepted range for systole: 0-300
    bp_df = bp_df[(bp_df['systole'] >= 0) & (bp_df['systole'] <= 300)]
    # accepted range for diastole: 0-200
    bp_df = bp_df[(bp_df['diastole'] >= 0) & (bp_df['diastole'] <= 200)]
    # accepted range for mitteldruck: 0-250
    bp_df = bp_df[(bp_df['mitteldruck'] >= 0) & (bp_df['mitteldruck'] <= 250)]

    if verbose:
        print(f'Number of systole values out of range: {n_sys_out_of_range}')
        print(f'Number of diastole values out of range: {n_dia_out_of_range}')
        print(f'Number of mitteldruck values out of range: {n_mitt_out_of_range}')

    # Convert to datetime
    bp_df['timeBd'] = pd.to_datetime(bp_df['timeBd'], format='%Y-%m-%d %H:%M:%S.%f')

    # Preprocess outcome data
    # check that DCI_YN_verified contains only 0, 1 or NaN
    assert registry_df[target].apply(lambda x: x in [0, 1, np.nan]).all()

    patients_with_bp_but_no_outcome = bp_df[~bp_df['pNr'].isin(registry_df['pNr'])].pNr.drop_duplicates()
    patients_with_outcome_but_no_bp = registry_df[~((registry_df['pNr'].isin(bp_df['pNr'])) & (~registry_df['pNr'].isnull()))]
    n_patients_with_bp_but_no_outcome = patients_with_bp_but_no_outcome.shape[0]
    n_patients_with_outcome_but_no_bp = patients_with_outcome_but_no_bp.shape[0]
    if verbose:
        print(f'Number of patients with BP but no outcome: {n_patients_with_bp_but_no_outcome}')
        print(f'Number of patients with outcome but no BP: {n_patients_with_outcome_but_no_bp}')

    # Construct target events df
    registry_df[target] = registry_df[target].astype(int)
    target_events_df = registry_df[registry_df[target] == 1]
    target_events_df['full_date_target'] = target_events_df[f'Date_{target}_first_image'].astype(str) + ' ' + \
                                                 target_events_df[f'Time_{target}_first_image'].astype(str)
    # replace NaT nan with nan
    target_events_df['full_date_target'] = target_events_df['full_date_target'].replace('NaT nan', pd.NaT)
    target_events_df['full_date_target'] = target_events_df['full_date_target'].apply(
        safe_conversion_to_datetime)

    # Label positive timebins
    # loop through all events and label bp data with event
    for index, row in target_events_df.iterrows():
        # verify that patient is in pupillometry data
        if not row['pNr'] in bp_df['pNr'].values:
            if verbose:
                print(f'Patient {row["Name"]} not in pupillometry data')
            continue

        timebin_begin = pd.to_datetime(row['full_date_target']) - pd.Timedelta(hours=timebin_hours, unit='h')
        timebin_end = pd.to_datetime(row['full_date_target'])

        bp_df.loc[(bp_df['pNr'] == row['pNr'])
                  & (bp_df['timeBd'] >= timebin_begin)
                  & (bp_df['timeBd'] <= timebin_end),
                 'within_event_timebin'] = 1
        bp_df.loc[(bp_df['pNr'] == row['pNr'])
                  & (bp_df['timeBd'] >= timebin_begin)
                  & (bp_df['timeBd'] <= timebin_end),
                    'associated_event_time'] = row['full_date_target']

        # if no bp data within timebin, print warning
        if bp_df.loc[(bp_df['pNr'] == row['pNr'])
                  & (bp_df['timeBd'] >= timebin_begin)
                  & (bp_df['timeBd'] <= timebin_end)].shape[0] == 0:
            if verbose:
                print(f'No BP data within timebin for patient {row["Name"]}')

        # Censor data after first event
        if censor_data_after_first_event:
            # drop rows for pNr with timeBd > timebin_end
            bp_df = bp_df[~((bp_df['pNr'] == row['pNr']) & (bp_df['timeBd'] > timebin_end))]

    bp_df['within_event_timebin'] = bp_df['within_event_timebin'].fillna(0).astype(int)

    n_patients_with_event_and_bp_data = target_events_df[target_events_df['pNr'].isin(bp_df['pNr'])].pNr.nunique()
    n_patients_with_event_and_bp_in_timebin = bp_df[bp_df['within_event_timebin'] == 1].pNr.nunique()
    if verbose:
        print(f'Number of patients with event and BP data: {n_patients_with_event_and_bp_data}')
        print(f'Number of patients with event and BP in timebin: {n_patients_with_event_and_bp_in_timebin}')

    # Label negative timebins
    # for every patient in bp_df, add a column with the index of the timebin (starting at admission, timebin_hours apart, ending at last measurement or at start of positive timebin)
    bp_df['negative_timebin'] = np.nan

    n_negative_timebins = 0
    for patient in tqdm(bp_df['pNr'].unique()):
        patient_df = bp_df[bp_df['pNr'] == patient]
        patient_last_measurement = patient_df['timeBd'].max()
        patient_first_measurement = patient_df['timeBd'].min()
        patient_start_positive_timebin = patient_df[patient_df['within_event_timebin'] == 1]['associated_event_time'].min() - pd.Timedelta(hours=timebin_hours, unit='h')
        # positive and negative timebins should not overlap
        patient_start_positive_timebin = patient_start_positive_timebin - pd.Timedelta(minutes=1)
        patient_end_negative_timebins = patient_start_positive_timebin if not pd.isnull(patient_start_positive_timebin) else patient_last_measurement
        n_patient_negative_timebins = math.ceil((patient_end_negative_timebins - patient_first_measurement).total_seconds() / (timebin_hours * 3600))

        timebin_end = patient_end_negative_timebins
        for i in range(n_patient_negative_timebins):
            timebin_begin = timebin_end - pd.Timedelta(hours=timebin_hours, unit='h')
            bp_df.loc[(bp_df['pNr'] == patient)
                      & (bp_df['timeBd'] > timebin_begin)
                      & (bp_df['timeBd'] <= timebin_end),
                      'negative_timebin'] = n_patient_negative_timebins - i
            timebin_end = timebin_begin
        n_negative_timebins += n_patient_negative_timebins

    n_positive_timebins = bp_df[bp_df['within_event_timebin'] == 1].pNr.nunique()

    if verbose:
        print(f'Number of negative timebins: {n_negative_timebins}')
        print(f'Number of positive timebins: {n_positive_timebins}')

    n_patients = bp_df['pNr'].nunique()

    log_df = pd.DataFrame({'n_sys_out_of_range': [n_sys_out_of_range],
                            'n_dia_out_of_range': [n_dia_out_of_range],
                            'n_mitt_out_of_range': [n_mitt_out_of_range],
                            'n_patients_before_filtering_year': [n_patients_before_filtering_year],
                            'n_patients_after_filtering_year': [n_patients_after_filtering_year],
                            'n_patients_with_bp_but_no_outcome': [n_patients_with_bp_but_no_outcome],
                            'n_patients_with_outcome_but_no_bp': [n_patients_with_outcome_but_no_bp],
                            'n_patients_with_event_and_bp_data': [n_patients_with_event_and_bp_data],
                            'n_patients_with_event_and_bp_in_timebin': [n_patients_with_event_and_bp_in_timebin],
                            'n_patients': [n_patients],
                            'n_negative_timebins': [n_negative_timebins],
                            'n_positive_timebins': [n_positive_timebins]})

    return bp_df, log_df, patients_with_bp_but_no_outcome, patients_with_outcome_but_no_bp


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', '--registry_data_path', type=str, required=True)
    parser.add_argument('-bp', '--bp_data_path', type=str, required=True)
    parser.add_argument('-c', '--registry_pdms_correspondence_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-t', '--timebin_hours', type=int, required=True)
    parser.add_argument('-cen', '--censor_before', type=str, required=True)
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    args = parser.parse_args()

    bp_df, log_df, missing_outcomes_df, missing_bp_data_df = bp_timeseries_decomposition(
        args.registry_data_path,
        args.bp_data_path,
        args.registry_pdms_correspondence_path,
        timebin_hours=args.timebin_hours,
        target='DCI_ischemia',
        censor_data_after_first_event=True,
        password=None,
        censor_before=args.censor_before,
        verbose=args.verbose
    )

    local_args = {'registry_data_path': args.registry_data_path,
            'bp_data_path': args.bp_data_path,
            'registry_correspondence_path': args.registry_pdms_correspondence_path,
            'timebin_hours': args.timebin_hours,
            'target': 'DCI_ischemia',
            'censor_data_after_first_event': True,
            'censor_before': args.censor_before,
            'verbose': True}

    folder_name = f'bp_timebin_{args.timebin_hours}h'
    ensure_dir(os.path.join(args.output_dir, folder_name))
    ensure_dir(os.path.join(args.output_dir, folder_name, 'logs'))
    bp_df.to_csv(os.path.join(args.output_dir, folder_name, f'bp_timebins_{args.timebin_hours}h.csv'), index=False)
    log_df.to_csv(os.path.join(args.output_dir, folder_name, 'logs', f'log_{args.timebin_hours}h.csv'), index=False)
    missing_outcomes_df.to_csv(
        os.path.join(args.output_dir, folder_name, 'logs', f'missing_outcomes_{args.timebin_hours}h.csv'), index=False)
    missing_bp_data_df.to_csv(
        os.path.join(args.output_dir, folder_name, 'logs', f'missing_bp_data_{args.timebin_hours}h.csv'), index=False)
    with open(os.path.join(args.output_dir, folder_name, 'logs', f'args_{args.timebin_hours}h.txt'), 'w') as f:
        f.write(str(local_args))






