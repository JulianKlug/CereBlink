import getpass
from utils import load_encrypted_xlsx, safe_conversion_to_datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def patient_flow(registry_data_path, pupillometry_data_path, registry_pdms_correspondence_path,
                 timebin_hours, target, password=None, exclude_nan_outcome=True, verbose=True):
    patient_flow_df = pd.DataFrame()
    patient_flow_df['exclude_nan_outcome'] = [exclude_nan_outcome]
    patient_flow_df['timebin_hours'] = timebin_hours
    patient_flow_df['target'] = target


    registry_df = load_encrypted_xlsx(registry_data_path, password=password)
    pupillometry_df = pd.read_csv(pupillometry_data_path, sep=';', decimal='.')
    registry_pdms_correspondence_df = pd.read_csv(registry_pdms_correspondence_path)
    registry_pdms_correspondence_df['Date_birth'] = pd.to_datetime(registry_pdms_correspondence_df['Date_birth'],
                                                                   format='%Y-%m-%d')

    registry_pdms_correspondence_df.rename(columns={'JoinedName': 'Name'}, inplace=True)
    pupillometry_df = pupillometry_df.merge(registry_pdms_correspondence_df, on='pNr', how='left')
    registry_df = registry_df.merge(registry_pdms_correspondence_df, on=['SOS-CENTER-YEAR-NO.', 'Name', 'Date_birth'],
                                    how='left')

    # count number of patients in registry_df with Date_admission after pupillometry_df.timePupil.min()
    start_of_pupillometry = pupillometry_df.timePupil.min()
    end_of_pupillometry = pupillometry_df.timePupil.max()

    n_patients_in_registry_after_pupillometry = \
    registry_df[registry_df['Date_admission'] > pupillometry_df.timePupil.min()]['Name'].nunique()

    patient_flow_df['start_of_pupillometry'] = start_of_pupillometry
    patient_flow_df['end_of_pupillometry'] = end_of_pupillometry
    patient_flow_df['n_patients_in_registry_after_pupillometry'] = n_patients_in_registry_after_pupillometry

    if verbose:
        print(f'Start of pupillometry: {start_of_pupillometry}, end of pupillometry: {end_of_pupillometry}')
        print(f'Number of patients in registry_df after start of pupillometry: {n_patients_in_registry_after_pupillometry}')

    registry_df['DCI_YN_verified'] = registry_df['DCI_YN_verified'].replace('Yes', 1).fillna(registry_df['DCI_YN'])
    # Optional: exclude all measures from patients with undefined outcome
    if exclude_nan_outcome:
        # exclude all measures from patients with nan outcome
        for index, row in registry_df[registry_df['DCI_YN_verified'].isnull()].iterrows():
            pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                                & (pupillometry_df['Date_birth'] == row['Date_birth'])
                                & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']),
            'to_drop'] = 1
        pupillometry_df = pupillometry_df[pupillometry_df['to_drop'] != 1]
        pupillometry_df.drop(columns=['to_drop'], inplace=True)

    n_patients_with_nan_outcome = registry_df[registry_df[target].isnull()][['Name', 'Date_birth', 'SOS-CENTER-YEAR-NO.']].nunique().mode()[0]
    patient_flow_df['n_patients_with_nan_outcome'] = n_patients_with_nan_outcome
    n_patients_with_event = registry_df[registry_df[target] == 1][['Name', 'Date_birth', 'SOS-CENTER-YEAR-NO.']].nunique().mode()[0]
    patient_flow_df['n_patients_with_event'] = n_patients_with_event

    if verbose:
        print(f'Number of patients with nan outcome: {n_patients_with_nan_outcome}')
        print(f'Number of patients with event: {n_patients_with_event}')

    # Preprocessing
    # in NPI_r_value NPI_l_value, CV_r_value and CV_l_value, if contains '-' set to np.nan, then replace , with . and ".." with "."
    pupillometry_df['NPI_r_value'] = pd.to_numeric(
        pupillometry_df['NPI_r_value'].apply(lambda x: np.nan if '-' in str(x) else x).str.replace(',',
                                                                                                   '.').str.replace(
            '..', '.'), errors='coerce')
    pupillometry_df['NPI_l_value'] = pd.to_numeric(
        pupillometry_df['NPI_l_value'].apply(lambda x: np.nan if '-' in str(x) else x).str.replace(',',
                                                                                                   '.').str.replace(
            '..', '.'), errors='coerce')
    pupillometry_df['CV_r_value'] = pd.to_numeric(
        pupillometry_df['CV_r_value'].apply(lambda x: np.nan if '-' in str(x) else x).str.replace(',', '.').str.replace(
            '..', '.'), errors='coerce')
    pupillometry_df['CV_l_value'] = pd.to_numeric(
        pupillometry_df['CV_l_value'].apply(lambda x: np.nan if '-' in str(x) else x).str.replace(',', '.').str.replace(
            '..', '.'), errors='coerce')

    # exclude NPI values outside of 0-5
    n_npi_out_of_range = pupillometry_df[(pupillometry_df['NPI_r_value'] < 0) | (pupillometry_df['NPI_r_value'] > 5) | (
                pupillometry_df['NPI_l_value'] < 0) | (pupillometry_df['NPI_l_value'] > 5)].shape[0]
    print(f'Excluding {n_npi_out_of_range} NPI values outside of 0-5')
    pupillometry_df.loc[
        (pupillometry_df['NPI_r_value'] < 0) | (pupillometry_df['NPI_r_value'] > 5), 'NPI_r_value'] = np.nan
    pupillometry_df.loc[
        (pupillometry_df['NPI_l_value'] < 0) | (pupillometry_df['NPI_l_value'] > 5), 'NPI_l_value'] = np.nan

    # exclude CV values outside of 0 - 10
    n_cv_out_of_range = pupillometry_df[(pupillometry_df['CV_r_value'] < 0) | (pupillometry_df['CV_r_value'] > 10) | (
                pupillometry_df['CV_l_value'] < 0) | (pupillometry_df['CV_l_value'] > 10)].shape[0]
    print(f'Excluding {n_cv_out_of_range} CV values outside of 0-10')
    pupillometry_df.loc[
        (pupillometry_df['CV_r_value'] < 0) | (pupillometry_df['CV_r_value'] > 10), 'CV_r_value'] = np.nan
    pupillometry_df.loc[
        (pupillometry_df['CV_l_value'] < 0) | (pupillometry_df['CV_l_value'] > 10), 'CV_l_value'] = np.nan

    # if all Npi and CV values are nan, exclude
    pupillometry_df['all_nan'] = pupillometry_df[['NPI_r_value', 'NPI_l_value', 'CV_r_value', 'CV_l_value']].isnull().all(axis=1)
    n_all_nan = pupillometry_df[pupillometry_df['all_nan'] == 1].shape[0]
    patient_flow_df['n_values_excluded_because_all_nan'] = n_all_nan
    if verbose:
        print(f'Excluding {n_all_nan} values because all NPI and CV values are nan')
    pupillometry_df = pupillometry_df[pupillometry_df['all_nan'] != 1]
    pupillometry_df.drop(columns=['all_nan'], inplace=True)

    pupillometry_df['timePupil'] = pd.to_datetime(pupillometry_df['timePupil'], format='%Y-%m-%d %H:%M:%S.%f')

    # add column to registry_df indicating if pupillometry data is available
    registry_df['pupillometry_available'] = 0
    for index, row in registry_df.iterrows():
        if row.pNr is np.nan:
            continue
        if pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr'])].shape[0] > 0:
            registry_df.loc[(registry_df['pNr'] == row['pNr']), 'pupillometry_available'] = 1

    n_patients_with_pupillometry_but_no_outcome = \
    registry_df[(registry_df['DCI_YN_verified'].isnull()) & (registry_df['pupillometry_available'] == 1)][
        'Name'].nunique()
    n_patients_with_pupillometry_and_outcome = \
    registry_df[(registry_df['DCI_YN_verified'].notnull()) & (registry_df['pupillometry_available'] == 1)][
        'Name'].nunique()
    n_patients_with_pupillometry_and_event = \
    registry_df[(registry_df[target] == 1) & (registry_df['pupillometry_available'] == 1)][
        'Name'].nunique()

    tmp_df = pupillometry_df.merge(registry_df, on='pNr', how='left')
    pupillometry_unvailable_pnrs = registry_df[registry_df['pupillometry_available'] == 0]['pNr'].unique()
    pupillometry_df[pupillometry_df.pNr.isin(pupillometry_unvailable_pnrs)]

    patient_flow_df['n_patients_with_pupillometry_but_no_outcome'] = n_patients_with_pupillometry_but_no_outcome
    patient_flow_df['n_patients_with_pupillometry_and_outcome'] = n_patients_with_pupillometry_and_outcome
    patient_flow_df['n_patients_with_pupillometry_and_event'] = n_patients_with_pupillometry_and_event

    if verbose:
        print(f'Number of patients with pupillometry but no outcome: {n_patients_with_pupillometry_but_no_outcome}')
        print(f'Number of patients with pupillometry and outcome: {n_patients_with_pupillometry_and_outcome}')
        print(f'Number of patients with pupillometry and event: {n_patients_with_pupillometry_and_event}')

    registry_df[target] = registry_df[target].astype(int)
    target_events_df = registry_df[registry_df[target] == 1]
    # add Date_DCI_ischemia_first_image and Time_DCI_ischemia_first_image to get the full date
    target_events_df['full_date_dci_ischemia'] = target_events_df['Date_DCI_ischemia_first_image'].astype(str) + ' ' + \
                                                 target_events_df['Time_DCI_ischemia_first_image'].astype(str)
    # replace NaT nan with nan
    target_events_df['full_date_dci_ischemia'] = target_events_df['full_date_dci_ischemia'].replace('NaT nan', pd.NaT)
    target_events_df['full_date_dci_ischemia'] = target_events_df['full_date_dci_ischemia'].apply(
        safe_conversion_to_datetime)

    target_events_df['full_date_dci_infarct'] = target_events_df['Date_DCI_infarct_first_image'].astype(str) + ' ' + \
                                                target_events_df['Time_DCI_infarct_first_image'].astype(str)
    # replace NaT nan with nan
    target_events_df['full_date_dci_infarct'] = target_events_df['full_date_dci_infarct'].replace('NaT nan', pd.NaT)
    target_events_df['full_date_dci_infarct'] = target_events_df['full_date_dci_infarct'].apply(
        safe_conversion_to_datetime)

    # loop through all events and label pupillometry data with event
    for index, row in target_events_df.iterrows():
        # verify that patient is in pupillometry data
        if not row['pNr'] in pupillometry_df['pNr'].values:
            print(f'Patient {row["Name"]} not in pupillometry data')
            continue

        target_event_time_column = f'full_date_{target.lower()}'
        timebin_begin = pd.to_datetime(row[target_event_time_column]) - pd.Timedelta(hours=timebin_hours, unit='h')
        timebin_end = pd.to_datetime(row[target_event_time_column])

        # for all associated pupillometry entries add a 'within_event_timebin' column
        pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                            & (pupillometry_df['Date_birth'] == row['Date_birth'])
                            & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.'])
                            & (pupillometry_df['timePupil'] >= timebin_begin)
                            & (pupillometry_df['timePupil'] <= timebin_end), 'within_event_timebin'] = 1

        # for all associated pupillometry entries add a 'associated_CT_time' column
        pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                            & (pupillometry_df['Date_birth'] == row['Date_birth'])
                            & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.'])
                            & (pupillometry_df['timePupil'] >= timebin_begin)
                            & (pupillometry_df['timePupil'] <= timebin_end), 'associated_CT_time'] = row[
            target_event_time_column]

        # if no pupillometry data within timebin, print warning
        if pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                               & (pupillometry_df['Date_birth'] == row['Date_birth'])
                               & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.'])
                               & (pupillometry_df['timePupil'] >= timebin_begin)
                               & (pupillometry_df['timePupil'] <= timebin_end)].shape[0] == 0:
            print(f'No pupillometry data within timebin for patient {row["Name"]}')

    pupillometry_df['within_event_timebin'] = pupillometry_df['within_event_timebin'].fillna(0).astype(int)

    n_patients_with_event_and_pupillometry_in_timebin = pupillometry_df[pupillometry_df.within_event_timebin == 1].pNr.nunique()

    patient_flow_df['n_patients_with_event_and_pupillometry_in_timebin'] = n_patients_with_event_and_pupillometry_in_timebin

    if verbose:
        print(f'Number of patients with event and pupillometry in timebin: {n_patients_with_event_and_pupillometry_in_timebin}')

    return patient_flow_df


if __name__ == '__main__':
    import argparse

    registry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/sos_sah_data/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx'
    pupillometry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/Transfer Urs.pietsch@kssg.ch 22.01.24, 15_34/20240117_SAH_SOS_Pupillometrie.csv'
    registry_pdms_correspondence_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/registry_pdms_correspondence.csv'
    output_dir = '/Users/jk1/Downloads/data_saving/'
    password = getpass.getpass()

    timebin_hours_choices = [24, 12, 8, 6]
    targets = ['DCI_infarct', 'DCI_ischemia']
    exclude_nan_outcome = [False]

    overall_patient_flow_df = pd.DataFrame()
    for timebin in tqdm(timebin_hours_choices):
        for target in targets:
            for exclude_nan in exclude_nan_outcome:
                patient_flow_df = patient_flow(registry_data_path, pupillometry_data_path, registry_pdms_correspondence_path,
                                               timebin, target, password=password, exclude_nan_outcome=exclude_nan)
                overall_patient_flow_df = pd.concat([overall_patient_flow_df, patient_flow_df])

    overall_patient_flow_df.to_csv(os.path.join(output_dir, 'patient_flow.csv'), index=False)

