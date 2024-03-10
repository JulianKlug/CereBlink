import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import getpass
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
from rpy2.rinterface_lib.embedded import RRuntimeError
from pymer4.models import Lmer
import statsmodels.stats.multitest
from multipy.fdr import qvalue

from utils import load_encrypted_xlsx
from utils import safe_conversion_to_datetime

def plot_timeseries_decomposition(registry_data_path, pupillometry_data_path, registry_pdms_correspondence_path,
                                  output_dir,
                                  timebin_hours, target, censure_data_after_first_positive_CT, use_span, password=None,
                                  exclude_nan_outcome=True, normalise_to_prior_max=True, use_R=False, save_data=False):
    # pandas version > 2 needed for correct string processing and sns > 12 for correct histplot
    assert int(pd.__version__[0]) >= 2, 'Please update pandas to version 2 or higher'
    assert int(sns.__version__.split('.')[1]) >= 12, 'Please update seaborn to version 0.12.0 or higher'

    # Timeseries decomposition
    # Goal: decompose timeseries into timebins of X hours and seperate if end of timebin includes CT showing DCI / vasospasm or not
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
    print(f'Start of pupillometry: {start_of_pupillometry}, end of pupillometry: {end_of_pupillometry}')
    n_patients_in_registry_after_pupillometry = \
    registry_df[registry_df['Date_admission'] > pupillometry_df.timePupil.min()]['Name'].nunique()
    print(f'Number of patients in registry_df after start of pupillometry: {n_patients_in_registry_after_pupillometry}')


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

    # plot histogram of all values
    hist_fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(pupillometry_df['NPI_r_value'], ax=ax[0, 0])
    sns.histplot(pupillometry_df['NPI_l_value'], ax=ax[0, 1])
    sns.histplot(pupillometry_df['CV_r_value'], ax=ax[1, 0])
    sns.histplot(pupillometry_df['CV_l_value'], ax=ax[1, 1])
    # save histogram
    hist_fig.savefig(os.path.join(output_dir, f'pupillometry_histograms_{timebin_hours}h_timebin.png'), dpi=300,
                     bbox_inches='tight')

    pupillometry_df['timePupil'] = pd.to_datetime(pupillometry_df['timePupil'], format='%Y-%m-%d %H:%M:%S.%f')

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
    print(f'Number of patients with pupillometry but no outcome: {n_patients_with_pupillometry_but_no_outcome}')
    n_patients_with_pupillometry_and_outcome = \
    registry_df[(registry_df['DCI_YN_verified'].notnull()) & (registry_df['pupillometry_available'] == 1)][
        'Name'].nunique()
    print(f'Number of patients with pupillometry and outcome: {n_patients_with_pupillometry_and_outcome}')

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
    print(f'Number of patients with event and pupillometry in timebin: {n_patients_with_event_and_pupillometry_in_timebin}')


    # Normalisation
    # normalise to prior max: for every patient, normalise to the max value of the same measure prior to the current timepoint
    # if no prior value, normalise to 1
    def normalisation(x, prior_max):
        epsilon = 1
        # avoid division by zero by adding epsilon
        # as epsilon is 1 no great shift in values
        return (x + epsilon) / (prior_max + epsilon)

    if normalise_to_prior_max:
        # add columns for normalised values
        pupillometry_df['NPI_r_value_normalised'] = np.nan
        pupillometry_df['NPI_l_value_normalised'] = np.nan
        pupillometry_df['CV_r_value_normalised'] = np.nan
        pupillometry_df['CV_l_value_normalised'] = np.nan
        # for every patient, normalise to prior max
        for patient_id in tqdm(pupillometry_df['pNr'].unique()):
            patient_pupillometry_df = pupillometry_df[pupillometry_df['pNr'] == patient_id]
            for index, row in patient_pupillometry_df.iterrows():
                time_of_pupillometry = row['timePupil']
                prior_pupillometry = patient_pupillometry_df[
                    patient_pupillometry_df['timePupil'] < time_of_pupillometry]
                # if prior_pupillometry.shape[0] > 0:
                max_NPI_r_value = prior_pupillometry['NPI_r_value'].max()
                max_NPI_l_value = prior_pupillometry['NPI_l_value'].max()
                max_CV_r_value = prior_pupillometry['CV_r_value'].max()
                max_CV_l_value = prior_pupillometry['CV_l_value'].max()

                if (np.isnan(max_NPI_r_value)) & (not np.isnan(row['NPI_r_value'])):
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (
                                pupillometry_df['timePupil'] == time_of_pupillometry), 'NPI_r_value_normalised'] = 1
                else:
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (pupillometry_df[
                                                                                      'timePupil'] == time_of_pupillometry), 'NPI_r_value_normalised'] = normalisation(
                        row['NPI_r_value'], max_NPI_r_value)

                if np.isnan(max_NPI_l_value) & (not np.isnan(row['NPI_l_value'])):
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (
                                pupillometry_df['timePupil'] == time_of_pupillometry), 'NPI_l_value_normalised'] = 1
                else:
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (pupillometry_df[
                                                                                      'timePupil'] == time_of_pupillometry), 'NPI_l_value_normalised'] = normalisation(
                        row['NPI_l_value'], max_NPI_l_value)

                if np.isnan(max_CV_r_value) & (not np.isnan(row['CV_r_value'])):
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (
                                pupillometry_df['timePupil'] == time_of_pupillometry), 'CV_r_value_normalised'] = 1
                else:
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (pupillometry_df[
                                                                                      'timePupil'] == time_of_pupillometry), 'CV_r_value_normalised'] = normalisation(
                        row['CV_r_value'], max_CV_r_value)

                if np.isnan(max_CV_l_value) & (not np.isnan(row['CV_l_value'])):
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (
                                pupillometry_df['timePupil'] == time_of_pupillometry), 'CV_l_value_normalised'] = 1
                else:
                    pupillometry_df.loc[(pupillometry_df['pNr'] == patient_id) & (pupillometry_df[
                                                                                      'timePupil'] == time_of_pupillometry), 'CV_l_value_normalised'] = normalisation(
                        row['CV_l_value'], max_CV_l_value)

        norm_hist_fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(pupillometry_df['NPI_r_value_normalised'], ax=ax[0, 0])
        sns.histplot(pupillometry_df['NPI_l_value_normalised'], ax=ax[0, 1])
        sns.histplot(pupillometry_df['CV_r_value_normalised'], ax=ax[1, 0])
        sns.histplot(pupillometry_df['CV_l_value_normalised'], ax=ax[1, 1])
        # save histogram
        norm_hist_fig.savefig(os.path.join(output_dir, f'normalised_pupillometry_histograms_{timebin_hours}h_timebin.png'),
                                dpi=300, bbox_inches='tight')

        if save_data:
            pupillometry_df.to_csv(os.path.join(output_dir, f'{target}_normalised_pupillometry_df.csv'))

        pupillometry_df.drop(columns=['NPI_r_value', 'NPI_l_value', 'CV_r_value', 'CV_l_value'], inplace=True)
        # replace non normalized values with normalized values
        pupillometry_df.rename(
            columns={'NPI_r_value_normalised': 'NPI_r_value', 'NPI_l_value_normalised': 'NPI_l_value',
                     'CV_r_value_normalised': 'CV_r_value', 'CV_l_value_normalised': 'CV_l_value'}, inplace=True)


    ### For every pupillometry entry add metrics for the timebin it ends
    # Gist: every new measure represents the end of a timebin of X hours
    # Metrics:
        # - For every two sided measure: mean, min, max, delta
        # - Over time: median, min, max, span

    # add inter eye metrics for every pupillometry entry
    # NPI
    pupillometry_df['NPI_inter_eye_mean'] = pupillometry_df[['NPI_r_value', 'NPI_l_value']].mean(axis=1)
    pupillometry_df['NPI_inter_eye_min'] = pupillometry_df[['NPI_r_value', 'NPI_l_value']].min(axis=1)
    pupillometry_df['NPI_inter_eye_max'] = pupillometry_df[['NPI_r_value', 'NPI_l_value']].max(axis=1)
    pupillometry_df['NPI_inter_eye_delta'] = np.abs(pupillometry_df['NPI_r_value'] - pupillometry_df['NPI_l_value'])

    # CV
    pupillometry_df['CV_inter_eye_mean'] = pupillometry_df[['CV_r_value', 'CV_l_value']].mean(axis=1)
    pupillometry_df['CV_inter_eye_min'] = pupillometry_df[['CV_r_value', 'CV_l_value']].min(axis=1)
    pupillometry_df['CV_inter_eye_max'] = pupillometry_df[['CV_r_value', 'CV_l_value']].max(axis=1)
    pupillometry_df['CV_inter_eye_delta'] = np.abs(pupillometry_df['CV_r_value'] - pupillometry_df['CV_l_value'])

    pupillometry_metrics = ['NPI', 'CV']
    inter_eye_metrics = ['mean', 'min', 'max', 'delta']
    # combine to get all metrics
    single_timepoint_metrics = [f'{metric}_inter_eye_{metric_type}' for metric in pupillometry_metrics for metric_type
                                in inter_eye_metrics]

    over_time_metrics = ['max', 'min', 'median', 'span']
    if not use_span:
        over_time_metrics.remove('span')
    # combine to get all metrics
    timebin_metrics = [f'{metric}_timebin_{metric_type}' for metric in single_timepoint_metrics for metric_type in
                       over_time_metrics]

    # add timebin metrics for every pupillometry entry
    for index, row in tqdm(pupillometry_df.iterrows(), total=len(pupillometry_df)):
        timebin_begin = row['timePupil'] - pd.Timedelta(hours=timebin_hours, unit='h')
        timebin_end = row['timePupil']

        # compute timebin metrics for every single timepoint metric
        for metric in single_timepoint_metrics:
            # get all values within timebin
            values_within_timebin = pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr'])
                                                        & (pupillometry_df['timePupil'] >= timebin_begin)
                                                        & (pupillometry_df['timePupil'] <= timebin_end), metric]

            n_values_within_timebin = len(values_within_timebin)

            # if no values within timebin, skip
            if n_values_within_timebin == 0:
                continue

            # add timebin metrics
            pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr']) &
                                (pupillometry_df['timePupil'] == row[
                                    'timePupil']), f'{metric}_timebin_median'] = values_within_timebin.median()
            pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr']) &
                                (pupillometry_df['timePupil'] == row[
                                    'timePupil']), f'{metric}_timebin_min'] = values_within_timebin.min()
            pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr']) &
                                (pupillometry_df['timePupil'] == row[
                                    'timePupil']), f'{metric}_timebin_max'] = values_within_timebin.max()
            pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr']) &
                                (pupillometry_df['timePupil'] == row[
                                    'timePupil']), f'{metric}_timebin_span'] = values_within_timebin.max() - values_within_timebin.min()

        pupillometry_df.loc[(pupillometry_df['pNr'] == row['pNr']) &
                            (pupillometry_df['timePupil'] == row[
                                'timePupil']), 'n_values_within_timebin'] = n_values_within_timebin


    ### Build negative pupillometry dataset (only pupillometry data outside of timebins containing target event)
    negative_pupillometry_df = pupillometry_df[pupillometry_df['within_event_timebin'] == 0]
    negative_pupillometry_df['label'] = 0
    negative_pupillometry_df['timebin_end'] = pd.to_datetime(negative_pupillometry_df['timePupil'])

    ### Build positive pupillometry dataset (only pupillometry data within CT timebin)
    # loop through CTs and collect all pupillometry data within CT timebin
    for index, row in tqdm(target_events_df.iterrows(), total=len(target_events_df)):
        # verify that patient is in pupillometry data
        if not row['Name'] in pupillometry_df['Name'].values:
            print(f'Patient {row["Name"]} not in pupillometry data')
            target_events_df.loc[(target_events_df['Name'] == row['Name']), 'pupillometry_available'] = 0
            continue

        target_events_df.loc[(target_events_df['Name'] == row['Name']), 'pupillometry_available'] = 1

        target_event_time_column = f'full_date_{target.lower()}'
        timebin_begin = pd.to_datetime(row[target_event_time_column]) - pd.Timedelta(hours=timebin_hours, unit='h')
        timebin_end = pd.to_datetime(row[target_event_time_column])

        # add timebin metrics comprising all data within timebin
        values_within_timebin = pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                                                    & (pupillometry_df['Date_birth'] == row['Date_birth'])
                                                    & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row[
            'SOS-CENTER-YEAR-NO.'])
                                                    & (pupillometry_df['timePupil'] >= timebin_begin)
                                                    & (pupillometry_df['timePupil'] <= timebin_end)]

        n_values_within_timebin = len(values_within_timebin)
        # if no values within timebin, skip
        if n_values_within_timebin == 0:
            continue

        target_events_df.loc[(target_events_df['Name'] == row['Name'])
                             & (target_events_df['Date_birth'] == row['Date_birth']) & (
                                         target_events_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']) &
                             (target_events_df[target_event_time_column] == row[
                                 target_event_time_column]), 'n_values_within_timebin'] = n_values_within_timebin

        for metric in single_timepoint_metrics:
            # add timebin metrics
            target_events_df.loc[(target_events_df['Name'] == row['Name'])
                                 & (target_events_df['Date_birth'] == row['Date_birth']) & (
                                             target_events_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']) &
                                 (target_events_df[target_event_time_column] == row[
                                     target_event_time_column]), f'{metric}_timebin_median'] = values_within_timebin[
                metric].median()
            target_events_df.loc[(target_events_df['Name'] == row['Name'])
                                 & (target_events_df['Date_birth'] == row['Date_birth']) & (
                                             target_events_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']) &
                                 (target_events_df[target_event_time_column] == row[
                                     target_event_time_column]), f'{metric}_timebin_min'] = values_within_timebin[
                metric].min()
            target_events_df.loc[(target_events_df['Name'] == row['Name'])
                                 & (target_events_df['Date_birth'] == row['Date_birth']) & (
                                             target_events_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']) &
                                 (target_events_df[target_event_time_column] == row[
                                     target_event_time_column]), f'{metric}_timebin_max'] = values_within_timebin[
                metric].max()
            target_events_df.loc[(target_events_df['Name'] == row['Name'])
                                 & (target_events_df['Date_birth'] == row['Date_birth']) & (
                                             target_events_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']) &
                                 (target_events_df[target_event_time_column] == row[
                                     target_event_time_column]), f'{metric}_timebin_span'] = values_within_timebin[
                                                                                                 metric].max() - \
                                                                                             values_within_timebin[
                                                                                                 metric].min()


    positive_pupillometry_df = target_events_df[target_events_df['pupillometry_available'] == 1]
    positive_pupillometry_df['label'] = 1
    target_event_time_column = f'full_date_{target.lower()}'
    positive_pupillometry_df['timebin_end'] = positive_pupillometry_df[target_event_time_column]

    reassembled_pupillometry_df = pd.concat([
        positive_pupillometry_df[['pNr', 'Name', 'Date_birth', 'label', 'timebin_end'] + timebin_metrics],
        negative_pupillometry_df[['pNr', 'Name', 'Date_birth', 'label', 'timebin_end'] + timebin_metrics]
    ])
    assert reassembled_pupillometry_df.pNr.isnull().sum() == 0

    # For every subject with a positive CT, censure data after CT
    if censure_data_after_first_positive_CT:
        # get all patient ids with a positive CT
        patients_with_positive_event = reassembled_pupillometry_df[reassembled_pupillometry_df['label'] == 1][
            'pNr'].unique()

        # for every subject with a positive event, censure data after event
        for pid in tqdm(patients_with_positive_event):
            # get time of first positive CT
            time_of_first_positive_event = reassembled_pupillometry_df[
                (reassembled_pupillometry_df['pNr'] == pid) & (reassembled_pupillometry_df['label'] == 1)][
                'timebin_end'].min()

            # censure all data after time of first positive event
            reassembled_pupillometry_df.loc[(reassembled_pupillometry_df['pNr'] == pid) & (
                        reassembled_pupillometry_df['timebin_end'] > time_of_first_positive_event), 'to_drop'] = 1

        reassembled_pupillometry_df = reassembled_pupillometry_df[reassembled_pupillometry_df['to_drop'] != 1]
        reassembled_pupillometry_df.drop(columns=['to_drop'], inplace=True)


    for metric in timebin_metrics:
        print(
            f'Number of nan values for {metric}: {reassembled_pupillometry_df[reassembled_pupillometry_df[metric].isnull()].shape[0]}')

    reassembled_pupillometry_df.reset_index(drop=True, inplace=True)
    reassembled_pupillometry_df['label'] = reassembled_pupillometry_df['label'].astype(int)

    # Count number of positive and negative timebins per metric
    max_n_positive_timebins = np.nan
    min_n_positive_timebins = np.nan
    max_n_negative_timebins = np.nan
    min_n_negative_timebins = np.nan
    for metric in timebin_metrics:
        n_positive_timebins = reassembled_pupillometry_df[
            (reassembled_pupillometry_df.label == 1) & (~reassembled_pupillometry_df[metric].isnull())].shape[0]
        n_negative_timebins = reassembled_pupillometry_df[
            (reassembled_pupillometry_df.label == 0) & (~reassembled_pupillometry_df[metric].isnull())].shape[0]
        if np.isnan(max_n_positive_timebins) or n_positive_timebins > max_n_positive_timebins:
            max_n_positive_timebins = n_positive_timebins
        if np.isnan(min_n_positive_timebins) or n_positive_timebins < min_n_positive_timebins:
            min_n_positive_timebins = n_positive_timebins
        if np.isnan(max_n_negative_timebins) or n_negative_timebins > max_n_negative_timebins:
            max_n_negative_timebins = n_negative_timebins
        if np.isnan(min_n_negative_timebins) or n_negative_timebins < min_n_negative_timebins:
            min_n_negative_timebins = n_negative_timebins
        print(f'Number of positive timebins for {metric}: {n_positive_timebins}')
        print(f'Number of negative timebins for {metric}: {n_negative_timebins}')

    reassembled_pupillometry_df['Name'] = reassembled_pupillometry_df['Name'].astype(str)
    reassembled_pupillometry_df['pNr'] = reassembled_pupillometry_df['pNr'].astype(int).astype(str)

    if save_data:
        # Save reassembled pupillometry data
        pupillometry_timebins_file_name = f'{target}_reassembled_pupillometry_{timebin_hours}h_timebin'
        if normalise_to_prior_max:
            pupillometry_timebins_file_name += '_normalised'
        if use_span:
            pupillometry_timebins_file_name += '_with_span'
        reassembled_pupillometry_df.to_csv(os.path.join(output_dir, pupillometry_timebins_file_name + '.csv'))

    ### Stats
    # Use pymer4
    if not use_R:
        pvals_per_metric = {}
        model_warnings_df = pd.DataFrame(columns=['metric', 'warning'])
        for metric in tqdm(timebin_metrics, total=len(timebin_metrics)):
            metric_df = reassembled_pupillometry_df[[metric, 'label', 'pNr']]
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
                    model_warnings_df = pd.concat([model_warnings_df, pd.DataFrame({'metric': [metric], 'warning': [model.warnings]})])
            pvals_per_metric[metric] = model.coefs['P-val'].to_dict()[metric]
    # directly call R
    else:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        import rpy2.robjects as ro

        stats = importr('stats')
        lme4 = importr('lme4')
        base = importr('base')
        lmerT = importr('lmerTest')

        r_pvals_per_metric = {}
        r_model_warnings_df = pd.DataFrame(columns=['metric', 'warning'])

        for metric in tqdm(timebin_metrics, total=len(timebin_metrics)):
            metric_df = reassembled_pupillometry_df[[metric, 'label', 'pNr']]
            metric_df.dropna(subset=[metric], inplace=True)
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

    if not normalise_to_prior_max:
        pval_file_name = f'{target}_pvals_{timebin_hours}h_timebin'
    else:
        pval_file_name = f'{target}_pvals_{timebin_hours}h_timebin_normalised'
    if use_span:
        pval_file_name += '_with_span'
    pvals_per_metric_df.to_csv(os.path.join(output_dir, pval_file_name + '.csv'))

    ### Plot
    # create a plot with a subplot for every timebin metric, with a scatterplot of metric vs label
    # add legend with p-value
    n_columns = len(over_time_metrics)
    n_rows = int(np.ceil(len(timebin_metrics) / n_columns))
    if normalise_to_prior_max:
        plot_type = 'box'
    else:
        plot_type = 'violin'

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(20, 60))

    for i, metric in enumerate(timebin_metrics):
        if plot_type == 'violin':
            sns.violinplot(data=reassembled_pupillometry_df, y=metric, hue='label', palette='pastel', split=True,
                           gap=0.1,
                           ax=axes[i // n_columns, i % n_columns])
        elif plot_type == 'box':
            sns.boxplot(data=reassembled_pupillometry_df, y=metric, hue='label', palette='pastel',
                        ax=axes[i // n_columns, i % n_columns], showfliers=False)
        else:
            print('plot type not recognized')
        axes[i // n_columns, i % n_columns].set_title(metric)
        axes[i // n_columns, i % n_columns].set_ylabel(metric)
        axes[i // n_columns, i % n_columns].set_ylabel('')
        axes[i // n_columns, i % n_columns].legend(title='DCI', loc='upper right')

        # add text on lower right with pval/qval
        t = axes[i // n_columns, i % n_columns].text(0.98, 0.05,
                                                     f'adj. p-val. = {pvals_per_metric_df.loc[metric, "adjusted_pval"]:.2f}\nq-val. = {pvals_per_metric_df.loc[metric, "qval"]:.2f}',
                                                     horizontalalignment='right', verticalalignment='center',
                                                     transform=axes[i // n_columns][i % n_columns].transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.5, boxstyle="round"))

    # add figure suptitle
    if normalise_to_prior_max:
        fig.suptitle(f'{target} normalised pupillometry data ({timebin_hours}h timebin)', fontsize=16, y=0.9)
    else:
        fig.suptitle(f'{target} pupillometry data ({timebin_hours}h timebin)', fontsize=16, y=0.9)

    # save figure
    if normalise_to_prior_max:
        figure_name = f'{target}_pupillometry_data_{timebin_hours}h_timebin_normalised'
    else:
        figure_name = f'{target}_pupillometry_data_{timebin_hours}h_timebin'

    if use_span:
        figure_name += '_with_span'

    fig.savefig(os.path.join(output_dir, figure_name + '.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    registry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/sos_sah_data/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx'
    pupillometry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/Transfer Urs.pietsch@kssg.ch 22.01.24, 15_34/20240117_SAH_SOS_Pupillometrie.csv'
    registry_pdms_correspondence_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/registry_pdms_correspondence.csv'
    output_dir = '/Users/jk1/Downloads/data_saving'
    password = getpass.getpass()

    timebin_hours_choices = [24, 12, 8, 6]
    targets = ['DCI_ischemia', 'DCI_infarct']
    use_span = [False]
    censure_data_after_first_positive_CT = True
    normalise_to_prior_max = [True, False]
    exclude_nan_outcome = [False]

    use_R = True
    save_data = True

    # generate all combinations of parameters
    for exclude in exclude_nan_outcome:
        dir = os.path.join(output_dir, f'exclude_nan_outcome_{exclude}')
        os.makedirs(dir, exist_ok=True)
        for timebin_hours in timebin_hours_choices:
            for target in targets:
                for span in use_span:
                    for normalise in normalise_to_prior_max:
                        if normalise:
                            target_figure_name = f'{target}_pupillometry_data_{timebin_hours}h_timebin_normalised'
                        else:
                            target_figure_name = f'{target}_pupillometry_data_{timebin_hours}h_timebin'
                        if span:
                            target_figure_name += '_with_span'
                        # check if figure already exists
                        if os.path.exists(os.path.join(dir, target_figure_name + '.png')):
                            print(f'Figure {target_figure_name} already exists, skipping')
                            continue

                        print(f'Generating data and figure {target_figure_name}')
                        plot_timeseries_decomposition(registry_data_path=registry_data_path, pupillometry_data_path=pupillometry_data_path,
                                                      registry_pdms_correspondence_path=registry_pdms_correspondence_path,
                                                      output_dir=dir,
                                                      timebin_hours=timebin_hours,
                                                      target=target, censure_data_after_first_positive_CT=censure_data_after_first_positive_CT,
                                                      use_span=span, password=password,
                                                      exclude_nan_outcome=exclude,
                                                      normalise_to_prior_max=normalise,
                                                      use_R=use_R,
                                                      save_data=save_data)




