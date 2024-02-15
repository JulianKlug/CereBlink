import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import getpass

from utils import load_encrypted_xlsx
from utils import safe_conversion_to_datetime

def plot_timeseries_decomposition(registry_data_path, pupillometry_data_path, registry_pdms_correspondence_path, timebin_hours, target, censure_data_after_first_positive_CT, use_span, password=None,
                                  exclude_nan_outcome=True):
    # %% md
    # Timeseries decomposition
    # Goal: decompose timeseries into timebins of X hours and seperate if end of timebin includes CT showing DCI / vasospasm or not
    # %%
    output_dir = '/Users/jk1/Downloads'

    registry_df = load_encrypted_xlsx(registry_data_path, password=password)
    pupillometry_df = pd.read_csv(pupillometry_data_path, sep=';', decimal='.')
    registry_pdms_correspondence_df = pd.read_csv(registry_pdms_correspondence_path)
    registry_pdms_correspondence_df['Date_birth'] = pd.to_datetime(registry_pdms_correspondence_df['Date_birth'],
                                                                   format='%Y-%m-%d')
    # %%
    pupillometry_df = pupillometry_df.merge(registry_pdms_correspondence_df, on='pNr', how='left')
    pupillometry_df.rename(columns={'JoinedName': 'Name'}, inplace=True)
    # %% md
    # Preprocessing
    # %%
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
    # %%
    # exclude NPI values outside of 0-5
    pupillometry_df = pupillometry_df[(pupillometry_df['NPI_r_value'] >= 0) & (pupillometry_df['NPI_r_value'] <= 5)]
    pupillometry_df = pupillometry_df[(pupillometry_df['NPI_l_value'] >= 0) & (pupillometry_df['NPI_l_value'] <= 5)]
    # %%
    # exclude CV values outside of 0 - 10
    pupillometry_df = pupillometry_df[(pupillometry_df['CV_r_value'] >= 0) & (pupillometry_df['CV_r_value'] <= 10)]
    pupillometry_df = pupillometry_df[(pupillometry_df['CV_l_value'] >= 0) & (pupillometry_df['CV_l_value'] <= 10)]
    # %%
    # plot histogram of all values

    hist_fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(pupillometry_df['NPI_r_value'], ax=ax[0, 0])
    sns.histplot(pupillometry_df['NPI_l_value'], ax=ax[0, 1])
    sns.histplot(pupillometry_df['CV_r_value'], ax=ax[1, 0])
    sns.histplot(pupillometry_df['CV_l_value'], ax=ax[1, 1])
    # save histogram
    hist_fig.savefig(os.path.join(output_dir, f'pupillometry_histograms_{timebin_hours}h_timebin.png'), dpi=300,
                     bbox_inches='tight')
    # %%
    pupillometry_df['timePupil'] = pd.to_datetime(pupillometry_df['timePupil'], format='%Y-%m-%d %H:%M:%S.%f')
    # %%
    registry_df[target] = registry_df[target].astype(int)
    target_events_df = registry_df[registry_df[target] == 1]

    registry_df['DCI_YN_verified'] = registry_df['DCI_YN_verified'].replace('Yes', 1).fillna(registry_df['DCI_YN'])
    if exclude_nan_outcome:
        # exclude all measures from patients with nan outcome
        for index, row in registry_df[registry_df['DCI_YN_verified'].isnull()].iterrows():
            pupillometry_df.loc[(pupillometry_df['Name'] == row['Name'])
                                & (pupillometry_df['Date_birth'] == row['Date_birth'])
                                & (pupillometry_df['SOS-CENTER-YEAR-NO.'] == row['SOS-CENTER-YEAR-NO.']),
                                'to_drop'] = 1
        pupillometry_df = pupillometry_df[pupillometry_df['to_drop'] != 1]
        pupillometry_df.drop(columns=['to_drop'], inplace=True)

    # %%
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

    # %%
    # loop through all events and label pupillometry data with event
    for index, row in target_events_df.iterrows():
        # verify that patient is in pupillometry data
        if not row['Name'] in pupillometry_df['Name'].values:
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

    pupillometry_df['within_event_timebin'] = pupillometry_df['within_event_timebin'].fillna(0).astype(int)
    # %% md
    ### For every pupillometry entry add metrics for the timebin it ends

    # Gist: every new measure represents the end of a timebin of X hours
    # Metrics: - For every two sided measure: mean, min, max, delta - Over time: median, min, max, span
    # %%
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
    # %%
    pupillometry_metrics = ['NPI', 'CV']
    inter_eye_metrics = ['mean', 'min', 'max', 'delta']
    # combine to get all metrics
    single_timepoint_metrics = [f'{metric}_inter_eye_{metric_type}' for metric in pupillometry_metrics for metric_type
                                in inter_eye_metrics]
    # %%
    over_time_metrics = ['max', 'min', 'median', 'span']
    if not use_span:
        over_time_metrics.remove('span')
    # combine to get all metrics
    timebin_metrics = [f'{metric}_timebin_{metric_type}' for metric in single_timepoint_metrics for metric_type in
                       over_time_metrics]
    # %%
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
    # %% md
    ### Build negative pupillometry dataset (only pupillometry data outside of timebins containing target event)
    # %%
    negative_pupillometry_df = pupillometry_df[pupillometry_df['within_event_timebin'] == 0]
    negative_pupillometry_df['label'] = 0
    negative_pupillometry_df['timebin_end'] = pd.to_datetime(negative_pupillometry_df['timePupil'])
    # %% md
    ### Build positive pupillometry dataset (only pupillometry data within CT timebin)
    # loop through CTs and collect all pupillometry data within CT timebin
    # %%
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

            # %%
    # %%
    positive_pupillometry_df = target_events_df[target_events_df['pupillometry_available'] == 1]
    positive_pupillometry_df['label'] = 1
    target_event_time_column = f'full_date_{target.lower()}'
    positive_pupillometry_df['timebin_end'] = positive_pupillometry_df[target_event_time_column]
    # %%
    # %%
    reassembled_pupillometry_df = pd.concat([
        positive_pupillometry_df[['Name', 'Date_birth', 'label', 'timebin_end'] + timebin_metrics],
        negative_pupillometry_df[['Name', 'Date_birth', 'label', 'timebin_end'] + timebin_metrics]
    ])
    # %%
    reassembled_pupillometry_df.dropna(subset='Name', inplace=True)
    # %% md
    # For every subject with a positive CT, censure data after CT

    # %%
    if censure_data_after_first_positive_CT:
        # get all patient names with a positive CT
        patients_with_positive_event = reassembled_pupillometry_df[reassembled_pupillometry_df['label'] == 1][
            'Name'].unique()

        # for every subject with a positive event, censure data after event
        for patient_name in tqdm(patients_with_positive_event):
            # get time of first positive CT
            time_of_first_positive_event = reassembled_pupillometry_df[
                (reassembled_pupillometry_df['Name'] == patient_name) & (reassembled_pupillometry_df['label'] == 1)][
                'timebin_end'].min()

            # censure all data after time of first positive event
            reassembled_pupillometry_df.loc[(reassembled_pupillometry_df['Name'] == patient_name) & (
                        reassembled_pupillometry_df['timebin_end'] > time_of_first_positive_event), 'to_drop'] = 1

        reassembled_pupillometry_df = reassembled_pupillometry_df[reassembled_pupillometry_df['to_drop'] != 1]
        reassembled_pupillometry_df.drop(columns=['to_drop'], inplace=True)
    # %%
    reassembled_pupillometry_df.reset_index(drop=True, inplace=True)
    reassembled_pupillometry_df['label'] = reassembled_pupillometry_df['label'].astype(int)
    # %%
    # %% md
    # Stats
    # %%
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
    from pymer4.models import Lmer
    # %%
    reassembled_pupillometry_df['Name'] = reassembled_pupillometry_df['Name'].astype(str)
    # %%
    pvals_per_metric = {}
    for metric in tqdm(timebin_metrics, total=len(timebin_metrics)):
        model = Lmer(f"label  ~ {metric}  + (1|Name)",
                     data=reassembled_pupillometry_df, family='binomial')
        model.fit()
        pvals_per_metric[metric] = model.coefs['P-val'].to_dict()[metric]
    # %%
    pvals_per_metric_df = pd.DataFrame.from_dict(pvals_per_metric, orient='index', columns=['pval'])
    # %%
    # plot pvals as barplot (lowest 10) - horizontal
    # ax = pvals_per_metric_df.sort_values(by='pval').iloc[:10].plot(kind='barh', figsize=(10, 5))
    # pval_fig = ax.get_figure()

    # %% md
    ## Correct for multiple comparisons
    # %%
    # correct for with Reiner et al 2003 (independence of measures not needed)
    import statsmodels.stats.multitest
    sign_flags, adj_pvals, alpha_sidak, alphacBonf = statsmodels.stats.multitest.multipletests(
        pvals_per_metric_df['pval'].values, alpha=0.05, method='fdr_by')
    pvals_per_metric_df['adjusted_pval'] = adj_pvals
    pvals_per_metric_df['significance'] = sign_flags
    # %%
    # correct using Storey 2003 (qvalue)

    from multipy.fdr import qvalue
    significance_flags, qvals = qvalue(pvals_per_metric_df['pval'].values)
    pvals_per_metric_df['qval'] = qvals
    # %% md
    # Plot
    # %%
    # create a plot with a subplot for every timebin metric, with a scatterplot of metric vs label
    # add legend with p-value

    # create a plot with a subplot for every timebin metric, with a scatterplot of metric vs label
    n_columns = len(over_time_metrics)
    n_rows = int(np.ceil(len(timebin_metrics) / n_columns))
    plot_type = 'violin'

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(20, 60))

    for i, metric in enumerate(timebin_metrics):
        if plot_type == 'violin':
            sns.violinplot(data=reassembled_pupillometry_df, y=metric, hue='label', palette='pastel', split=True,
                           gap=0.1,
                           ax=axes[i // n_columns, i % n_columns])
        elif plot_type == 'box':
            sns.boxplot(data=reassembled_pupillometry_df, y=metric, hue='label', palette='pastel',
                        ax=axes[i // n_columns, i % n_columns])
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
    fig.suptitle(f'{target} pupillometry data ({timebin_hours}h timebin)', fontsize=16, y=0.9)
    # %%
    # save figure
    figure_name = f'{target}_pupillometry_data_{timebin_hours}h_timebin'
    if use_span:
        figure_name += '_with_span'

    fig.savefig(os.path.join(output_dir, figure_name + '.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    registry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/sos_sah_data/post_hoc_modified_aSAH_DATA_2009_2023_24122023.xlsx'
    pupillometry_data_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/Transfer Urs.pietsch@kssg.ch 22.01.24, 15_34/20240117_SAH_SOS_Pupillometrie.csv'
    registry_pdms_correspondence_path = '/Users/jk1/Library/CloudStorage/OneDrive-unige.ch/icu_research/dci_sah/data/pdms_data/registry_pdms_correspondence.csv'
    password = getpass.getpass()

    timebin_hours_choices = [24, 12, 8, 6]
    targets = ['DCI_infarct', 'DCI_ischemia']
    use_span = [True, False]
    censure_data_after_first_positive_CT = True

    # generate all combinations of parameters
    for timebin_hours in timebin_hours_choices:
        for target in targets:
            for span in use_span:
                plot_timeseries_decomposition(registry_data_path, pupillometry_data_path, registry_pdms_correspondence_path,
                                              timebin_hours, target, censure_data_after_first_positive_CT, span, password)




