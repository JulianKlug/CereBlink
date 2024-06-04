import numpy as np
import asyncio
from tqdm.asyncio import tqdm
import pandas as pd

from utils.utils import background


def normalisation(x, prior_median):
    epsilon = 1
    # avoid division by zero by adding epsilon
    # as epsilon is 1 no great shift in values
    return (x + epsilon) / (prior_median + epsilon)

@background
def normalise_single_patient(patient_bp_df, bp_metrics=['systole', 'diastole', 'mitteldruck']):
    for index, row in patient_bp_df.iterrows():
        time_of_bd = row['timeBd']
        prior_bp_df = patient_bp_df[
            patient_bp_df['timeBd'] < time_of_bd]

        prior_medians = {}
        for bp_metric in bp_metrics:
            prior_medians[bp_metric] = prior_bp_df[f'{bp_metric}'].median()

            if (np.isnan(prior_medians[bp_metric])) & (not np.isnan(row[bp_metric])):
                patient_bp_df.loc[(patient_bp_df['timeBd'] == time_of_bd), f'{bp_metric}_normalised'] = 1
            else:
                patient_bp_df.loc[(patient_bp_df['timeBd'] == time_of_bd), f'{bp_metric}_normalised'] = normalisation(
                    row[bp_metric], prior_medians[bp_metric])

    return patient_bp_df


def parallel_normalise(bp_df, bp_metrics=['systole', 'diastole', 'mitteldruck']):
    # add columns for normalised values
    for bp_metric in bp_metrics:
        bp_df[f'{bp_metric}_normalised'] = np.nan

    loop = asyncio.get_event_loop()  # Have a new event loop
    looper = tqdm.gather(*[normalise_single_patient(bp_df[bp_df['pNr'] == patient_id], bp_metrics) for patient_id in bp_df['pNr'].unique()])  # Run the loop
    pt_dfs_list = loop.run_until_complete(looper)

    normalised_bp_df = pd.concat(pt_dfs_list, axis=0)

    return normalised_bp_df

def normalise(bp_df, bp_metrics=['systole', 'diastole', 'mitteldruck']):
    # add columns for normalised values
    for bp_metric in bp_metrics:
        bp_df[f'{bp_metric}_normalised'] = np.nan
    # for every patient, normalise to prior median
    for patient_id in tqdm(bp_df['pNr'].unique()):
        patient_bp_df = bp_df[bp_df['pNr'] == patient_id]
        for index, row in patient_bp_df.iterrows():
            time_of_bd = row['timeBd']
            prior_bp_df = patient_bp_df[
                patient_bp_df['timeBd'] < time_of_bd]

            prior_medians = {}
            for bp_metric in bp_metrics:
                prior_medians[bp_metric] = prior_bp_df[f'{bp_metric}'].median()

                if (np.isnan(prior_medians[bp_metric])) & (not np.isnan(row[bp_metric])):
                    bp_df.loc[(bp_df['pNr'] == patient_id) & (
                            bp_df['timeBd'] == time_of_bd), f'{bp_metric}_normalised'] = 1
                else:
                    bp_df.loc[(bp_df['pNr'] == patient_id) & (bp_df[
                            'timeBd'] == time_of_bd), f'{bp_metric}_normalised'] = normalisation(
                                                                                    row[bp_metric], prior_medians[bp_metric])


    return bp_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    args = parser.parse_args()

    bp_df = pd.read_csv(args.input_file)
    normalised_bp_df = parallel_normalise(bp_df)
    normalised_bp_df.to_csv(args.input_file.replace('.csv', '_normalised.csv'), index=False)