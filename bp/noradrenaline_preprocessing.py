from tqdm import tqdm
import pandas as pd


def annotate_concomitant_noradrenaline(bp_df, nor_df)->pd.DataFrame:
    """
    Annotate concomitant noradrenaline data to blood pressure data
    :param bp_df:
    :param nor_df:
    :return:
    """

    # only accept noradrenaline data with Einheit 'MICROGRAM' and 'MILLIGRAM'
    nor_df = nor_df[nor_df['Einheit'].isin(['MICROGRAM', 'MILLIGRAM'])]
    # convert Menge to MICROGRAM where Einheit is MILLIGRAM
    nor_df.loc[nor_df['Einheit'] == 'MILLIGRAM', 'Menge'] = nor_df['Menge'] * 1000
    nor_df = nor_df.drop(columns=['Einheit'])

    # filter out rows where Menge or Dauer is 0 or NaN
    nor_df = nor_df[(nor_df['Menge'] != 0) & (nor_df['Dauer'] != 0)]
    nor_df = nor_df.dropna(subset=['Menge', 'Dauer'])

    for index, row in tqdm(nor_df.iterrows(), total=nor_df.shape[0]):
        pNr = row['pNr']
        if pNr not in bp_df['pNr'].unique():
            continue

        start = row['Start']
        end = row['Ende']
        bp_df.loc[((bp_df['pNr'] == pNr) & (bp_df['timeBd'] >= start) & (
                    bp_df['timeBd'] <= end)), 'noradrenaline_concomitant'] = 1
        bp_df.loc[((bp_df['pNr'] == pNr) & (bp_df['timeBd'] >= start) & (
                    bp_df['timeBd'] <= end)), 'noradrenaline_dose_mcg_min'] = row['Menge'] / row['Dauer']

    bp_df['noradrenaline_concomitant'] = bp_df['noradrenaline_concomitant'].fillna(0)

    return bp_df


def filter_out_concomitant_noradrenaline(annotated_timebin_df, verbose=False):
    """
    Filter out rows where noradrenaline is concomitant
    :param annotated_timebin_df:
    :return:
    """

    if 'noradrenaline_concomitant' not in annotated_timebin_df.columns:
        raise ValueError('noradrenaline_concomitant column not found in the dataframe')
        
        
    n_patients_with_concomitant_noradrenaline = annotated_timebin_df[annotated_timebin_df['noradrenaline_concomitant'] == 1][
        'pNr'].nunique()
    n_measures_with_concomitant_noradrenaline = annotated_timebin_df[annotated_timebin_df['noradrenaline_concomitant'] == 1].shape[0]
    n_pos_measures_with_concomitant_noradrenaline = \
    annotated_timebin_df[(annotated_timebin_df['noradrenaline_concomitant'] == 1) & (annotated_timebin_df['within_event_timebin'] == 1)].shape[0]
    n_neg_measures_with_concomitant_noradrenaline = \
    annotated_timebin_df[(annotated_timebin_df['noradrenaline_concomitant'] == 1) & (annotated_timebin_df['within_event_timebin'] == 0)].shape[0]

    # percentages
    f_pos_measures_with_concomitant_noradrenaline = n_pos_measures_with_concomitant_noradrenaline / annotated_timebin_df[
        annotated_timebin_df["within_event_timebin"] == 1].shape[0]
    f_neg_measures_with_concomitant_noradrenaline = n_neg_measures_with_concomitant_noradrenaline / annotated_timebin_df[
        annotated_timebin_df["within_event_timebin"] == 0].shape[0]

    if verbose:
        print(
            f'Percentage of patients with concomitant noradrenaline: {n_patients_with_concomitant_noradrenaline / annotated_timebin_df["pNr"].nunique()}')
        print(
            f'Percentage of measures with concomitant noradrenaline: {n_measures_with_concomitant_noradrenaline / annotated_timebin_df.shape[0]}')
        print(
            f'Percentage of positive measures with concomitant noradrenaline: {n_pos_measures_with_concomitant_noradrenaline / annotated_timebin_df[annotated_timebin_df["within_event_timebin"] == 1].shape[0]}')
        print(
            f'Percentage of negative measures with concomitant noradrenaline: {n_neg_measures_with_concomitant_noradrenaline / annotated_timebin_df[annotated_timebin_df["within_event_timebin"] == 0].shape[0]}')

    annotated_timebin_df = annotated_timebin_df[annotated_timebin_df['noradrenaline_concomitant'] != 1]

    log_df = pd.DataFrame({
        'n_patients_with_concomitant_noradrenaline': [n_patients_with_concomitant_noradrenaline],
        'n_measures_with_concomitant_noradrenaline': [n_measures_with_concomitant_noradrenaline],
        'n_pos_measures_with_concomitant_noradrenaline': [n_pos_measures_with_concomitant_noradrenaline],
        'n_neg_measures_with_concomitant_noradrenaline': [n_neg_measures_with_concomitant_noradrenaline],
        'f_pos_measures_with_concomitant_noradrenaline': [f_pos_measures_with_concomitant_noradrenaline],
        'f_neg_measures_with_concomitant_noradrenaline': [f_neg_measures_with_concomitant_noradrenaline]
    })

    return annotated_timebin_df, log_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bp', type=str, help='Path to blood pressure data')
    parser.add_argument('--nor', type=str, help='Path to noradrenaline data')
    args = parser.parse_args()

    bp_df = pd.read_csv(args.bp)
    nor_df = pd.read_csv(args.nor, sep=';', decimal='.')

    annotated_bp_df = annotate_concomitant_noradrenaline(bp_df, nor_df)

    annotated_bp_df.to_csv(args.bp.replace('.csv', '_nor_annotated.csv'), index=False)


