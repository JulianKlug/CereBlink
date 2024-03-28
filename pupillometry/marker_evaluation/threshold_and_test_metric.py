import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from testing_utils import test_predictor, youdens_index, bootstrapped_roc_auc


def test_pupillometry_metrics(df, aim_for_sp80 = False):
    """
    Test pupillometry metrics through cross-validation
    - For each metric, calculate the Youden's index on the training set, and test the performance on the test set
    :param df: DataFrame with pupillometry metrics (label + metrics)
    :return: DataFrame with results
    """
    id_columns = ['Unnamed: 0', 'pNr', 'Name', 'Date_birth', 'label', 'timebin_end']
    # all other columns are features
    feature_columns = [col for col in df.columns if col not in id_columns]

    results_df = pd.DataFrame()

    for metric in tqdm(feature_columns):
        metric_results_df = pd.DataFrame()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(df, df['label']):
            train_df = df.iloc[train_index]
            train_df.dropna(subset=[metric], inplace=True)

            test_df = df.iloc[test_index]
            test_df.dropna(subset=[metric], inplace=True)

            train_n_pos = train_df['label'].sum()
            test_n_pos = test_df['label'].sum()

            if train_df[train_df['label'] == 1][metric].median() > train_df[train_df['label'] == 0][metric].median():
                youdens = youdens_index(train_df['label'], train_df[metric])
            else:
                youdens = -1 * youdens_index(train_df['label'], -1 * train_df[metric])

            # find threshold clostest to 80% specificity (FPR = 20%) in training set
            fpr, tpr, thresholds = roc_curve(train_df['label'], train_df[metric])
            sp80_fpr_idx = np.abs(fpr - 0.2).argmin()
            sp80_threshold = thresholds[sp80_fpr_idx]

            if aim_for_sp80:
                threshold = sp80_threshold
            else:
                threshold = youdens


            # check direction of comparison (to know which if should be thresholded above or below)
            # median of label 1 > median of label 0 -> threshold above; else threshold below
            if train_df[train_df['label'] == 1][metric].median() > train_df[train_df['label'] == 0][metric].median():
                y_pred_binary = test_df[metric] > threshold
            else:
                y_pred_binary = test_df[metric] <= threshold
            y_pred_binary = y_pred_binary.astype(int)

            test_fold_results = test_predictor(test_df['label'], y_pred_binary)
            train_fold_results = test_predictor(train_df['label'], train_df[metric] > threshold)
            fold_test_roc_auc = roc_auc_score(test_df['label'], test_df[metric])
            fold_train_roc_auc = roc_auc_score(train_df['label'], train_df[metric])
            if fold_test_roc_auc < 0.5:
                fold_test_roc_auc = 1 - fold_test_roc_auc
            if fold_train_roc_auc < 0.5:
                fold_train_roc_auc = 1 - fold_train_roc_auc


            fold_results = test_fold_results
            # prepend test_ to all keys
            fold_results = {f'test_{k}': v for k, v in fold_results.items()}

            train_fold_results = {f'train_{k}': v for k, v in train_fold_results.items()}
            fold_results.update(train_fold_results)

            fold_results['youdens'] = youdens
            fold_results['threshold'] = threshold
            fold_results['test_roc_auc'] = fold_test_roc_auc
            fold_results['train_roc_auc'] = fold_train_roc_auc
            fold_results['test_n_pos'] = test_n_pos
            fold_results['test_n_neg'] = len(test_df) - test_n_pos
            fold_results['train_n_pos'] = train_n_pos
            fold_results['train_n_neg'] = len(train_df) - train_n_pos
            fold_results['fold'] = len(metric_results_df)

            metric_results_df = pd.concat([metric_results_df, pd.DataFrame(fold_results, index=[0])])

        overall_roc_auc = roc_auc_score(df.dropna(subset=[metric])['label'], df.dropna(subset=[metric])[metric])
        if overall_roc_auc < 0.5:
            overall_roc_auc = 1 - overall_roc_auc
        metric_results_df['overall_roc_auc'] = overall_roc_auc

        bs_median_roc_auc, bs_lower_ci_roc_auc, bs_upper_ci_roc_auc = bootstrapped_roc_auc(df.dropna(subset=[metric])['label'], df.dropna(subset=[metric])[metric])
        if bs_median_roc_auc < 0.5:
            bs_median_roc_auc = 1 - bs_median_roc_auc
            bs_lower_ci_roc_auc = 1 - bs_lower_ci_roc_auc
            bs_upper_ci_roc_auc = 1 - bs_upper_ci_roc_auc

        metric_results_df['bs_median_roc_auc'] = bs_median_roc_auc
        metric_results_df['bs_lower_ci_roc_auc'] = bs_lower_ci_roc_auc
        metric_results_df['bs_upper_ci_roc_auc'] = bs_upper_ci_roc_auc

        metric_results_df['metric'] = metric
        metric_results_df.reset_index(drop=True, inplace=True)

        results_df = pd.concat([results_df, metric_results_df])

    results_df.reset_index(drop=True, inplace=True)
    return results_df


def test_all_timebins(data_dir:str, aim_for_sp80 = False):
    data_filenames = [f for f in os.listdir(data_dir) if
                      f.endswith('.csv') and 'timebin' in f and 'reassembled_pupillometry' in f]

    results_df = pd.DataFrame()
    for data_filename in data_filenames:
        # find timebin size with regex identifying pattern : _xh_
        timebin_size = int(re.search(r'_(\d+)h_', data_filename).group(1))
        data_is_normalized = int(('normalized' in data_filename) or ('normalised' in data_filename))
        outcome = '_'.join(data_filename.split('_')[0:2])

        df = pd.read_csv(os.path.join(data_dir, data_filename))
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        timebin_results_df = test_pupillometry_metrics(df, aim_for_sp80=aim_for_sp80)
        timebin_results_df['timebin_size'] = timebin_size
        timebin_results_df['data_is_normalized'] = data_is_normalized
        timebin_results_df['outcome'] = outcome
        results_df = pd.concat([results_df, timebin_results_df])

    results_df.reset_index(drop=True, inplace=True)

    return results_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-sp', '--aim_for_sp80', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    results_df = test_all_timebins(data_dir, aim_for_sp80=args.aim_for_sp80)
    results_df.to_csv(os.path.join(output_dir, 'pupillometry_metrics_results.csv'), index=False)
