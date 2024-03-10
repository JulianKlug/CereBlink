import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix, brier_score_loss, confusion_matrix
import numpy as np

def bootstrapped_roc_auc(y, y_pred, n_bootstraps=1000, random_seed=42):
    '''Calculate bootstrapped ROC AUC

    Parameters
    ----------

    y : array, shape = [n_samples]
        True binary labels.

    y_pred : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    n_bootstraps : int, default=1000
        Number of bootstraps.

    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------

    result_dict : dict
        Dictionary with keys 'auc_test', 'auc_test_lower_ci', 'auc_test_upper_ci'.
    '''
    np.random.seed(random_seed)

    if type(y_pred) == pd.Series:
        y_pred = y_pred.values
    if type(y) == pd.Series:
        y = y.values

    bootstrapped_auc = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(y)), len(y), replace=True)
        y_bs = y[indices]
        y_pred_bs = y_pred[indices]

        if not np.all(y_bs == 0) and not np.all(y_bs == 1):
            roc_auc_bs = roc_auc_score(y_bs, y_pred_bs)
        else:
            roc_auc_bs = np.nan
        bootstrapped_auc.append(roc_auc_bs)

    # get medians
    median_roc_auc = np.nanpercentile(bootstrapped_auc, 50)

    # get 95% interval
    alpha = 100 - 95
    lower_ci_roc_auc = np.nanpercentile(bootstrapped_auc, alpha / 2)
    upper_ci_roc_auc = np.nanpercentile(bootstrapped_auc, 100 - alpha / 2)

    return (median_roc_auc, lower_ci_roc_auc, upper_ci_roc_auc)


def test_predictor(y, y_pred_binary):
    results_dict = {}

    results_dict['matthews'] = matthews_corrcoef(y, y_pred_binary)

    results_dict['accuracy'] = accuracy_score(y, y_pred_binary)

    precision = precision_score(y, y_pred_binary)  # == PPV
    results_dict['precision'] = precision

    recall = recall_score(y, y_pred_binary)  # == sensitivity
    results_dict['recall'] = recall

    mcm = confusion_matrix(y, y_pred_binary)
    tn = mcm[0, 0]
    tp = mcm[1, 1]
    fn = mcm[1, 0]
    fp = mcm[0, 1]
    specificity = tn / (tn + fp)
    results_dict['specificity'] = specificity
    neg_pred_value = tn / (tn + fn)
    results_dict['neg_pred_value'] = neg_pred_value

    return results_dict


def youdens_index(y_true, y_score):
    '''Find data-driven cut-off for classification

    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    References
    ----------

    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.

    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.

    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]