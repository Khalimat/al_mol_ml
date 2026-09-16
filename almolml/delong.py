"""DeLong's method for the variance of a ROC AUC estimate, used to report a
confidence interval alongside every AUC this pipeline computes.

Reference: Sun, X., and Xu, W. "Fast Implementation of DeLong's Algorithm
for Comparing the Areas Under Correlated Receiver Operating Characteristic
Curves." IEEE Signal Processing Letters 21.11 (2014): 1389-1393.
"""

import numpy as np
from scipy import stats


def compute_midrank(x):
    """Midranks of `x`, used by the DeLong AUC-variance formula."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1)
        i = j
    ranks = np.empty(n, dtype=float)
    # +1 is due to Python using 0-based indexing instead of 1-based in the
    # AUC formula in the paper.
    ranks[order] = midranks + 1
    return ranks


def fast_delong(predictions_sorted_transposed, label_1_count):
    """AUC and its DeLong covariance for one or more classifiers.

    Parameters
    ----------
    predictions_sorted_transposed: 2D array [n_classifiers, n_examples],
        with the label "1" examples sorted first
    label_1_count: number of label "1" examples

    Returns
    -------
    (aucs, delong_covariance)
    """
    # Short variable names match the paper's notation.
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    delong_covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, delong_covariance


def _sort_by_label(ground_truth):
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """AUC and its DeLong variance for a single set of predictions.

    Parameters
    ----------
    ground_truth: np.array of 0 and 1
    predictions: np.array of floats, the predicted probability of class 1
    """
    order, label_1_count = _sort_by_label(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delong_covariance = fast_delong(predictions_sorted_transposed, label_1_count)
    return aucs[0], delong_covariance


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    """AUC and its (lower, upper) confidence bound at level `alpha`."""
    if len(np.unique(y_true)) < 2:
        # AUC is undefined without both classes present, which happens for
        # the deliberately tiny "almost no validation" splits used by some
        # splitters. Report NaN; callers already treat NaN as a degenerate
        # metric (see ActiveLearningModel.run's np.isnan handling).
        return np.nan, np.array([np.nan, np.nan])
    auc, auc_variance = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_variance)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    return auc, ci
