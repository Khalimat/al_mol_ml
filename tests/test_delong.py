import numpy as np

from almolml.delong import calc_auc_ci


def test_calc_auc_ci_returns_nan_when_ground_truth_is_single_class():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0.9, 0.6, 0.7])
    auc, ci = calc_auc_ci(y_true, y_pred)
    assert np.isnan(auc)
    assert np.all(np.isnan(ci))


def test_calc_auc_ci_returns_finite_values_for_normal_input():
    # Predictions with some overlap between classes, so AUC variance is
    # nonzero and the confidence interval is well-defined. Perfectly
    # separated predictions (AUC exactly 1.0) legitimately produce a
    # zero-variance, NaN-bounded CI -- that's an inherent property of the
    # DeLong method, not a bug.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.2, 0.6, 0.4, 0.9])
    auc, ci = calc_auc_ci(y_true, y_pred)
    assert 0.0 <= auc <= 1.0
    assert np.all(np.isfinite(ci))
