import numpy as np

from jcim.validation import Validation


class _StubModel:
    """Minimal predict/predict_proba stub, avoiding a real classifier fit."""

    def __init__(self, predictions):
        self._predictions = np.asarray(predictions)

    def predict(self, X):
        return self._predictions[: len(X)]

    def predict_proba(self, X):
        p1 = self._predictions[: len(X)].astype(float)
        return np.column_stack([1 - p1, p1])


def test_validation_handles_balanced_two_class_input():
    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])
    model = _StubModel([0, 0, 1, 1])
    results = Validation(model, X, y, "Test").results
    row = results.iloc[0]
    assert row["accuracy"] == 1.0
    assert row["F1"] == 1.0
    assert np.isfinite(row["AUC"])


def test_validation_does_not_crash_on_single_sample_single_class():
    # This is the "almost no validation" edge case: a 1-row validation set
    # is always single-class, which used to crash confusion_matrix().ravel()
    # (expected 4 values, got 1) and the DeLong AUC assert.
    X = np.zeros((1, 1))
    y = np.array([1])
    model = _StubModel([1])
    results = Validation(model, X, y, "Validation").results
    row = results.iloc[0]
    assert np.isnan(row["AUC"])
    assert row["accuracy"] == 1.0


def test_validation_handles_wrong_single_class_prediction():
    X = np.zeros((1, 1))
    y = np.array([1])
    model = _StubModel([0])
    results = Validation(model, X, y, "Validation").results
    row = results.iloc[0]
    assert row["accuracy"] == 0.0
