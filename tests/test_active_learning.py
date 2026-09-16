import numpy as np

from jcim.active_learning import ActiveLearner


class _StubEstimator:
    """Fakes predict_proba with a fixed table, and records fit() calls so
    tests can check what the learner actually retrains on.
    """

    def __init__(self, proba_table):
        self.proba_table = proba_table
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((np.array(X).copy(), np.array(y).copy()))
        return self

    def predict_proba(self, X):
        return self.proba_table[: len(X)]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def test_active_learner_fits_on_initial_data():
    estimator = _StubEstimator(np.array([[0.5, 0.5]]))
    X_initial = np.array([[0.0], [1.0]])
    y_initial = np.array([0, 1])

    ActiveLearner(estimator, X_initial, y_initial)

    assert len(estimator.fit_calls) == 1
    fit_X, fit_y = estimator.fit_calls[0]
    np.testing.assert_array_equal(fit_X, X_initial)
    np.testing.assert_array_equal(fit_y, y_initial)


def test_active_learner_teach_retrains_on_full_accumulated_set():
    estimator = _StubEstimator(np.array([[0.5, 0.5]]))
    learner = ActiveLearner(
        estimator, X_initial=np.array([[0.0], [1.0]]), y_initial=np.array([0, 1])
    )

    learner.teach(X_new=np.array([[2.0]]), y_new=np.array([1]))

    assert len(estimator.fit_calls) == 2
    fit_X, fit_y = estimator.fit_calls[-1]
    np.testing.assert_array_equal(fit_X, np.array([[0.0], [1.0], [2.0]]))
    np.testing.assert_array_equal(fit_y, np.array([0, 1, 1]))


def test_active_learner_query_passes_current_training_set_to_strategy():
    estimator = _StubEstimator(np.array([[0.5, 0.5]] * 3))
    learner = ActiveLearner(
        estimator, X_initial=np.array([[0.0], [1.0]]), y_initial=np.array([0, 1])
    )

    seen = {}

    def spy_strategy(estimator, X_pool, X_training=None, y_training=None):
        seen["X_training"] = X_training
        seen["y_training"] = y_training
        return np.array([0])

    learner.query_strategy = spy_strategy
    learner.query(X_pool=np.zeros((3, 1)))

    np.testing.assert_array_equal(seen["X_training"], learner.X_training)
    np.testing.assert_array_equal(seen["y_training"], learner.y_training)
