"""A minimal, pluggable pool-based active learner.

This replaces the modAL-python dependency: modAL's stock ActiveLearner
retrains its estimator with `self.estimator.fit(X, y, **fit_kwargs)`, which
works perfectly well for a plain scikit-learn Pipeline (scaler, then
classifier) -- no monkey-patching of a third-party library needed for the
handful of lines actually used here.

See query_strategies.py for the acquisition functions (entropy, BALD,
Core-Set, DIRECT) that decide which pool point to label next.
"""

import numpy as np

from .query_strategies import entropy_query


class ActiveLearner:
    """Wraps an sklearn-compatible estimator (fit/predict/predict_proba) in
    an active-learning loop: query() asks `query_strategy` which pool
    sample is most informative, teach() adds it to the training set and
    retrains on the full accumulated set.
    """

    def __init__(self, estimator, X_initial, y_initial, query_strategy=entropy_query):
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.X_training = X_initial
        self.y_training = y_initial
        self.estimator.fit(self.X_training, self.y_training)

    def query(self, X_pool):
        return self.query_strategy(
            self.estimator, X_pool, X_training=self.X_training, y_training=self.y_training
        )

    def teach(self, X_new, y_new):
        self.X_training = np.append(self.X_training, X_new, axis=0)
        self.y_training = np.append(self.y_training, y_new)
        self.estimator.fit(self.X_training, self.y_training)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
