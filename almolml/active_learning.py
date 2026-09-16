"""A minimal, pluggable pool-based active learner.

This replaces the modAL-python dependency: modAL's stock ActiveLearner
retrains its estimator with `self.estimator.fit(X, y, **fit_kwargs)`, which
works perfectly well for a plain scikit-learn Pipeline (scaler, then
classifier) -- no monkey-patching of a third-party library needed for the
handful of lines actually used here.

See query_strategies.py for the acquisition functions (entropy, BALD,
Core-Set, DIRECT, and batch versions of BALD/Core-Set/DIRECT) that decide
which pool point(s) to label next.
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

    def query(self, X_pool, batch_size=1):
        """Ask `query_strategy` which pool row(s) to label next. Passes
        `batch_size` through only when it's not the default 1, so a
        single-point strategy (whose signature doesn't accept batch_size)
        keeps working unchanged when nothing asks it to batch.
        """
        kwargs = dict(X_training=self.X_training, y_training=self.y_training)
        if batch_size != 1:
            kwargs["batch_size"] = batch_size
        return self.query_strategy(self.estimator, X_pool, **kwargs)

    def teach(self, X_new, y_new):
        self.X_training = np.append(self.X_training, X_new, axis=0)
        self.y_training = np.append(self.y_training, y_new)
        self.estimator.fit(self.X_training, self.y_training)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
