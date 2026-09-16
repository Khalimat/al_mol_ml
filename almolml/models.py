import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .active_learning import ActiveLearner
from .query_strategies import entropy_query
from .utilities import perf_columns
from .validation import Validation


def seed_everything(seed: int) -> None:
    """Seed numpy and torch's global RNGs.

    Call this once per run (e.g. once per pipeline iteration) rather than
    once at import time: a single import-time seed makes every run's first
    iteration identical regardless of how many times the process is
    restarted, and gives later iterations a RNG state that depends on
    incidental prior random calls rather than on a controlled, reproducible
    seed of their own.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


class Model:
    """Template for a classifier trained on (X_train, Y_train) and scored on
    both a held-out test set and a validation set. Subclasses set
    `self.model` (anything with fit/predict/predict_proba) and implement
    `train()`.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.model = None
        self.test_performance = None
        self.validation_performance = None

    def run(self):
        self.train()
        self.test_performance = Validation(
            self.model, self.X_test, self.Y_test, "Test"
        ).results
        self.validation_performance = Validation(
            self.model, self.X_validation, self.Y_validation, "Validation"
        ).results

    def train(self):
        raise NotImplementedError


class DeepSCAMsModel(Model):
    """A scikit-learn MLP, matching the hyperparameters of the original
    DeepSCAMs paper.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
        super().__init__(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)
        self.model = Pipeline(
            [("scaler", MinMaxScaler()), ("mlp", self._build_classifier())]
        )
        self.run()

    @staticmethod
    def _build_classifier():
        return MLPClassifier(
            activation="tanh",
            alpha=0.0001,
            batch_size="auto",
            beta_1=0.9,
            beta_2=0.999,
            early_stopping=False,
            epsilon=1e-08,
            hidden_layer_sizes=(100, 1000, 1000),
            learning_rate="constant",
            learning_rate_init=0.001,
            max_iter=200,
            momentum=0.9,
            n_iter_no_change=10,
            nesterovs_momentum=True,
            power_t=0.5,
            random_state=1234,
            shuffle=True,
            solver="sgd",
            tol=0.0001,
            validation_fraction=0.1,
            verbose=False,
            warm_start=False,
        )

    def train(self):
        self.model.fit(self.X_train, self.Y_train.T)


class SCAMsNet(nn.Module):
    """MLP architecture originally defined as a Keras Sequential model.

    Outputs a single logit (no sigmoid) since it is paired with
    NeuralNetBinaryClassifier, which applies the sigmoid internally.
    `features`/`head` are split apart (rather than one Sequential) so that
    `embed()` -- the penultimate-layer representation -- is available for
    embedding-space active-learning strategies (see query_strategies.py).
    """

    def __init__(self, shape=2255, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(shape, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(10, 1)

    def embed(self, X):
        return self.features(X.float())

    def forward(self, X):
        return self.head(self.embed(X)).squeeze(-1)


class TorchBinaryClassifier(NeuralNetBinaryClassifier):
    """NeuralNetBinaryClassifier that tolerates integer 0/1 labels.

    BCEWithLogitsLoss requires float targets, but labels flow in here as
    plain int arrays from pandas/rdkit, including through the
    active-learning teach()/query() loop where we don't control the dtype.
    """

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        return super().get_loss(y_pred, y_true, *args, **kwargs)


def make_mlp_pipeline(shape, epochs=50):
    """A (scaler, neural net) pipeline: the shared model architecture used
    by both TorchMLPModel and ActiveLearningModel.
    """
    classifier = TorchBinaryClassifier(
        module=SCAMsNet,
        module__shape=shape,
        max_epochs=epochs,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
    )
    return Pipeline(
        [("scaler", RobustScaler(quantile_range=(25, 75))), ("mlp", classifier)]
    )


class TorchMLPModel(Model):
    """The neural-network model (originally TensorFlow/Keras, now PyTorch
    via skorch) trained without active learning.
    """

    def __init__(
        self, X_train, Y_train, X_test, Y_test, X_validation, Y_validation, epochs=50
    ):
        super().__init__(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)
        self.epochs = epochs
        self.model = make_mlp_pipeline(X_train.shape[1], epochs=epochs)
        self.run()

    def train(self):
        self.model.fit(self.X_train, self.Y_train.T)


class ActiveLearningModel:
    """Trains the same neural-network architecture as TorchMLPModel, but
    picks its training examples via active learning instead of using the
    whole training set up front. `query_strategy` selects which acquisition
    function decides the next point to label (see query_strategies.py).
    """

    def __init__(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        X_validation,
        Y_validation,
        n_queries,
        results_dir,
        n_initial=10,
        epochs=50,
        query_strategy=entropy_query,
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.n_initial = n_initial
        self.n_queries = n_queries
        self.epochs = epochs
        self.query_strategy = query_strategy
        self.results_dir = results_dir
        self.model = make_mlp_pipeline(X_train.shape[1], epochs=epochs)
        self.test_performance = None
        self.validation_performance = None
        self.run()

    @staticmethod
    def _initial_sample(X_train, Y_train, n_initial):
        """Pick `n_initial` random training rows, retrying until both
        classes are represented (the active learner needs both to start).
        """
        while True:
            idx = np.random.choice(len(X_train), size=n_initial, replace=False)
            if len(set(Y_train[idx])) == 2:
                return idx

    @staticmethod
    def _score(learner, X, Y):
        return Validation(learner, X, Y, "").results.iloc[0].tolist()

    @staticmethod
    def _zero_out_nans(scores):
        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        return scores

    @staticmethod
    def _final_row(scores, label):
        """The last query step's scores, i.e. the model actually left
        trained at the end of the loop -- not the best of the whole
        trajectory, which would be a look-elsewhere-biased estimate (the
        network is retrained from scratch every step, so each step's score
        is a noisy, independent draw).
        """
        return pd.DataFrame([scores[-1]], index=[label], columns=perf_columns)

    def _write_trace_csvs(self, test_scores, validation_scores):
        """Write the query-by-query trace so far. Called every
        CHECKPOINT_EVERY queries (not just at the end) so a long run's
        progress can be inspected from the filesystem while it's still
        going, not just once it finishes.
        """
        pd.DataFrame(self._zero_out_nans(test_scores), columns=perf_columns).to_csv(
            self.results_dir / "test_initial_stats.csv"
        )
        pd.DataFrame(
            self._zero_out_nans(validation_scores), columns=perf_columns
        ).to_csv(self.results_dir / "validation_initial_stats.csv")

    CHECKPOINT_EVERY = 200

    def run(self):
        initial_idx = self._initial_sample(self.X_train, self.Y_train, self.n_initial)
        X_initial, y_initial = self.X_train[initial_idx], self.Y_train[initial_idx]
        X_pool, y_pool = (
            np.delete(self.X_train, initial_idx, axis=0),
            np.delete(self.Y_train, initial_idx, axis=0),
        )
        learner = ActiveLearner(
            estimator=self.model,
            X_initial=X_initial,
            y_initial=y_initial,
            query_strategy=self.query_strategy,
        )

        test_scores = [self._score(learner, self.X_test, self.Y_test)]
        validation_scores = [self._score(learner, self.X_validation, self.Y_validation)]

        for i in range(self.n_queries - 1):
            query_idx = learner.query(X_pool)
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            X_pool, y_pool = (
                np.delete(X_pool, query_idx, axis=0),
                np.delete(y_pool, query_idx, axis=0),
            )

            test_scores.append(self._score(learner, self.X_test, self.Y_test))
            validation_scores.append(
                self._score(learner, self.X_validation, self.Y_validation)
            )

            if (i + 1) % self.CHECKPOINT_EVERY == 0:
                self._write_trace_csvs(test_scores, validation_scores)

        self._write_trace_csvs(test_scores, validation_scores)
        test_scores = self._zero_out_nans(test_scores)
        validation_scores = self._zero_out_nans(validation_scores)
        self.test_performance = self._final_row(test_scores, "test performance")
        self.validation_performance = self._final_row(
            validation_scores, "validation performance"
        )
