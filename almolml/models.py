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
    function decides the next point(s) to label (see query_strategies.py).

    `batch_size` controls how many pool points are queried and taught per
    round: 1 (default) reproduces the original point-at-a-time loop; > 1
    requires a batch-capable `query_strategy` (batch_bald_query,
    batch_core_set_query, or batch_direct_query), and retrains the model
    once per batch instead of once per point.
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
        batch_size=1,
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
        self.batch_size = batch_size
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

    def _write_trace_csvs(self, test_scores, validation_scores, n_labeled_history):
        """Write the round-by-round trace so far. Called every
        CHECKPOINT_EVERY newly-labeled points (not just at the end) so a
        long run's progress can be inspected from the filesystem while
        it's still going, not just once it finishes.

        `n_labeled_history` (one entry per row, cumulative training-set
        size at that point) matters once `batch_size > 1`: each row is then
        a whole batch, not a single query, so the row index alone no
        longer says how many points have been labeled.
        """
        test_df = pd.DataFrame(self._zero_out_nans(test_scores), columns=perf_columns)
        test_df.insert(0, "n_labeled", n_labeled_history)
        test_df.to_csv(self.results_dir / "test_initial_stats.csv")

        validation_df = pd.DataFrame(
            self._zero_out_nans(validation_scores), columns=perf_columns
        )
        validation_df.insert(0, "n_labeled", n_labeled_history)
        validation_df.to_csv(self.results_dir / "validation_initial_stats.csv")

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
        n_labeled_history = [len(X_initial)]

        remaining = self.n_queries - 1
        labeled_since_checkpoint = 0
        while remaining > 0 and len(X_pool) > 0:
            this_batch = min(self.batch_size, remaining, len(X_pool))
            query_idx = learner.query(X_pool, batch_size=this_batch)
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            X_pool, y_pool = (
                np.delete(X_pool, query_idx, axis=0),
                np.delete(y_pool, query_idx, axis=0),
            )

            test_scores.append(self._score(learner, self.X_test, self.Y_test))
            validation_scores.append(
                self._score(learner, self.X_validation, self.Y_validation)
            )
            n_labeled_history.append(len(learner.X_training))

            remaining -= this_batch
            labeled_since_checkpoint += this_batch
            if labeled_since_checkpoint >= self.CHECKPOINT_EVERY:
                self._write_trace_csvs(test_scores, validation_scores, n_labeled_history)
                labeled_since_checkpoint = 0

        self._write_trace_csvs(test_scores, validation_scores, n_labeled_history)
        test_scores = self._zero_out_nans(test_scores)
        validation_scores = self._zero_out_nans(validation_scores)
        self.test_performance = self._final_row(test_scores, "test performance")
        self.validation_performance = self._final_row(
            validation_scores, "validation performance"
        )
