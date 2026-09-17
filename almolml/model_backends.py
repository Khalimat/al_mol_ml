"""Architecture backends: a uniform interface that lets the active learner
(active_learning.py) and the query strategies (query_strategies.py) work
identically regardless of which architecture is plugged in.

Every backend implements:
    fit(X, y)
    predict(X) -> (n,) hard class labels
    predict_proba(X) -> (n, 2)
    mc_probabilities(X, n_samples) -> (n_samples, n), P(class=1) per sample
    embed(X) -> (n, d)

`mc_probabilities` is what bald_query/batch_bald_query (query_strategies.py)
need: `n_samples` independent-ish probability estimates per pool point, so
BALD's predictive-vs-expected-entropy gap can be computed the same way no
matter what produced the samples. The two neural-net backends get this from
MC-Dropout (stochastic forward passes, dropout left on at inference); the
Random Forest backend gets it from per-tree predictions instead -- a
different mechanism for the same idea (epistemic uncertainty as
disagreement across an ensemble of hypotheses: dropout masks vs. bagged
trees).

`embed` is what core_set_query/batch_core_set_query need: a representation
space for the greedy k-center diversity step. The two neural-net backends
use their learned penultimate-layer representation; the Random Forest uses
each row's per-tree leaf index as a categorical code -- the standard
random-forest "proximity" representation (Breiman), used here as a
model-derived embedding in place of a learned one.

This module intentionally has no import of `almolml.models`: the higher-
level Model/ActiveLearningModel classes (models.py) import backends from
here, not the other way around.
"""

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetBinaryClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .graph_featurization import ATOM_FEATURE_DIM

try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GINConv, global_add_pool
except ImportError:  # pragma: no cover - exercised only without the 'graph' extra
    Batch = None


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


# --------------------------------------------------------------------------
# MLP-on-fingerprint backend (the original architecture)
# --------------------------------------------------------------------------


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
    """A (scaler, neural net) pipeline: the architecture used by
    TorchMLPBackend.
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


class TorchMLPBackend:
    """Wraps the (scaler, torch MLP) pipeline behind the shared backend
    interface. Behavior-identical to the code that, pre-refactor, lived
    inline in models.py/query_strategies.py -- this is a pure interface
    extraction, not a change to the MLP architecture or training loop.
    """

    def __init__(self, shape, epochs=50):
        self.pipeline = make_mlp_pipeline(shape, epochs=epochs)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def mc_probabilities(self, X, n_samples):
        classifier = self.pipeline.named_steps["mlp"]
        X_scaled = self.pipeline.named_steps["scaler"].transform(X).astype(np.float32)
        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)

        module = classifier.module_
        module.train()  # keep dropout active even though we're not training
        with torch.no_grad():
            probs = [torch.sigmoid(module(X_tensor)).numpy() for _ in range(n_samples)]
        module.eval()
        return np.stack(probs, axis=0)

    def embed(self, X):
        classifier = self.pipeline.named_steps["mlp"]
        X_scaled = self.pipeline.named_steps["scaler"].transform(X).astype(np.float32)
        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)

        module = classifier.module_
        module.eval()
        with torch.no_grad():
            return module.embed(X_tensor).numpy()


# --------------------------------------------------------------------------
# Random Forest backend
# --------------------------------------------------------------------------


class RandomForestBackend:
    """Random Forest on the same fixed-length fingerprint+descriptor
    features the MLP backend uses -- no new featurization needed. See
    module docstring for how `mc_probabilities`/`embed` are defined for a
    tree ensemble.
    """

    def __init__(self, n_estimators=300, random_state=None):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def fit(self, X, y):
        # A fresh RandomForestClassifier every call (warm_start defaults to
        # False), matching the active-learning loop's "retrain from scratch
        # on the accumulated labeled set" contract -- not a warm-started
        # update of the previous round's forest.
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.model.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def _tree_p1(tree, X):
        """P(class=1) from one tree, robust to a bootstrap sample that
        happened to contain only one class (common early in the active-
        learning loop, when the labeled set is tiny and imbalanced) -- in
        that case the tree's own `classes_` is a single value, and
        `predict_proba` returns one column, not two.
        """
        proba = tree.predict_proba(X)
        classes = list(tree.classes_)
        if 1 not in classes:
            return np.zeros(len(X))
        return proba[:, classes.index(1)]

    def mc_probabilities(self, X, n_samples):
        trees = self.model.estimators_[: min(n_samples, len(self.model.estimators_))]
        return np.stack([self._tree_p1(tree, X) for tree in trees], axis=0)

    def embed(self, X):
        # Per-row, per-tree leaf index: rows landing in the same leaf
        # across most trees are similar by the forest's own construction
        # (Breiman's random-forest proximity measure).
        return self.model.apply(X).astype(np.float32)


# --------------------------------------------------------------------------
# Graph neural network backend
# --------------------------------------------------------------------------


class _GINNet(nn.Module):
    """Two-layer GIN over the molecular graph, mirroring SCAMsNet's
    depth/dropout pattern (dense layers -> graph convolutions). `embed`
    returns the graph-level (sum-pooled node) readout -- the GNN's analog
    of SCAMsNet's penultimate layer; dropout stays on for MC sampling
    exactly like SCAMsNet.
    """

    def __init__(self, in_dim=ATOM_FEATURE_DIM, hidden=64, dropout=0.4):
        super().__init__()

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))

        self.conv1 = GINConv(mlp(in_dim, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def embed(self, batch):
        x = torch.relu(self.conv1(batch.x, batch.edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch.edge_index))
        x = self.dropout(x)
        return global_add_pool(x, batch.batch)

    def forward(self, batch):
        return self.head(self.embed(batch)).squeeze(-1)


class GNNBackend:
    """Graph neural network (GIN) operating directly on molecular graphs
    (see graph_featurization.py) instead of a fixed fingerprint. `X` here
    is a 1-D object array of `torch_geometric.data.Data`, not a 2-D numeric
    matrix -- see graph_featurization.py's module docstring for why that's
    safe to slice/append/delete through the same numpy operations the rest
    of the active-learning loop already uses.
    """

    def __init__(self, epochs=50, lr=1e-3, dropout=0.4, hidden=64):
        if Batch is None:
            raise ImportError(
                "The GNN architecture requires torch_geometric. Install the "
                "'graph' extra (e.g. `uv sync --extra cpu --extra graph`)."
            )
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.hidden = hidden
        self.module = _GINNet(dropout=dropout, hidden=hidden)

    @staticmethod
    def _batch(X):
        return Batch.from_data_list(list(X))

    def fit(self, X, y):
        # Fresh module + optimizer every call, matching the active-learning
        # loop's "retrain from scratch" contract (see RandomForestBackend).
        self.module = _GINNet(dropout=self.dropout, hidden=self.hidden)
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        batch = self._batch(X)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32)

        self.module.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.module(batch), y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        self.module.eval()
        with torch.no_grad():
            p1 = torch.sigmoid(self.module(self._batch(X))).numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def mc_probabilities(self, X, n_samples):
        batch = self._batch(X)
        self.module.train()  # keep dropout active even though we're not training
        with torch.no_grad():
            probs = [torch.sigmoid(self.module(batch)).numpy() for _ in range(n_samples)]
        self.module.eval()
        return np.stack(probs, axis=0)

    def embed(self, X):
        self.module.eval()
        with torch.no_grad():
            return self.module.embed(self._batch(X)).numpy()


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

ARCHITECTURES = ("mlp", "rf", "gnn")


def make_backend(architecture, X_train, epochs=50, seed=None):
    """Construct the backend for `architecture` ("mlp", "rf", or "gnn"),
    sized/seeded appropriately for `X_train`.
    """
    if architecture == "mlp":
        return TorchMLPBackend(shape=X_train.shape[1], epochs=epochs)
    if architecture == "rf":
        return RandomForestBackend(random_state=seed)
    if architecture == "gnn":
        return GNNBackend(epochs=epochs)
    raise ValueError(
        f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}"
    )
