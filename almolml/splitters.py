from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .featurization import butina_cluster, generate_scaffolds


class BaseSplitter(ABC):
    """`train_idx`/`test_idx` (positional indices into the original,
    pre-split `X`/`Y`/`mols` ordering) let a second, parallel feature
    representation -- e.g. graph_featurization.describe_graphs' per-molecule
    graphs, for the GNN backend -- be split identically to `X`/`Y`, without
    recomputing the split: `graphs[splitter.train_idx]` is the same set of
    molecules as `splitter.X_train`, in the same order.
    """

    def __init__(self, X, Y, SMILES, mols, ID, test_split_r=0.3):
        self.X = X
        self.Y = Y
        self.SMILES = SMILES
        self.mols = mols
        self.ID = ID
        self.test_split_r = test_split_r
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.train_idx = None
        self.test_idx = None
        self.split()

    @abstractmethod
    def split(self):
        pass


class TTSSplitter(BaseSplitter):
    """Plain random train/test split."""

    def split(self):
        n = self.Y.shape[0]
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=self.test_split_r
        )
        self.train_idx = np.array(train_idx)
        self.test_idx = np.array(test_idx)
        self.X_train = np.array(self.X)[self.train_idx]
        self.X_test = np.array(self.X)[self.test_idx]
        self.Y_train = np.array(self.Y)[self.train_idx]
        self.Y_test = np.array(self.Y)[self.test_idx]


class BSplitter(BaseSplitter):
    """Butina-clustering split: the test set is drawn from clusters that
    contain only a single molecule, so it holds out structurally distinct
    compounds rather than near-duplicates of the training set.
    """

    def split(self):
        n_samples_test = int(self.Y.shape[0] * self.test_split_r)

        # self.mols/self.ID are pandas Series indexed by ID (see Dataset),
        # so building this frame with an explicit positional index would
        # silently reindex them to all-NaN; np.asarray each one first to
        # drop that mismatched index and assign every column positionally.
        data = pd.DataFrame(
            {
                "X": self.X,
                "Y": self.Y,
                "mols": np.asarray(self.mols, dtype=object),
                "ID": np.asarray(self.ID),
            },
            index=np.arange(self.Y.shape[0]),
        )
        data["cluster"] = butina_cluster(data["mols"])
        singleton_cluster_ids = (
            data["cluster"].value_counts().loc[lambda counts: counts == 1].index.tolist()
        )
        singleton_rows = data[data["cluster"].isin(singleton_cluster_ids)]
        if len(singleton_rows) < n_samples_test:
            print(
                f"Only {len(singleton_rows)} structurally-distinct (singleton-cluster) "
                f"molecules available, fewer than the {n_samples_test} requested for the "
                "test set -- using all of them instead of the full requested ratio."
            )
            n_samples_test = len(singleton_rows)

        test_set = singleton_rows.sample(n_samples_test)
        train_set = data[~data["ID"].isin(test_set["ID"])]

        self.train_idx = train_set.index.to_numpy()
        self.test_idx = test_set.index.to_numpy()
        self.X_train = np.array(train_set["X"].tolist())
        self.X_test = np.array(test_set["X"].tolist())
        self.Y_train = np.array(train_set["Y"].tolist())
        self.Y_test = np.array(test_set["Y"].tolist())


class SSplitter(BaseSplitter):
    """Scaffold split: greedily fills the training set with whole
    Bemis-Murcko scaffold groups, spilling the smallest, most distinctive
    scaffolds into the test set.
    """

    def split(self):
        data = pd.DataFrame(
            {
                "X": self.X,
                "Y": self.Y,
                "Smiles String": self.SMILES,
                "mols": self.mols,
                "ID": self.ID,
            },
            index=self.ID,
        )
        scaffold_sets = generate_scaffolds(data)
        train_cutoff = (1 - self.test_split_r) * self.Y.shape[0]

        train_inds, test_inds = [], []
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set

        self.train_idx = np.array(train_inds)
        self.test_idx = np.array(test_inds)
        self.X_train = np.array(data.iloc[train_inds]["X"].tolist())
        self.X_test = np.array(data.iloc[test_inds]["X"].tolist())
        self.Y_train = np.array(data.iloc[train_inds]["Y"].tolist())
        self.Y_test = np.array(data.iloc[test_inds]["Y"].tolist())


class SplitView:
    """Minimal stand-in for a BaseSplitter's public train/test attributes,
    for a split assembled from index arrays computed elsewhere (see
    three_way_split below) rather than by running split() itself.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, train_idx, test_idx):
        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test
        self.train_idx, self.test_idx = train_idx, test_idx


def three_way_split(dataset, splitter_cls, test_split_r=0.2, validation_split_r=0.3):
    """Carve `dataset` into (test, train, validation) using `splitter_cls`
    twice: once on the whole dataset to hold out `test`, then again on the
    remainder to hold out `validation` from `train`.

    For datasets with a separately curated external test file (SCAM's
    test_DLS.csv), that file already *is* the test set -- this function is
    for the datasets added in the generalization study (BBBP, ClinTox,
    Tox21, HIV-subsample), none of which come with one. Splitting the same
    file twice with the same splitter class means the test set is held out
    by the same structural-holdout logic (scaffold/Butina/TTS) as
    train/validation, not just a uniformly random subset of it.

    Returns
    -------
    (test_view, train_validation_view) -- both expose the same
    X_train/Y_train/X_test/Y_test/train_idx/test_idx attributes a
    BaseSplitter does (test_view's X_train/Y_train/train_idx are left
    None: it has no further internal split of its own). train_idx/test_idx
    are positions in `dataset`'s *original* row ordering throughout -- not
    positions within the "remainder" subset stage 2 operates on -- so a
    parallel representation (e.g. graph_featurization.describe_graphs'
    per-molecule graphs, for the GNN backend) can be sliced consistently
    with either stage using the same indices as the numeric features.
    """
    mols = np.asarray(dataset.dataset["mols"].to_numpy(), dtype=object)
    smiles = np.asarray(dataset.SMILES.to_numpy())
    ids = np.asarray(dataset.ID_name.to_numpy())
    # Left as the plain list dataset.X already is (not wrapped in an
    # ndarray): BSplitter/SSplitter put it straight into a DataFrame column
    # (which requires a list of row-vectors, not a genuine 2-D ndarray --
    # pandas can't make one column out of a 2-D array), while TTSSplitter's
    # own `np.array(self.X)` step is what turns it into a real numeric
    # matrix. That conversion only happens once, at the point each
    # splitter actually builds X_train/X_test -- doing it here up front
    # would break the other two splitters.
    X = dataset.X
    Y = dataset.Y

    stage1 = splitter_cls(X, Y, smiles, mols, ids, test_split_r=test_split_r)
    test_view = SplitView(
        X_train=None,
        Y_train=None,
        X_test=stage1.X_test,
        Y_test=stage1.Y_test,
        train_idx=None,
        test_idx=stage1.test_idx,
    )

    rest_idx = stage1.train_idx
    stage2 = splitter_cls(
        [X[i] for i in rest_idx],
        Y[rest_idx],
        smiles[rest_idx],
        mols[rest_idx],
        ids[rest_idx],
        test_split_r=validation_split_r,
    )
    train_validation_view = SplitView(
        X_train=stage2.X_train,
        Y_train=stage2.Y_train,
        X_test=stage2.X_test,
        Y_test=stage2.Y_test,
        train_idx=rest_idx[stage2.train_idx],
        test_idx=rest_idx[stage2.test_idx],
    )
    return test_view, train_validation_view
