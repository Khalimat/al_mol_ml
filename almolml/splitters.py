from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .featurization import butina_cluster, generate_scaffolds


class BaseSplitter(ABC):
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
        self.split()

    @abstractmethod
    def split(self):
        pass


class TTSSplitter(BaseSplitter):
    """Plain random train/test split."""

    def split(self):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, test_size=self.test_split_r
        )
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.Y_train = np.array(Y_train)
        self.Y_test = np.array(Y_test)


class BSplitter(BaseSplitter):
    """Butina-clustering split: the test set is drawn from clusters that
    contain only a single molecule, so it holds out structurally distinct
    compounds rather than near-duplicates of the training set.
    """

    def split(self):
        n_samples_test = int(self.Y.shape[0] * self.test_split_r)

        data = pd.DataFrame(
            {"X": self.X, "Y": self.Y, "mols": self.mols, "ID": self.ID}, index=self.ID
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

        self.X_train = np.array(data.iloc[train_inds]["X"].tolist())
        self.X_test = np.array(data.iloc[test_inds]["X"].tolist())
        self.Y_train = np.array(data.iloc[train_inds]["Y"].tolist())
        self.Y_test = np.array(data.iloc[test_inds]["Y"].tolist())
