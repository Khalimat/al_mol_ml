from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from utilities import butina_cluster, generate_scaffolds

class BaseSplitter:
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        self.X = X
        self.Y = Y
        self.SMILES = SMILES
        self.mols = mols
        self.ID = ID
        self.type = type
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.test_split_r = test_split_r
        self.split()

    @abstractmethod
    def split(self):
        pass

class TTSSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):
        split = train_test_split(self.X, self.Y,
                                 test_size=self.test_split_r)
        X_train, X_test, Y_train, Y_test = split
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.Y_train = np.array(Y_train)
        self.Y_test = np.array(Y_test)


class BSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):

        n_samples_test = int(self.Y.shape[0] * self.test_split_r)

        dataset_w_mol_obj = pd.DataFrame({'X': self.X, 'Y': self.Y,
                                          'mols': self.mols, 'ID': self.ID},
                                         index=self.ID)
        dataset_w_mol_obj['cluster'] = butina_cluster(dataset_w_mol_obj['mols'])
        uniq_cluster_ids = dataset_w_mol_obj['cluster'].value_counts().loc[lambda x: x == 1].index.tolist()
        if len(uniq_cluster_ids) < int(dataset_w_mol_obj.shape[0] * self.test_split_r):
            print('Unable to split dataset based on butina clustering')
        test_set = dataset_w_mol_obj[dataset_w_mol_obj.cluster.isin(uniq_cluster_ids)].sample(n_samples_test)
        train_set = dataset_w_mol_obj[~dataset_w_mol_obj['ID'].isin(test_set['ID'].tolist())]
        data = [train_set['X'], test_set['X'], train_set['Y'], test_set['Y']]
        data = [np.array(x.tolist()) for x in data]
        X_train, X_test, Y_train, Y_test = data
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


class SSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):
        dataset_w_mol_obj = pd.DataFrame({'X': self.X, 'Y': self.Y,
                                          'Smiles String': self.SMILES,
                                          'mols': self.mols, 'ID': self.ID},
                                           index=self.ID)
        scaffold_sets = generate_scaffolds(dataset_w_mol_obj)
        train_cutoff = (1 - self.test_split_r) * self.Y.shape[0]
        train_inds = []
        test_inds = []
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set
        X_train = dataset_w_mol_obj.iloc[train_inds]['X']
        X_test = dataset_w_mol_obj.iloc[test_inds]['X']
        Y_train = dataset_w_mol_obj.iloc[train_inds]['Y']
        Y_test = dataset_w_mol_obj.iloc[test_inds]['Y']
        data = [np.array(x.tolist()) for x in [X_train, X_test, Y_train, Y_test]]
        X_train, X_test, Y_train, Y_test = data
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train =Y_train
        self.Y_test = Y_test


class AlmostNoValidation(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):
        mask = np.array([False for i in self.Y])
        mask[np.random.choice(np.where(self.Y == 0)[0])] = True
        mask[np.random.choice(np.where(self.Y == 1)[0])] = True
        self.Y_test = self.Y[mask]
        self.Y_train = self.Y[np.invert(mask)]
        self.X_test = np.array(self.X)[mask]
        self.X_train = np.array(self.X)[np.invert(mask)]
