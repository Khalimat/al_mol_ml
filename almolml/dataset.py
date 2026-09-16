import numpy as np
import pandas as pd
from rdkit import Chem

from .featurization import describe


class Dataset:
    def __init__(self, dataset, ID_name, X_column_name, Y_column_name):
        self.dataset = dataset.set_index(pd.Index(dataset[ID_name]))
        self.ID_name = self.dataset[ID_name]
        self.SMILES = self.dataset[X_column_name]
        self.Y_column_name = Y_column_name
        self.X = None
        self.Y = None
        self.calculate_descriptors()

    def calculate_descriptors(self):
        self.dataset["mols"] = self.SMILES.apply(Chem.MolFromSmiles)
        self.X = describe(self.dataset["mols"])
        self.Y = np.array(self.dataset[self.Y_column_name])
