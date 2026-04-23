import numpy as np
import pandas as pd
from rdkit import Chem

from .utilities import describe


class Dataset:
    def __init__(self, dataset, ID_name, X_column_name, Y_column_name):
        self.dataset = dataset
        self.dataset = self.dataset.set_index([pd.Index(self.dataset[ID_name])])
        self.ID_name = self.dataset[ID_name]
        self.SMILES = self.dataset[X_column_name]
        self.Y_column_name = Y_column_name
        self.X = None
        self.Y = None
        self.run()

    def run(self):
        self.calculate_descriptors()

    def calculate_descriptors(self):
        mol_obj = [Chem.MolFromSmiles(s) for s in self.SMILES]
        self.dataset["mols"] = self.SMILES.apply(lambda x: Chem.MolFromSmiles(x))
        descriptors = describe(mol_obj)
        self.X = descriptors
        self.Y = np.array(self.dataset[self.Y_column_name])
