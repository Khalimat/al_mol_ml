"""Molecule -> graph representation for the GNN architecture backend.

`featurization.describe` turns each molecule into one flat feature vector
(Morgan fingerprint + RDKit descriptors) -- what the MLP and Random Forest
backends train on. The GNN backend instead needs the molecular graph
itself (atoms as nodes with feature vectors, bonds as edges), so it can
learn its own structural representation rather than relying on a fixed
hash-based fingerprint. This module builds that representation, as
`torch_geometric.data.Data` objects, from the same rdkit `Mol` objects
`Dataset` already parses.

Returned as a 1-D numpy object array (one `Data` per molecule) rather than
a plain Python list: `ActiveLearner`/`ActiveLearningModel`'s query/teach
loop slices, deletes from, and appends to the pool with plain numpy
operations (`np.delete`, `np.append`, fancy indexing) that were written
against fixed-length numeric feature matrices. Those operations are
dtype-agnostic on a genuine 1-D object ndarray -- but np.array() called on
a raw Python list of Data objects can misbehave, since numpy's sequence
auto-detection tries to introspect each element's own `__len__`/
`__getitem__` (which Data defines, for its *attributes*, not for numpy's
benefit). Building the object array directly with `np.empty(..., dtype=
object)` sidesteps that inspection entirely.
"""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Common organic elements first, with an explicit "other" bucket so an
# unexpected element degrades gracefully instead of raising.
_ATOM_SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si"]
_DEGREES = [0, 1, 2, 3, 4]
_TOTAL_HS = [0, 1, 2, 3]
_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
]


def _one_hot(value, choices):
    """One-hot over `choices`, plus a trailing "not in choices" bit -- so
    an out-of-vocabulary value is represented explicitly rather than
    silently mapped to the zero vector of some other category.
    """
    return [float(value == choice) for choice in choices] + [
        float(value not in choices)
    ]


def _atom_features(atom):
    return np.array(
        _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), _DEGREES)
        + _one_hot(atom.GetTotalNumHs(), _TOTAL_HS)
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
            float(atom.IsInRing()),
        ],
        dtype=np.float32,
    )


def _atom_feature_dim():
    return len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


ATOM_FEATURE_DIM = _atom_feature_dim()


def mol_to_graph(mol):
    """One rdkit Mol -> one torch_geometric.data.Data (undirected: each
    bond contributes both (i, j) and (j, i) edges).
    """
    atom_features = np.stack([_atom_features(atom) for atom in mol.GetAtoms()])

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    edge_index = (
        np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    )

    return Data(
        x=torch.as_tensor(atom_features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
    )


def describe_graphs(mols):
    """Parallel to `featurization.describe`, but for the GNN backend: one
    `Data` object per molecule instead of one flat feature vector.

    Parameters
    ----------
    mols: list (or array) of rdkit.Chem.rdchem.Mol

    Returns
    -------
    numpy.ndarray, dtype=object, shape (len(mols),) -- one Data per row
    """
    graphs = np.empty(len(mols), dtype=object)
    for i, mol in enumerate(mols):
        graphs[i] = mol_to_graph(mol)
    return graphs
