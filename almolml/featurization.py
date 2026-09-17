"""Turn molecules into fixed-length feature vectors, and group them by
structural similarity for the clustering-based train/test splitters.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina


def describe(mols):
    """Featurize each molecule as a 2048-bit Morgan fingerprint concatenated
    with rdkit's built-in descriptors (the approach used by DeepSCAMs).

    Parameters
    ----------
    mols: list of rdkit.Chem.rdchem.Mol

    Returns
    -------
    list of feature vectors, one per molecule
    """
    descriptor_fns = [fn for _, fn in Descriptors._descList[0:2] + Descriptors._descList[3:]]
    descrs = []
    for mol in mols:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        fingerprint_bits = [float(bit) for bit in fingerprint.ToBitString()]
        descriptor_values = []
        for descriptor_fn in descriptor_fns:
            value = descriptor_fn(mol)
            if not np.isfinite(value):
                # A handful of rdkit descriptors (e.g. Ipc) can overflow to
                # +/-inf, or come back NaN, for specific structures --
                # not something the three original SCAM datasets happened
                # to trigger, but real once the study covers more varied
                # molecules (see docs/generalization_study_design.md).
                # Zero it out rather than let a NaN/inf reach the
                # downstream models, which reject non-finite input.
                value = 0.0
            elif value > np.finfo(np.float32).max:
                value = np.finfo(np.float32).max
            descriptor_values.append(np.float32(value))
        descrs.append(fingerprint_bits + descriptor_values)
    return descrs


def butina_cluster(mol_list, cutoff=0.35):
    """Cluster molecules by Tanimoto similarity of their Morgan fingerprints.

    Reference
    ---------
    Butina, D. "Unsupervised data base clustering based on daylight's
    fingerprint and Tanimoto similarity: A fast and automated way to
    cluster small and large data sets." J. Chem. Inf. Comput. Sci. 39.4
    (1999): 747-750.

    Adapted from:
    https://github.com/PatWalters/Learning_Cheminformatics/blob/master/clustering.ipynb

    Parameters
    ----------
    mol_list: list of rdkit.Chem.rdchem.Mol to cluster
    cutoff: Tanimoto distance cutoff

    Returns
    -------
    list of int, a cluster id per molecule in `mol_list`
    """
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in mol_list
    ]
    distances = []
    n_fingerprints = len(fingerprints)
    for i in range(1, n_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], fingerprints[:i]
        )
        distances.extend(1 - s for s in similarities)
    clusters = Butina.ClusterData(distances, n_fingerprints, cutoff, isDistData=True)
    cluster_ids = [0] * n_fingerprints
    for cluster_id, cluster in enumerate(clusters, 1):
        for member in cluster:
            cluster_ids[member] = cluster_id
    return cluster_ids


def generate_scaffolds(dataset):
    """Group row indices of `dataset` by Bemis-Murcko scaffold.

    Adapted from deepchem. Larger scaffold groups come first, which is what
    lets SSplitter greedily fill the training set before spilling the
    smallest, most distinctive scaffolds into the test set.

    Parameters
    ----------
    dataset: DataFrame with a "Smiles String" column

    Returns
    -------
    list of list of int, row indices grouped by scaffold, largest first
    """
    scaffolds = {}
    for row_idx, smiles in enumerate(dataset["Smiles String"]):
        scaffold = _bemis_murcko_scaffold(smiles)
        scaffolds.setdefault(scaffold, []).append(row_idx)

    return [
        group
        for _, group in sorted(
            scaffolds.items(), key=lambda item: (len(item[1]), item[1][0]), reverse=True
        )
    ]


def _bemis_murcko_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string: the rings and
    the linker atoms between them.

    Reference: Bemis, G. W., and Murcko, M. A. "The properties of known
    drugs. 1. Molecular frameworks." J. Med. Chem. 39.15 (1996): 2887-2893.
    """
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
