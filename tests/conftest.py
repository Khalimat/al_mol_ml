import numpy as np
import pandas as pd
import pytest

# A mix of small, valid, drug-like SMILES strings used to build synthetic
# datasets that are fast to featurize/train on but still exercise real
# rdkit descriptor and clustering code paths.
_SMILES_POOL = [
    "CCO",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "CC(=O)C",
    "O=Cc1ccccc1",
    "c1ccc2ccccc2c1",
    "c1ccncc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "OCC(O)CO",
    "CCN(CC)CC",
    "ClC(Cl)Cl",
    "CC(C)O",
    "CCOC(=O)C",
    "CC(N)C(=O)O",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "Fc1ccccc1",
    "Brc1ccccc1",
    "CC1=CC=CC=C1",
    "CCCCCCCC",
]


def _make_dataset(n_rows, id_prefix):
    smiles = [_SMILES_POOL[i % len(_SMILES_POOL)] for i in range(n_rows)]
    labels = [i % 2 for i in range(n_rows)]
    ids = ["{}-{:03d}".format(id_prefix, i) for i in range(n_rows)]
    return pd.DataFrame({"ID": ids, "Smiles String": smiles, "agg?": labels})


@pytest.fixture
def train_validation_df():
    return _make_dataset(40, "TRAIN")


@pytest.fixture
def test_df():
    return _make_dataset(12, "TEST")


@pytest.fixture
def synthetic_datasets_dir(tmp_path, train_validation_df, test_df):
    datasets_dir = tmp_path / "Datasets"
    datasets_dir.mkdir()
    train_validation_df.to_csv(datasets_dir / "SCAMS_filtered.csv", index=False)
    test_df.to_csv(datasets_dir / "test_DLS.csv", index=False)
    return datasets_dir


@pytest.fixture
def rng():
    return np.random.default_rng(0)
