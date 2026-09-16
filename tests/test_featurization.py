import pandas as pd
from rdkit import Chem

from jcim.featurization import butina_cluster, describe, generate_scaffolds


def test_describe_shape_matches_fingerprint_plus_descriptor_count():
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1"]]
    descrs = describe(mols)
    assert len(descrs) == 2
    # 2048-bit Morgan fingerprint plus whatever descriptors this rdkit
    # version ships. The point of this test is to guard against silently
    # hardcoding a specific rdkit version's descriptor count elsewhere in
    # the codebase (that mismatch used to crash model construction).
    from rdkit.Chem import Descriptors

    expected_len = 2048 + len(Descriptors._descList[0:2] + Descriptors._descList[3:])
    assert len(descrs[0]) == expected_len


def test_butina_cluster_groups_identical_molecules_together():
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCO", "c1ccccc1"]]
    clusters = butina_cluster(mols)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_generate_scaffolds_partitions_all_rows():
    df = pd.DataFrame({"Smiles String": ["CCO", "CCN", "c1ccccc1", "Cc1ccccc1"]})
    scaffold_sets = generate_scaffolds(df)
    all_indices = sorted(i for group in scaffold_sets for i in group)
    assert all_indices == [0, 1, 2, 3]
