"""Downloads and cleans BBBP, ClinTox, Tox21 (NR-AR, SR-p53), and a
stratified HIV subsample from MoleculeNet's public distribution, into the
CSV shape almolml.dataset.Dataset already expects: 'ID', 'Smiles String',
plus a label column.

Source: the same raw CSVs `deepchem.molnet.load_*` fetches, mirrored at
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/ -- no `deepchem`
dependency needed, just plain CSV/gzip.

These datasets don't come with a separately curated external test set the
way SCAM's test_DLS.csv does, so (per the generalization design doc,
docs/generalization_study_design.md) the pipeline carves train/validation/
test out of the one file itself, via `almolml.splitters.three_way_split` --
this script's only job is to produce one clean (ID, Smiles String, label)
CSV per dataset.

Run once, from the repo root:
    uv run python scripts/prepare_new_datasets.py

Re-running skips any file that already exists -- delete a file under
Datasets/ first if you want to rebuild it (e.g. after changing
HIV_SUBSAMPLE_N/HIV_SUBSAMPLE_SEED below).
"""

import gzip
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "Datasets"
SOURCE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

# ~6-8k rows, stratified on the source dataset's own positive rate, fixed
# seed -- keeps HIV's "large pool, rare positives" character (see design
# doc section 2) without the full 41k-row runtime cost.
HIV_SUBSAMPLE_N = 7000
HIV_SUBSAMPLE_SEED = 0


def _download(name):
    url = f"{SOURCE}/{name}"
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if name.endswith(".gz"):
        content = gzip.decompress(content)
    return pd.read_csv(io.BytesIO(content))


def _clean(df, smiles_col, label_col, id_prefix):
    """Drop rows with a missing label or an unparseable SMILES string,
    assign a stable ID, and return the (ID, Smiles String, label) frame.
    """
    df = df[[smiles_col, label_col]].rename(
        columns={smiles_col: "Smiles String", label_col: "label"}
    )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    parsable = df["Smiles String"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    n_dropped = int((~parsable).sum())
    if n_dropped:
        print(f"  dropping {n_dropped} rows with an unparseable SMILES string")
    df = df[parsable].reset_index(drop=True)

    df.insert(0, "ID", [f"{id_prefix}-{i:05d}" for i in range(len(df))])
    return df


def prepare_bbbp():
    df = _download("BBBP.csv")
    out = _clean(df, "smiles", "p_np", "BBBP")
    print(f"BBBP: {len(out)} rows, positive rate {out['label'].mean():.3f}")
    return out


def prepare_clintox():
    df = _download("clintox.csv.gz")
    out = _clean(df, "smiles", "CT_TOX", "CLINTOX")
    print(f"ClinTox: {len(out)} rows, positive rate {out['label'].mean():.3f}")
    return out


def prepare_tox21(assay, id_prefix):
    df = _download("tox21.csv.gz")
    out = _clean(df, "smiles", assay, id_prefix)
    print(f"Tox21/{assay}: {len(out)} rows, positive rate {out['label'].mean():.3f}")
    return out


def prepare_hiv_subsample(n=HIV_SUBSAMPLE_N, seed=HIV_SUBSAMPLE_SEED):
    df = _download("HIV.csv")
    out = _clean(df, "smiles", "HIV_active", "HIVSUB")

    positive = out[out["label"] == 1]
    negative = out[out["label"] == 0]
    true_rate = len(positive) / len(out)
    n_pos = max(1, round(n * true_rate))
    n_neg = n - n_pos

    sampled = pd.concat(
        [
            positive.sample(n=min(n_pos, len(positive)), random_state=seed),
            negative.sample(n=min(n_neg, len(negative)), random_state=seed),
        ]
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled["ID"] = [f"HIVSUB-{i:05d}" for i in range(len(sampled))]

    print(
        f"HIV subsample: {len(sampled)} rows (of {len(out)}), "
        f"positive rate {sampled['label'].mean():.3f} "
        f"(source positive rate {true_rate:.3f}, seed {seed})"
    )
    return sampled


# Maps the almolml dataset-registry key (see pipeline.py's DATASETS) to its
# output filename and how to build it.
DATASET_BUILDERS = {
    "BBBP.csv": prepare_bbbp,
    "CLINTOX.csv": prepare_clintox,
    "TOX21NRAR.csv": lambda: prepare_tox21("NR-AR", "TOX21NRAR"),
    "TOX21SRP53.csv": lambda: prepare_tox21("SR-p53", "TOX21SRP53"),
    "HIVSUB.csv": prepare_hiv_subsample,
}


def main():
    DATASETS_DIR.mkdir(exist_ok=True)
    for filename, builder in DATASET_BUILDERS.items():
        out_path = DATASETS_DIR / filename
        if out_path.exists():
            print(f"{filename} already exists, skipping (delete it to rebuild)")
            continue
        df = builder()
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
