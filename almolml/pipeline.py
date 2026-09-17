"""Orchestrates one benchmark study: load data, split it, then train and
score the classical, neural-network, and active-learning models over
`iterations` repeats.
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import RDLogger

from .dataset import Dataset
from .model_backends import ARCHITECTURES
from .models import ActiveLearningModel, DeepSCAMsModel, TorchMLPModel, seed_everything
from .paths import REPO_ROOT
from .query_strategies import (
    bald_query,
    batch_bald_query,
    batch_core_set_query,
    batch_direct_query,
    core_set_query,
    direct_query,
    entropy_query,
)
from .splitters import BSplitter, SSplitter, TTSSplitter, three_way_split
from .utilities import dataset_to_splitter, prepare_results_dir, require_file

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class DatasetConfig:
    """One entry in DATASETS below.

    `external_test_file`, when set (the three original SCAM variants),
    means the dataset ships its own separately curated held-out test file
    (test_DLS.csv) -- test/train/validation are three different files.
    When None (every dataset added by the generalization study: BBBP,
    ClinTox, Tox21, HIV-subsample), there is no such file, and
    `three_way_split` (splitters.py) carves all three out of the one file
    instead -- see docs/generalization_study_design.md section 2 for why.
    """

    file: str
    x_col: str = "Smiles String"
    y_col: str = "agg?"
    external_test_file: Optional[str] = None


# A study name has the form "<DATASET>_<SPLIT>", e.g. "SF_TTS". Dataset
# keys must not themselves contain "_" (study_name.split("_") assumes
# exactly one split point).
DATASETS = {
    "SF": DatasetConfig("SCAMS_filtered.csv", external_test_file="test_DLS.csv"),
    "SP1": DatasetConfig(
        "SCAMS_balanced_with_positive.csv", external_test_file="test_DLS.csv"
    ),
    "SP2": DatasetConfig(
        "SCAMS_added_positives_653_1043.csv", external_test_file="test_DLS.csv"
    ),
    # Added by the generalization study (docs/generalization_study_design.md):
    # small-to-medium pools with scarce/imbalanced labels, like SCAM, plus
    # HIVSUB for a "large pool, rare positives" regime SCAM doesn't cover.
    # Built by scripts/prepare_new_datasets.py; y_col is "label" for all of
    # them (a uniform name chosen for the new datasets, distinct from SCAM's
    # native "agg?").
    "BBBP": DatasetConfig("BBBP.csv", y_col="label"),
    "CLINTOX": DatasetConfig("CLINTOX.csv", y_col="label"),
    "TOX21NRAR": DatasetConfig("TOX21NRAR.csv", y_col="label"),
    "TOX21SRP53": DatasetConfig("TOX21SRP53.csv", y_col="label"),
    "HIVSUB": DatasetConfig("HIVSUB.csv", y_col="label"),
}
SPLITTERS = {"TTS": TTSSplitter, "SS": SSplitter, "B": BSplitter}
AL_STRATEGIES = {
    "entropy": entropy_query,
    "bald": bald_query,
    "core_set": core_set_query,
    "direct": direct_query,
    "bald_batch": batch_bald_query,
    "core_set_batch": batch_core_set_query,
    "direct_batch": batch_direct_query,
}

RUN_STATS_COLUMNS = [
    "Iteration",
    "AUC_LB_test",
    "AUC_test",
    "AUC_UB_test",
    "Accuracy_test",
    "F1_test",
    "MCC_test",
    "AUC_LB_validation",
    "AUC_validation",
    "AUC_UB_validation",
    "Accuracy_validation",
    "F1_validation",
    "MCC_validation",
]


def _stats_row(iteration: int, model) -> list:
    """Flatten one model's validation + test Validation results into a row
    matching RUN_STATS_COLUMNS.
    """
    return (
        [iteration]
        + model.test_performance.iloc[0].tolist()
        + model.validation_performance.iloc[0].tolist()
    )


def _append_run(accumulated: Optional[pd.DataFrame], row: list) -> pd.DataFrame:
    """Append one run's stats row, creating the DataFrame on the first call."""
    new_row = pd.DataFrame([row], columns=RUN_STATS_COLUMNS)
    if accumulated is None:
        return new_row
    return pd.concat([accumulated, new_row], ignore_index=True)


class SCAMsPipeline:
    def __init__(
        self,
        study_name: str,
        datasets_path,
        ID_name: str = "ID",
        iterations: int = 10,
        epochs: int = 50,
        max_queries: Optional[int] = None,
        al_strategy: str = "entropy",
        batch_size: int = 1,
        architecture: str = "mlp",
        overwrite: bool = False,
        test_name: str = "test_DLS.csv",
        test_split_r: float = 0.2,
        validation_split_r: float = 0.3,
        seed: int = 0,
    ):
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {architecture!r}; expected one of {ARCHITECTURES}"
            )
        self.iterations = iterations
        self.epochs = epochs
        self.max_queries = max_queries
        self.al_strategy = AL_STRATEGIES[al_strategy]
        self.batch_size = batch_size
        self.architecture = architecture
        self.study_name = study_name
        self.seed = seed

        dataset_key, split_key = study_name.split("_")
        config = DATASETS[dataset_key]
        self.splitter = SPLITTERS[split_key]

        self.results_dir = prepare_results_dir(
            REPO_ROOT / "Results" / study_name, overwrite, allow_rename=True
        )

        if config.external_test_file:
            self._load_external_test_file_study(
                config, datasets_path, ID_name, test_name
            )
        else:
            self._load_single_file_study(
                config, datasets_path, ID_name, test_split_r, validation_split_r
            )

        self.deepscams_results = None
        self.mlp_results = None
        self.active_learning_results = None

        seed_everything(seed)
        self._write_split_csvs()
        self._run_all_iterations()

    def _load_external_test_file_study(self, config, datasets_path, ID_name, test_name):
        """SCAM's original layout: a separately curated external test file
        (test_DLS.csv), distinct from the train/validation pool.
        """
        test_dataset_df = pd.read_csv(require_file(datasets_path, test_name))
        train_validation_df = pd.read_csv(require_file(datasets_path, config.file))

        self.test_dataset = Dataset(test_dataset_df, ID_name, config.x_col, config.y_col)
        self.train_validation_dataset = Dataset(
            train_validation_df, ID_name, config.x_col, config.y_col
        )
        self.train_validation = dataset_to_splitter(
            self.splitter, self.train_validation_dataset
        )

        self.X_test_numeric, self.Y_test = self.test_dataset.X, self.test_dataset.Y
        self.X_train_numeric = self.train_validation.X_train
        self.Y_train = self.train_validation.Y_train
        self.X_validation_numeric = self.train_validation.X_test
        self.Y_validation = self.train_validation.Y_test

        if self.architecture == "gnn":
            from .graph_featurization import describe_graphs

            test_graphs = describe_graphs(self.test_dataset.dataset["mols"])
            train_validation_graphs = describe_graphs(
                self.train_validation_dataset.dataset["mols"]
            )
            self.X_test_model = test_graphs
            self.X_train_model = train_validation_graphs[self.train_validation.train_idx]
            self.X_validation_model = train_validation_graphs[
                self.train_validation.test_idx
            ]
        else:
            self.X_test_model = self.X_test_numeric
            self.X_train_model = self.X_train_numeric
            self.X_validation_model = self.X_validation_numeric

    def _load_single_file_study(
        self, config, datasets_path, ID_name, test_split_r, validation_split_r
    ):
        """A dataset with no separately curated external test file (every
        dataset added by the generalization study): train/validation/test
        are all carved out of the one file, via `three_way_split`.
        """
        df = pd.read_csv(require_file(datasets_path, config.file))
        dataset = Dataset(df, ID_name, config.x_col, config.y_col)

        test_view, train_validation_view = three_way_split(
            dataset,
            self.splitter,
            test_split_r=test_split_r,
            validation_split_r=validation_split_r,
        )
        self.train_validation = train_validation_view

        self.X_test_numeric, self.Y_test = test_view.X_test, test_view.Y_test
        self.X_train_numeric = train_validation_view.X_train
        self.Y_train = train_validation_view.Y_train
        self.X_validation_numeric = train_validation_view.X_test
        self.Y_validation = train_validation_view.Y_test

        if self.architecture == "gnn":
            from .graph_featurization import describe_graphs

            graphs = describe_graphs(dataset.dataset["mols"])
            self.X_test_model = graphs[test_view.test_idx]
            self.X_train_model = graphs[train_validation_view.train_idx]
            self.X_validation_model = graphs[train_validation_view.test_idx]
        else:
            self.X_test_model = self.X_test_numeric
            self.X_train_model = self.X_train_numeric
            self.X_validation_model = self.X_validation_numeric

    def _write_split_csvs(self):
        """Persist the numeric-feature split -- the same split every
        architecture trains on, just encoded differently for "gnn" -- so
        it's inspectable from the filesystem regardless of architecture
        (a graph object array isn't something you can usefully dump to CSV).
        """
        pd.DataFrame(self.X_train_numeric).to_csv(self.results_dir / "X_train.csv")
        pd.DataFrame(self.Y_train).to_csv(self.results_dir / "Y_train.csv")
        pd.DataFrame(self.X_validation_numeric).to_csv(
            self.results_dir / "X_validation.csv"
        )
        pd.DataFrame(self.Y_validation).to_csv(self.results_dir / "Y_validation.csv")

    def _run_all_iterations(self):
        for i in range(self.iterations):
            run = PipelineRun(
                i,
                self.X_train_numeric,
                self.X_train_model,
                self.Y_train,
                self.X_validation_numeric,
                self.X_validation_model,
                self.Y_validation,
                self.X_test_numeric,
                self.X_test_model,
                self.Y_test,
                self.results_dir,
                epochs=self.epochs,
                max_queries=self.max_queries,
                al_strategy=self.al_strategy,
                batch_size=self.batch_size,
                architecture=self.architecture,
                seed=self.seed + i,
            )
            self.deepscams_results = _append_run(
                self.deepscams_results, run.deepscams_stats
            )
            self.mlp_results = _append_run(self.mlp_results, run.mlp_stats)
            self.active_learning_results = _append_run(
                self.active_learning_results, run.active_learning_stats
            )

        self.deepscams_results.to_csv(self.results_dir / "DeepSCAMs.csv")
        self.mlp_results.to_csv(self.results_dir / "TF_ML_non_AL.csv")
        self.active_learning_results.to_csv(self.results_dir / "TF_ML_AL.csv")


class PipelineRun:
    """Trains and scores every model variant once, on one train/validation
    split, writing per-iteration active-learning traces under
    `results_dir_par/<iteration>/`.

    Two parallel feature representations flow through here: `X_*_numeric`
    (Morgan fingerprint + RDKit descriptors, always) and `X_*_model`
    (identical to the numeric one unless `architecture="gnn"`, in which
    case it's the per-molecule graph representation instead -- see
    graph_featurization.py). DeepSCAMsModel is a fixed reference baseline
    (the original DeepSCAMs paper's architecture) independent of whichever
    architecture the study is comparing, so it always trains on the numeric
    features; TorchMLPModel/ActiveLearningModel train on whichever
    representation `architecture` calls for.
    """

    def __init__(
        self,
        iteration: int,
        X_train_numeric,
        X_train_model,
        Y_train,
        X_validation_numeric,
        X_validation_model,
        Y_validation,
        X_test_numeric,
        X_test_model,
        Y_test,
        results_dir_par: Path,
        epochs: int = 50,
        max_queries: Optional[int] = None,
        al_strategy=entropy_query,
        batch_size: int = 1,
        architecture: str = "mlp",
        seed: int = 0,
    ):
        self.iteration = iteration
        self.X_train_numeric = X_train_numeric
        self.X_train_model = X_train_model
        self.Y_train = Y_train
        self.X_validation_numeric = X_validation_numeric
        self.X_validation_model = X_validation_model
        self.Y_validation = Y_validation
        self.X_test_numeric = X_test_numeric
        self.X_test_model = X_test_model
        self.Y_test = Y_test
        self.epochs = epochs
        self.max_queries = max_queries
        self.al_strategy = al_strategy
        self.batch_size = batch_size
        self.architecture = architecture
        self.seed = seed
        self.results_dir = prepare_results_dir(
            results_dir_par / str(iteration), overwrite=True
        )

        self.deepscams_stats = None
        self.mlp_stats = None
        self.active_learning_stats = None
        self._run()

    def _run(self):
        start = time.time()
        seed_everything(self.seed)
        self._run_non_active_learning()
        self._run_active_learning()
        minutes = int((time.time() - start) / 60)
        print("The run {} took {} minutes".format(self.iteration, minutes))

    def _run_non_active_learning(self):
        deepscams = DeepSCAMsModel(
            self.X_train_numeric,
            self.Y_train,
            self.X_test_numeric,
            self.Y_test,
            self.X_validation_numeric,
            self.Y_validation,
        )
        mlp = TorchMLPModel(
            self.X_train_model,
            self.Y_train,
            self.X_test_model,
            self.Y_test,
            self.X_validation_model,
            self.Y_validation,
            epochs=self.epochs,
            architecture=self.architecture,
            seed=self.seed,
        )

        self.deepscams_stats = _stats_row(self.iteration, deepscams)
        self.mlp_stats = _stats_row(self.iteration, mlp)

    def _run_active_learning(self):
        # Set generously high enough that ActiveLearningModel.run()'s own
        # `len(X_pool) > 0` check is what actually stops the loop -- i.e.
        # this always consumes the entire pool -- regardless of n_initial.
        # (Previously `self.X_train.shape[0] - 11`, which only reproduced
        # the true full pool when n_initial was 12; with the current
        # n_initial=10 default it silently left the last 2 pool examples
        # unlabeled on every "full pool" run.)
        n_queries = self.X_train_model.shape[0]
        if self.max_queries is not None:
            n_queries = min(n_queries, self.max_queries)

        active_learner = ActiveLearningModel(
            self.X_train_model,
            self.Y_train,
            self.X_test_model,
            self.Y_test,
            self.X_validation_model,
            self.Y_validation,
            n_queries=n_queries,
            results_dir=self.results_dir,
            epochs=self.epochs,
            query_strategy=self.al_strategy,
            batch_size=self.batch_size,
            architecture=self.architecture,
            seed=self.seed,
        )
        self.active_learning_stats = _stats_row(self.iteration, active_learner)


if __name__ == "__main__":
    from .cli import main

    main()
