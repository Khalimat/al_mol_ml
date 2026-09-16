"""Orchestrates one benchmark study: load data, split it, then train and
score the classical, neural-network, and active-learning models over
`iterations` repeats.
"""

import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import RDLogger

from .dataset import Dataset
from .models import ActiveLearningModel, DeepSCAMsModel, TorchMLPModel, seed_everything
from .paths import REPO_ROOT
from .query_strategies import bald_query, core_set_query, direct_query, entropy_query
from .splitters import AlmostNoValidation, BSplitter, SSplitter, TTSSplitter
from .utilities import dataset_to_splitter, prepare_results_dir, require_file

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# A study name has the form "<DATASET>_<SPLIT>", e.g. "SF_TTS".
DATASETS = {
    "SF": "SCAMS_filtered.csv",
    "SP1": "SCAMS_balanced_with_positive.csv",
    "SP2": "SCAMS_added_positives_653_1043.csv",
}
SPLITTERS = {"TTS": TTSSplitter, "SS": SSplitter, "B": BSplitter, "ANV": AlmostNoValidation}
AL_STRATEGIES = {
    "entropy": entropy_query,
    "bald": bald_query,
    "core_set": core_set_query,
    "direct": direct_query,
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
        overwrite: bool = False,
        test_name: str = "test_DLS.csv",
        X_column_name: str = "Smiles String",
        Y_column_name: str = "agg?",
        seed: int = 0,
    ):
        self.iterations = iterations
        self.epochs = epochs
        self.max_queries = max_queries
        self.al_strategy = AL_STRATEGIES[al_strategy]
        self.study_name = study_name
        self.seed = seed

        dataset_key, split_key = study_name.split("_")
        dataset_file = DATASETS[dataset_key]
        self.splitter = SPLITTERS[split_key]

        test_dataset = pd.read_csv(require_file(datasets_path, test_name))
        train_validation_dataset = pd.read_csv(require_file(datasets_path, dataset_file))
        self.results_dir = prepare_results_dir(
            REPO_ROOT / "Results" / study_name, overwrite, allow_rename=True
        )

        self.test_dataset = Dataset(test_dataset, ID_name, X_column_name, Y_column_name)
        self.train_validation_dataset = Dataset(
            train_validation_dataset, ID_name, X_column_name, Y_column_name
        )
        self.train_validation = None
        self.deepscams_results = None
        self.mlp_results = None
        self.active_learning_results = None

        seed_everything(seed)
        self._split_train_validation()
        self._run_all_iterations()

    def _split_train_validation(self):
        split = dataset_to_splitter(self.splitter, self.train_validation_dataset)
        self.train_validation = split
        pd.DataFrame(split.X_train).to_csv(self.results_dir / "X_train.csv")
        pd.DataFrame(split.Y_train).to_csv(self.results_dir / "Y_train.csv")
        pd.DataFrame(split.X_test).to_csv(self.results_dir / "X_validation.csv")
        pd.DataFrame(split.Y_test).to_csv(self.results_dir / "Y_validation.csv")

    def _run_all_iterations(self):
        for i in range(self.iterations):
            run = PipelineRun(
                i,
                self.test_dataset.X,
                self.test_dataset.Y,
                self.train_validation,
                self.results_dir,
                epochs=self.epochs,
                max_queries=self.max_queries,
                al_strategy=self.al_strategy,
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
    """

    def __init__(
        self,
        iteration: int,
        X_test,
        Y_test,
        train_validation,
        results_dir_par: Path,
        epochs: int = 50,
        max_queries: Optional[int] = None,
        al_strategy=entropy_query,
        seed: int = 0,
    ):
        self.iteration = iteration
        self.X_train = train_validation.X_train
        self.Y_train = train_validation.Y_train
        self.X_validation = train_validation.X_test
        self.Y_validation = train_validation.Y_test
        self.X_test = X_test
        self.Y_test = Y_test
        self.epochs = epochs
        self.max_queries = max_queries
        self.al_strategy = al_strategy
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
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test,
            self.X_validation,
            self.Y_validation,
        )
        mlp = TorchMLPModel(
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test,
            self.X_validation,
            self.Y_validation,
            epochs=self.epochs,
        )

        self.deepscams_stats = _stats_row(self.iteration, deepscams)
        self.mlp_stats = _stats_row(self.iteration, mlp)

    def _run_active_learning(self):
        n_queries = self.X_train.shape[0] - 11
        if self.max_queries is not None:
            n_queries = min(n_queries, self.max_queries)

        active_learner = ActiveLearningModel(
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test,
            self.X_validation,
            self.Y_validation,
            n_queries=n_queries,
            results_dir=self.results_dir,
            epochs=self.epochs,
            query_strategy=self.al_strategy,
        )
        self.active_learning_stats = _stats_row(self.iteration, active_learner)


if __name__ == "__main__":
    from .cli import main

    main()
