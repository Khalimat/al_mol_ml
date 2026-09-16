import pandas as pd
import pytest

from almolml.dataset import Dataset
from almolml.models import ActiveLearningModel, DeepSCAMsModel, TorchMLPModel
from almolml.query_strategies import (
    bald_query,
    batch_bald_query,
    batch_core_set_query,
    batch_direct_query,
    core_set_query,
    direct_query,
    entropy_query,
)
from almolml.splitters import TTSSplitter
from almolml.utilities import dataset_to_splitter


def _features_and_labels(train_validation_df):
    dataset = Dataset(train_validation_df, "ID", "Smiles String", "agg?")
    return dataset_to_splitter(TTSSplitter, dataset)


def test_deepscams_model_trains_and_scores(train_validation_df):
    split = _features_and_labels(train_validation_df)
    model = DeepSCAMsModel(
        split.X_train, split.Y_train, split.X_train, split.Y_train, split.X_test, split.Y_test
    )
    assert model.test_performance is not None
    assert 0.0 <= model.test_performance.iloc[0]["accuracy"] <= 1.0


def test_torch_mlp_model_trains_and_scores(train_validation_df):
    split = _features_and_labels(train_validation_df)
    model = TorchMLPModel(
        split.X_train,
        split.Y_train,
        split.X_train,
        split.Y_train,
        split.X_test,
        split.Y_test,
        epochs=2,
    )
    assert model.test_performance is not None
    assert 0.0 <= model.test_performance.iloc[0]["accuracy"] <= 1.0


@pytest.mark.parametrize(
    "query_strategy", [entropy_query, bald_query, core_set_query, direct_query]
)
def test_active_learning_model_runs_a_short_query_loop(
    train_validation_df, tmp_path, query_strategy
):
    split = _features_and_labels(train_validation_df)
    results_dir = tmp_path / "al_run"
    results_dir.mkdir()
    model = ActiveLearningModel(
        split.X_train,
        split.Y_train,
        split.X_train,
        split.Y_train,
        split.X_test,
        split.Y_test,
        n_queries=3,
        results_dir=results_dir,
        epochs=2,
        query_strategy=query_strategy,
    )
    assert model.test_performance is not None
    assert (results_dir / "test_initial_stats.csv").exists()
    assert (results_dir / "validation_initial_stats.csv").exists()


@pytest.mark.parametrize(
    "query_strategy", [batch_bald_query, batch_core_set_query, batch_direct_query]
)
def test_active_learning_model_runs_a_batched_query_loop(
    train_validation_df, tmp_path, query_strategy
):
    split = _features_and_labels(train_validation_df)
    results_dir = tmp_path / "al_batch_run"
    results_dir.mkdir()
    n_initial = 10
    batch_size = 2

    model = ActiveLearningModel(
        split.X_train,
        split.Y_train,
        split.X_train,
        split.Y_train,
        split.X_test,
        split.Y_test,
        n_queries=5,  # initial score row + 2 batches of 2
        results_dir=results_dir,
        epochs=2,
        n_initial=n_initial,
        query_strategy=query_strategy,
        batch_size=batch_size,
    )
    assert model.test_performance is not None

    trace = pd.read_csv(results_dir / "test_initial_stats.csv", index_col=0)
    assert "n_labeled" in trace.columns
    # One row for the initial model, then one row per batch (not per point):
    # n_labeled should jump by batch_size each row, not by 1.
    assert trace["n_labeled"].tolist() == [
        n_initial,
        n_initial + batch_size,
        n_initial + 2 * batch_size,
    ]
