from almolml.dataset import Dataset
from almolml.splitters import BSplitter, SSplitter, TTSSplitter
from almolml.utilities import dataset_to_splitter


def _build_dataset(train_validation_df):
    return Dataset(train_validation_df, "ID", "Smiles String", "agg?")


def _assert_real_split(split, total, expected_ratio=0.3, tolerance=0.1):
    """dataset_to_splitter must use the splitter's own real test_split_r
    (default 0.3) when the caller doesn't override it -- not silently
    collapse to a near-empty holdout, which happened once already (see
    TRAIN_VALIDATION_SPLIT_RATIO in git history).
    """
    assert split.X_train.shape[0] + split.X_test.shape[0] == total
    actual_ratio = split.X_test.shape[0] / total
    assert abs(actual_ratio - expected_ratio) < tolerance, (
        f"expected roughly a {expected_ratio:.0%} test split, got "
        f"{split.X_test.shape[0]}/{total} ({actual_ratio:.0%})"
    )


def test_tts_splitter_splits_all_rows(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(TTSSplitter, dataset)
    _assert_real_split(split, len(train_validation_df))


def test_butina_splitter_splits_all_rows(train_validation_df):
    # Unlike TTS/scaffold, Butina's test set is capped by how many
    # structurally-distinct (singleton-cluster) molecules exist, which the
    # small synthetic fixture doesn't have 30% worth of -- it should use
    # all of what's available rather than crash (see BSplitter.split's
    # fallback) or silently produce a near-empty split as before.
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(BSplitter, dataset)
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(train_validation_df)
    assert split.X_test.shape[0] >= 5


def test_scaffold_splitter_splits_all_rows(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(SSplitter, dataset)
    _assert_real_split(split, len(train_validation_df))
