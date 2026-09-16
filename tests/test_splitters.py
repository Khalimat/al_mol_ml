from almolml.dataset import Dataset
from almolml.splitters import AlmostNoValidation, BSplitter, SSplitter, TTSSplitter
from almolml.utilities import dataset_to_splitter


def _build_dataset(train_validation_df):
    return Dataset(train_validation_df, "ID", "Smiles String", "agg?")


def test_tts_splitter_splits_all_rows(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(TTSSplitter, dataset)
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(train_validation_df)
    assert split.X_test.shape[0] >= 1


def test_almost_no_validation_keeps_exactly_one_sample_per_class(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(AlmostNoValidation, dataset)
    assert split.Y_test.shape[0] == 2
    assert set(split.Y_test.tolist()) == {0, 1}
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(train_validation_df)


def test_butina_splitter_splits_all_rows(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(BSplitter, dataset)
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(train_validation_df)


def test_scaffold_splitter_splits_all_rows(train_validation_df):
    dataset = _build_dataset(train_validation_df)
    split = dataset_to_splitter(SSplitter, dataset)
    assert split.X_train.shape[0] + split.X_test.shape[0] == len(train_validation_df)
