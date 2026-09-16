import pandas as pd
import pytest

from almolml.pipeline import SCAMsPipeline


def test_full_pipeline_runs_end_to_end_and_writes_results(
    tmp_path, monkeypatch, synthetic_datasets_dir
):
    # SCAMsPipeline resolves its own output directory relative to REPO_ROOT
    # (imported from almolml.paths), so redirect that to a temp location for
    # the duration of this test instead of touching the real repo's
    # Results/ directory.
    import almolml.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "REPO_ROOT", tmp_path)

    pipeline = SCAMsPipeline(
        "SF_TTS",
        str(synthetic_datasets_dir),
        iterations=1,
        epochs=2,
        max_queries=2,
        overwrite=True,
    )

    results_dir = tmp_path / "Results" / "SF_TTS"
    assert (results_dir / "DeepSCAMs.csv").exists()
    assert (results_dir / "TF_ML_non_AL.csv").exists()
    assert (results_dir / "TF_ML_AL.csv").exists()

    deepscams = pd.read_csv(results_dir / "DeepSCAMs.csv")
    assert len(deepscams) == 1
    assert (
        deepscams.loc[0, "Accuracy_test"]
        == pipeline.deepscams_results.iloc[0]["Accuracy_test"]
    )


def test_pipeline_refuses_to_clobber_existing_results_without_overwrite(
    tmp_path, monkeypatch, synthetic_datasets_dir
):
    import almolml.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "REPO_ROOT", tmp_path)
    (tmp_path / "Results" / "SF_TTS").mkdir(parents=True)

    # Non-interactive (pytest's stdin isn't a TTY): should fail fast with a
    # clear error instead of hanging on input().
    with pytest.raises(FileExistsError):
        SCAMsPipeline(
            "SF_TTS",
            str(synthetic_datasets_dir),
            iterations=1,
            epochs=1,
            max_queries=1,
            overwrite=False,
        )
