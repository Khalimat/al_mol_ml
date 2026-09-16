import pandas as pd
import pytest

from jcim.pipeline import RUN_STATS_COLUMNS, _append_run, _stats_row


class _StubModel:
    def __init__(self, test_performance, validation_performance):
        self.test_performance = pd.DataFrame([test_performance])
        self.validation_performance = pd.DataFrame([validation_performance])


def test_stats_row_puts_test_values_under_test_columns():
    # RUN_STATS_COLUMNS lists all "_test" columns before all "_validation"
    # columns, so the row must put test_performance values first. This is a
    # regression test for a bug where the row put validation_performance
    # first, silently swapping every "test" metric with the corresponding
    # "validation" metric in every result CSV the pipeline ever wrote.
    model = _StubModel(
        test_performance=[1, 2, 3, 4, 5, 6],
        validation_performance=[7, 8, 9, 10, 11, 12],
    )
    row = _stats_row(iteration=0, model=model)

    assert row == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert len(row) == len(RUN_STATS_COLUMNS)


def test_append_run_preserves_iteration_order():
    accumulated = None
    for i in range(3):
        accumulated = _append_run(accumulated, [i] + [0] * (len(RUN_STATS_COLUMNS) - 1))

    assert accumulated["Iteration"].tolist() == [0, 1, 2]
