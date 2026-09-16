import numpy as np

from jcim.query_strategies import (
    bald_query,
    bald_scores,
    core_set_query,
    direct_query,
    entropy_query,
    farthest_point_index,
    separation_threshold,
)


class _StubEstimator:
    def __init__(self, proba_table):
        self.proba_table = proba_table

    def predict_proba(self, X):
        return self.proba_table[: len(X)]


def test_entropy_query_picks_the_most_uncertain_row():
    estimator = _StubEstimator(np.array([[0.99, 0.01], [0.5, 0.5], [0.1, 0.9]]))
    query_idx = entropy_query(estimator, X_pool=np.zeros((3, 1)))
    assert query_idx.tolist() == [1]


def test_bald_scores_is_high_for_disagreement_low_for_consistent_confidence():
    # Column 0: every MC pass says ~0.99 -- confident and consistent, low
    # epistemic uncertainty despite each pass individually being "sure".
    # Column 1: passes disagree wildly (0.05, 0.95, 0.05, 0.95) -- the mean
    # looks uncertain (~0.5) *because* of disagreement between passes, the
    # signature of high epistemic uncertainty BALD is meant to catch.
    probs = np.array(
        [
            [0.99, 0.05],
            [0.99, 0.95],
            [0.99, 0.05],
            [0.99, 0.95],
        ]
    )
    scores = bald_scores(probs)
    assert scores[1] > scores[0]


def test_bald_query_picks_the_pool_row_with_highest_disagreement(monkeypatch):
    import jcim.query_strategies as qs

    # Column 0: both MC passes say exactly 0.5 -- consistently uncertain
    # (aleatoric), no disagreement between passes -> BALD score 0.
    # Column 1: passes are each individually confident (0.99, 0.01) but
    # disagree with each other -> high epistemic uncertainty -> high BALD.
    fake_probs = np.array([[0.5, 0.99], [0.5, 0.01]])
    monkeypatch.setattr(qs, "_mc_dropout_probabilities", lambda *a, **k: fake_probs)

    query_idx = bald_query(estimator=None, X_pool=np.zeros((2, 1)))
    assert query_idx.tolist() == [1]


def test_farthest_point_index_picks_the_most_isolated_pool_point():
    labeled_embed = np.array([[0.0, 0.0]])
    pool_embed = np.array([[0.1, 0.0], [5.0, 0.0], [0.2, 0.0]])
    assert farthest_point_index(pool_embed, labeled_embed) == 1


def test_core_set_query_falls_back_to_random_when_no_labeled_points_given():
    query_idx = core_set_query(estimator=None, X_pool=np.zeros((5, 1)), X_training=None)
    assert 0 <= query_idx[0] < 5


def test_core_set_query_uses_embedding_distance(monkeypatch):
    import jcim.query_strategies as qs

    embeddings = {
        "pool": np.array([[0.1, 0.0], [9.0, 0.0]]),
        "labeled": np.array([[0.0, 0.0]]),
    }

    def fake_embed(estimator, X):
        return embeddings["pool"] if len(X) == 2 else embeddings["labeled"]

    monkeypatch.setattr(qs, "_embed", fake_embed)

    query_idx = core_set_query(
        estimator=None, X_pool=np.zeros((2, 1)), X_training=np.zeros((1, 1))
    )
    assert query_idx.tolist() == [1]


def test_separation_threshold_lands_between_perfectly_separated_clusters():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    is_minority = np.array([False, False, False, False, True, True])
    threshold = separation_threshold(scores, is_minority)
    assert 0.4 < threshold < 0.8


def test_direct_query_falls_back_to_entropy_without_both_classes(monkeypatch):
    import jcim.query_strategies as qs

    called = {}
    monkeypatch.setattr(
        qs, "entropy_query", lambda estimator, X_pool: called.setdefault("called", True)
        or np.array([0])
    )

    estimator = _StubEstimator(np.array([[0.5, 0.5]]))
    direct_query(estimator, X_pool=np.zeros((1, 1)), X_training=None, y_training=None)
    assert called.get("called")

    direct_query(
        estimator,
        X_pool=np.zeros((1, 1)),
        X_training=np.zeros((3, 1)),
        y_training=np.array([0, 0, 0]),  # only one class present
    )
    assert called.get("called")


def test_direct_query_picks_pool_point_nearest_the_separation_threshold():
    class _KeyedProbaEstimator:
        """predict_proba keyed by array identity, since direct_query calls
        it on both X_training and X_pool with different expected scores.
        """

        def __init__(self, proba_by_id):
            self.proba_by_id = proba_by_id

        def predict_proba(self, X):
            return self.proba_by_id[id(X)]

    X_training = np.arange(6).reshape(6, 1)
    y_training = np.array([0, 0, 0, 0, 1, 1])
    training_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    X_pool = np.arange(3).reshape(3, 1)
    pool_scores = np.array([0.05, 0.65, 0.95])  # index 1 is closest to ~0.6 threshold

    estimator = _KeyedProbaEstimator(
        {
            id(X_training): np.column_stack([1 - training_scores, training_scores]),
            id(X_pool): np.column_stack([1 - pool_scores, pool_scores]),
        }
    )

    query_idx = direct_query(estimator, X_pool, X_training=X_training, y_training=y_training)
    assert query_idx.tolist() == [1]
