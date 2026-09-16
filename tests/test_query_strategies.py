import numpy as np

from almolml.query_strategies import (
    bald_query,
    bald_scores,
    batch_bald_query,
    batch_core_set_query,
    batch_direct_query,
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
    import almolml.query_strategies as qs

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
    import almolml.query_strategies as qs

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
    import almolml.query_strategies as qs

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


def test_batch_bald_query_returns_batch_size_distinct_indices(monkeypatch):
    import almolml.query_strategies as qs

    fake_probs = np.array(
        [[0.99, 0.99, 0.99], [0.99, 0.01, 0.99], [0.01, 0.99, 0.01], [0.01, 0.01, 0.01]]
    )
    monkeypatch.setattr(qs, "_mc_dropout_probabilities", lambda *a, **k: fake_probs)

    query_idx = batch_bald_query(estimator=None, X_pool=np.zeros((3, 1)), batch_size=2)
    assert len(query_idx) == 2
    assert len(set(query_idx.tolist())) == 2


def test_batch_bald_query_prefers_a_complementary_point_over_a_redundant_duplicate(monkeypatch):
    """A and B are exact duplicates across every MC-dropout mask (perfectly
    redundant with each other); C disagrees with A under exactly the masks
    where A is confident, and agrees where A is unsure -- so C's label
    tells you something A's doesn't. Individually, A, B, and C all have
    identical bald_scores (same 0.99/0.01 magnitude pattern), so a strategy
    that just took the top-2 individual scores couldn't distinguish B from
    C for the second slot. BatchBALD's joint objective can, and should
    pick A then C -- not A then its own duplicate B.
    """
    import almolml.query_strategies as qs

    fake_probs = np.array(
        [
            [0.99, 0.99, 0.99],  # mask 0: A, B, C all confident "1"
            [0.99, 0.99, 0.01],  # mask 1: A, B confident "1"; C confident "0"
            [0.01, 0.01, 0.99],  # mask 2: A, B confident "0"; C confident "1"
            [0.01, 0.01, 0.01],  # mask 3: A, B, C all confident "0"
        ]
    )  # columns: A, B, C
    monkeypatch.setattr(qs, "_mc_dropout_probabilities", lambda *a, **k: fake_probs)

    query_idx = batch_bald_query(estimator=None, X_pool=np.zeros((3, 1)), batch_size=2)
    assert query_idx.tolist() == [0, 2]


def test_batch_core_set_query_avoids_reselecting_near_the_first_pick(monkeypatch):
    """idx2 is farthest from the single labeled point, so it's picked
    first. idx1 sits right next to idx2 and would be the "top-2 by
    distance from the original labeled set" pick, but batch_core_set_query
    should prefer the more distant-from-idx2 idx3 instead, since idx1 adds
    little once idx2 is already selected.
    """
    import almolml.query_strategies as qs

    pool_embed = np.array([[0.1, 0.0], [10.0, 0.0], [10.05, 0.0], [5.0, 0.0]])
    labeled_embed = np.array([[0.0, 0.0]])

    def fake_embed(estimator, X):
        return pool_embed if len(X) == 4 else labeled_embed

    monkeypatch.setattr(qs, "_embed", fake_embed)

    query_idx = batch_core_set_query(
        estimator=None, X_pool=np.zeros((4, 1)), X_training=np.zeros((1, 1)), batch_size=2
    )
    assert query_idx.tolist() == [2, 3]


def test_batch_core_set_query_falls_back_to_random_first_pick_when_no_labeled_points(monkeypatch):
    import almolml.query_strategies as qs

    pool_embed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    monkeypatch.setattr(qs, "_embed", lambda estimator, X: pool_embed)

    query_idx = batch_core_set_query(
        estimator=None, X_pool=np.zeros((3, 1)), X_training=None, batch_size=2
    )
    assert len(query_idx) == 2
    assert len(set(query_idx.tolist())) == 2


def test_batch_direct_query_falls_back_to_entropy_batch_without_both_classes():
    estimator = _StubEstimator(
        np.array([[0.9, 0.1], [0.6, 0.4], [0.99, 0.01], [0.5, 0.5]])
    )
    query_idx = batch_direct_query(
        estimator, X_pool=np.zeros((4, 1)), X_training=None, y_training=None, batch_size=2
    )
    # Highest entropy (closest to 0.5/0.5) rows first: index 3, then index 1.
    assert query_idx.tolist() == [3, 1]


def test_batch_direct_query_straddles_the_threshold_from_both_sides():
    class _KeyedProbaEstimator:
        def __init__(self, proba_by_id):
            self.proba_by_id = proba_by_id

        def predict_proba(self, X):
            return self.proba_by_id[id(X)]

    X_training = np.arange(6).reshape(6, 1)
    y_training = np.array([0, 0, 0, 0, 1, 1])
    training_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])  # threshold lands at 0.6

    X_pool = np.arange(6).reshape(6, 1)
    pool_scores = np.array([0.55, 0.65, 0.45, 0.75, 0.35, 0.85])

    estimator = _KeyedProbaEstimator(
        {
            id(X_training): np.column_stack([1 - training_scores, training_scores]),
            id(X_pool): np.column_stack([1 - pool_scores, pool_scores]),
        }
    )

    query_idx = batch_direct_query(
        estimator, X_pool, X_training=X_training, y_training=y_training, batch_size=4
    )
    # Nearest-below (0.55 -> idx 0), nearest-above (0.65 -> idx 1), then the
    # next-nearest on each side (0.45 -> idx 2, 0.75 -> idx 3) -- straddling
    # the threshold from both directions rather than walking one way.
    assert query_idx.tolist() == [0, 1, 2, 3]
