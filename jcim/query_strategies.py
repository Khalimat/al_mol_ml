"""Acquisition functions ("query strategies") for the active learner in
active_learning.py. Each has the signature
`strategy(estimator, X_pool, X_training=None, y_training=None) ->
np.ndarray([index])`, picking the single pool row to label next.
`y_training` (the true labels of the currently labeled set) is only used
by direct_query; the others ignore it.

`estimator` is the (scaler, classifier) sklearn Pipeline built by
make_mlp_pipeline in models.py; bald_query and core_set_query reach into
its "mlp" step's underlying PyTorch module (`.module_`, populated by
skorch after fitting), so they only work with that specific pipeline
shape -- they are not generic like entropy_query.

References:
- entropy_query: classic uncertainty sampling (Lewis & Gale, SIGIR 1994).
- bald_query: Bayesian Active Learning by Disagreement, estimated via
  MC-Dropout (Gal, Islam & Ghahramani, "Deep Bayesian Active Learning with
  Image Data", ICML 2017).
- core_set_query: greedy k-center diversity sampling in the model's
  learned embedding space (Sener & Savarese, "Active Learning for
  Convolutional Neural Networks: A Core-Set Approach", ICLR 2018).
- direct_query: adapted from DIRECT, an imbalance-aware acquisition
  function that reduces active learning to a 1-D separation-threshold
  problem (Zhang, Katz-Samuels & Nowak, "Improved Algorithms for Deep
  Active Learning under Imbalance via Optimal Separation", ICML 2025,
  arXiv:2312.09196). See direct_query's docstring for how this adapts
  their batch/multi-round algorithm to our single-point query loop.
"""

import numpy as np
import torch


def entropy_query(estimator, X_pool, X_training=None, y_training=None):
    """Index of the pool row with the highest predictive entropy -- the
    one the model's single forward pass is least sure about.

    Classic uncertainty sampling (Lewis & Gale, "A Sequential Algorithm
    for Training Text Classifiers", SIGIR 1994).
    """
    proba = np.clip(estimator.predict_proba(X_pool), 1e-12, 1.0)
    entropy = -(proba * np.log(proba)).sum(axis=1)
    return np.array([np.argmax(entropy)])


def _mc_dropout_probabilities(estimator, X, n_samples):
    """`n_samples` stochastic forward passes (dropout left ON) through the
    torch classifier at the end of `estimator`, an sklearn Pipeline of
    (scaler, classifier). Returns P(class=1) with shape (n_samples, len(X)).
    """
    classifier = estimator.named_steps["mlp"]
    X_scaled = estimator.named_steps["scaler"].transform(X).astype(np.float32)
    X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)

    module = classifier.module_
    module.train()  # keep dropout active even though we're not training
    with torch.no_grad():
        probs = [
            torch.sigmoid(module(X_tensor)).numpy() for _ in range(n_samples)
        ]
    module.eval()
    return np.stack(probs, axis=0)


def _binary_entropy(p):
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bald_scores(probs):
    """BALD score per column of `probs` (shape (n_mc_samples, n_points)):
    the mutual information between the prediction and the model's
    (dropout) posterior. High when individual stochastic passes are each
    confident but disagree with each other -- i.e. epistemic uncertainty --
    as opposed to entropy_query, which is also high for points that are
    genuinely ambiguous even to a single, certain model.

    Bayesian Active Learning by Disagreement, estimated via MC-Dropout
    (Gal, Islam & Ghahramani, "Deep Bayesian Active Learning with Image
    Data", ICML 2017).
    """
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    predictive_entropy = _binary_entropy(probs.mean(axis=0))
    expected_entropy = _binary_entropy(probs).mean(axis=0)
    return predictive_entropy - expected_entropy


def bald_query(estimator, X_pool, X_training=None, y_training=None, n_mc_samples=20):
    """Index of the pool row with the highest BALD score (see bald_scores
    for the citation)."""
    probs = _mc_dropout_probabilities(estimator, X_pool, n_mc_samples)
    return np.array([np.argmax(bald_scores(probs))])


def _embed(estimator, X):
    """Penultimate-layer representation from the torch classifier at the
    end of `estimator`: the model's learned feature space.
    """
    classifier = estimator.named_steps["mlp"]
    X_scaled = estimator.named_steps["scaler"].transform(X).astype(np.float32)
    X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)

    module = classifier.module_
    module.eval()
    with torch.no_grad():
        return module.embed(X_tensor).numpy()


def farthest_point_index(pool_embed, labeled_embed):
    """Index into `pool_embed` of the row farthest from its nearest row in
    `labeled_embed` (greedy k-center, one step): a pure diversity
    heuristic, no model uncertainty involved.
    """
    distances = np.linalg.norm(
        pool_embed[:, None, :] - labeled_embed[None, :, :], axis=2
    )
    nearest_labeled_distance = distances.min(axis=1)
    return np.argmax(nearest_labeled_distance)


def core_set_query(estimator, X_pool, X_training=None, y_training=None):
    """Index of the pool row farthest (in the model's embedding space)
    from its nearest already-labeled point: picks whichever unlabeled
    region of the input space the labeled set covers most poorly.

    Greedy k-center diversity sampling (Sener & Savarese, "Active Learning
    for Convolutional Neural Networks: A Core-Set Approach", ICLR 2018).
    """
    if X_training is None or len(X_training) == 0:
        return np.array([np.random.randint(len(X_pool))])
    pool_embed = _embed(estimator, X_pool)
    labeled_embed = _embed(estimator, X_training)
    return np.array([farthest_point_index(pool_embed, labeled_embed)])


def separation_threshold(scores, is_minority):
    """The score value that best splits `scores` into "predicted majority"
    (low scores) / "predicted minority" (high scores), by minimizing the
    balanced misclassification count against the known `is_minority`
    labels -- DIRECT's reduction of active learning to a 1-D
    threshold-finding problem, evaluated here against whatever labeled
    set is currently available (their loss(s) restricted to r in L; see
    module docstring for the paper reference).

    Parameters
    ----------
    scores: array of a "propensity toward the minority class" score
        (e.g. predicted P(minority)) for each labeled point
    is_minority: bool array, whether each labeled point's true label is
        the minority class

    Returns
    -------
    float, the estimated separating score threshold
    """
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_is_minority = is_minority[order]

    total_majority = (~sorted_is_minority).sum()
    cum_minority = np.cumsum(sorted_is_minority)
    cum_majority = np.cumsum(~sorted_is_minority)
    # loss[s]: minority points misclassified as majority (positions <= s)
    #        + majority points misclassified as minority (positions > s)
    loss = cum_minority + (total_majority - cum_majority)

    best_split = np.argmin(loss)
    next_split = min(best_split + 1, len(sorted_scores) - 1)
    return (sorted_scores[best_split] + sorted_scores[next_split]) / 2


def direct_query(estimator, X_pool, X_training=None, y_training=None):
    """Query the pool point whose predicted probability of the minority
    class is closest to the current best estimate of the class-separation
    threshold.

    Adapted from DIRECT (Zhang, Katz-Samuels & Nowak, "Improved Algorithms
    for Deep Active Learning under Imbalance via Optimal Separation", ICML
    2025, arXiv:2312.09196): it targets the boundary between the two
    classes rather than generic uncertainty (entropy_query) or diversity
    (core_set_query).

    This adapts DIRECT's batch, multi-round version-space search (their
    VReduce subroutine, which spends part of a fixed per-round budget
    narrowing a candidate threshold interval before annotating near it)
    into a single-point-per-call strategy: each call re-estimates the
    threshold from whatever labeled set the active learner has
    accumulated so far (separation_threshold), then queries the one pool
    point closest to it. Falls back to entropy_query if the labeled set
    doesn't yet contain both classes (undefined threshold).
    """
    if X_training is None or y_training is None or len(set(y_training)) < 2:
        return entropy_query(estimator, X_pool)

    minority_class = 0 if (y_training == 0).sum() < (y_training == 1).sum() else 1

    labeled_scores = estimator.predict_proba(X_training)[:, minority_class]
    is_minority = y_training == minority_class
    threshold = separation_threshold(labeled_scores, is_minority)

    pool_scores = estimator.predict_proba(X_pool)[:, minority_class]
    return np.array([np.argmin(np.abs(pool_scores - threshold))])
