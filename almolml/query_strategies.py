"""Acquisition functions ("query strategies") for the active learner in
active_learning.py. The single-point strategies have the signature
`strategy(estimator, X_pool, X_training=None, y_training=None) ->
np.ndarray([index])`. The batch strategies additionally take `batch_size`
and return `batch_size` indices at once: `strategy(estimator, X_pool,
X_training=None, y_training=None, batch_size=5) -> np.ndarray([idx, ...])`.
`y_training` (the true labels of the currently labeled set) is only used
by the DIRECT strategies; the others ignore it.

`estimator` is the (scaler, classifier) sklearn Pipeline built by
make_mlp_pipeline in models.py; every strategy except entropy_query and
batch_direct_query's entropy fallback reaches into its "mlp" step's
underlying PyTorch module (`.module_`, populated by skorch after
fitting), so they only work with that specific pipeline shape -- they are
not generic like entropy_query.

References:
- entropy_query: classic uncertainty sampling (Lewis & Gale, SIGIR 1994).
- bald_query: Bayesian Active Learning by Disagreement, estimated via
  MC-Dropout (Gal, Islam & Ghahramani, "Deep Bayesian Active Learning with
  Image Data", ICML 2017).
- batch_bald_query: BatchBALD (Kirsch, van Amersfoort & Gal, NeurIPS
  2019), which selects a *jointly* informative batch instead of the
  independently-highest-scoring points bald_query would pick one at a
  time -- see its docstring for why that distinction matters.
- core_set_query / batch_core_set_query: greedy k-center diversity
  sampling in the model's learned embedding space (Sener & Savarese,
  "Active Learning for Convolutional Neural Networks: A Core-Set
  Approach", ICLR 2018). The batch version is the paper's actual
  batch algorithm; the single-point version is just one greedy step of it.
- direct_query / batch_direct_query: adapted from DIRECT, an
  imbalance-aware acquisition function that reduces active learning to a
  1-D separation-threshold problem (Zhang, Katz-Samuels & Nowak, "Improved
  Algorithms for Deep Active Learning under Imbalance via Optimal
  Separation", ICML 2025, arXiv:2312.09196). See each docstring for how
  they adapt the paper's batch/multi-round algorithm.
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


def batch_bald_query(
    estimator, X_pool, X_training=None, y_training=None, batch_size=5, n_mc_samples=20
):
    """BatchBALD: greedily builds a batch of `batch_size` pool points that
    *jointly* maximize mutual information with the model's (MC-Dropout)
    posterior -- I(y_1, ..., y_b; theta) -- rather than independently
    ranking points by their own bald_scores and taking the top `batch_size`
    (which is what repeatedly calling bald_query would do). Taking the
    top-k independently tends to pick redundant points that are all
    uncertain for the *same* reason (e.g. several points clustered right on
    one decision boundary): each one's individual BALD score says nothing
    about how much labeling it tells you beyond what a similar, already-
    selected point already would. BatchBALD's joint objective explicitly
    penalizes that redundancy, favoring a batch that's diverse in *why*
    each point is uncertain.

    Implementation note: conditional on one dropout mask (one MC sample),
    pool points are independent, so a batch's expected per-mask entropy is
    just the sum of each member's own expected entropy -- exactly the
    `expected_entropy` term bald_scores already computes per point, no
    extra work needed there. The batch's *predictive* entropy is not
    similarly decomposable, though: it requires enumerating the joint
    distribution over all 2**batch_size label configurations for the
    candidate batch, evaluated via the same MC samples used per-point. This
    is built up incrementally, one point at a time, over `batch_size`
    greedy steps.

    Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). "BatchBALD: Efficient
    and Diverse Batch Acquisition for Deep Bayesian Active Learning."
    NeurIPS 2019.

    Cost is exponential in `batch_size` (2**batch_size joint label
    configurations per candidate per greedy step), so this is only
    practical for modest batch sizes.
    """
    probs = np.clip(_mc_dropout_probabilities(estimator, X_pool, n_mc_samples), 1e-6, 1 - 1e-6)
    n_pool = probs.shape[1]
    batch_size = min(batch_size, n_pool)

    expected_entropy_per_point = _binary_entropy(probs).mean(axis=0)

    selected = []
    remaining = list(range(n_pool))
    # joint_probs[k, config]: P(y_selected = config | dropout mask k). One
    # column per possible joint label assignment of the batch selected so
    # far; starts as a single (trivial, empty-batch) config of probability
    # 1 under every mask.
    joint_probs = np.ones((probs.shape[0], 1))

    for _ in range(batch_size):
        best_gain, best_idx, best_joint = -np.inf, None, None
        for idx in remaining:
            p1 = probs[:, idx]
            # Expand each existing joint config into two: append a 0 bit
            # (weight by 1 - p1) or a 1 bit (weight by p1) for this point.
            candidate_joint = np.concatenate(
                [joint_probs * (1 - p1)[:, None], joint_probs * p1[:, None]], axis=1
            )
            mean_joint = candidate_joint.mean(axis=0)
            predictive_entropy = -(mean_joint * np.log(mean_joint + 1e-12)).sum()
            expected_entropy = expected_entropy_per_point[selected + [idx]].sum()
            gain = predictive_entropy - expected_entropy
            if gain > best_gain:
                best_gain, best_idx, best_joint = gain, idx, candidate_joint

        selected.append(best_idx)
        remaining.remove(best_idx)
        joint_probs = best_joint

    return np.array(selected)


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


def batch_core_set_query(estimator, X_pool, X_training=None, y_training=None, batch_size=5):
    """Greedy k-center batch selection: the actual published Core-Set batch
    algorithm (Algorithm 1 in Sener & Savarese), rather than core_set_query's
    single greedy step of it. Repeats the same farthest-point step
    `batch_size` times, adding each pick's embedding to the reference set
    before choosing the next -- so later picks in the batch are diverse
    with respect to *each other*, not just the already-labeled set. Taking
    the top `batch_size` farthest-from-labeled points independently (in one
    shot, without this update) would instead tend to cluster all of them in
    whichever single region is currently most poorly covered.

    Falls back to a random first pick if there's no labeled set yet to
    measure distance from (same as core_set_query), then builds the rest of
    the batch around that pick.

    Does not implement the paper's optional robustness refinement (their
    Algorithm 2, a mixed-integer program relaxing the k-center objective to
    tolerate outliers) -- just the core greedy k-center batch procedure.

    Sener, O., & Savarese, S. (2018). "Active Learning for Convolutional
    Neural Networks: A Core-Set Approach." ICLR 2018.
    """
    batch_size = min(batch_size, len(X_pool))
    pool_embed = _embed(estimator, X_pool)

    if X_training is None or len(X_training) == 0:
        selected = [int(np.random.randint(len(X_pool)))]
        reference_embed = pool_embed[selected]
    else:
        selected = []
        reference_embed = _embed(estimator, X_training)

    remaining = [i for i in range(len(X_pool)) if i not in selected]
    while len(selected) < batch_size and remaining:
        pick = remaining[farthest_point_index(pool_embed[remaining], reference_embed)]
        selected.append(pick)
        remaining.remove(pick)
        reference_embed = np.concatenate([reference_embed, pool_embed[[pick]]], axis=0)

    return np.array(selected)


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


def batch_direct_query(estimator, X_pool, X_training=None, y_training=None, batch_size=5):
    """Batched adaptation of DIRECT: queries a batch straddling the current
    separation-threshold estimate from both sides -- alternating nearest-
    below, nearest-above, next-nearest-below, next-nearest-above, and so on
    -- instead of just the single nearest point (direct_query). One round
    of labels can then confirm or correct the threshold's location from
    both directions at once, rather than nudging it a single point at a
    time in whichever direction happened to be nearest.

    This is still an adaptation, not literal DIRECT: the paper's VReduce
    subroutine chooses queries to shrink a formal version-space interval
    over multiple bisection rounds using the model's induced ordering
    directly, which is a more specific and better-justified procedure than
    "closest points on each side of the current estimate" -- what's
    implemented here.

    Zhang, S., Katz-Samuels, J., & Nowak, R. (2025). "Improved Algorithms
    for Deep Active Learning under Imbalance via Optimal Separation." ICML
    2025. arXiv:2312.09196.

    Falls back to the `batch_size` highest-entropy pool points if the
    labeled set doesn't yet contain both classes (undefined threshold).
    """
    batch_size = min(batch_size, len(X_pool))
    if X_training is None or y_training is None or len(set(y_training)) < 2:
        proba = np.clip(estimator.predict_proba(X_pool), 1e-12, 1.0)
        entropy = -(proba * np.log(proba)).sum(axis=1)
        return np.argsort(entropy)[::-1][:batch_size]

    minority_class = 0 if (y_training == 0).sum() < (y_training == 1).sum() else 1
    labeled_scores = estimator.predict_proba(X_training)[:, minority_class]
    is_minority = y_training == minority_class
    threshold = separation_threshold(labeled_scores, is_minority)

    pool_scores = estimator.predict_proba(X_pool)[:, minority_class]
    order = np.argsort(pool_scores)
    split = np.searchsorted(pool_scores[order], threshold)

    below = order[:split][::-1]  # walking down from just below the threshold
    above = order[split:]  # walking up from just above the threshold

    selected = []
    i = j = 0
    take_below = True
    while len(selected) < batch_size and (i < len(below) or j < len(above)):
        if take_below and i < len(below):
            selected.append(below[i])
            i += 1
        elif j < len(above):
            selected.append(above[j])
            j += 1
        elif i < len(below):
            selected.append(below[i])
            i += 1
        take_below = not take_below

    return np.array(selected)
