# Study Design: Generalizing the Active-Learning Benchmark Beyond SCAM

Status: draft for review. No code changes yet -- this is the design doc requested before
touching `almolml/`.

## 1. Research question

The existing study (see main README) asks, for SCAM classification only: *does an
active-learning query strategy trained on a strategy-chosen subset beat a model trained on
the full labeled pool?* Answer so far: no, not beyond run-to-run noise, across 3 query
strategies x 3 dataset variants x 3 splitters.

This generalization asks two further questions:

1. Does that "no" hold on other datasets where AL is plausibly useful -- i.e. small-to-medium
   pools with scarce or imbalanced labels, which is the regime AL papers usually target and
   the regime SCAM happens to be in?
2. Does model architecture change the answer? The current study only ever tested one
   architecture family (an MLP on Morgan fingerprint + RDKit descriptors). If AL's value (or
   lack of it) is architecture-dependent, testing only one architecture silently overfits the
   conclusion to that architecture's inductive bias.

A negative result that replicates across 4-5 datasets and 3 architecture families is a much
stronger claim than the current single-dataset, single-architecture one. A positive result
(AL helps somewhere) would be just as interesting, and this design is meant to be equally
capable of finding either.

## 2. Datasets

All are single-task binary classification, matching the existing pipeline's assumptions.
Sizes/positive-rates below are from published MoleculeNet figures and should be confirmed
against the actually-downloaded file before locking in (a re-processing pass, a different
canonicalization step, or a dataset version bump can shift these slightly).

| Key | Dataset | Source task | n (approx) | Positive rate (approx) | Why it's here |
|---|---|---|---|---|---|
| `SF`/`SP1`/`SP2` | SCAM (existing) | aggregator flag | 916 / 1306 / 1696 | 29% / 50% / 61% | baseline, already run |
| `BBBP` | BBBP | blood-brain-barrier permeability (`p_np`) | ~2039 | ~76% (imbalanced toward permeable) | small pool, imbalance in the *opposite* direction from SCAM -- tests whether conclusions are an artifact of "minority = actives" |
| `CLINTOX` | ClinTox | clinical-trial toxicity failure (`CT_TOX`) | ~1478 | ~7% | closest in shape to SCAM (small pool, strong imbalance, real safety-liability label) |
| `TOX21_*` | Tox21 | pick 1-2 of 12 assays, e.g. `NR-AR` (~4% actives) and `SR-p53` (~7% actives, more label noise) | ~6-8k per assay (after dropping missing labels) | 4-15% depending on assay | multi-task dataset used as several independent single-task studies; larger pool than SCAM at similar imbalance |
| `HIV_SUB` | HIV (subsampled) | HIV replication inhibition (`HIV_active`) | full ~41k, subsample to ~6-8k keeping the true ~3.5% actives rate | ~3.5% | "large pool, rare positives" regime -- the classic case AL papers claim to help most, and one the current study never tested (its pools are all <1700) |

Open decisions (need your input before implementation):
- Exact Tox21 assay(s) -- recommend one clean/well-studied assay (`NR-AR`) plus one noisier one
  (`SR-p53`) rather than all 12, to keep the matrix bounded.
- HIV subsample size and method (recommend stratified random subsample preserving the true
  positive rate, fixed seed, documented in the prep script -- not a convenience truncation that
  happens to change the class balance).
- Whether to keep SCAM's pattern of a separate curated external test set (`test_DLS.csv`), or
  simplify to a single file with train/validation/test all carved from it by the splitter. New
  datasets don't have a curated held-out set like SCAM's, so **recommend the latter**: one file,
  test set held out by the same splitter used for train/validation, with a fixed proportion
  (e.g. keep test_split_r for validation vs. pool, plus a second held-out slice for test).

## 3. Splitters

`TTSSplitter`, `BSplitter` (Butina), `SSplitter` (scaffold) already operate on generic
X/Y/SMILES/mol/ID arrays (`almolml/splitters.py`) -- no changes needed, they generalize as-is.

Recommend **not** running all 3 splitters x all datasets (that's 5 datasets x 3 splitters = 15
split configs before even multiplying by architecture and strategy). Scope down to:
- Scaffold split (`SS`) as primary -- most realistic for prospective drug discovery, and the
  split most likely to reveal architecture differences (train/test structurally disjoint).
- TTS as a naive-baseline sanity check.
- Drop Butina for the new datasets (redundant with scaffold split's purpose: both hold out
  structurally distinct compounds; keeping both only matters for the existing SCAM ablation).

## 4. Architectures

### 4.1 A shared interface

Today `query_strategies.py`'s BALD and Core-Set functions reach directly into
`estimator.named_steps["mlp"].module_` (a skorch/torch internal) to get MC-dropout samples and
penultimate-layer embeddings. That's the one real blocker to adding architectures: a Random
Forest or a GNN can't be plugged into `bald_query`/`core_set_query` as written.

Fix: define a small backend interface every architecture implements, and have
`active_learning.py`/`query_strategies.py` call *that* instead of reaching into torch
internals directly:

```text
ModelBackend:
    fit(X, y)
    predict_proba(X) -> (n, 2)
    mc_probabilities(X, n_samples) -> (n_samples, n)   # P(class=1) per stochastic sample
    embed(X) -> (n, d)                                  # a representation space for Core-Set
```

`entropy_query` and `direct_query`/`batch_direct_query` only need `predict_proba` and already
work against anything sklearn-shaped -- no change required for those. `bald_query`/
`batch_bald_query` and `core_set_query`/`batch_core_set_query` change to call
`mc_probabilities`/`embed` on the backend instead of the current torch-specific helpers. The
existing `TorchMLPModel`/`ActiveLearningModel` wraps into `TorchMLPBackend`, preserving current
behavior exactly (MC-dropout via `module.train()` at inference, embedding via `SCAMsNet.embed`)
-- this refactor is meant to be behavior-preserving for the existing SCAM experiments, i.e. a
pure interface extraction, not a rewrite of the MLP path.

### 4.2 Tree ensemble (Random Forest)

Operates on the same Morgan-fingerprint + RDKit-descriptor features already produced by
`featurization.describe`. No new featurization needed.

- `predict_proba`: `RandomForestClassifier.predict_proba` directly.
- `mc_probabilities`: per-tree `predict_proba` across `.estimators_`, one "sample" per tree.
  This is the standard ensemble-disagreement analog to MC-dropout: both estimate epistemic
  uncertainty as disagreement across an ensemble of hypotheses (dropout masks vs. bagged
  trees) rather than a single point estimate's own entropy -- same BALD formula
  (`predictive_entropy - expected_entropy`) applies unchanged once you have a
  `(n_samples, n_points)` probability array, regardless of what produced the samples.
- `embed`: leaf-index representation. `RandomForestClassifier.apply(X)` returns, per sample,
  which leaf of each tree it landed in -- a categorical code vector `(n, n_estimators)`. Treat
  Hamming distance between leaf-code rows as the distance metric for Core-Set's greedy
  k-center step (two points that land in the same leaf across most trees are, by the forest's
  own construction, similar). This is a well-established idea (Breiman's random-forest
  proximity measure), not an ad hoc substitute.

This architecture is the cheapest to add (no new dependency, CPU-fast, no epoch/dropout
hyperparameters) and should be implemented first to validate the backend-interface refactor
before touching anything torch- or graph-related.

### 4.3 Graph neural network

Molecules as graphs (atoms = nodes, bonds = edges) rather than a fixed fingerprint -- lets the
model learn its own structural representation instead of relying on Morgan hashing.

Needs new featurization (`almolml/graph_featurization.py`): per-atom feature vectors (atomic
number, degree, formal charge, aromaticity, H count -- all cheap RDKit atom queries, one-hot
encoded) and an adjacency matrix per molecule, from the same SMILES already loaded by
`Dataset`.

Dependency decision (needs your input): PyTorch Geometric (mature, standard, but a real
dependency with version-matched compiled extensions that can be finicky on a CPU-only Mac) vs.
a small hand-rolled message-passing net in plain PyTorch, using dense per-molecule adjacency
matrices with padding/masking (molecules are small -- under ~100 atoms -- so dense adjacency is
completely tractable; no need for the sparse/scatter machinery PyG exists for at graph-classification
scale). **Recommend hand-rolled**, consistent with this project's existing stated preference
(`active_learning.py`'s docstring: replaced modAL specifically to avoid depending on a
third-party library for a handful of lines) and with the fact that these datasets are all
small enough that dense adjacency has no real cost.

Sketch: a GCN/GIN-style stack (2-3 message-passing layers + ReLU + dropout, exactly mirroring
`SCAMsNet`'s structure), a graph-level readout (sum or mean pool over node embeddings) as the
`embed()` output, then a linear head -- structurally almost a drop-in replacement for
`SCAMsNet`, just with graph convolution in place of the first dense layers. Dropout stays on
at inference for MC sampling exactly like the current model, so `mc_probabilities` needs no new
mechanism, just reads dropout-perturbed forward passes from the graph model instead of the MLP.

This is the most engineering-heavy of the three additions (new featurization, new batching
logic for variable-size graphs, new model) and should come last, once the RF backend has
proven the interface refactor works.

### 4.4 Architecture-selection stage

Per your requirement to find the best architecture per dataset: before running the full AL
comparison, run all 3 architectures on the full labeled pool (no AL) for each dataset x split,
a handful of iterations (5, cheaper than the full 10), and report mean AUC/MCC per
(architecture, dataset). Report the full grid (transparency -- this itself is a useful result,
independent of AL), and additionally use the best-per-dataset architecture as the headline
comparison in the main AL-vs-full-pool study, so the "does AL help" answer isn't handicapped by
a weak architecture choice on any given dataset.

## 5. Experiment matrix and compute budget

Full combinatorics, if unscoped: 5 datasets x 2 splits x 3 architectures x (1 baseline + 3
batch strategies) x 10 iterations = 1200 runs. That's before even counting the 4 single-point
strategies. Not tractable as a first pass.

Recommend a **phased rollout**, each phase a checkpoint to confirm before continuing:

1. **Interface refactor**: extract `ModelBackend`, wrap existing torch model into
   `TorchMLPBackend`, re-run one existing SCAM study end-to-end and confirm identical results
   to current `Results/` outputs (regression check on the refactor itself).
2. **RF backend on SCAM datasets**: validates the backend interface against a second
   architecture using data/splitters already known to work. Small iteration count (3-5) first.
3. **New datasets, existing architectures (MLP + RF)**: BBBP and ClinTox first (smallest, most
   similar in scale to SCAM), scaffold split only, pilot iteration count (5), baseline (no AL)
   + entropy + best batch strategy from the SCAM results (batched DIRECT, per the current
   README) -- not the full strategy list yet.
4. **GNN backend**: add once 1-3 are solid, same reduced scope first.
5. **Tox21 + HIV_SUB**: add once the pipeline handles the smaller new datasets cleanly (their
   larger pool size means longer AL loops -- worth confirming runtime before committing).
6. **Full matrix at 10 iterations**: only after 1-5 have validated the pipeline; scope (which
   splits, which strategies, which architectures) revisited based on what's been most
   informative so far, rather than committed to upfront.

## 6. Metrics and comparison methodology

Unchanged from the existing study, applied identically to every new (dataset, architecture)
combination: ROC AUC with DeLong CI, accuracy, F1, MCC; 10 independent iterations (fresh
initial sample + model init each time, seed = base + iteration index); final-trajectory-step
score only (not best-along-the-way, which would be look-elsewhere-biased); batch AL vs.
full-pool baseline compared via error bars that must not overlap to count as a real effect --
the same standard that overturned this project's own earlier single-run "wins" (see README
section "Results: Does Batch Active Learning Beat Full-Data Training?").

## 7. Proposed repository changes

```text
almolml/
├── model_backends.py       new: ModelBackend interface + TorchMLPBackend, RFBackend, GNNBackend
├── graph_featurization.py  new: SMILES -> per-molecule atom features + adjacency
├── query_strategies.py     modified: bald_query/core_set_query (+ batch variants) call
│                           backend.mc_probabilities()/embed() instead of torch internals
├── models.py                modified: architecture classes become thin wrappers around backends
├── pipeline.py              modified: DATASETS registry -> richer per-dataset config
│                           (file, x_col, y_col, has separate test file y/n); new
│                           --architecture flag alongside --al_strategy
├── cli.py                   modified: --architecture {mlp,rf,gnn}
Datasets/
└── (new files from a prep script, not committed raw from source without checking license/size)
scripts/
└── prepare_new_datasets.py  new: download + clean MoleculeNet CSVs into the existing
                            ID/Smiles String/<label> shape; documents provenance + seed for
                            the HIV subsample
```

## 8. Decisions (locked 2026-09-17)

1. Tox21 assays: `NR-AR` + `SR-p53` -- confirmed.
2. HIV subsample: ~6-8k, stratified on the true ~3.5% positive rate, fixed seed -- confirmed.
3. GNN library: **PyTorch Geometric** (not the hand-rolled dense-adjacency net originally
   recommended -- user has H100 cluster access, so the dependency-weight and CPU-cost concerns
   that motivated the hand-rolled recommendation don't apply; PyG's maturity and sparse-graph
   batching win instead).
4. Per-dataset test set: **single-file 3-way split** for all new datasets (train/validation/test
   all carved from the one downloaded file by the same splitter) -- "important they validate on
   the same data," i.e. no separate curated external test file like SCAM's `test_DLS.csv`.
5. Compute: H100 cluster available, so iteration count and architecture scope are not
   compute-bound the way they'd be on a laptop. First deliverable is a **pilot run script**
   (see `scripts/run_pilot.py`) covering all 3 architectures (MLP, RF, GNN) x all 5 new-dataset
   entries x scaffold split, at reduced iteration count, to validate the whole extended
   pipeline end-to-end before committing to the full 10-iteration matrix.
