"""
toytask.py — the Gaussian-mixture classification toy task.

THE TASK
    One "problem instance" is K Gaussians in d dimensions. Each Gaussian is one
    class. Points are sampled from them and carry the label of the Gaussian that
    produced them.

    The "network" being optimised is not a neural net - it IS the Gaussian
    parameters. Classification is by likelihood: given x, predict whichever
    Gaussian makes x most probable.

    So the parameter vector the diffusion model must generate is

        w = [ mu_1, logsigma_1, mu_2, logsigma_2, ..., mu_K, logsigma_K ]

    with 2*d numbers per component and 2*K*d numbers in total.
    For K=3, d=2 that is 12 parameters (matching the supervisor's "3 Gaussians
    with 2D means" example, plus the standard deviations).

WHY LOG-SIGMA
    Diffusion adds unconstrained Gaussian noise, but a standard deviation must
    be positive. Storing log(sigma) means EVERY vector the diffusion model can
    produce is a valid parameter set - no clipping, no rejection.

WHY COMPONENTS ARE THE UNIT
    Each component's mean and log-std are kept adjacent, so one component is one
    contiguous block. Swapping two components swaps two blocks and leaves the
    model unchanged - the exchangeability that paramtokens.py is built to
    represent.

CLOSED-FORM OPTIMUM
    For labelled data the best parameters are just the per-class empirical mean
    and standard deviation. No SGD needed. This makes generating the pool of
    "trained networks" for Stage 1 essentially free, and gives an exact yardstick
    to score the diffusion model's output against.
"""

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# A problem instance
# ---------------------------------------------------------------------------
@dataclass
class Task:
    """One classification problem: K Gaussians in d dimensions."""
    means: torch.Tensor     # (K, d) the true means
    stds: torch.Tensor      # (K, d) the true standard deviations

    @property
    def K(self):
        return self.means.shape[0]

    @property
    def d(self):
        return self.means.shape[1]

    @property
    def n_params(self):
        return 2 * self.K * self.d


def sample_task(K=3, d=2, spread=2.5, std_low=0.4, std_high=1.0, generator=None):
    """Draw a random problem instance.

    spread controls how far apart the Gaussians are: larger spread -> easier
    classification. The defaults give an optimal accuracy of roughly 0.85-0.95,
    which is meaningful (not trivially 1.0, not chance).
    """
    g = generator
    means = torch.randn(K, d, generator=g) * spread
    stds = torch.rand(K, d, generator=g) * (std_high - std_low) + std_low
    return Task(means=means, stds=stds)


def sample_data(task, n_per_class=64, generator=None):
    """Sample a labelled dataset from a task.

    Returns X (n, d) and y (n,), where y[i] is the index of the Gaussian that
    produced X[i]. This is the labelled mini-batch the diffusion model
    conditions on.
    """
    g = generator
    xs, ys = [], []
    for k in range(task.K):
        pts = torch.randn(n_per_class, task.d, generator=g)
        pts = pts * task.stds[k] + task.means[k]
        xs.append(pts)
        ys.append(torch.full((n_per_class,), k, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Packing: (means, log-stds) <-> flat parameter vector
# ---------------------------------------------------------------------------
def pack(means, log_stds):
    """(K, d), (K, d) -> flat (2*K*d,) with each component contiguous."""
    K, d = means.shape
    return torch.cat([means, log_stds], dim=1).reshape(-1)


def unpack(w, K, d):
    """flat (2*K*d,) -> (means (K,d), log_stds (K,d))."""
    block = w.reshape(K, 2 * d)
    return block[:, :d], block[:, d:]


# ---------------------------------------------------------------------------
# The closed-form optimum, and the loss that defines "good"
# ---------------------------------------------------------------------------
def optimal_params(X, y, K, d, eps=1e-3):
    """The best parameters for labelled data: per-class empirical mean and std.

    This is the Stage-1 training target w_0 - available WITHOUT gradient
    descent, which is what makes pool generation cheap for this toy task.
    """
    means = torch.zeros(K, d)
    log_stds = torch.zeros(K, d)
    for k in range(K):
        pts = X[y == k]
        if pts.shape[0] < 2:                       # degenerate class
            means[k] = 0.0
            log_stds[k] = 0.0
            continue
        means[k] = pts.mean(dim=0)
        log_stds[k] = torch.log(pts.std(dim=0).clamp_min(eps))
    return pack(means, log_stds)


def log_likelihoods(w, X, K, d):
    """Per-class log-likelihood of each point. Returns (n, K).

    log_stds are clamped before exponentiating. Without this, a badly-generated
    log-sigma of, say, -20 gives sigma ~ 2e-9, and the likelihood of any point
    not exactly at the mean explodes to ~1e17. That single number then dominates
    any average over tasks. The clamp bounds sigma to roughly [0.007, 150],
    which spans far more than the true range (~0.4 to 1.0) while keeping the
    metric finite and comparable across models.
    """
    means, log_stds = unpack(w, K, d)
    log_stds = log_stds.clamp(-5.0, 5.0)
    stds = torch.exp(log_stds).clamp_min(1e-6)
    # (n, 1, d) vs (1, K, d) -> (n, K, d)
    z = (X.unsqueeze(1) - means.unsqueeze(0)) / stds.unsqueeze(0)
    ll = -0.5 * (z ** 2) - log_stds.unsqueeze(0) - 0.5 * np.log(2 * np.pi)
    return ll.sum(dim=-1)                          # (n, K)


def classify(w, X, K, d):
    """Predict the class of each point by highest likelihood."""
    return log_likelihoods(w, X, K, d).argmax(dim=-1)


def accuracy(w, X, y, K, d):
    """Fraction of points classified correctly."""
    return (classify(w, X, K, d) == y).float().mean().item()


def nll_loss(w, X, y, K, d):
    """Negative log-likelihood of the true labels - the R_n(w) of this task.

    This is the loss whose gradient IS the score, via the bridge identity.
    Lower is better.
    """
    ll = log_likelihoods(w, X, K, d)
    return -ll.gather(1, y.unsqueeze(1)).mean()


# ---------------------------------------------------------------------------
# Pool generation: many (parameters, mini-batch) pairs for Stage 1
# ---------------------------------------------------------------------------
def make_pool(n_tasks=256, K=3, d=2, n_per_class=64, seed=0, **task_kw):
    """Build a pool of solved problem instances.

    Each entry is (w_0, X, y, task): the optimal parameter vector, the labelled
    mini-batch that defines the problem, and the underlying task. These are the
    training targets for Algorithm 1.
    """
    g = torch.Generator().manual_seed(seed)
    pool = []
    for _ in range(n_tasks):
        task = sample_task(K=K, d=d, generator=g, **task_kw)
        X, y = sample_data(task, n_per_class=n_per_class, generator=g)
        w0 = optimal_params(X, y, K, d)
        pool.append((w0, X, y, task))
    return pool


# ---------------------------------------------------------------------------
# The architecture spec, for paramtokens.py
# ---------------------------------------------------------------------------
def task_spec(K, d):
    """Describe this parameter layout to the tokenizer.

    One token per Gaussian component, each of width 2*d (mean then log-std).
    Changing K or d changes the token structure - which is exactly the
    '3 Gaussians x 2D vs 2 Gaussians x 3D' distinction.
    """
    from paramtokens import Spec, UnitGroup
    return Spec([UnitGroup("gaussian_component", count=K, width=2 * d)])


# ---------------------------------------------------------------------------
# Conditioning input: how the labelled mini-batch is presented to enc(B)
# ---------------------------------------------------------------------------
def batch_features(X, y, K, max_k=None):
    """Turn a labelled mini-batch into the (N, feature_dim) set that enc(B)
    consumes: each point's coordinates concatenated with a one-hot of its label.

    The label must be included - without it the model cannot know WHICH cluster
    is which class, and the problem becomes clustering rather than
    classification.

    max_k pads the one-hot to a fixed width so that tasks with different numbers
    of classes still produce the SAME feature dimension. Without this, a model
    conditioned on K=3 data could not accept K=2 data at all.
    """
    max_k = max_k or K
    onehot = torch.zeros(X.shape[0], max_k)
    onehot[torch.arange(X.shape[0]), y] = 1.0
    return torch.cat([X, onehot], dim=1)           # (N, d + max_k)
