"""A synthetic corpus whose optimal perplexity is computable exactly.

Perplexity on scraped text tells you a model got worse, not how much worse it
was allowed to get. Here the corpus comes from a seeded order-k Markov chain, so
the best achievable perplexity on any given sample is available in closed form
by evaluating the generating distribution on that sample.

**Order was tuned to make the benchmark able to detect anything at all.** At
order 1 the task is a 24-entry lookup table: the model landed 0.023 perplexity
above the oracle and 2-bit weights reproduced it almost exactly, so int4
per-tensor scored *better* than fp32. That is not a quantization result, it is
noise on a saturated task.

Order 3 overcorrected — 13,824 contexts leaves 28 training samples each, the
model captured 4.6% of the available headroom, and an undertrained model has
just as little to damage. Order 2 gives 576 contexts at ~694 samples each:
enough capacity demand that the weights carry real information, few enough that
the model actually learns it.
"""

from __future__ import annotations

import math

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwx"      # 24 symbols
V = len(ALPHABET)
ORDER = 2


def transition_tensor(order: int = ORDER, seed: int = 0,
                      concentration: float = 0.25) -> np.ndarray:
    """Row-stochastic (V**order, V) transitions drawn from a Dirichlet.

    Low concentration gives peaked rows — each context has a few likely
    successors — so a model that captures context does dramatically better than
    one that does not. That spread is the headroom quantization can eat into.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(V, concentration), size=V ** order)


def _ctx_index(window: np.ndarray) -> np.ndarray:
    """Base-V index of each context window, vectorised over positions."""
    order = window.shape[1]
    powers = V ** np.arange(order - 1, -1, -1)
    return (window * powers).sum(axis=1)


def sample(P: np.ndarray, n: int, order: int = ORDER, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.int64)
    out[:order] = rng.integers(V, size=order)
    cdf = np.cumsum(P, axis=1)
    u = rng.random(n)
    powers = V ** np.arange(order - 1, -1, -1)
    for t in range(order, n):
        ctx = int((out[t - order:t] * powers).sum())
        out[t] = int(np.searchsorted(cdf[ctx], u[t]))
    return out


def oracle_perplexity(P: np.ndarray, data: np.ndarray,
                      order: int = ORDER) -> float:
    """Perplexity of the *true* generator on this exact sequence.

    No predictor can beat this on this data, so it is the bound a finite
    evaluation actually has. An asymptotic entropy rate is not: a 20,000-token
    sample sits either side of it, and an earlier version of this file reported
    a model that appeared to beat information theory for exactly that reason.
    """
    n = len(data)
    idx = np.lib.stride_tricks.sliding_window_view(data[:-1], order)[:n - order]
    ctx = _ctx_index(idx)
    nxt = data[order:]
    with np.errstate(divide="ignore"):
        nll = -np.log(np.clip(P[ctx, nxt], 1e-300, None))
    return float(np.exp(nll.mean()))


def context_blind_perplexity(data: np.ndarray) -> float:
    """What a model that ignores context achieves — the other end of the range.

    Reported alongside the oracle so the headroom is explicit: everything
    between this and the oracle is what the model learned, and therefore all
    that quantization has to damage.
    """
    counts = np.bincount(data, minlength=V).astype(float)
    p = counts / counts.sum()
    with np.errstate(divide="ignore"):
        nll = -np.log(np.clip(p[data], 1e-300, None))
    return float(np.exp(nll.mean()))


def entropy_rate(P: np.ndarray, data: np.ndarray, order: int = ORDER) -> float:
    """Nats per symbol under the generator, estimated on `data`."""
    return math.log(oracle_perplexity(P, data, order))


def corpus(n_train: int = 400_000, n_val: int = 40_000, seed: int = 0,
           order: int = ORDER):
    P = transition_tensor(order, seed)
    train = sample(P, n_train, order, seed=seed + 1)
    val = sample(P, n_val, order, seed=seed + 2)
    return P, train, val
