import numpy as np
import pytest
import torch

from ptq.data import (V, context_blind_perplexity, oracle_perplexity, sample,
                      transition_tensor)
from ptq.evaluate import collect_calibration
from ptq.methods import awq, gptq
from ptq.model import Config, TinyGPT
from ptq.quant import (PER_CHANNEL, PER_TENSOR, QSpec, fp16_bytes, packed_bytes,
                       quantize, rel_error)


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------

def test_oracle_is_not_beatable_by_the_generator_itself():
    """Sanity on the bound: the true distribution scores its own samples, and
    an empirical count-based model of the same order cannot do better in
    expectation."""
    P = transition_tensor(2, seed=3)
    data = sample(P, 60_000, 2, seed=4)
    assert oracle_perplexity(P, data, 2) < context_blind_perplexity(data)


def test_oracle_beats_a_shuffled_generator():
    """If the oracle were not actually using context, permuting the context
    axis would not hurt it."""
    P = transition_tensor(2, seed=3)
    data = sample(P, 40_000, 2, seed=4)
    rng = np.random.default_rng(0)
    Pshuf = P[rng.permutation(P.shape[0])]
    assert oracle_perplexity(P, data, 2) < oracle_perplexity(Pshuf, data, 2)


def test_headroom_is_large_enough_to_measure_damage():
    """Guards the mistake that made the first version of this benchmark
    useless: an order-1 source left 0.023 perplexity of headroom, so nothing
    quantization did was visible."""
    P = transition_tensor(2, seed=0)
    data = sample(P, 40_000, 2, seed=2)
    o = oracle_perplexity(P, data, 2)
    b = context_blind_perplexity(data)
    assert (b - o) / o > 1.0


# ---------------------------------------------------------------------------
# quantizers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [8, 4, 3, 2])
@pytest.mark.parametrize("group", [PER_TENSOR, PER_CHANNEL, 128, 32])
def test_quantized_values_are_representable(bits, group):
    """Output must hold at most 2**bits distinct values per group — the whole
    claim of the format."""
    W = torch.randn(64, 256)
    Wq = quantize(W, QSpec(bits, group))
    assert Wq.shape == W.shape
    if group == 128:
        for g in Wq.reshape(-1, 128):
            assert len(torch.unique(g)) <= 2 ** bits


def test_finer_granularity_never_hurts_weight_error():
    """The core claim of the granularity sweep, as a property."""
    W = torch.randn(128, 512) * torch.linspace(0.1, 4.0, 512)
    prev = None
    for group in (PER_TENSOR, PER_CHANNEL, 128, 32):
        err = rel_error(W, quantize(W, QSpec(3, group)))
        if prev is not None:
            assert err <= prev + 1e-6, group
        prev = err


def test_more_bits_never_hurts():
    W = torch.randn(64, 256)
    errs = [rel_error(W, quantize(W, QSpec(b, 128))) for b in (2, 3, 4, 8)]
    assert errs == sorted(errs, reverse=True)


def test_int8_is_near_lossless():
    W = torch.randn(64, 256)
    assert rel_error(W, quantize(W, QSpec(8, 128))) < 0.01


def test_packed_size_counts_the_scales():
    """Scale overhead is why the finest group is not automatically best. At
    group 32 in 4-bit it adds real bytes and the accounting must show it."""
    W = torch.randn(256, 512)
    fine = packed_bytes(W, QSpec(4, 32))
    coarse = packed_bytes(W, QSpec(4, 128))
    assert fine > coarse
    assert coarse < fp16_bytes(W)


def test_group_size_must_divide_the_input_dimension():
    with pytest.raises(ValueError, match="divisible"):
        quantize(torch.randn(8, 100), QSpec(4, 32))


def test_symmetric_keeps_zero_exact():
    """Symmetric quantization must represent 0 exactly — sparsity depends on it."""
    W = torch.randn(32, 128)
    W[:, 0] = 0.0
    Wq = quantize(W, QSpec(4, 128, symmetric=True))
    assert torch.allclose(Wq[:, 0], torch.zeros(32), atol=1e-9)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def _hessian(in_f, n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_f, generator=g)
    return 2.0 * (X.T @ X) / n


def test_gptq_refuses_an_empty_hessian():
    """The bug this guards: a forward hook that never fired gave an all-zero
    Hessian, GPTQ read every channel as dead, zeroed the weight, and reported a
    relative error of exactly 1.0 as though it had worked."""
    W = torch.randn(32, 128)
    with pytest.raises(ValueError, match="no calibration input"):
        gptq(W, torch.zeros(128, 128), QSpec(4, 128))


def test_gptq_beats_rtn_in_output_space():
    """GPTQ's whole purpose. Note it is *not* asked to win on weight error —
    it usually loses there, which is the point of the finding."""
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    H = _hessian(256)
    spec = QSpec(3, 128)
    X = torch.randn(256, 256)

    out = X @ W.T
    err_rtn = torch.linalg.vector_norm(X @ quantize(W, spec).T - out)
    err_gptq = torch.linalg.vector_norm(X @ gptq(W, H, spec).T - out)
    assert err_gptq < err_rtn


def test_gptq_honours_the_declared_granularity():
    """Scales are fixed at group boundaries. Deriving one per column would be a
    finer scheme than the QSpec names and would flatter GPTQ throughout."""
    W = torch.randn(16, 128)
    Wq = gptq(W, _hessian(128), QSpec(3, 128))
    assert len(torch.unique(Wq)) <= 16 * (2 ** 3)


def test_awq_returns_the_identity_scale_when_channels_are_uniform():
    """With no salient channels there is nothing to protect, so the search
    should not invent a benefit."""
    W = torch.randn(32, 128)
    s_x = torch.ones(128)
    _, scale, _ = awq(W, s_x, QSpec(4, 128))
    assert torch.allclose(scale, torch.ones(128), atol=1e-5)


def test_awq_helps_when_one_channel_dominates():
    torch.manual_seed(0)
    W = torch.randn(32, 128)
    s_x = torch.ones(128)
    s_x[:3] = 400.0                      # a few channels carry the signal
    spec = QSpec(2, 128)

    Wq_awq, _, alpha = awq(W, s_x, spec)
    Wq_rtn = quantize(W, spec)
    weighted = lambda Q: float((((Q - W) * s_x.unsqueeze(0)) ** 2).sum())
    assert weighted(Wq_awq) < weighted(Wq_rtn)
    assert alpha > 0.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_every_quantizable_layer_receives_calibration_data():
    """torch's MultiheadAttention calls out_proj through the functional path, so
    a hook on it never fires. That silently produced an all-zero Hessian; this
    is the regression pin."""
    P = transition_tensor(2, seed=0)
    data = torch.tensor(sample(P, 20_000, 2, seed=1))
    m = TinyGPT(Config())
    H, S = collect_calibration(m, data, n_batches=4, bs=8)
    assert set(H) == set(m.quantizable())
    for name, h in H.items():
        assert float(torch.diag(h).abs().sum()) > 0, name
