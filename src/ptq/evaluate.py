"""Applying a quantization scheme to a trained model, and scoring what it cost.

Every measurement is a distance from something known: perplexity from the
corpus's entropy floor, weights from their fp32 originals, memory from fp16.
Nothing here reports a number that needs another number to interpret.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .methods import HessianAccumulator, act_scale, awq, gptq
from .model import TinyGPT, batches, perplexity
from .quant import QSpec, fp16_bytes, packed_bytes, quantize, rel_error


@dataclass
class Result:
    method: str
    spec: str
    ppl: float
    ppl_floor: float
    fp32_ppl: float
    bytes_q: int
    bytes_fp16: int
    layer_err: dict[str, float]

    @property
    def compression(self) -> float:
        return self.bytes_fp16 / max(self.bytes_q, 1)

    @property
    def d_ppl(self) -> float:
        """Absolute perplexity given away against the fp32 model."""
        return self.ppl - self.fp32_ppl

    @property
    def pct(self) -> float:
        return (self.ppl / self.fp32_ppl - 1.0) * 100.0

    @property
    def above_oracle(self) -> float:
        """Distance from the true generator's perplexity on this sample.

        Deliberately *not* used as the denominator of a ratio. The trained model
        sits 0.023 above the oracle, so dividing by that headroom turns a
        negligible absolute change into a frightening multiple and would be a
        gameable metric rather than an informative one.
        """
        return self.ppl - self.ppl_floor

    @property
    def mean_err(self) -> float:
        return sum(self.layer_err.values()) / max(len(self.layer_err), 1)


@torch.no_grad()
def collect_calibration(model: TinyGPT, data: torch.Tensor,
                        n_batches: int = 16, bs: int = 16, seed: int = 7):
    """Run calibration data and capture each quantizable layer's inputs.

    Hessians are accumulated streaming rather than storing activations, because
    storing them is what makes GPTQ memory-hungry on real models and there is no
    reason to pretend otherwise here.
    """
    layers = model.quantizable()
    hess = {n: HessianAccumulator(m.in_features) for n, m in layers.items()}
    scales = {n: torch.zeros(m.in_features) for n, m in layers.items()}
    counts = {n: 0 for n in layers}
    handles = []

    def hook(name):
        def fn(_mod, inp, _out):
            x = inp[0].detach()
            hess[name].add(x)
            scales[name] += act_scale(x)
            counts[name] += 1
        return fn

    for name, mod in layers.items():
        handles.append(mod.register_forward_hook(hook(name)))
    for x, _ in batches(data, model.c.block, bs, seed, n_batches):
        model(x)
    for h in handles:
        h.remove()

    missed = [n for n, c in counts.items() if c == 0]
    if missed:
        # A hook that never fires yields an all-zero Hessian, which GPTQ reads
        # as "every input channel is dead" and answers by zeroing the weight.
        # It looks like a working run. Fail loudly instead.
        raise RuntimeError(
            f"no calibration data captured for {missed} — the forward hook never "
            f"fired, so these layers cannot be calibrated")

    H = {n: hess[n].finalize() for n in layers}
    S = {n: scales[n] / max(counts[n], 1) for n in layers}
    return H, S


@torch.no_grad()
def apply_quantization(model: TinyGPT, spec: QSpec, method: str = "rtn",
                       H=None, S=None) -> tuple[TinyGPT, dict[str, float], int]:
    """Return a quantized copy, per-layer weight error, and packed size."""
    q = copy.deepcopy(model)
    errs, total = {}, 0

    for name, mod in q.quantizable().items():
        W = mod.weight.data
        if method == "rtn":
            Wq = quantize(W, spec)
        elif method == "gptq":
            Wq = gptq(W, H[name], spec)
        elif method == "awq":
            Wq, _, _ = awq(W, S[name], spec)
        else:
            raise ValueError(f"unknown method {method!r}")

        errs[name] = rel_error(W, Wq)
        mod.weight.data = Wq
        total += packed_bytes(W, spec)

    return q, errs, total


def evaluate(model: TinyGPT, val: torch.Tensor, spec: QSpec, method: str,
             floor: float, fp32_ppl: float, H=None, S=None) -> Result:
    q, errs, nbytes = apply_quantization(model, spec, method, H, S)
    base_bytes = sum(fp16_bytes(m.weight.data) for m in model.quantizable().values())
    return Result(
        method=method,
        spec=spec.label(),
        ppl=perplexity(q, val),
        ppl_floor=floor,
        fp32_ppl=fp32_ppl,
        bytes_q=nbytes,
        bytes_fp16=base_bytes,
        layer_err=errs,
    )
