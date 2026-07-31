"""Weight quantizers, and the granularity that decides whether they work.

Everything here is round-to-nearest. The interesting variable is not the
rounding — it is what shares a scale. A single scale for a whole tensor is
hostage to its largest magnitude; one scale per group of 128 input channels is
not. Most of the accuracy in practical 4-bit quantization comes from that choice
rather than from any clever algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PER_TENSOR = -1
PER_CHANNEL = 0        # one scale per output row


@dataclass(frozen=True)
class QSpec:
    bits: int = 4
    group_size: int = 128      # PER_TENSOR, PER_CHANNEL, or a positive width
    symmetric: bool = False

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def label(self) -> str:
        g = {PER_TENSOR: "per-tensor", PER_CHANNEL: "per-channel"}.get(
            self.group_size, f"g{self.group_size}")
        return f"int{self.bits}/{g}/{'sym' if self.symmetric else 'asym'}"


def _reshape_groups(W: torch.Tensor, group_size: int):
    """Return a view of shape (n_groups, group_width) plus how to invert it.

    Weights are (out_features, in_features). Grouping runs along the input
    dimension, which is the one a matmul contracts over.
    """
    out_f, in_f = W.shape
    if group_size == PER_TENSOR:
        return W.reshape(1, -1), (out_f, in_f)
    if group_size == PER_CHANNEL:
        return W.reshape(out_f, in_f), (out_f, in_f)
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group {group_size}")
    return W.reshape(-1, group_size), (out_f, in_f)


def quantize(W: torch.Tensor, spec: QSpec) -> torch.Tensor:
    """Quantize then immediately dequantize — 'fake quantization'.

    The returned tensor is float and holds only representable values, which is
    what makes the error measurable in the same units as the original. Real
    deployment packs these into int4; the packing is a storage question and is
    measured separately in `packed_bytes`.
    """
    G, shape = _reshape_groups(W.float(), spec.group_size)
    qmax = spec.levels - 1

    if spec.symmetric:
        s = G.abs().amax(dim=1, keepdim=True) / (spec.levels / 2 - 1)
        s = torch.clamp(s, min=1e-9)
        q = torch.clamp(torch.round(G / s), -(spec.levels // 2), spec.levels // 2 - 1)
        deq = q * s
    else:
        lo = G.amin(dim=1, keepdim=True)
        hi = G.amax(dim=1, keepdim=True)
        s = torch.clamp((hi - lo) / qmax, min=1e-9)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(G / s) + z, 0, qmax)
        deq = (q - z) * s

    return deq.reshape(shape).to(W.dtype)


def packed_bytes(W: torch.Tensor, spec: QSpec) -> int:
    """Storage a real implementation would use: packed codes plus the scales
    (and zero points) that make them meaningful.

    Scale overhead is not a rounding detail. At group 32 in 4-bit it adds 2 bits
    per weight in fp16 — half again on top of the payload — which is exactly why
    the finest granularity is not automatically the best trade.
    """
    n = W.numel()
    payload = n * spec.bits / 8
    G, _ = _reshape_groups(W, spec.group_size)
    n_groups = G.shape[0]
    per_group = 2 if spec.symmetric else 4      # fp16 scale (+ fp16 zero point)
    return int(payload + n_groups * per_group)


def fp16_bytes(W: torch.Tensor) -> int:
    return W.numel() * 2


def rel_error(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Relative Frobenius error, the standard weight-space distance."""
    d = torch.linalg.vector_norm(got.float() - ref.float())
    n = torch.linalg.vector_norm(ref.float())
    return float(d / torch.clamp(n, min=1e-12))
