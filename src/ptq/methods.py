"""GPTQ and AWQ.

Both attack the same problem from opposite ends. GPTQ accepts the rounding and
repairs the damage: quantize one column, then push its error into the columns
not yet done, weighted by the inverse Hessian of the layer's own inputs. AWQ
never lets the damage happen: rescale input channels so the ones the
activations actually use are represented well, then quantize.

Neither is magic, and the benchmark reports where each is worth its cost and
where round-to-nearest at the right granularity matches both.
"""

from __future__ import annotations

import torch

from .quant import QSpec, quantize


# ---------------------------------------------------------------------------
# Hessian accumulation, shared by both methods
# ---------------------------------------------------------------------------

class HessianAccumulator:
    """H = 2 XᵀX over calibration inputs to one linear layer.

    Accumulated in a streaming fashion because the calibration set does not fit
    a single matmul at any useful size, and because this is exactly what a real
    implementation has to do.
    """

    def __init__(self, in_features: int):
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.n = 0

    def add(self, X: torch.Tensor) -> None:
        X = X.reshape(-1, X.shape[-1]).float()
        self.n += X.shape[0]
        self.H += 2.0 * (X.T @ X)

    def finalize(self) -> torch.Tensor:
        return self.H / max(self.n, 1)


def act_scale(X: torch.Tensor) -> torch.Tensor:
    """Mean absolute activation per input channel — what AWQ calls salience."""
    return X.reshape(-1, X.shape[-1]).abs().float().mean(dim=0)


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------

def gptq(W: torch.Tensor, H: torch.Tensor, spec: QSpec,
         damp_frac: float = 0.01, block: int = 128) -> torch.Tensor:
    """Quantize `W` (out, in) column by column, compensating as it goes.

    The damping is not optional. A layer whose calibration inputs never excite
    some direction leaves H singular there, and the Cholesky fails or produces
    garbage that the error feedback then amplifies. Damping by a fraction of the
    mean diagonal is what the reference implementation does and what makes this
    stable on real layers.
    """
    Wq = W.detach().clone().float()
    out_f, in_f = Wq.shape
    H = H.clone().float()

    dead = torch.diag(H) == 0
    if bool(dead.all()):
        raise ValueError(
            "Hessian is entirely zero: this layer saw no calibration input. "
            "Zeroing the weight here would look like a successful quantization.")
    H[dead, dead] = 1.0
    Wq[:, dead] = 0.0

    damp = damp_frac * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp

    # upper Cholesky factor of H⁻¹: row j gives how column j's error spreads
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    # width of one scale group along the input dimension
    gw = in_f if spec.group_size in (-1, 0) else spec.group_size
    grp_s = grp_z = None

    Q = torch.zeros_like(Wq)
    for start in range(0, in_f, block):
        end = min(start + block, in_f)
        W_blk = Wq[:, start:end].clone()
        Q_blk = torch.zeros_like(W_blk)
        E_blk = torch.zeros_like(W_blk)

        for j in range(end - start):
            col_index = start + j
            col = W_blk[:, j]
            d = Hinv[col_index, col_index]

            # Scales are fixed at group boundaries and held for the whole group,
            # so GPTQ is compared against RTN at the *same* granularity. Deriving
            # a fresh scale per column would quietly be a finer scheme than
            # anything in the QSpec, and would flatter GPTQ throughout.
            if col_index % gw == 0:
                s, z = _row_scales(Wq[:, col_index:col_index + gw], spec)
                grp_s, grp_z = s, z

            q = _apply(col, grp_s, grp_z, spec)
            Q_blk[:, j] = q

            err = (col - q) / d
            # push this column's error onto the ones still to be quantized
            W_blk[:, j + 1:] -= err.unsqueeze(1) @ Hinv[start + j, start + j + 1:end].unsqueeze(0)
            E_blk[:, j] = err

        Q[:, start:end] = Q_blk
        if end < in_f:
            Wq[:, end:] -= E_blk @ Hinv[start:end, end:]

    return Q.to(W.dtype)


def _row_scales(Wslice: torch.Tensor, spec: QSpec):
    """Scale (and zero point) for each output row over one group of columns.

    `per-tensor` collapses to a single shared scale; everything else keeps one
    per row, which is the granularity the QSpec names.
    """
    qmax = spec.levels - 1
    if spec.symmetric:
        amax = Wslice.abs().amax(dim=1, keepdim=True)
        if spec.group_size == -1:
            amax = amax.max().expand_as(amax).clone()
        s = torch.clamp(amax / (spec.levels / 2 - 1), min=1e-9)
        return s, torch.zeros_like(s)

    lo = Wslice.amin(dim=1, keepdim=True)
    hi = Wslice.amax(dim=1, keepdim=True)
    if spec.group_size == -1:
        lo = lo.min().expand_as(lo).clone()
        hi = hi.max().expand_as(hi).clone()
    s = torch.clamp((hi - lo) / qmax, min=1e-9)
    return s, torch.round(-lo / s)


def _apply(col: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
           spec: QSpec) -> torch.Tensor:
    c = col.unsqueeze(1)
    if spec.symmetric:
        q = torch.clamp(torch.round(c / s), -(spec.levels // 2), spec.levels // 2 - 1)
        return (q * s).squeeze(1)
    q = torch.clamp(torch.round(c / s) + z, 0, spec.levels - 1)
    return ((q - z) * s).squeeze(1)


# ---------------------------------------------------------------------------
# AWQ
# ---------------------------------------------------------------------------

def awq(W: torch.Tensor, s_x: torch.Tensor, spec: QSpec,
        grid: int = 20) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Search a per-input-channel scale that minimises output error.

    Returns (quantized weight in the *original* space, the scale, the chosen α).

    The scale is folded back out afterwards so the result is directly comparable
    with RTN and GPTQ. In deployment it would instead be folded into the
    preceding layer, which costs nothing at inference — that is AWQ's actual
    selling point and this benchmark does not model it.
    """
    W = W.detach().float()
    s_x = s_x.float().clamp(min=1e-8)

    best, best_alpha, best_scale = None, 0.0, None
    ref_out = W  # error measured in weight space, weighted by activation scale

    for i in range(grid + 1):
        alpha = i / grid
        s = s_x.pow(alpha)
        s = s / s.mean().clamp(min=1e-8)          # keep magnitudes comparable

        Wq = quantize(W * s.unsqueeze(0), spec) / s.unsqueeze(0)
        # weight the error by how much the activations actually use each channel
        err = float((((Wq - ref_out) * s_x.unsqueeze(0)) ** 2).sum())

        if best is None or err < best:
            best, best_alpha, best_scale = err, alpha, s

    Wq = quantize(W * best_scale.unsqueeze(0), spec) / best_scale.unsqueeze(0)
    return Wq.to(W.dtype), best_scale, best_alpha
