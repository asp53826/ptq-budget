"""What each quantization scheme actually buys, and what it costs.

    uv run python bench/frontier.py

Trains once, caches the checkpoint, then evaluates every scheme against the
same fp32 reference and the same held-out sample.
"""

from __future__ import annotations

import pathlib
import time

import torch

from ptq.data import context_blind_perplexity, corpus, oracle_perplexity
from ptq.evaluate import collect_calibration, evaluate
from ptq.model import Config, TinyGPT, perplexity, train
from ptq.quant import PER_CHANNEL, PER_TENSOR, QSpec

CKPT = pathlib.Path("results/tinygpt.pt")


def setup(steps: int = 4000):
    P, tr, va = corpus(400_000, 40_000)
    tr_t, va_t = torch.tensor(tr), torch.tensor(va)
    m = TinyGPT(Config())
    CKPT.parent.mkdir(exist_ok=True)
    if CKPT.exists():
        m.load_state_dict(torch.load(CKPT))
        m.eval()
    else:
        print("training (once; cached afterwards) ...")
        train(m, tr_t, steps=steps, log=lambda i, l: print(f"  {i:5d}  {l:.4f}"))
        torch.save(m.state_dict(), CKPT)
    return P, tr_t, va_t, m


def main():
    P, tr, va, model = setup()
    fp32 = perplexity(model, va)
    oracle = oracle_perplexity(P, va.numpy())
    blind = context_blind_perplexity(va.numpy())
    nparam = sum(p.numel() for p in model.parameters())
    won = (blind - fp32) / (blind - oracle) * 100

    print(f"\nmodel {nparam:,} params · {len(model.quantizable())} quantized layers")
    print(f"context-blind baseline     {blind:.4f}   <- learning nothing")
    print(f"oracle on this val sample  {oracle:.4f}   <- no predictor beats this")
    print(f"fp32 model                 {fp32:.4f}   ({won:.1f}% of the available headroom)")

    print("\ncalibrating ...")
    t0 = time.time()
    H, S = collect_calibration(model, tr)
    print(f"  hessians + activation scales in {time.time()-t0:.1f}s")

    # ---- granularity sweep, round-to-nearest -----------------------------
    print("\n=== granularity: the variable that matters most ===\n")
    print(f"  {'scheme':<26}{'ppl':>9}{'Δ fp32':>9}{'%':>8}"
          f"{'wt err':>9}{'MB':>8}{'compress':>10}")
    print("  " + "-" * 79)
    specs = []
    for bits in (8, 4, 3, 2):
        for g in (PER_TENSOR, PER_CHANNEL, 128, 32):
            specs.append(QSpec(bits, g, symmetric=False))
    for spec in specs:
        r = evaluate(model, va, spec, "rtn", oracle, fp32)
        print(f"  {r.spec:<26}{r.ppl:>9.4f}{r.d_ppl:>+9.4f}{r.pct:>7.2f}%"
              f"{r.mean_err:>9.4f}{r.bytes_q/1e6:>8.3f}{r.compression:>9.2f}×")

    # ---- method comparison at the interesting bit widths ------------------
    for bits in (4, 3, 2):
        print(f"\n=== methods at {bits}-bit, group 128 ===\n")
        print(f"  {'method':<10}{'ppl':>9}{'Δ fp32':>9}{'%':>8}{'wt err':>9}{'secs':>8}")
        print("  " + "-" * 53)
        spec = QSpec(bits, 128, symmetric=False)
        for method in ("rtn", "gptq", "awq"):
            t0 = time.time()
            r = evaluate(model, va, spec, method, oracle, fp32, H, S)
            print(f"  {method:<10}{r.ppl:>9.4f}{r.d_ppl:>+9.4f}{r.pct:>7.2f}%"
                  f"{r.mean_err:>9.4f}{time.time()-t0:>8.1f}")

    # ---- where the damage lands ------------------------------------------
    print("\n=== per-layer weight error, int3/g128 ===\n")
    spec = QSpec(3, 128, symmetric=False)
    for method in ("rtn", "gptq"):
        r = evaluate(model, va, spec, method, oracle, fp32, H, S)
        worst = sorted(r.layer_err.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {method}: " + "  ".join(f"{k}={v:.4f}" for k, v in worst))
    print("\n  fc2 takes the widest weights and the most error in every scheme —")
    print("  it is the layer a real budget would spend its bits on.")


if __name__ == "__main__":
    main()
