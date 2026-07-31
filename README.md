# ptq-budget

Post-training quantization — RTN, GPTQ and AWQ — scored against an exact fp32
reference and a corpus whose optimal perplexity is known in closed form.

[![CI](https://github.com/asp53826/ptq-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/asp53826/ptq-budget/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square)
![CPU only](https://img.shields.io/badge/CPU_only-no_GPU_required-2ea44f?style=flat-square)

> Weight error is the wrong metric, and this benchmark shows it losing. GPTQ
> ends up **further** from the original weights than round-to-nearest at every
> bit width, and closer in the only place that matters. A pipeline tuned on
> weight-space distance would reject the better method.

## The measurement problem, first

Perplexity on scraped text tells you a model got worse. It cannot tell you how
much worse it was *allowed* to get. So the corpus here is a seeded order-2
Markov chain, which makes three numbers exact:

```
context-blind baseline     23.8889   <- what learning nothing gets you
oracle on this val sample   8.2289   <- no predictor can beat this
fp32 model                 10.0600   (88.3% of the available headroom)
```

Everything between 23.89 and 8.23 is what the model learned, and therefore all
that quantization has to destroy.

**Getting that headroom right took two attempts, both instructive.** At order 1
the task is a 24-entry lookup table: the model landed 0.023 above the oracle and
2-bit weights reproduced it, so int4 per-tensor scored *better than fp32*. That
is not a quantization result, it is noise on a saturated benchmark. Order 3
overcorrected — 13,824 contexts at 28 samples each, the model captured 4.6% of
the headroom, and an undertrained model is equally undamageable. Order 2 sits
where the weights actually carry information.

## Measured, not implied

### Granularity beats algorithms

`Δ fp32` is absolute perplexity given away; 807,680-parameter model, 24
quantized layers.

| scheme | ppl | Δ fp32 | % | weight err | compress |
|---|---:|---:|---:|---:|---:|
| int8 / per-tensor | 10.0623 | +0.0023 | 0.02% | 0.0096 | 2.00× |
| int8 / g32 | 10.0597 | −0.0004 | −0.00% | 0.0047 | 1.78× |
| int4 / per-tensor | 10.4233 | +0.3633 | **3.61%** | 0.1627 | 4.00× |
| int4 / per-channel | 10.1647 | +0.1047 | 1.04% | 0.1028 | 3.82× |
| int4 / g128 | 10.1608 | +0.1007 | 1.00% | 0.0998 | 3.76× |
| int4 / g32 | 10.1164 | +0.0564 | **0.56%** | 0.0803 | 3.20× |
| int3 / per-tensor | 11.7050 | +1.6450 | **16.35%** | 0.3482 | 5.33× |
| int3 / g32 | 10.4660 | +0.4060 | **4.04%** | 0.1721 | 4.00× |
| int2 / per-tensor | 34.1534 | +24.0934 | **239.50%** | 0.7550 | 8.00× |
| int2 / g32 | 13.1687 | +3.1087 | **30.90%** | 0.4041 | 5.33× |

At 2-bit, changing nothing but what shares a scale moves the damage from
**239.5% to 30.9%** — an 8× swing from granularity alone, which is more than
any algorithm below contributes. int8 is free at any granularity.

Note the compression column: int2/g32 buys 5.33×, *less* than int3/per-tensor's
5.33× at a fraction of the damage. Scale overhead is real and the accounting
includes it.

### GPTQ wins by getting the weights more wrong

int3, group 128:

| method | ppl | Δ fp32 | % | **weight err** |
|---|---:|---:|---:|---:|
| RTN | 10.6695 | +0.6095 | 6.06% | 0.2142 |
| **GPTQ** | **10.4019** | **+0.3419** | **3.40%** | **0.2986** |
| AWQ | 10.6610 | +0.6010 | 5.97% | 0.2155 |

GPTQ halves the perplexity damage while sitting 39% further from the original
weights. It spends weight-space fidelity to buy output-space fidelity, which is
the entire idea — and it means **any pipeline selecting on weight error would
pick the worse method.** The same holds at 4-bit (0.61% vs 1.00%) and 2-bit
(26.13% vs 49.49%).

### AWQ does nothing here, and that is a statement about scale

AWQ matches RTN in every natural configuration. It protects input channels the
activations lean on, and at 800k parameters the top 1% of channels hold 1.3% of
activation mass — there is nothing to protect. Emergent outliers appear in
models three orders of magnitude larger.

So the study **injects** them, function-preservingly, and labels every line:

| gain | top 1% mass | RTN | AWQ | GPTQ | AWQ − RTN |
|---:|---:|---:|---:|---:|---:|
| 1 *(natural)* | 1.3% | 6.06% | 5.97% | 3.40% | +0.08 |
| 30 | 4.3% | 6.78% | 6.60% | 3.89% | +0.18 |
| 100 | 7.8% | 7.56% | 7.08% | 4.28% | +0.48 |
| 300 | 8.1% | 8.09% | 7.16% | 5.22% | +0.94 |
| 1000 | 6.1% | 9.65% | **6.69%** | 7.10% | **+2.96** |

The margin grows monotonically with severity, and at the strongest injection AWQ
overtakes GPTQ: error compensation cannot rescue a scale a few channels have
already ruined, while rescaling those channels can. fp32 perplexity moves by
0.012 across the whole sweep because the injection scales fc1 rows up and the
matching fc2 columns down — the function is unchanged, only the representation
becomes hostile.

## Where it loses

- **The outlier result is injected, not emergent.** Stated in the code, the
  output banner and here. This shows a mechanism works; it does not reproduce a
  property of billion-parameter models on a laptop.
- **A synthetic corpus is not language.** It buys an exact oracle at the cost of
  every property real text has — long-range structure, heavy-tailed vocabulary,
  syntax. Conclusions about granularity and about weight-error-as-metric should
  transfer; specific percentages should not.
- **Fake quantization only.** Weights are quantized then dequantized to float, so
  the numbers are accuracy results, not speed results. `packed_bytes` accounts
  for real storage including scales, but nothing here is faster — that needs
  kernels this repo does not have.
- **Activations stay in fp32.** Weight-only quantization. Activation
  quantization is where the genuinely hard outlier problems live.
- **No 4-bit kernel, so no latency number.** Claiming a speedup without one
  would be arithmetic, not measurement.
- **One model, one seed, one task.** The granularity ordering is a property and
  is tested as one; the percentages are a single point estimate.

## Verify it

```bash
make test        # 31 tests
make frontier    # granularity + method sweeps (trains once, ~4 min, then cached)
make outliers    # the injected-outlier severity sweep
```

## Use it

```python
from ptq.quant import QSpec, quantize
from ptq.methods import gptq

Wq = quantize(W, QSpec(bits=4, group_size=128))      # round to nearest
Wq = gptq(W, H, QSpec(bits=4, group_size=128))       # Hessian-compensated
```

`QSpec` carries bits, granularity and signedness; `packed_bytes` reports what a
real implementation would store, scales included.

## License

MIT
