"""A small GPT, trained far enough that quantization damage is the only thing
left to measure.

The model is deliberately modest — four layers, 128 wide. That is enough for
error to compound through depth and for granularity to matter, and small enough
that the whole benchmark runs on a laptop CPU in minutes. It is *not* enough for
the emergent activation outliers that dominate quantization of billion-parameter
models; see `bench/outliers.py`, which injects them and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import V


@dataclass
class Config:
    vocab: int = V
    block: int = 64
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    seed: int = 0


class Attention(nn.Module):
    """Written out rather than using nn.MultiheadAttention.

    Torch's MHA runs through `F.multi_head_attention_forward`, which uses
    `out_proj.weight` directly instead of calling `out_proj` as a module — so a
    forward hook on it never fires. That produced an all-zero Hessian, every
    input channel looked dead, and GPTQ silently zeroed the entire layer while
    reporting a tidy relative error of exactly 1.0. Explicit projections keep
    every quantizable weight on a path a hook can see.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.d_head = c.d_model // c.n_head
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, mask):
        B, T, C = x.shape
        def split(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + mask
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, 4 * c.d_model)
        self.fc2 = nn.Linear(4 * c.d_model, c.d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        torch.manual_seed(c.seed)
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.block, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lnf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.full((c.block, c.block), float("-inf")), 1))

    def forward(self, idx):
        _, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.lnf(x))

    def quantizable(self) -> dict[str, nn.Linear]:
        """The layers a real pipeline would quantize: the big projections.

        Embeddings, LayerNorm and the output head are left alone — that is
        standard practice, and quantizing the head in particular is known to be
        disproportionately damaging.
        """
        out = {}
        for i, b in enumerate(self.blocks):
            out[f"block{i}.fc1"] = b.fc1
            out[f"block{i}.fc2"] = b.fc2
            out[f"block{i}.attn.q"] = b.attn.q
            out[f"block{i}.attn.k"] = b.attn.k
            out[f"block{i}.attn.v"] = b.attn.v
            out[f"block{i}.attn.out"] = b.attn.out
        return out


def batches(data: torch.Tensor, block: int, bs: int, seed: int, n: int):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        ix = torch.randint(0, len(data) - block - 1, (bs,), generator=g)
        x = torch.stack([data[i:i + block] for i in ix])
        y = torch.stack([data[i + 1:i + block + 1] for i in ix])
        yield x, y


def train(model: TinyGPT, data: torch.Tensor, steps: int = 1500,
          bs: int = 48, lr: float = 3e-3, log=None) -> TinyGPT:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for i, (x, y) in enumerate(batches(data, model.c.block, bs, 1234, steps)):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.c.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (i % 250 == 0 or i == steps - 1):
            log(i, loss.item())
    model.eval()
    return model


@torch.no_grad()
def perplexity(model: TinyGPT, data: torch.Tensor, bs: int = 64,
               n_batches: int = 40, seed: int = 99) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for x, y in batches(data, model.c.block, bs, seed, n_batches):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.c.vocab), y.reshape(-1))
        tot += float(loss) * y.numel()
        cnt += y.numel()
    return math.exp(tot / cnt)
