"""
chronos_lite.py
===============
A faithful, compact re-implementation of the CHRONOS recipe (Ansari et al.,
TMLR 2024) for time-series forecasting, used as the *time-series foundation
model* (TSFM) whose learning performance we tie to the Fisher information of
the underlying LQG system.

Chronos recipe, verbatim in spirit:
  1. mean-scale each series:  s = mean(|x_context|),  x~ = x / s          (m=0)
  2. quantize scaled values into B uniform bins -> tokens in {1,...,B}
  3. train an off-the-shelf sequence model on the tokens with cross-entropy
     ("regression via classification" -- a categorical predictive law)
  4. forecast by autoregressive sampling of tokens, then de-quantize + un-scale.

The system is multivariate (m=100 channels) but -- exactly as Chronos does --
the model is *univariate*: one shared model is trained across all channels and
all series (the foundation-model / global-model regime) and applied
channel-independently at inference.

Backbone: a small causal Transformer (a decoder-only 'time-series language
model').  Kept intentionally tiny so the whole stability sweep runs on CPU.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")          # torch 2.0.1 <-> numpy 2.0 ABI warning
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))


def _long(arr) -> torch.Tensor:
    """numpy int array -> torch long tensor WITHOUT the (broken under numpy 2.0)
    torch.from_numpy bridge: go through a Python list."""
    return torch.tensor(np.asarray(arr).tolist(), dtype=torch.long)


# --------------------------------------------------------------------------- #
#  Chronos tokenizer: mean scaling + uniform quantization
# --------------------------------------------------------------------------- #
class MeanScaleQuantizer:
    """Mean scaling (m=0, s=mean|x|) followed by uniform quantization into B
    bins over [q_lo, q_hi] in scaled space.  Tokens are ids in {0,...,B-1}."""

    def __init__(self, B: int = 128, q_lo: float = -6.0, q_hi: float = 6.0):
        self.B = B
        self.q_lo, self.q_hi = q_lo, q_hi
        self.centers = np.linspace(q_lo, q_hi, B)          # bin centers c_j
        self.edges = 0.5 * (self.centers[1:] + self.centers[:-1])  # B-1 edges

    def scale(self, ctx: np.ndarray) -> float:
        s = np.mean(np.abs(ctx))
        return float(s) if s > 1e-6 else 1.0

    def quantize(self, x: np.ndarray, s: float) -> np.ndarray:
        xs = np.asarray(x) / s
        return np.digitize(xs, self.edges).astype(np.int64)   # -> {0,...,B-1}

    def dequantize(self, ids: np.ndarray, s: float) -> np.ndarray:
        return self.centers[np.asarray(ids)] * s


# --------------------------------------------------------------------------- #
#  Tiny causal Transformer over the time-series vocabulary
# --------------------------------------------------------------------------- #
class TinyTSLM(nn.Module):
    """A minimal decoder-only Transformer language model over B time-series
    tokens: token + positional embeddings, a few causal self-attention blocks,
    and a linear head producing categorical logits over the B bins."""

    def __init__(self, B: int = 128, d_model: int = 64, n_layer: int = 2,
                 n_head: int = 4, ctx: int = 48, dropout: float = 0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(B, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, B)
        mask = torch.triu(torch.ones(ctx, ctx) * float("-inf"), diagonal=1)
        self.register_buffer("attn_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, L) token ids, L <= ctx
        B_, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        h = self.tok(idx) + self.pos(pos)[None]
        h = self.blocks(h, mask=self.attn_mask[:L, :L])
        return self.head(self.ln(h))          # (batch, L, B) logits


# --------------------------------------------------------------------------- #
#  Build a token corpus from multivariate observation trajectories
# --------------------------------------------------------------------------- #
def make_token_windows(trajectories, quant: MeanScaleQuantizer, ctx: int,
                       stride: int = 8, rng=None):
    """trajectories: list of (T, m) arrays.  Each channel of each trajectory is
    an independent univariate series (Chronos global-model regime).  Returns a
    (N, ctx+1) int64 array of token windows (context + next token)."""
    wins = []
    for traj in trajectories:
        T, m = traj.shape
        for ch in range(m):
            x = traj[:, ch]
            s = quant.scale(x[:max(ctx, 8)])       # scale from an initial window
            tok = quant.quantize(x, s)
            for st in range(0, T - ctx - 1, stride):
                wins.append(tok[st:st + ctx + 1])
    W = np.asarray(wins, dtype=np.int64)
    if rng is not None:
        rng.shuffle(W)
    return W


def train_tslm(windows: np.ndarray, B: int, ctx: int, epochs: int = 6,
               d_model: int = 64, n_layer: int = 2, batch: int = 256,
               lr: float = 3e-3, seed: int = 0, verbose: bool = False):
    """Train the tiny TS language model by cross-entropy next-token loss."""
    torch.manual_seed(seed)
    model = TinyTSLM(B=B, d_model=d_model, n_layer=n_layer, ctx=ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X = _long(windows)
    n = X.shape[0]
    lossfn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = X[perm[i:i + batch]]
            inp, tgt = idx[:, :-1], idx[:, 1:]
            logits = model(inp)
            loss = lossfn(logits.reshape(-1, B), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * idx.shape[0]
        if verbose:
            print(f"    epoch {ep + 1}/{epochs}  CE = {tot / n:.4f} nats")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Forecasting: autoregressive token sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forecast_channel(model: TinyTSLM, quant: MeanScaleQuantizer,
                     context: np.ndarray, H: int, n_samples: int = 20,
                     temperature: float = 1.0):
    """Probabilistic forecast of one univariate channel.  Returns (n_samples,H)
    de-quantized sample paths in the ORIGINAL observation units."""
    ctx = model.ctx
    s = quant.scale(context)
    ctok = quant.quantize(context, s)
    if len(ctok) < ctx:
        ctok = np.concatenate([np.full(ctx - len(ctok), ctok[0]), ctok])
    ctok = ctok[-ctx:]
    base = _long(ctok)[None].repeat(n_samples, 1)              # (S, ctx)
    out = np.zeros((n_samples, H), dtype=np.int64)
    for h in range(H):
        logits = model(base)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                      # (S,1)
        out[:, h] = np.asarray(nxt[:, 0].tolist(), dtype=np.int64)
        base = torch.cat([base[:, 1:], nxt], dim=1)
    return quant.dequantize(out, s)                            # (S, H) real units


@torch.no_grad()
def expected_next(model: TinyTSLM, windows: np.ndarray,
                  centers: np.ndarray) -> np.ndarray:
    """Mean of the predictive categorical over de-quantized bin CENTERS, given
    a batch of token windows (N, L).  Returns (N,) expected next value in
    SCALED units (multiply by each window's scale to get original units).
    This is the tokenized model's minimum-MSE one-step point forecast."""
    logits = model(_long(windows))[:, -1, :]           # (N, B)
    p = torch.softmax(logits, dim=-1)
    ctr = torch.tensor(np.asarray(centers).tolist(), dtype=torch.float32)
    ev = (p * ctr[None]).sum(-1)                        # (N,)
    return np.asarray(ev.tolist())


@torch.no_grad()
def one_step_ce(model: TinyTSLM, windows: np.ndarray, B: int) -> float:
    """Mean one-step cross-entropy (nats/token) on held-out token windows --
    the Chronos training objective, an empirical predictive-entropy proxy."""
    X = _long(windows)
    inp, tgt = X[:, :-1], X[:, 1:]
    logits = model(inp)
    ce = nn.functional.cross_entropy(logits.reshape(-1, B), tgt.reshape(-1))
    return float(ce)


if __name__ == "__main__":
    # smoke test on one LQG system
    import time
    from lqg_system import build_modal_lqg

    sys = build_modal_lqg(seed=0)
    rng = np.random.default_rng(0)

    def rollout(sys, T, rng):
        n, m = sys.n, sys.m
        Wc = np.linalg.cholesky(sys.W); Vc = np.linalg.cholesky(sys.V)
        x = rng.standard_normal(n)
        O = np.zeros((T, m))
        for t in range(T):
            O[t] = sys.C @ x + Vc @ rng.standard_normal(m)
            x = sys.A @ x + Wc @ rng.standard_normal(n)
        return O

    print("generating corpus...")
    train_traj = [rollout(sys, 160, rng) for _ in range(6)]
    test_traj = [rollout(sys, 160, rng) for _ in range(2)]

    B, ctx = 128, 48
    quant = MeanScaleQuantizer(B=B)
    Wtr = make_token_windows(train_traj, quant, ctx, stride=8, rng=rng)
    Wte = make_token_windows(test_traj, quant, ctx, stride=16)
    print(f"train windows = {Wtr.shape}, test windows = {Wte.shape}")

    t0 = time.time()
    model = train_tslm(Wtr, B=B, ctx=ctx, epochs=5, verbose=True, seed=0)
    print(f"trained in {time.time() - t0:.1f}s")
    print(f"held-out one-step CE = {one_step_ce(model, Wte, B):.4f} nats")

    # forecast one channel
    O = test_traj[0]
    ctxlen = 64
    fc = forecast_channel(model, quant, O[:ctxlen, 0], H=12, n_samples=30)
    truth = O[ctxlen:ctxlen + 12, 0]
    pred = fc.mean(0)
    nmse = np.mean((pred - truth) ** 2) / (np.var(O[:, 0]) + 1e-9)
    print(f"1-channel 12-step forecast NMSE = {nmse:.3f}")
