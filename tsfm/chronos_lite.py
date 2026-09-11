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

import multiprocessing
import os

cores = multiprocessing.cpu_count()
torch.set_num_threads(cores)
os.environ['OMP_NUM_THREADS'] = str(cores)
os.environ['MKL_NUM_THREADS'] = str(cores)

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

    def quantize(self, x: np.ndarray, s) -> np.ndarray:
        s = np.asarray(s)
        if s.ndim > 0:
            xs = np.asarray(x) / s[..., None]
        else:
            xs = np.asarray(x) / s
            
        dx = (self.q_hi - self.q_lo) / (self.B - 1)
        idx = np.floor((xs - self.edges[0]) / dx) + 1
        return np.clip(idx, 0, self.B - 1).astype(np.int64)

    def dequantize(self, ids: np.ndarray, s) -> np.ndarray:
        s = np.asarray(s)
        if s.ndim > 0:
            return self.centers[np.asarray(ids)] * s[..., None]
        else:
            return self.centers[np.asarray(ids)] * s


# --------------------------------------------------------------------------- #
#  Tiny causal Transformer over the time-series vocabulary
# --------------------------------------------------------------------------- #
class TinyTSLM(nn.Module):
    """A minimal decoder-only Transformer language model over B time-series
    tokens: token + positional embeddings, a few causal self-attention blocks,
    and a linear head producing categorical logits over the B bins."""

    def __init__(self, B: int = 1024, d_model: int = 256, n_layer: int = 4,
                 n_head: int = 8, ctx: int = 256, dropout: float = 0.1):
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
        mask = nn.Transformer.generate_square_subsequent_mask(ctx)
        self.register_buffer("attn_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, L) token ids, L <= ctx
        B_, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        h = self.tok(idx) + self.pos(pos)[None]
        # Pass BOTH mask and is_causal=True for PyTorch 2.1+ compliance
        h = self.blocks(h, mask=self.attn_mask[:L, :L], is_causal=True)
        return self.head(self.ln(h))          # (batch, L, B) logits


# --------------------------------------------------------------------------- #
#  Build a token corpus from multivariate observation trajectories
# --------------------------------------------------------------------------- #
def make_token_windows(trajectories: list, quant: MeanScaleQuantizer, ctx: int,
                       stride: int = 8, rng=None) -> np.ndarray:
    """Convert a list of multi-channel trajectories into a flat dataset of tokenized sliding windows.
    Each window (length ctx+1) is scaled independently using the mean of its first ctx steps."""
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows_list = []
    
    for traj in trajectories:
        T_i, m = traj.shape
        if T_i <= ctx:
            continue
            
        # Extract sliding windows in real units
        # traj.T is (m, T_i). sliding_window_view gives (m, num_windows, ctx+1)
        windows = sliding_window_view(traj.T, window_shape=ctx+1, axis=1)
        
        if stride > 1:
            windows = windows[:, ::stride, :]
            
        # Flatten across channels and windows -> (m * num_windows, ctx+1)
        flat_windows = windows.reshape(-1, ctx + 1)
        
        # Compute local scale 's' PER WINDOW using only the context part (first ctx steps)
        s_arr = np.mean(np.abs(flat_windows[:, :ctx]), axis=1)
        s_arr = np.where(s_arr > 1e-6, s_arr, 1.0)
        
        # Quantize all windows dynamically using their own independent scale
        tok_windows = quant.quantize(flat_windows, s_arr)
        
        windows_list.append(tok_windows)
        
    if not windows_list:
        W = np.empty((0, ctx+1), dtype=np.int64)
    else:
        W = np.concatenate(windows_list, axis=0)
        
    if rng is not None:
        rng.shuffle(W)
    return W


def train_tslm(windows: np.ndarray, B: int, ctx: int, epochs: int = 6,
               d_model: int = 64, n_layer: int = 2, batch: int = 256,
               max_batches: int = None,
               lr: float = 3e-4, seed: int = 0, resume_checkpoint: str = None, verbose: bool = False):
    """Train the tiny TS language model by cross-entropy next-token loss."""
    torch.manual_seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTSLM(B=B, d_model=d_model, n_layer=n_layer, ctx=ctx).to(device)
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if verbose:
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    n = len(windows)
    
    total_batches = (n + batch - 1) // batch
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * total_batches, eta_min=1e-5)
    
    lossfn = nn.CrossEntropyLoss()
    
    import time
    import matplotlib.pyplot as plt
    import os
    last_print_time = time.time()
    model.train()
    
    epoch_losses = []
    
    for ep in range(epochs):
        perm = torch.randperm(n) # CPU permutation
        tot = 0.0
        samples_processed = 0
        for i in range(0, n, batch):
            if max_batches is not None and (i // batch) >= max_batches:
                break
            # Safely get indices as numpy array to prevent memory leaks when indexing
            batch_indices = perm[i:i + batch].numpy()
            idx_np = windows[batch_indices]
            
            # Move only the minibatch to the GPU
            idx = torch.tensor(idx_np, dtype=torch.long, device=device)
            inp, tgt = idx[:, :-1], idx[:, 1:]
            logits = model(inp)
            loss = lossfn(logits.reshape(-1, B), tgt.reshape(-1))
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            batch_loss = float(loss)
            tot += batch_loss * idx.shape[0]
            samples_processed += idx.shape[0]
            
            # Explicitly free memory to prevent Mac swap thrashing
            del idx, inp, tgt, logits, loss
            if torch.backends.mps.is_available() and (i // batch) % 100 == 0:
                torch.mps.empty_cache()
            
            if verbose and (i // batch) % 20 == 0:
                current_time = time.time()
                elapsed = current_time - last_print_time
                total_batches = (n + batch - 1) // batch
                if max_batches is not None:
                    total_batches = min(total_batches, max_batches)
                print(f"      Batch {i // batch + 1}/{total_batches}, Current Loss: {batch_loss:.4f}, Time: {elapsed:.2f}s")
                last_print_time = current_time
                
        if verbose:
            epoch_loss = tot / max(1, samples_processed)
            print(f"    epoch {ep + 1}/{epochs}  CE = {epoch_loss:.4f} nats")
            epoch_losses.append(epoch_loss)
            
            # Save the training loss plot
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b')
            plt.title("Training Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss (nats)")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(os.path.dirname(__file__), 'tsfm_training_loss.png')
            plt.savefig(out_path)
            plt.close()
            
            # Save model checkpoint
            ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'tsfm_model_epoch_{ep+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"      Saved checkpoint to {ckpt_path}")
            
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Forecasting: autoregressive token sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forecast_channel(model: TinyTSLM, quant: MeanScaleQuantizer,
                     context: np.ndarray, H: int, n_samples: int = 20,
                     temperature: float = 1.0):
    """Probabilistic forecast of one or more univariate channels. 
    If context is (L,), returns (n_samples, H).
    If context is (C, L), returns (C, n_samples, H) in ORIGINAL units."""
    device = next(model.parameters()).device
    ctx = model.ctx
    
    is_1d = (context.ndim == 1)
    if is_1d:
        context = context[None, :]
        
    C, L = context.shape
    s_arr = np.mean(np.abs(context), axis=1)
    s_arr = np.where(s_arr > 1e-6, s_arr, 1.0)
    
    ctok = quant.quantize(context, s_arr)
    if L < ctx:
        pad = np.repeat(ctok[:, 0:1], ctx - L, axis=1)
        ctok = np.concatenate([pad, ctok], axis=1)
    ctok = ctok[:, -ctx:]
    
    base = _long(ctok).repeat_interleave(n_samples, dim=0).to(device)
    out = np.zeros((C * n_samples, H), dtype=np.int64)
    
    for h in range(H):
        logits = model(base)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                      
        out[:, h] = np.asarray(nxt[:, 0].cpu().tolist(), dtype=np.int64)
        base = torch.cat([base[:, 1:], nxt], dim=1)
        
    s_rep = np.repeat(s_arr, n_samples)
    out_real = quant.dequantize(out, s_rep)
    out_real = out_real.reshape(C, n_samples, H)
    
    if is_1d:
        return out_real[0]
    return out_real


@torch.no_grad()
def expected_next(model: TinyTSLM, windows: np.ndarray,
                  centers: np.ndarray) -> np.ndarray:
    """Mean of the predictive categorical over de-quantized bin CENTERS, given
    a batch of token windows (N, L).  Returns (N,) expected next value in
    SCALED units (multiply by each window's scale to get original units).
    This is the tokenized model's minimum-MSE one-step point forecast."""
    device = next(model.parameters()).device
    logits = model(_long(windows).to(device))[:, -1, :]        # (N, B)
    p = torch.softmax(logits, dim=-1)
    ctr = torch.tensor(np.asarray(centers).tolist(), dtype=torch.float32, device=device)
    ev = (p * ctr[None]).sum(-1)                        # (N,)
    return np.asarray(ev.cpu().tolist())


@torch.no_grad()
def one_step_ce(model: TinyTSLM, windows: np.ndarray, B: int) -> float:
    """Mean one-step cross-entropy (nats/token) on held-out token windows --
    the Chronos training objective, an empirical predictive-entropy proxy."""
    device = next(model.parameters()).device
    X = _long(windows).to(device)
    inp, tgt = X[:, :-1], X[:, 1:]
    logits = model(inp)
    ce = nn.functional.cross_entropy(logits.reshape(-1, B), tgt.reshape(-1))
    return float(ce)


