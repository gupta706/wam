# TSFM–LQG: Fisher information of a time-series foundation model

Numerical companion to **`tsfm_fisher.tex`**
*("Time-Series Foundation Models as Approximate-Information-State World Models:
A Fisher-Information Theory of Stability, Observability, and Learnability").*

The code demonstrates that a time-series foundation model (TSFM, here a faithful
[CHRONOS](https://github.com/amazon-science/chronos-forecasting)-style tokenized
transformer) is an **autonomous world-action model**: it fits the observation
channel of a partially observed linear system, and its learning behavior is
governed by a single Fisher information matrix (FIM) whose eigenstructure is set
by the system's **stability margins** and **observability**.

## The system

A large, stable, partially observed **LQG** system in modal coordinates
(`lqg_system.py`):

```
x_{t+1} = A x_t + w_t,   w ~ N(0, σ_w² I)     n = 200 latent states (K = 100 modes)
o_t     = C x_t + v_t,   v ~ N(0, σ_v² I)     m = 100 observations  (C is 100×200, fat)
```

Each mode is a 2×2 damped oscillator with magnitude `r_j ∈ (0,1)` (its
*stability*) and frequency `ω_j`. `ρ(A) = max_j r_j` is the spectral radius and
`1 − ρ(A)` the stability margin. Eight of the fastest modes are left
**unobserved** (`gain = 0`) to realize the sensor-limited case.

## What each file does

| file | contents |
|------|----------|
| `lqg_system.py` | builds the modal LQG system; Lyapunov / DARE utilities; a spectral-radius rescaler (the stability knob); an "add a sensor" operator. |
| `fisher.py` | the innovations / system-ID FIM `Γ = I_θθ` about the modal magnitudes, via a fast tangent Kalman filter (the parameter-sensitivity observability Gramian); eigen-analysis; small-eigenvector → mode attribution; the analytic `gain²/(1−r²)` law. |
| `chronos_lite.py` | the CHRONOS recipe: mean-scaling + uniform quantization → tokens, a small causal Transformer trained by cross-entropy, autoregressive / expected-value forecasting. |
| `experiments.py` | theory figures 1–3 + `results_theory.json`. |
| `experiments_tsfm.py` | trained-model figures 4–5 + `results_tsfm.json`. |

## Running

```bash
python experiments.py          # figs 1–3 (Fisher theory), ~45 s
python experiments_tsfm.py     # figs 4–5 (trained TSFM), ~7 min on CPU
QUICK=1 python experiments_tsfm.py   # fast smoke run
```

Figures are written to `figs/` (ingested by the paper); numbers to
`results_theory.json` and `results_tsfm.json`.

## The figures / claims

1. **`fig_spectrum.pdf`** — the FIM spectrum spans ~15 decades with exactly 8
   zero eigenvalues (one per unobserved mode); per-mode information collapses
   onto the `1/(1−r²)` stability law (log–log corr ≈ 0.97); each small
   eigenvector is a single physical mode.
2. **`fig_dichotomy.pdf`** — **data vs sensor**: more data cures excitation-
   limited (fast, observed) directions but not sensor-limited (unobserved)
   ones; adding channels that see the hidden modes lifts the null eigenvalue by
   ~8 orders of magnitude.
3. **`fig_stability_fisher.pdf`** — sweeping `ρ(A)`: information (`log det Γ`,
   `λ_max`) grows toward the unit circle while conditioning worsens;
   `λ_max ∝ 1/(1−ρ)`.
4. **`fig_tsfm_stability.pdf`** — the trained CHRONOS-lite's one-step forecast
   R² rises with stability toward the Kalman-optimal ceiling and tracks
   `log det Γ`.
5. **`fig_learning_curve.pdf`** — forecast R² rises toward the ceiling with
   data; the Cramér–Rao `1/√(Tλ)` law shows sloppy modes are learned last, in
   proportion to their Fisher eigenvalue.

## Notes

* `torch 2.0.1` here has an ABI break with `numpy 2.0`, so `chronos_lite.py`
  avoids the `torch.from_numpy` / `.numpy()` bridge (it crosses via Python
  lists). Everything else is float64 numpy/scipy.
* All runs are seeded and reproducible.
