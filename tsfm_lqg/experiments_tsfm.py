"""
experiments_tsfm.py
===================
The empirical half of the study: train the CHRONOS-lite time-series foundation
model on data from the LQG system and tie its LEARNING PERFORMANCE to the
system's stability margin and Fisher-information properties.

  fig_tsfm_stability.pdf -- forecast skill vs stability (spectral radius), and
                            forecast skill vs Fisher log-det.
  fig_learning_curve.pdf -- forecast skill vs training-data budget, next to the
                            Cramer-Rao 1/sqrt(T*lambda) law for a stiff vs a
                            sloppy Fisher direction.

Run:  python experiments_tsfm.py           (full; ~6-10 min on CPU)
      QUICK=1 python experiments_tsfm.py    (fast smoke run)
"""
from __future__ import annotations
import os, json, time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lqg_system import build_modal_lqg, rescale_spectral_radius
from fisher import fisher_information, fim_summaries, eig_analysis
from chronos_lite import (MeanScaleQuantizer, make_token_windows, train_tslm,
                          expected_next)
from experiments import (rollouts, kalman_one_step_r2, C_WAM, C_VLA, C_CROSS,
                         C_PURP, C_INK, C_INK2, FIG)

QUICK = os.environ.get("QUICK", "0") == "1"
RESULTS: dict = {}

# corpus / model sizes (shrunk in QUICK mode)
B, CTX = 128, 40
D_MODEL, N_LAYER = 64, 2
EPOCHS = 3 if QUICK else 4
N_TRAIN = 4 if QUICK else 6
N_TEST = 2
T_LEN = 140 if QUICK else 200
STRIDE = 8


def build_corpus(sys, n_train, T, seed):
    rng = np.random.default_rng(seed)
    tr = rollouts(sys, n_train, T, rng)
    te = rollouts(sys, N_TEST, T, rng)
    return tr, te


def tsfm_one_step_r2(model, quant, test_traj, ctx=CTX, max_ch=None):
    """One-step forecast R^2 (original units), averaged over channels, using the
    tokenized model's expected-value point forecast in teacher-forcing mode."""
    m = test_traj[0].shape[1]
    chans = range(m if max_ch is None else min(m, max_ch))
    num = 0.0; den = 0.0
    # gather windows across all test series / channels, predict in batches
    W = []; scales = []; targets = []
    for O in test_traj:
        T = O.shape[0]
        for ch in chans:
            x = O[:, ch]
            s = quant.scale(x[:ctx])
            tok = quant.quantize(x, s)
            for st in range(0, T - ctx - 1, 2):
                W.append(tok[st:st + ctx]); scales.append(s)
                targets.append(x[st + ctx])
    W = np.asarray(W, dtype=np.int64)
    scales = np.asarray(scales); targets = np.asarray(targets)
    ev = np.zeros(len(W))
    bs = 4096
    for i in range(0, len(W), bs):
        ev[i:i + bs] = expected_next(model, W[i:i + bs], quant.centers)
    pred = ev * scales
    num = np.sum((pred - targets) ** 2)
    den = np.sum((targets - targets.mean()) ** 2)
    return float(1.0 - num / max(den, 1e-9))


# --------------------------------------------------------------------------- #
#  Figure 4 : TSFM learning performance vs stability & Fisher
# --------------------------------------------------------------------------- #
def fig_tsfm_stability(seed=0):
    print("[fig_tsfm_stability] sweeping stability for the TSFM ...")
    base = build_modal_lqg(seed=seed)
    rhos = ([0.55, 0.75, 0.92] if QUICK
            else [0.45, 0.6, 0.72, 0.82, 0.9, 0.95, 0.975])
    tsfm_r2, kal_r2, fish_logdet, fish_lammax = [], [], [], []
    for rho in rhos:
        t0 = time.time()
        sysr = rescale_spectral_radius(base, rho)
        tr, te = build_corpus(sysr, N_TRAIN, T_LEN, seed=100 + int(rho * 100))
        quant = MeanScaleQuantizer(B=B)
        rng = np.random.default_rng(1)
        Wtr = make_token_windows(tr, quant, CTX, stride=STRIDE, rng=rng)
        model = train_tslm(Wtr, B=B, ctx=CTX, epochs=EPOCHS, d_model=D_MODEL,
                           n_layer=N_LAYER, seed=0)
        r2 = tsfm_one_step_r2(model, quant, te)
        kr2 = kalman_one_step_r2(sysr, te)
        G = fisher_information(sysr, T=250, burn=70, M=14, seed=5)
        s = fim_summaries(G)
        tsfm_r2.append(r2); kal_r2.append(kr2)
        fish_logdet.append(s["logdet"]); fish_lammax.append(s["lam_max"])
        print(f"    rho={rho:.3f}: TSFM R2={r2:.3f}  Kalman R2={kr2:.3f}  "
              f"logdetG={s['logdet']:.1f}  ({time.time()-t0:.0f}s)")

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(rhos, kal_r2, "s--", color=C_INK2, lw=1.5, ms=5,
               label="Kalman-optimal (all channels)")
    ax[0].plot(rhos, tsfm_r2, "o-", color=C_VLA, lw=1.8, ms=6,
               label="CHRONOS-lite TSFM (learned)")
    ax[0].set_xlabel(r"spectral radius  $\rho(A)$  (stability)")
    ax[0].set_ylabel(r"one-step forecast $R^2$")
    ax[0].set_title("(a) learning improves toward the unit circle", fontsize=9.5)
    ax[0].legend(fontsize=7.8, frameon=False, loc="upper left")

    sc = ax[1].scatter(fish_logdet, tsfm_r2, c=rhos, cmap="viridis", s=60,
                       zorder=3, edgecolor="white", lw=0.6)
    ax[1].plot(fish_logdet, tsfm_r2, "-", color=C_INK2, lw=0.8, zorder=2)
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label(r"$\rho(A)$", fontsize=8)
    ax[1].set_xlabel(r"Fisher information  $\log\det\Gamma$")
    ax[1].set_ylabel(r"TSFM forecast $R^2$")
    ax[1].set_title("(b) skill tracks Fisher information", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_tsfm_stability.pdf"))
    plt.close(fig)
    RESULTS["tsfm_stability"] = dict(
        rhos=list(map(float, rhos)), tsfm_r2=list(map(float, tsfm_r2)),
        kalman_r2=list(map(float, kal_r2)),
        fisher_logdet=list(map(float, fish_logdet)),
        fisher_lammax=list(map(float, fish_lammax)))
    print("[fig_tsfm_stability] done.")


# --------------------------------------------------------------------------- #
#  Figure 5 : learning curves and the Cramer-Rao law
# --------------------------------------------------------------------------- #
def fig_learning_curve(seed=0):
    print("[fig_learning_curve] learning curve + Cramer-Rao ...")
    rho = 0.9
    sysr = rescale_spectral_radius(build_modal_lqg(seed=seed), rho)
    # (a) forecast R^2 vs training-data budget
    Ns = ([1, 2, 4] if QUICK else [1, 2, 4, 8, 16])
    _, te = build_corpus(sysr, N_TEST, T_LEN, seed=999)
    r2_curve = []
    for N in Ns:
        rng = np.random.default_rng(2000 + N)
        tr = rollouts(sysr, N, T_LEN, rng)
        quant = MeanScaleQuantizer(B=B)
        Wtr = make_token_windows(tr, quant, CTX, stride=STRIDE,
                                 rng=np.random.default_rng(0))
        model = train_tslm(Wtr, B=B, ctx=CTX, epochs=EPOCHS, d_model=D_MODEL,
                           n_layer=N_LAYER, seed=0)
        r2 = tsfm_one_step_r2(model, quant, te, max_ch=60)
        r2_curve.append(r2)
        print(f"    N_traj={N:2d} (~{N*T_LEN*sysr.m} obs): R2={r2:.3f}")

    # Cramer-Rao: estimation std along a stiff vs a sloppy Fisher direction
    G = fisher_information(sysr, T=300, burn=80, M=18, seed=5)
    w, _ = eig_analysis(G)
    wpos = w[w > 1e-9 * w[-1]]
    lam_stiff = wpos[-1]           # stiffest direction
    lam_sloppy = wpos[0]           # sloppiest identifiable direction
    Tgrid = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000], float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    obs_counts = [N * T_LEN for N in Ns]
    ax[0].semilogx(obs_counts, r2_curve, "o-", color=C_VLA, lw=1.8, ms=6,
                   label="CHRONOS-lite TSFM")
    ax[0].axhline(kalman_one_step_r2(sysr, te), color=C_INK2, ls="--", lw=1.2,
                  label="Kalman-optimal ceiling")
    ax[0].set_xlabel("training length per channel (steps)")
    ax[0].set_ylabel(r"one-step forecast $R^2$")
    ax[0].set_title("(a) more data $\\rightarrow$ closer to the ceiling",
                    fontsize=9.5)
    ax[0].legend(fontsize=7.8, frameon=False, loc="lower right")

    ax[1].loglog(Tgrid, 1 / np.sqrt(Tgrid * lam_stiff), "o-", color=C_WAM,
                 lw=1.6, ms=4,
                 label=r"stiff dir ($\lambda_{\max}$): fast")
    ax[1].loglog(Tgrid, 1 / np.sqrt(Tgrid * lam_sloppy), "s-", color=C_PURP,
                 lw=1.6, ms=4,
                 label=r"sloppy dir ($\lambda_{\min}^{+}$): slow")
    ax[1].set_xlabel(r"series length  $T$")
    ax[1].set_ylabel(r"parameter std  $1/\sqrt{T\lambda}$")
    ax[1].set_title("(b) Cramér–Rao: sloppy modes learned last",
                    fontsize=9.5)
    ax[1].legend(fontsize=7.6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_learning_curve.pdf"))
    plt.close(fig)
    RESULTS["learning_curve"] = dict(
        rho=rho, N_traj=list(Ns), obs_per_channel=list(map(int, obs_counts)),
        tsfm_r2=list(map(float, r2_curve)),
        lam_stiff=float(lam_stiff), lam_sloppy=float(lam_sloppy),
        cond_identifiable=float(lam_stiff / lam_sloppy))
    print(f"[fig_learning_curve] done. stiff/sloppy Fisher ratio = "
          f"{lam_stiff/lam_sloppy:.1f}")


if __name__ == "__main__":
    t0 = time.time()
    fig_tsfm_stability()
    fig_learning_curve()
    out = os.path.join(os.path.dirname(__file__), "results_tsfm.json")
    json.dump(RESULTS, open(out, "w"), indent=2)
    print(f"\nTSFM figures done in {time.time()-t0:.1f}s. Wrote {out}")
