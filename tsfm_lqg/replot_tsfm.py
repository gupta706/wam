"""
replot_tsfm.py
==============
Redraw the trained-model figures (fig_tsfm_stability, fig_learning_curve) from
the saved results_tsfm.json -- NO retraining.  Used to apply figure/label fixes
without re-running the ~7-minute neural sweep.  Compute is decoupled from plots.
"""
from __future__ import annotations
import os, json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lqg_system import build_modal_lqg, rescale_spectral_radius
from experiments import (rollouts, kalman_one_step_r2, C_WAM, C_VLA, C_PURP,
                         C_INK2, FIG)

HERE = os.path.dirname(__file__)
R = json.load(open(os.path.join(HERE, "results_tsfm.json")))
plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.edgecolor": C_INK2,
    "axes.grid": True, "grid.color": "#d8d7d2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig4():
    d = R["tsfm_stability"]
    rhos, tsfm, kal = d["rhos"], d["tsfm_r2"], d["kalman_r2"]
    logdet = d["fisher_logdet"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(rhos, kal, "s--", color=C_INK2, lw=1.5, ms=5,
               label="Kalman-optimal (all channels)")
    ax[0].plot(rhos, tsfm, "o-", color=C_VLA, lw=1.8, ms=6,
               label="CHRONOS-lite TSFM (learned)")
    ax[0].set_xlabel(r"spectral radius  $\rho(A)$  (stability)")
    ax[0].set_ylabel(r"one-step forecast $R^2$")
    ax[0].set_title("(a) learning improves toward the unit circle", fontsize=9.5)
    ax[0].legend(fontsize=7.8, frameon=False, loc="upper left")
    sc = ax[1].scatter(logdet, tsfm, c=rhos, cmap="viridis", s=60, zorder=3,
                       edgecolor="white", lw=0.6)
    ax[1].plot(logdet, tsfm, "-", color=C_INK2, lw=0.8, zorder=2)
    fig.colorbar(sc, ax=ax[1]).set_label(r"$\rho(A)$", fontsize=8)
    ax[1].set_xlabel(r"Fisher information  $\log\det\Gamma$")
    ax[1].set_ylabel(r"TSFM forecast $R^2$")
    ax[1].set_title("(b) skill tracks Fisher information", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tsfm_stability.pdf"))
    plt.close(fig); print("redrew fig_tsfm_stability.pdf")


def fig5():
    d = R["learning_curve"]
    obs = d["obs_per_channel"]; r2 = d["tsfm_r2"]
    lam_stiff, lam_sloppy = d["lam_stiff"], d["lam_sloppy"]
    # recompute the (cheap) Kalman ceiling for the horizontal reference
    sysr = rescale_spectral_radius(build_modal_lqg(seed=0), d["rho"])
    _, te = rollouts(sysr, 2, 200, np.random.default_rng(999)), None
    te = rollouts(sysr, 2, 200, np.random.default_rng(999))
    ceil = kalman_one_step_r2(sysr, te)
    Tgrid = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000], float)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].semilogx(obs, r2, "o-", color=C_VLA, lw=1.8, ms=6,
                   label="CHRONOS-lite TSFM")
    ax[0].axhline(ceil, color=C_INK2, ls="--", lw=1.2,
                  label="Kalman-optimal ceiling")
    ax[0].set_xlabel("training length per channel (steps)")
    ax[0].set_ylabel(r"one-step forecast $R^2$")
    ax[0].set_title("(a) more data → closer to the ceiling", fontsize=9.5)
    ax[0].legend(fontsize=7.8, frameon=False, loc="lower right")
    ax[1].loglog(Tgrid, 1 / np.sqrt(Tgrid * lam_stiff), "o-", color=C_WAM,
                 lw=1.6, ms=4, label=r"stiff dir ($\lambda_{\max}$): fast")
    ax[1].loglog(Tgrid, 1 / np.sqrt(Tgrid * lam_sloppy), "s-", color=C_PURP,
                 lw=1.6, ms=4, label=r"sloppy dir ($\lambda_{\min}^{+}$): slow")
    ax[1].set_xlabel(r"series length  $T$")
    ax[1].set_ylabel(r"parameter std  $1/\sqrt{T\lambda}$")
    ax[1].set_title("(b) Cramér–Rao: sloppy modes learned last", fontsize=9.5)
    ax[1].legend(fontsize=7.6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_learning_curve.pdf"))
    plt.close(fig); print("redrew fig_learning_curve.pdf")


if __name__ == "__main__":
    fig4(); fig5()
