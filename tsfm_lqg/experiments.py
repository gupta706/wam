"""
experiments.py
==============
Numerical study for the paper
"Time-Series Foundation Models as Approximate-Information-State World Models:
 A Fisher-Information Theory of Stability, Observability, and Learnability."

Ground truth: a large stable LQG system (n=200 latent states, m=100 observations,
K=100 physical modes) built in modal coordinates (lqg_system.py).

Learner: CHRONOS-lite, a faithful tokenized time-series foundation model
(chronos_lite.py) -- an AUTONOMOUS world-action model (no policy/action block).

Theory: the innovations / system-identification Fisher information block
Gamma = I_theta_theta about the modal magnitudes theta = (r_1,...,r_K)
(fisher.py), which for a pure forecaster is the ONLY Fisher block.

Figures produced in figs/:
  fig_spectrum.pdf         -- FIM sloppy spectrum + the 1/(1-r^2) stability law
  fig_dichotomy.pdf        -- small-eigenvalue eigenvectors: data vs sensor
  fig_stability_fisher.pdf -- FIM properties vs stability margin
  fig_tsfm_stability.pdf   -- TSFM learning performance vs stability
  fig_learning_curve.pdf   -- learning curves and the Cramer-Rao 1/(T lambda) law
and a machine-readable results.json consumed by the paper.
"""
from __future__ import annotations
import os, json, time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lqg_system import (build_modal_lqg, rescale_spectral_radius,
                        add_velocity_sensors, ModalLQG)
from fisher import (fisher_information, analytic_mode_fisher, eig_analysis,
                    small_eigenvector_report, fim_summaries)

# ---- figure style: matches the WAM/VLA paper figures ---------------------- #
C_WAM = "#2a78d6"    # blue   (world / Fisher)
C_VLA = "#eb6834"    # orange (learner)
C_CROSS = "#1baf7a"  # aqua
C_PURP = "#7a5cc0"   # purple
C_INK, C_INK2, C_GRID = "#0b0b0b", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.edgecolor": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_INK2, "ytick.color": C_INK2, "axes.grid": True,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
})
FIG = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIG, exist_ok=True)
RESULTS: dict = {}


# --------------------------------------------------------------------------- #
#  Simulation utilities
# --------------------------------------------------------------------------- #
def rollout(sys: ModalLQG, T: int, rng) -> np.ndarray:
    """One observation trajectory (T, m) from the autonomous LQG system."""
    n, m = sys.n, sys.m
    Wc = np.linalg.cholesky(sys.W + 1e-12 * np.eye(n))
    Vc = np.linalg.cholesky(sys.V + 1e-12 * np.eye(m))
    x = rng.standard_normal(n) * 0.5
    O = np.zeros((T, m))
    for t in range(T):
        O[t] = sys.C @ x + Vc @ rng.standard_normal(m)
        x = sys.A @ x + Wc @ rng.standard_normal(n)
    return O


def rollouts(sys, n_traj, T, rng):
    return [rollout(sys, T, rng) for _ in range(n_traj)]


def kalman_one_step_r2(sys: ModalLQG, test_traj, seed=0) -> float:
    """Bayes-optimal one-step forecast R^2 per channel, averaged over channels,
    using the steady-state Kalman filter with FULL (multivariate) information --
    the absolute predictability ceiling."""
    P, Sig, L, F = sys.kalman_steady_state()
    A, C = sys.A, sys.C
    num = np.zeros(sys.m); den = np.zeros(sys.m); ntot = 0
    for O in test_traj:
        xhat = np.zeros(sys.n)
        errs = []; ys = []
        for t in range(len(O)):
            ohat = C @ xhat
            errs.append(O[t] - ohat); ys.append(O[t])
            nu = O[t] - ohat
            xhat = A @ (xhat + L @ nu)
        errs = np.array(errs[20:]); ys = np.array(ys[20:])
        num += np.sum(errs ** 2, 0); den += np.sum((ys - ys.mean(0)) ** 2, 0)
        ntot += 1
    r2 = 1.0 - num / np.clip(den, 1e-9, None)
    return float(np.mean(r2))


def ar_oracle_r2(train_traj, test_traj, L=16, H=1) -> float:
    """Best univariate LINEAR predictor (order-L AR fit on abundant data):
    the achievable *univariate* forecast R^2 ceiling that a foundation model
    trained on one channel at a time can aspire to. H-step (iterated)."""
    m = train_traj[0].shape[1]
    r2s = []
    for ch in range(m):
        xs = [tr[:, ch] for tr in train_traj]
        # build AR design
        X, y = [], []
        for x in xs:
            T = len(x)
            for t in range(L, T):
                X.append(x[t - L:t][::-1]); y.append(x[t])
        X = np.array(X); y = np.array(y)
        X = np.column_stack([X, np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        # evaluate iterated H-step on test
        num = den = 0.0
        for x in test_traj_channels(test_traj, ch):
            T = len(x)
            for t in range(L, T - H):
                hist = list(x[t - L:t][::-1])
                pred = None
                for h in range(H):
                    feat = np.array(hist[:L] + [1.0])
                    pred = feat @ beta
                    hist = [pred] + hist[:-1]
                num += (pred - x[t + H - 1]) ** 2
                den += (x[t + H - 1] - x.mean()) ** 2
        r2s.append(1 - num / max(den, 1e-9))
    return float(np.mean(r2s))


def test_traj_channels(test_traj, ch):
    for tr in test_traj:
        yield tr[:, ch]


# --------------------------------------------------------------------------- #
#  Figure 1 : FIM sloppy spectrum and the stability (1/(1-r^2)) law
# --------------------------------------------------------------------------- #
def fig_spectrum(seed=0):
    print("[fig_spectrum] building system + Fisher ...")
    sys = build_modal_lqg(seed=seed)
    Gamma = fisher_information(sys, T=350, burn=90, M=24, seed=1)
    w, Vv = eig_analysis(Gamma)
    diag = np.diag(Gamma)
    ana = analytic_mode_fisher(sys)
    observed = sys.gains > 0

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.5))

    # (a) sorted eigenvalue spectrum (sloppy: many orders of magnitude)
    wpos = np.clip(w[::-1], 1e-16, None)
    ax[0].semilogy(np.arange(1, len(w) + 1), wpos, "o-", color=C_WAM,
                   ms=3, lw=1.0)
    n_zero = int(np.sum(w < 1e-9 * w[-1]))
    ax[0].axhline(1e-9 * w[-1], color=C_INK2, ls=":", lw=0.8)
    ax[0].set_xlabel("index (descending)")
    ax[0].set_ylabel(r"Fisher eigenvalue $\lambda_i(\Gamma)$")
    ax[0].set_title(f"(a) sloppy spectrum: {n_zero} sensor-limited zeros",
                    fontsize=9.5)

    # (b) gain-normalized per-mode Fisher vs stability margin: collapses onto
    #     the clean AR(1) law  Gamma_jj / ||C_j||^2  ~  (sigma_w^2/sigma_v^2)/(1-r^2)
    cnorm2 = np.array([np.sum(sys.C[:, 2 * j:2 * j + 2] ** 2)
                       for j in range(sys.K)])
    xg = 1 - sys.r[observed] ** 2
    yg = diag[observed] / cnorm2[observed]
    ax[1].loglog(xg, yg, "o", color=C_WAM, ms=4, alpha=0.8,
                 label=r"modes  $\Gamma_{jj}/\|C_j\|^2$")
    xs = np.linspace(xg.min(), xg.max(), 50)
    scale = np.median(yg * xg)
    ax[1].loglog(xs, scale / xs, "-", color=C_INK2, lw=1.3,
                 label=r"$\propto 1/(1-r^2)$")
    ax[1].set_xlabel(r"stability margin  $1-r_j^2$")
    ax[1].set_ylabel(r"gain-normalized Fisher")
    ax[1].set_title("(b) stability sets identifiability", fontsize=9.5)
    ax[1].legend(fontsize=7.6, frameon=False, loc="lower left")

    # (c) the smallest-eigenvalue eigenvectors, as |component| heat over modes
    k = 10
    order_mode = np.argsort(sys.r)      # modes fast -> slow
    Vsmall = np.abs(Vv[:, :k])[order_mode]     # (K, k)
    im = ax[2].imshow(Vsmall.T, aspect="auto", cmap="magma",
                      extent=[0, sys.K, k, 0], vmin=0, vmax=1)
    ax[2].set_xlabel("mode (fast $\\rightarrow$ slow)")
    ax[2].set_ylabel("small-eigenvalue rank")
    ax[2].set_title("(c) each sloppy eigvec $=$ one mode", fontsize=9.5)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02).set_label(
        r"$|v_i|$", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_spectrum.pdf"))
    plt.close(fig)

    rep = small_eigenvector_report(sys, Gamma, k=10)
    s = fim_summaries(Gamma)
    corr = float(np.corrcoef(np.log(diag[observed] + 1e-12),
                             np.log(ana[observed] + 1e-12))[0, 1])
    RESULTS["spectrum"] = dict(
        n=sys.n, m=sys.m, K=sys.K, rho=sys.spectral_radius(),
        n_hidden=int(np.sum(~observed)), n_zero_eig=n_zero,
        lam_min=s["lam_min"], lam_max=s["lam_max"], cond=s["cond"],
        logdet=s["logdet"], stability_law_loglog_corr=corr,
        small_eigvecs=rep)
    print(f"[fig_spectrum] done. {n_zero} zero eigenvalues (hidden modes), "
          f"cond={s['cond']:.2e}, stability-law corr={corr:.3f}")
    return sys, Gamma


# --------------------------------------------------------------------------- #
#  Figure 2 : the data-vs-sensor dichotomy
# --------------------------------------------------------------------------- #
def fig_dichotomy(sys, Gamma, seed=0):
    print("[fig_dichotomy] data-vs-sensor ...")
    w, Vv = eig_analysis(Gamma)
    # pick three representative directions:
    #   stiff (large Fisher, slow mode), excitation-limited (small +ve, fast
    #   observed mode), sensor-limited (zero, hidden mode)
    diag = np.diag(Gamma)
    observed = sys.gains > 0
    slow_obs = np.where(observed)[0][np.argmax(sys.r[observed])]
    fast_obs = np.where(observed)[0][np.argmin(sys.r[observed])]
    hidden = np.where(~observed)[0][0]
    dirs = {"stiff (slow, observed)": (slow_obs, C_WAM),
            "excitation-limited (fast, observed)": (fast_obs, C_VLA),
            "sensor-limited (unobserved)": (hidden, C_PURP)}

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5))

    # (a) Cramer-Rao std  1/sqrt(T * rate)  vs data length T
    Tgrid = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400])
    for name, (j, c) in dirs.items():
        rate = max(diag[j], 0.0)
        if rate <= 1e-12:
            std = np.full_like(Tgrid, np.nan, dtype=float)
            ax[0].plot(Tgrid, np.full_like(Tgrid, 1.0, dtype=float), "--",
                       color=c, lw=1.6, label=name + " ($\\infty$)")
        else:
            std = 1.0 / np.sqrt(Tgrid * rate)
            ax[0].loglog(Tgrid, std, "o-", color=c, lw=1.5, ms=4, label=name)
    ax[0].set_xlabel(r"series length  $T$")
    ax[0].set_ylabel("Cramér–Rao std  " + r"$1/\sqrt{T\,\Gamma_{vv}}$")
    ax[0].set_title("(a) data cures excitation-, not sensor-, limits",
                    fontsize=9.5)
    ax[0].legend(fontsize=7.0, frameon=False)

    # (b) add a sensor that sees the hidden modes -> null eigenvalues lift
    hidden_modes = list(np.where(~observed)[0])
    sys2 = add_velocity_sensors(sys, hidden_modes, gain=1.0, seed=7)
    Gamma2 = fisher_information(sys2, T=350, burn=90, M=24, seed=3)
    # smallest eigenvalue restricted to the previously-hidden subspace
    def hidden_lammin(G):
        sub = G[np.ix_(hidden_modes, hidden_modes)]
        return float(np.linalg.eigvalsh(0.5 * (sub + sub.T)).min())
    before = max(hidden_lammin(Gamma), 1e-8)
    after = hidden_lammin(Gamma2)
    ax[1].bar([0, 1], [before, after], color=[C_PURP, C_CROSS],
              edgecolor="white", width=0.6)
    ax[1].set_yscale("log")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["original\nsensors", "+ sensors that\nsee the modes"])
    ax[1].set_ylabel(r"$\lambda_{\min}$ over hidden subspace")
    ax[1].set_title("(b) only a new sensor cures the null", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dichotomy.pdf"))
    plt.close(fig)
    RESULTS["dichotomy"] = dict(
        stiff_mode=int(slow_obs), stiff_rate=float(diag[slow_obs]),
        fast_obs_mode=int(fast_obs), fast_obs_rate=float(diag[fast_obs]),
        hidden_mode=int(hidden), hidden_rate=float(diag[hidden]),
        hidden_lammin_before=before, hidden_lammin_after=after,
        n_added_sensors=len(hidden_modes))
    print(f"[fig_dichotomy] hidden-subspace lambda_min: "
          f"{before:.2e} -> {after:.2e} after adding sensors")


# --------------------------------------------------------------------------- #
#  Figure 3 : FIM properties vs stability margin
# --------------------------------------------------------------------------- #
def fig_stability_fisher(seed=0):
    print("[fig_stability_fisher] sweeping spectral radius ...")
    base = build_modal_lqg(seed=seed)
    rhos = np.array([0.40, 0.55, 0.68, 0.78, 0.86, 0.92, 0.96, 0.98])
    logdet, cond, lam_max, lam_min_pos, dom = [], [], [], [], []
    for rho in rhos:
        sysr = rescale_spectral_radius(base, rho)
        G = fisher_information(sysr, T=300, burn=80, M=18, seed=11)
        s = fim_summaries(G)
        logdet.append(s["logdet"]); cond.append(s["cond"])
        lam_max.append(s["lam_max"]); lam_min_pos.append(s["lam_min_pos"])
        # dominant-mode (slowest observed) Fisher
        dg = np.diag(G); obs = sysr.gains > 0
        dom.append(float(dg[np.where(obs)[0][np.argmax(sysr.r[obs])]]))
        print(f"    rho={rho:.2f}: logdet={s['logdet']:.1f} "
              f"cond={s['cond']:.2e} lam_max={s['lam_max']:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.5))
    margin = 1 - rhos
    ax[0].plot(rhos, lam_max, "o-", color=C_WAM, lw=1.6, ms=5,
               label=r"$\lambda_{\max}$")
    ax[0].plot(rhos, dom, "s--", color=C_CROSS, lw=1.4, ms=4,
               label="dominant-mode $\\Gamma_{jj}$")
    ax[0].set_xlabel(r"spectral radius  $\rho(A)$")
    ax[0].set_ylabel("Fisher magnitude")
    ax[0].set_yscale("log")
    ax[0].set_title("(a) information grows near unit circle", fontsize=9.5)
    ax[0].legend(fontsize=8, frameon=False)

    ax[1].plot(rhos, cond, "o-", color=C_VLA, lw=1.6, ms=5)
    ax[1].set_xlabel(r"spectral radius  $\rho(A)$")
    ax[1].set_ylabel(r"condition number  $\lambda_{\max}/\lambda_{\min}^{+}$")
    ax[1].set_yscale("log")
    ax[1].set_title("(b) but conditioning worsens", fontsize=9.5)

    ax[2].loglog(margin, lam_max, "o-", color=C_WAM, lw=1.6, ms=5,
                 label=r"measured $\lambda_{\max}$")
    ax[2].loglog(margin, lam_max[-1] * (margin[-1] / margin), "--",
                 color=C_INK2, lw=1.0, label=r"$\propto 1/(1-\rho)$")
    ax[2].set_xlabel(r"stability margin  $1-\rho(A)$")
    ax[2].set_ylabel(r"$\lambda_{\max}(\Gamma)$")
    ax[2].set_title("(c) the stability-margin law", fontsize=9.5)
    ax[2].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_stability_fisher.pdf"))
    plt.close(fig)
    RESULTS["stability_fisher"] = dict(
        rhos=list(map(float, rhos)), logdet=list(map(float, logdet)),
        cond=list(map(float, cond)), lam_max=list(map(float, lam_max)),
        lam_min_pos=list(map(float, lam_min_pos)), dominant=list(map(float, dom)))
    print("[fig_stability_fisher] done.")


if __name__ == "__main__":
    t0 = time.time()
    sys, Gamma = fig_spectrum()
    fig_dichotomy(sys, Gamma)
    fig_stability_fisher()
    json.dump(RESULTS, open(os.path.join(os.path.dirname(__file__),
              "results_theory.json"), "w"), indent=2)
    print(f"\nTheory figures done in {time.time()-t0:.1f}s. "
          f"Wrote results_theory.json")
