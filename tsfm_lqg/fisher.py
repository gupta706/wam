"""
fisher.py
=========
Fisher information of a time-series foundation model, viewed as an *autonomous
world-action model*.

A TSFM trained by next-observation prediction fits the environment channel
P_theta(o_t | history).  Its per-step score is a martingale difference and its
information about the dynamics theta = (r_1,...,r_K) (the modal magnitudes) is
the innovations / system-identification Fisher block I_theta_theta of the
WAM/VLA construction -- here with NO policy block, because an autonomous
forecaster takes no actions.

We use the *innovations state-sensitivity* Fisher information, which is exactly
the parameter-sensitivity observability Gramian of the WAM/VLA paper
(Corollary 'World block: unobservable parameter sensitivity'):

    Gamma_ij = (1/T) sum_t  E[ (C d_i xhat_{t|t-1})^T Sig^{-1} (C d_j xhat_{t|t-1}) ]

with d_i xhat propagated by the tangent Kalman filter at the steady-state gain:

    eta^i_{t+1} = F eta^i_t + (d_i A) xhat_{t|t},    F = A(I - L C)   (stable).

This module computes Gamma (per-step rate), its eigendecomposition, the mode
each eigenvector points at, the per-mode analytic Fisher ~ gain^2/(1-r^2), and
the data-vs-sensor diagnostic for the small eigenvalues.
"""
from __future__ import annotations
import numpy as np
from lqg_system import ModalLQG


# --------------------------------------------------------------------------- #
#  Tangent Kalman filter -> Fisher information (sensitivity Gramian)
# --------------------------------------------------------------------------- #
def fisher_information(sys: ModalLQG,
                       T: int = 400,
                       burn: int = 100,
                       M: int = 24,
                       seed: int = 0,
                       return_scores: bool = False):
    """Per-step Fisher information rate Gamma (K x K) about theta = modal r_j.

    Uses the steady-state predictor and the fast fixed-gain tangent recursion.
    The 'plug-in' accumulation (contracting with Sig^{-1} instead of the noisy
    innovation outer product) is used for low variance; it equals E[s s^T]
    because the innovation is independent of the past-measurable sensitivity.

    Set return_scores=True to also get a Monte-Carlo E[s s^T] estimate (the
    literal FIM = mean score outer product) for cross-validation.
    """
    rng = np.random.default_rng(seed)
    n, m, K = sys.n, sys.m, sys.K
    A, C, W, V = sys.A, sys.C, sys.W, sys.V
    P, Sig, L, F = sys.kalman_steady_state()
    Sinv = np.linalg.inv(Sig)
    Wc = np.linalg.cholesky(W + 1e-12 * np.eye(n))
    Vc = np.linalg.cholesky(V + 1e-12 * np.eye(m))
    ImLC = np.eye(n) - L @ C

    # precompute per-mode 2x2 rotations R_j so that (d_j A) picks mode j:
    #   (d_j A) xhat  has, in rows [2j,2j+1], the vector R_j @ xhat[2j:2j+2]
    ROT = np.zeros((K, 2, 2))
    for j in range(K):
        cj, sj = np.cos(sys.omega[j]), np.sin(sys.omega[j])
        ROT[j] = np.array([[cj, sj], [-sj, cj]])

    Gamma = np.zeros((K, K))
    Gamma_score = np.zeros((K, K))
    n_acc = 0
    for _ in range(M):
        x = np.zeros(n)                       # true state
        xhat = np.zeros(n)                    # xhat_{t|t-1}
        H = np.zeros((n, K))                  # columns eta^i_t = d_i xhat_{t|t-1}
        for t in range(T):
            # --- data from the true autonomous system --------------------- #
            o = C @ x + Vc @ rng.standard_normal(m)
            nu = o - C @ xhat                 # innovation
            xf = xhat + L @ nu                # xhat_{t|t}
            # --- accumulate Fisher (after burn-in) ------------------------ #
            if t >= burn:
                CH = C @ H                     # (m, K), columns C eta^i_t
                SiCH = Sinv @ CH
                Gamma += CH.T @ SiCH           # plug-in E[s s^T]
                if return_scores:
                    s = SiCH.T @ nu            # (K,) score increment
                    Gamma_score += np.outer(s, s)
                n_acc += 1
            # --- tangent recursion: H_{t+1} = F H + Phi(xf) --------------- #
            xf_pairs = xf.reshape(K, 2)
            forcing_pairs = np.einsum('kij,kj->ki', ROT, xf_pairs)   # (K,2)
            Phi = np.zeros((n, K))
            rows = np.arange(K)
            Phi[2 * rows, rows] = forcing_pairs[:, 0]
            Phi[2 * rows + 1, rows] = forcing_pairs[:, 1]
            H = F @ H + Phi
            # --- advance predictor and true state ------------------------- #
            xhat = A @ xf
            x = A @ x + Wc @ rng.standard_normal(n)

    Gamma /= n_acc
    Gamma = 0.5 * (Gamma + Gamma.T)
    if return_scores:
        Gamma_score /= n_acc
        Gamma_score = 0.5 * (Gamma_score + Gamma_score.T)
        return Gamma, Gamma_score
    return Gamma


# --------------------------------------------------------------------------- #
#  Analytic per-mode Fisher (the AR(1) stability law)  ~  gain^2 / (1 - r^2)
# --------------------------------------------------------------------------- #
def analytic_mode_fisher(sys: ModalLQG) -> np.ndarray:
    """Closed-form leading-order per-mode Fisher information about r_j.

    A single mode is a damped 2-D oscillator; its stationary variance scales as
    ~ sigma_w^2 / (1 - r_j^2) (the AR(1) law), and the one-step prediction's
    sensitivity to r_j is proportional to the mode amplitude, so its Fisher
    information is ~ (observation gain)^2 * modal_variance / innovation_scale.
    This is the diagonal that the measured Gamma should track."""
    K = sys.K
    sig_w2 = sys.W[0, 0]
    # per-mode output scale through C (squared Frobenius of the mode's columns)
    cnorm2 = np.array([np.sum(sys.C[:, 2 * j:2 * j + 2] ** 2) for j in range(K)])
    sv2 = sys.V[0, 0]
    modal_var = sig_w2 / (1.0 - sys.r ** 2)
    return (cnorm2 * modal_var) / sv2


# --------------------------------------------------------------------------- #
#  Eigen-analysis and small-eigenvalue -> mode attribution
# --------------------------------------------------------------------------- #
def eig_analysis(Gamma: np.ndarray):
    """Ascending eigenvalues and eigenvectors of a symmetric Fisher block."""
    w, Vv = np.linalg.eigh(0.5 * (Gamma + Gamma.T))
    return w, Vv                       # w ascending; Vv[:, i] eigenvector i


def dominant_mode(eigvec: np.ndarray) -> int:
    """Which modal parameter r_j an eigenvector points at (largest |component|)."""
    return int(np.argmax(np.abs(eigvec)))


def small_eigenvector_report(sys: ModalLQG, Gamma: np.ndarray, k: int = 8):
    """For the k smallest Fisher eigenvalues, report the mode each eigenvector
    names and whether that mode is observed (sensor-limited if not)."""
    w, Vv = eig_analysis(Gamma)
    rows = []
    for i in range(k):
        v = Vv[:, i]
        j = dominant_mode(v)
        rows.append(dict(rank=i, eigenvalue=float(w[i]), mode=j,
                         r_j=float(sys.r[j]), gain_j=float(sys.gains[j]),
                         concentration=float(v[j] ** 2),
                         sensor_limited=bool(sys.gains[j] == 0.0)))
    return rows


# --------------------------------------------------------------------------- #
#  Data-vs-sensor test on a single direction
# --------------------------------------------------------------------------- #
def directional_information(sys: ModalLQG, v: np.ndarray, **kw) -> float:
    """v^T Gamma v : the Fisher information along a unit direction v.
    (Used to test whether a small direction is excitation- or sensor-limited.)"""
    Gamma = fisher_information(sys, **kw)
    v = v / np.linalg.norm(v)
    return float(v @ Gamma @ v)


def fim_summaries(Gamma: np.ndarray) -> dict:
    """Scalar conditioning summaries of a Fisher block."""
    w = np.linalg.eigvalsh(0.5 * (Gamma + Gamma.T))
    w = np.clip(w, 0.0, None)
    wpos = w[w > 1e-14]
    logdet = float(np.sum(np.log(np.clip(w, 1e-14, None))))
    return dict(lam_min=float(w[0]),
                lam_max=float(w[-1]),
                lam_min_pos=float(wpos.min()) if wpos.size else 0.0,
                cond=float(w[-1] / max(w[0], 1e-14)),
                logdet=logdet,
                trace=float(np.sum(w)),
                rank_eff=float(np.sum(w) / w[-1]))


if __name__ == "__main__":
    from lqg_system import build_modal_lqg
    sys = build_modal_lqg(seed=0)
    print("computing Fisher information (this takes a few seconds)...")
    Gamma, Gscore = fisher_information(sys, T=300, burn=80, M=16,
                                       seed=1, return_scores=True)
    # cross-validate plug-in vs Monte-Carlo score FIM
    rel = np.linalg.norm(Gamma - Gscore) / np.linalg.norm(Gamma)
    print(f"plug-in vs MC-score FIM relative diff = {rel:.3f}")

    w, Vv = eig_analysis(Gamma)
    s = fim_summaries(Gamma)
    print(f"Fisher eigenvalues: min={w[0]:.3e}  max={w[-1]:.3e}  "
          f"cond={s['cond']:.2e}  logdet={s['logdet']:.1f}")
    print(f"# eigenvalues below 1e-6 * lam_max = "
          f"{int(np.sum(w < 1e-6 * w[-1]))} (expected ~ #hidden modes)")

    print("\nSmallest-eigenvalue eigenvectors (data-vs-sensor):")
    for row in small_eigenvector_report(sys, Gamma, k=10):
        tag = "SENSOR-limited" if row["sensor_limited"] else "excitation-limited"
        print(f"  rank {row['rank']:2d}: lam={row['eigenvalue']:.3e}  "
              f"mode {row['mode']:3d} (r={row['r_j']:.3f}, gain={row['gain_j']:.2f})"
              f"  conc={row['concentration']:.2f}  -> {tag}")

    # analytic diagonal vs measured diagonal (stability law)
    diag = np.diag(Gamma)
    ana = analytic_mode_fisher(sys)
    corr = np.corrcoef(np.log(diag + 1e-12), np.log(ana + 1e-12))[0, 1]
    print(f"\nlog-log corr(measured diag Fisher, analytic gain^2/(1-r^2)) = {corr:.3f}")
