"""
lqg_system.py
=============
A large, stable, partially observed linear--Gaussian (LQG) system in *modal*
coordinates, used as the ground-truth data-generating process for a time-series
foundation model (TSFM).

    x_{t+1} = A x_t + w_t,     w_t ~ N(0, W)          (n = 200 latent states)
    o_t     = C x_t + v_t,     v_t ~ N(0, V)          (m = 100 observations)

The state is organized as 100 physical *modes*, each a 2x2 damped oscillator
(scaled rotation) with eigenvalues  r_j e^{+- i w_j}.  The modal magnitude r_j
in (0,1) is the *stability* of mode j; its distance to the unit circle,
1 - r_j, is the mode's stability margin.  rho(A) = max_j r_j is the system
spectral radius and 1 - rho(A) the (global) stability margin.

This modal construction is what makes the whole study interpretable: the
Fisher information matrix about the modal magnitudes theta = (r_1, ..., r_100)
is (near) diagonal in the modes, so an eigenvector of a *small* Fisher
eigenvalue points at a *specific physical mode* -- either a fast mode
(excitation-limited) or an unobserved mode (sensor-limited).

Everything is float64 and seed-reproducible.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are


# --------------------------------------------------------------------------- #
#  Modal system container
# --------------------------------------------------------------------------- #
@dataclass
class ModalLQG:
    """A partially observed linear-Gaussian system in modal (block-2x2) form."""
    r: np.ndarray            # (K,) modal magnitudes in (0,1)      -- theta
    omega: np.ndarray        # (K,) modal angular frequencies
    A: np.ndarray            # (2K, 2K) block-diagonal dynamics
    C: np.ndarray            # (m, 2K) observation map
    W: np.ndarray            # (2K, 2K) process-noise covariance
    V: np.ndarray            # (m, m) sensor-noise covariance
    gains: np.ndarray        # (K,) per-mode observation gain (0 => unobserved)
    n: int = field(init=False)
    m: int = field(init=False)
    K: int = field(init=False)

    def __post_init__(self):
        self.n = self.A.shape[0]
        self.m = self.C.shape[0]
        self.K = len(self.r)

    # ---- spectral / stability summaries ---------------------------------- #
    def spectral_radius(self) -> float:
        return float(np.max(self.r))

    def stability_margin(self) -> float:
        return float(1.0 - self.spectral_radius())

    # ---- second-moment structure ----------------------------------------- #
    def state_covariance(self) -> np.ndarray:
        """Stationary state covariance Pi = A Pi A^T + W  (Lyapunov)."""
        return solve_discrete_lyapunov(self.A, self.W)

    def kalman_steady_state(self):
        """Steady-state *predicted* covariance P, innovation covariance Sig,
        Kalman gain L (update form) and predictor closed-loop F = A(I - LC).

        Returns (P, Sig, L, F).  Requires (A, C) detectable and (A, W^{1/2})
        stabilizable, which holds here (A stable, W > 0)."""
        A, C, W, V = self.A, self.C, self.W, self.V
        # DARE for the predicted covariance of the *predictor* form.
        P = solve_discrete_are(A.T, C.T, W, V)
        Sig = C @ P @ C.T + V
        Sig = 0.5 * (Sig + Sig.T)
        L = P @ C.T @ np.linalg.inv(Sig)          # update gain  x|t = x|t-1 + L nu
        F = A @ (np.eye(self.n) - L @ C)           # predictor closed loop (stable)
        return P, Sig, L, F

    # ---- derivative of A w.r.t. modal magnitude r_j ---------------------- #
    def dA_dr(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(rows, block) : the 2x2 block dA/dr_j and its state-index slice.
        Since A is block diagonal, dA/dr_j is nonzero only in mode j's 2x2
        block, equal to the rotation  [[cos w, sin w], [-sin w, cos w]]."""
        w = self.omega[j]
        rot = np.array([[np.cos(w), np.sin(w)],
                        [-np.sin(w), np.cos(w)]])
        sl = slice(2 * j, 2 * j + 2)
        return sl, rot


# --------------------------------------------------------------------------- #
#  Builder
# --------------------------------------------------------------------------- #
def build_modal_lqg(K: int = 100,
                    m: int = 100,
                    rho: float = 0.9,
                    r_lo: float = 0.15,
                    sigma_w: float = 1.0,
                    sigma_v: float = 0.1,
                    n_hidden: int = 8,
                    obs_gain_lo: float = 0.4,
                    obs_gain_hi: float = 1.0,
                    obs_slow_bias: float = 2.5,
                    seed: int = 0) -> ModalLQG:
    """Construct a 2K-state, m-observation modal LQG system.

    Parameters
    ----------
    K          : number of 2x2 modes  (state dimension n = 2K, default 200).
    m          : observation dimension (default 100 -> C is 100 x 200, *fat*,
                 so there is a nontrivial weakly/un-observed subspace).
    rho        : desired spectral radius (global stability); the largest modal
                 magnitude is set to rho, the stability margin is 1 - rho.
    r_lo       : smallest modal magnitude (the *fastest* mode).
    n_hidden   : this many (fast) modes get observation gain 0 -> structurally
                 unobserved -> *sensor-limited* null directions of the Fisher.
    obs_gain_* : range of per-mode observation gains for the observed modes.
    obs_slow_bias : exponent p >= 0 tilting observation energy toward the slow,
                 persistent modes: gain_j *= (r_j / rho)^p.  With p>0 the
                 predictable low-frequency modes dominate each channel (so the
                 univariate TSFM has real structure to learn), and the fast
                 modes become weakly observed (small Fisher).

    The modal magnitudes are log-spaced in [r_lo, rho]; slow modes (r -> rho)
    have small stability margin, fast modes (r -> r_lo) large margin.
    """
    rng = np.random.default_rng(seed)
    n = 2 * K

    # --- modal magnitudes: log-spaced from fast (r_lo) to slow (rho) ------ #
    r = np.geomspace(r_lo, rho, K)
    # random modal frequencies, biased low so the slow modes are also smooth
    # (low-frequency) and hence forecastable from a single channel
    omega = rng.uniform(0.05, 0.9, size=K) * (1.0 - 0.5 * (r / rho))
    omega = np.clip(omega, 0.03, np.pi - 0.05)

    # --- block-diagonal A ------------------------------------------------- #
    A = np.zeros((n, n))
    for j in range(K):
        cj, sj = np.cos(omega[j]), np.sin(omega[j])
        A[2 * j:2 * j + 2, 2 * j:2 * j + 2] = r[j] * np.array([[cj, sj],
                                                               [-sj, cj]])

    # --- observation gains: tilt toward slow modes; hide the fastest ------ #
    gains = rng.uniform(obs_gain_lo, obs_gain_hi, size=K)
    gains *= (r / rho) ** obs_slow_bias        # slow modes better observed
    hidden = np.argsort(r)[:n_hidden]          # fastest modes -> hidden
    gains[hidden] = 0.0

    # --- observation map C (m x 2K): random directions, per-mode gain ----- #
    C = np.zeros((m, n))
    for j in range(K):
        Cj = rng.standard_normal((m, 2)) / np.sqrt(m)   # random 2D readout
        C[:, 2 * j:2 * j + 2] = gains[j] * Cj

    # --- noise covariances ------------------------------------------------ #
    W = (sigma_w ** 2) * np.eye(n)
    V = (sigma_v ** 2) * np.eye(m)

    return ModalLQG(r=r, omega=omega, A=A, C=C, W=W, V=V, gains=gains)


def rescale_spectral_radius(sys: ModalLQG, rho_new: float, seed: int = 0) -> ModalLQG:
    """Return a copy of `sys` with every modal magnitude rescaled so that the
    spectral radius equals `rho_new` (relative modal spacing preserved).

    This is the single 'stability knob' gamma used for the stability sweep:
    it moves the whole spectrum toward (rho->1) or away from (rho->0) the unit
    circle without changing frequencies, observation map, or noise."""
    scale = rho_new / sys.spectral_radius()
    r_new = sys.r * scale
    K = sys.K
    A = np.zeros_like(sys.A)
    for j in range(K):
        cj, sj = np.cos(sys.omega[j]), np.sin(sys.omega[j])
        A[2 * j:2 * j + 2, 2 * j:2 * j + 2] = r_new[j] * np.array([[cj, sj],
                                                                   [-sj, cj]])
    return ModalLQG(r=r_new, omega=sys.omega.copy(), A=A, C=sys.C.copy(),
                    W=sys.W.copy(), V=sys.V.copy(), gains=sys.gains.copy())


def add_velocity_sensors(sys: ModalLQG, modes, gain: float = 1.0,
                         seed: int = 123) -> ModalLQG:
    """Return a copy with EXTRA observation rows that see the given `modes`
    (which may currently be unobserved).  Models 'adding a sensor': it appends
    rows to C (and to V) that observe the previously blind modal subspaces.

    Used for the data-vs-sensor experiment: a sensor-limited null direction of
    the Fisher can only be cured by a sensor that *sees* its mode."""
    rng = np.random.default_rng(seed)
    m_old = sys.m
    extra = []
    for j in modes:
        row = np.zeros((2, sys.n))
        blk = rng.standard_normal((2, 2))
        row[:, 2 * j:2 * j + 2] = gain * blk
        extra.append(row)
    Cadd = np.vstack(extra) if extra else np.zeros((0, sys.n))
    C_new = np.vstack([sys.C, Cadd])
    m_new = C_new.shape[0]
    V_new = np.eye(m_new) * (sys.V[0, 0])
    V_new[:m_old, :m_old] = sys.V
    gains_new = sys.gains.copy()
    for j in modes:
        gains_new[j] = max(gains_new[j], gain)
    return ModalLQG(r=sys.r.copy(), omega=sys.omega.copy(), A=sys.A.copy(),
                    C=C_new, W=sys.W.copy(), V=V_new, gains=gains_new)


if __name__ == "__main__":
    sys = build_modal_lqg(seed=0)
    print(f"state dim n = {sys.n}, obs dim m = {sys.m}, modes K = {sys.K}")
    print(f"spectral radius rho(A) = {sys.spectral_radius():.4f}, "
          f"stability margin = {sys.stability_margin():.4f}")
    print(f"# unobserved (hidden) modes = {int(np.sum(sys.gains == 0))}")
    Pi = sys.state_covariance()
    print(f"stationary state variance tr(Pi) = {np.trace(Pi):.2f}")
    P, Sig, L, F = sys.kalman_steady_state()
    print(f"predictor closed-loop spectral radius rho(F) = "
          f"{np.max(np.abs(np.linalg.eigvals(F))):.4f}  (should be < 1)")
    print(f"innovation covariance: tr(Sig) = {np.trace(Sig):.3f}, "
          f"log det(Sig) = {np.linalg.slogdet(Sig)[1]:.3f}")
