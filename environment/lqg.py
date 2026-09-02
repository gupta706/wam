import numpy as np
from numba import njit
import sys
import os

def create_lqg_params(K: int = 100, m: int = 100, rho: float = 0.9, 
                      r_lo: float = 0.15, sigma_w: float = 1.0, 
                      sigma_v: float = 0.1, n_hidden: int = 8, 
                      obs_gain_lo: float = 0.4, obs_gain_hi: float = 1.0, 
                      obs_slow_bias: float = 2.5, key_seed: int = 42):
    """
    Construct parameters for a 2K-state, m-observation modal LQG system.
    Returns (A, C, W, V).
    """
    rng = np.random.default_rng(key_seed)
    n = 2 * K

    # modal magnitudes: log-spaced from fast (r_lo) to slow (rho)
    r = np.geomspace(r_lo, rho, K)
    
    # random modal frequencies, biased low so the slow modes are also smooth
    omega = rng.uniform(0.05, 0.9, size=K) * (1.0 - 0.5 * (r / rho))
    omega = np.clip(omega, 0.03, np.pi - 0.05)

    # block-diagonal A
    A = np.zeros((n, n))
    for j in range(K):
        cj, sj = np.cos(omega[j]), np.sin(omega[j])
        A[2 * j:2 * j + 2, 2 * j:2 * j + 2] = r[j] * np.array([[cj, sj],
                                                               [-sj, cj]])

    # observation gains
    gains = rng.uniform(obs_gain_lo, obs_gain_hi, size=K)
    gains *= (r / rho) ** obs_slow_bias
    hidden = np.argsort(r)[:n_hidden]
    gains[hidden] = 0.0

    # observation map C
    C = np.zeros((m, n))
    for j in range(K):
        Cj = rng.standard_normal((m, 2)) / np.sqrt(m)
        C[:, 2 * j:2 * j + 2] = gains[j] * Cj

    # noise covariances
    W = (sigma_w ** 2) * np.eye(n)
    V = (sigma_v ** 2) * np.eye(m)

    return A, C, W, V

def get_initial_conditions(n_states=200, key_seed=99):
    rng = np.random.default_rng(key_seed)
    return rng.standard_normal(n_states)

@njit
def lqg_step(x, A, C, W_chol, V_chol, w_noise, v_noise):
    """
    One discrete step of the LQG system.
    x_{t+1} = A x_t + W_chol * w_noise
    o_t = C x_t + V_chol * v_noise
    """
    x_next = A @ x + W_chol @ w_noise
    o = C @ x + V_chol @ v_noise
    return x_next, o

@njit
def lqg_rollout_numba(x0, A, C, W_chol, V_chol, w_noise_seq, v_noise_seq):
    """
    Rollout the LQG system for T steps using numba.
    w_noise_seq: shape (T, n)
    v_noise_seq: shape (T, m)
    """
    T = w_noise_seq.shape[0]
    n = x0.shape[0]
    m = C.shape[0]
    
    states = np.zeros((T, n))
    obs = np.zeros((T, m))
    
    x = x0.copy()
    for t in range(T):
        x, o = lqg_step(x, A, C, W_chol, V_chol, w_noise_seq[t], v_noise_seq[t])
        states[t] = x
        obs[t] = o
        
    return states, obs

def simulate_lqg(T: int, params: tuple, key_seed: int = 42):
    """
    Helper to run a full simulation. Generates noise in numpy and passes to numba.
    """
    A, C, W, V = params
    n = A.shape[0]
    m = C.shape[0]
    
    rng = np.random.default_rng(key_seed)
    x0 = get_initial_conditions(n_states=n, key_seed=key_seed)
    
    # Pre-compute Cholesky factors
    W_chol = np.linalg.cholesky(W)
    V_chol = np.linalg.cholesky(V)
    
    w_noise_seq = rng.standard_normal((T, n))
    v_noise_seq = rng.standard_normal((T, m))
    
    states, obs = lqg_rollout_numba(x0, A, C, W_chol, V_chol, w_noise_seq, v_noise_seq)
    
    # Create time array to match integrators
    times = np.arange(T)
    return times, states, obs
