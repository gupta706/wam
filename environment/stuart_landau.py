import numpy as np
from numba import njit

def create_stuart_landau_params(n_oscillators=200, n_clusters=7, intra_coupling=2.0, inter_coupling=0.05, key_seed=42):
    np.random.seed(key_seed)
    
    pop_sizes = [n_oscillators // n_clusters] * n_clusters
    pop_sizes[-1] += n_oscillators - sum(pop_sizes)
    
    K = np.ones((n_oscillators, n_oscillators)) * inter_coupling
    start = 0
    for size in pop_sizes:
        K[start:start+size, start:start+size] = intra_coupling
        start += size
        
    K = K / n_oscillators
    
    lam = np.ones(n_oscillators)
    omega = np.random.normal(2.0, 0.2, size=(n_oscillators,))
    
    return (K, lam, omega)

@njit
def stuart_landau_derivative(t, z, params):
    K, lam, omega = params
    N = len(z)
    dz = np.zeros(N, dtype=np.complex128)
    
    for i in range(N):
        linear = (lam[i] + 1j * omega[i]) * z[i]
        nonlinear = - (np.abs(z[i])**2) * z[i]
        
        coupling = 0j
        for j in range(N):
            coupling += K[i, j] * (z[j] - z[i])
            
        dz[i] = linear + nonlinear + coupling
        
    return dz

def get_initial_conditions(n_oscillators=200, key_seed=99):
    np.random.seed(key_seed)
    r = np.random.uniform(0.1, 0.2, size=(n_oscillators,))
    theta = np.random.uniform(0, 2 * np.pi, size=(n_oscillators,))
    return r * np.exp(1j * theta)
