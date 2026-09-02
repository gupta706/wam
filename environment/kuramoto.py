import numpy as np
from numba import njit

def create_kuramoto_params(n_oscillators=200, n_populations=5, intra_coupling=5.0, inter_coupling=0.1, key_seed=42):
    np.random.seed(key_seed)
    
    pop_sizes = [n_oscillators // n_populations] * n_populations
    pop_sizes[-1] += n_oscillators - sum(pop_sizes)
    
    K = np.ones((n_oscillators, n_oscillators)) * inter_coupling
    start = 0
    for size in pop_sizes:
        K[start:start+size, start:start+size] = intra_coupling
        start += size
        
    K = K / n_oscillators
    omega = np.random.normal(0, 0.5, size=(n_oscillators,))
    
    return (K, omega)

@njit
def kuramoto_derivative(t, theta, params):
    K, omega = params
    N = len(theta)
    
    dtheta = np.zeros(N)
    for i in range(N):
        sum_k = 0.0
        for j in range(N):
            sum_k += K[i, j] * np.sin(theta[j] - theta[i])
        dtheta[i] = omega[i] + sum_k
        
    return dtheta

def get_initial_conditions(n_oscillators=200, key_seed=99):
    np.random.seed(key_seed)
    return np.random.uniform(0, 2 * np.pi, size=(n_oscillators,))
