import numpy as np
from numba import njit

def create_lorenz96_params(N=40, F=8.0, key_seed=42):
    """
    Create parameters for the Lorenz-96 system.
    N: dimensionality of the system (typically 40)
    F: forcing constant (typically 8.0 for chaotic behavior)
    """
    return (N, F)

@njit
def lorenz96_derivative(t, y, params):
    """
    Derivative for the Lorenz-96 system.
    y: state vector of shape (N,)
    params: tuple with (N, F)
    """
    N, F = params
    dy = np.zeros_like(y)
    
    # Lorenz-96 equations:
    # dy_i/dt = (y_{i+1} - y_{i-2}) * y_{i-1} - y_i + F
    for i in range(N):
        dy[i] = (y[(i + 1) % N] - y[(i - 2) % N]) * y[(i - 1) % N] - y[i] + F
        
    return dy

def get_initial_conditions(N, key_seed=42):
    """
    Generate initial conditions for the Lorenz-96 system.
    We typically start with equilibrium state F and add a small perturbation
    to one of the variables to kick off the dynamics.
    """
    np.random.seed(key_seed)
    y0 = np.full(N, 8.0) # Equilibrium is around F (typically 8)
    y0[0] += 0.01        # Add small perturbation to the first variable
    return y0
