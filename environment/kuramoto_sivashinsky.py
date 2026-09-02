import numpy as np

def create_ks_params(N=200, L=22.0):
    k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    L_k = k**2 - k**4
    return (k, L_k)

def get_initial_conditions_ks(N=200, key_seed=42):
    np.random.seed(key_seed)
    u0 = np.random.normal(0, 0.1, size=(N,))
    u0 = u0 - np.mean(u0)
    return np.fft.fft(u0)

def integrate_ks(y0_hat, t_span, params, dt=0.001, save_every=50):
    t0, t1 = t_span
    n_steps = int((t1 - t0) / dt)
    
    saved_steps = n_steps // save_every + 1
    trajectory_hat = np.zeros((saved_steps, len(y0_hat)), dtype=np.complex128)
    times = np.zeros(saved_steps)
    
    trajectory_hat[0] = y0_hat
    times[0] = t0
    
    y = y0_hat.copy()
    save_idx = 1
    
    k, L_k = params
    
    # Precompute IMEX Euler operator: (1 - dt * L_k)^(-1)
    imex_op = 1.0 / (1.0 - dt * L_k)
    
    for i in range(1, n_steps + 1):
        # Nonlinear term: N(u)
        u = np.fft.ifft(y)
        u_squared_hat = np.fft.fft(u**2)
        non_linear_term = -0.5 * 1j * k * u_squared_hat
        
        # IMEX Euler step
        y = imex_op * (y + dt * non_linear_term)
        
        if i % save_every == 0:
            if save_idx < saved_steps:
                trajectory_hat[save_idx] = y
                times[save_idx] = t0 + i * dt
                save_idx += 1
            
    return times[:save_idx], trajectory_hat[:save_idx]
