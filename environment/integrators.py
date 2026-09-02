import numpy as np
from numba import njit

def build_integrator(f, dt, save_every=1):
    @njit
    def step(t, y, params):
        k1 = dt * f(t, y, params)
        k2 = dt * f(t + dt / 2, y + k1 / 2, params)
        k3 = dt * f(t + dt / 2, y + k2 / 2, params)
        k4 = dt * f(t + dt, y + k3, params)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
    @njit
    def integrate(y0, t_span, params):
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt)
        
        saved_steps = n_steps // save_every + 1
        trajectory = np.zeros((saved_steps, len(y0)), dtype=y0.dtype)
        times = np.zeros(saved_steps)
        
        trajectory[0] = y0
        times[0] = t0
        
        y = y0.copy()
        save_idx = 1
        for i in range(1, n_steps + 1):
            t = t0 + (i - 1) * dt
            y = step(t, y, params)
            if i % save_every == 0:
                if save_idx < saved_steps:
                    trajectory[save_idx] = y
                    times[save_idx] = t0 + i * dt
                    save_idx += 1
                
        return times[:save_idx], trajectory[:save_idx]
        
    return integrate
