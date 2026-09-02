import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.kuramoto_sivashinsky import create_ks_params, get_initial_conditions_ks, integrate_ks

def main():
    print("Setting up Kuramoto-Sivashinsky model...")
    L = 22.0
    N = 200
    params = create_ks_params(N=N, L=L)
    
    dt = 0.001
    t_span = (0.0, 100.0)
    save_every = 50
    
    y0_hat = get_initial_conditions_ks(N)
    
    print("Simulating...")
    times, trajectory_hat = integrate_ks(y0_hat, t_span, params, dt=dt, save_every=save_every)
    
    print("Simulation complete. Inverse FFT to physical space...")
    
    trajectory_u = np.real(np.fft.ifft(trajectory_hat, axis=1))
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, L, N, endpoint=False)
    X, T = np.meshgrid(x, times)
    
    c = ax.pcolormesh(X, T, trajectory_u, cmap='RdBu_r', shading='gouraud')
    fig.colorbar(c, ax=ax, label='u(x,t)')
    ax.set_title(f'Kuramoto-Sivashinsky Equation\nSpatiotemporal Chaos (L={L})')
    ax.set_xlabel('x')
    ax.set_ylabel('Time')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'ks_results.png')
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
