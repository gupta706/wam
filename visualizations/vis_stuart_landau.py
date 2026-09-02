import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.stuart_landau import create_stuart_landau_params, stuart_landau_derivative, get_initial_conditions
from environment.integrators import build_integrator

def main():
    print("Setting up Stuart-Landau model...")
    n_clusters = 7
    params = create_stuart_landau_params(n_oscillators=200, n_clusters=n_clusters, intra_coupling=2.0, inter_coupling=0.01)
    
    dt = 0.01
    t_span = (0.0, 50.0)
    save_every = 5
    
    print(f"Building Numba integrator (dt={dt})...")
    integrate = build_integrator(stuart_landau_derivative, dt, save_every=save_every)
    
    y0 = get_initial_conditions(200)
    
    print("Simulating (first run might take a moment to compile)...")
    times, trajectory = integrate(y0, t_span, params)
    
    print("Simulation complete. Plotting...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    real_part = np.real(trajectory)
    imag_part = np.imag(trajectory)
    
    axes[0].plot(times, real_part, alpha=0.5, lw=0.5)
    axes[0].set_title(f'Stuart-Landau Clusters\nReal part vs Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Re(z)')
    
    last_steps = min(1000, len(times) // 2)
    
    pop_sizes = [200 // n_clusters] * n_clusters
    pop_sizes[-1] += 200 - sum(pop_sizes)
    
    start = 0
    colors = plt.cm.get_cmap('tab10', n_clusters)
    for i, size in enumerate(pop_sizes):
        axes[1].plot(real_part[-last_steps:, start:start+size], 
                     imag_part[-last_steps:, start:start+size], 
                     color=colors(i), alpha=0.3, lw=0.5)
        start += size
        
    axes[1].set_title('Phase Space (Re vs Im)\nLate Time Dynamics')
    axes[1].set_xlabel('Re(z)')
    axes[1].set_ylabel('Im(z)')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'stuart_landau_results.png')
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
