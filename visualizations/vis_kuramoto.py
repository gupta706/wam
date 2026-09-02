import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.kuramoto import create_kuramoto_params, kuramoto_derivative, get_initial_conditions
from environment.integrators import build_integrator

def main():
    print("Setting up Kuramoto model...")
    n_pop = 5
    params = create_kuramoto_params(n_oscillators=200, n_populations=n_pop, intra_coupling=5.0, inter_coupling=0.1)
    
    dt = 0.01
    t_span = (0.0, 50.0)
    save_every = 10
    
    print(f"Building Numba integrator (dt={dt})...")
    integrate = build_integrator(kuramoto_derivative, dt, save_every=save_every)
    
    y0 = get_initial_conditions(200)
    
    print("Simulating (first run might take a moment to compile)...")
    times, trajectory = integrate(y0, t_span, params)
    
    print("Simulation complete. Plotting...")
    
    trajectory_mod = trajectory % (2 * np.pi)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(times, trajectory_mod, alpha=0.5, lw=0.5)
    axes[0].set_title(f'Multi-Population Kuramoto\n({n_pop} populations)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Phase (mod 2$\pi$)')
    
    # We reconstruct pop_sizes from the parameters since it's just 200 // 5
    pop_sizes = [200 // n_pop] * n_pop
    pop_sizes[-1] += 200 - sum(pop_sizes)
    
    start = 0
    for i, size in enumerate(pop_sizes):
        pop_traj = trajectory[:, start:start+size]
        r = np.abs(np.mean(np.exp(1j * pop_traj), axis=1))
        axes[1].plot(times, r, label=f'Pop {i+1}')
        start += size
        
    axes[1].set_title('Order Parameter per Population')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('R')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'kuramoto_results.png')
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
