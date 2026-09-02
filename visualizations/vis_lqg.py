import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.lqg import create_lqg_params, simulate_lqg

def main():
    print("Setting up LQG system...")
    # Default params are K=100 (200 states), m=100 (100 observations)
    params = create_lqg_params()
    
    T = 200
    print(f"Simulating for {T} timesteps...")
    times, states, obs = simulate_lqg(T, params)
    
    print("Simulation complete. Plotting...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot a few observation channels
    num_plot_channels = 10
    axes[0].plot(times, obs[:, :num_plot_channels], alpha=0.7)
    axes[0].set_title(f'LQG Observations\n(First {num_plot_channels} channels)')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Observation Value')
    
    # Plot a few state variables (e.g., first 5 modes = 10 state dimensions)
    num_plot_states = 10
    axes[1].plot(times, states[:, :num_plot_states], alpha=0.7)
    axes[1].set_title(f'LQG Hidden States\n(First {num_plot_states} state dims)')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('State Value')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'lqg_results.png')
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
