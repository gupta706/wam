import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the wam directory to the path so we can import from environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.lqg import create_lqg_params, simulate_lqg
from environment.kuramoto import create_kuramoto_params, kuramoto_derivative, get_initial_conditions
from environment.stuart_landau import create_stuart_landau_params, stuart_landau_derivative, get_initial_conditions as get_initial_conditions_sl
from environment.kuramoto_sivashinsky import create_ks_params, get_initial_conditions_ks, integrate_ks
from environment.integrators import build_integrator
from tsfm_unified.chronos_lite import MeanScaleQuantizer, make_token_windows, train_tslm, forecast_channel

def generate_lqg_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the LQG system."""
    params = create_lqg_params(key_seed=seed)
    trajectories = []
    for i in range(n_traj):
        # We want independent trajectories, so vary the seed slightly
        _, _, obs = simulate_lqg(T, params, key_seed=seed + i)
        trajectories.append(obs)
    return trajectories

def generate_kuramoto_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Kuramoto system."""
    params = create_kuramoto_params(key_seed=seed)
    dt = 0.05 # larger step for efficiency
    save_every = 2
    # T steps total in output means we need T * save_every integration steps
    t_span = (0.0, dt * T * save_every)
    
    integrate = build_integrator(kuramoto_derivative, dt, save_every=save_every)
    
    trajectories = []
    for i in range(n_traj):
        np.random.seed(seed + i)
        y0 = get_initial_conditions(200, key_seed=seed+i)
        times, traj = integrate(y0, t_span, params)
        # Ensure it's exactly T steps
        trajectories.append(traj[:T] % (2*np.pi))
    return trajectories

def generate_stuart_landau_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Stuart-Landau system."""
    params = create_stuart_landau_params(key_seed=seed)
    dt = 0.05
    save_every = 2
    t_span = (0.0, dt * T * save_every)
    
    integrate = build_integrator(stuart_landau_derivative, dt, save_every=save_every)
    
    trajectories = []
    for i in range(n_traj):
        np.random.seed(seed + i)
        y0 = get_initial_conditions_sl(200, key_seed=seed+i)
        times, traj = integrate(y0, t_span, params)
        trajectories.append(np.real(traj[:T]))
    return trajectories

def generate_ks_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Kuramoto-Sivashinsky system."""
    params = create_ks_params(N=200, L=22.0)
    dt = 0.001
    save_every = 50
    t_span = (0.0, dt * save_every * T)
    
    trajectories = []
    for i in range(n_traj):
        y0_hat = get_initial_conditions_ks(200, key_seed=seed+i)
        times, traj_hat = integrate_ks(y0_hat, t_span, params, dt=dt, save_every=save_every)
        traj = np.real(np.fft.ifft(traj_hat, axis=1))
        trajectories.append(traj[:T])
    return trajectories

def train_and_eval(name, train_traj, test_traj):
    print(f"\n--- Training TSFM on {name} ---")
    B = 1024
    ctx = 64
    quant = MeanScaleQuantizer(B=B)
    
    # Use small stride and model for demo purposes
    Wtr = make_token_windows(train_traj, quant, ctx, stride=8)
    Wte = make_token_windows(test_traj, quant, ctx, stride=16)
    print(f"Tokenized: {len(Wtr)} training windows, {len(Wte)} test windows.")
    
    model = train_tslm(Wtr, B=B, ctx=ctx, epochs=30, d_model=256, n_layer=10, verbose=True, seed=0)
    
    # Evaluate by forecasting one channel
    O = test_traj[0]
    ctxlen = ctx
    H = 100
    test_channel = 0
    context = O[:ctxlen, test_channel]
    truth = O[ctxlen:ctxlen + H, test_channel]
    
    print(f"Forecasting {H} steps ahead...")
    fc = forecast_channel(model, quant, context, H=H, n_samples=30)
    pred_mean = fc.mean(0)
    
    rmse = np.sqrt(np.mean((pred_mean - truth)**2))
    print(f"Forecast RMSE: {rmse:.5f}")
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(8, 4))
    
    time_ctx = np.arange(ctxlen)
    time_pred = np.arange(ctxlen, ctxlen + H)
    
    ax.plot(time_ctx, context, color='black', label='Context')
    ax.plot(time_pred, truth, color='blue', label='Ground Truth')
    
    # Plot samples
    for i in range(min(10, fc.shape[0])):
        ax.plot(time_pred, fc[i], color='red', alpha=0.1)
    ax.plot(time_pred, pred_mean, color='red', label='TSFM Mean Forecast')
    
    ax.set_title(f'TSFM Forecast on {name}')
    ax.legend()
    
    out_path = os.path.join(os.path.dirname(__file__), f'tsfm_forecast_{name}.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved forecast plot to {out_path}")
    plt.close()

def main():
    N = 1000
    M = 50
    print("Generating LQG data...")
    lqg_train = generate_lqg_corpus(N, 500, seed=10)
    lqg_test = generate_lqg_corpus(M, 500, seed=99)
    train_and_eval("LQG", lqg_train, lqg_test)
    
    print("\nGenerating Kuramoto data...")
    kur_train = generate_kuramoto_corpus(N, 500, seed=20)
    kur_test = generate_kuramoto_corpus(M, 500, seed=199)
    train_and_eval("Kuramoto", kur_train, kur_test)

    print("\nGenerating Stuart-Landau data...")
    sl_train = generate_stuart_landau_corpus(N, 500, seed=30)
    sl_test = generate_stuart_landau_corpus(M, 500, seed=299)
    train_and_eval("Stuart-Landau", sl_train, sl_test)
    
    print("\nGenerating Kuramoto-Sivashinsky data...")
    ks_train = generate_ks_corpus(N, 500, seed=40)
    ks_test = generate_ks_corpus(M, 500, seed=399)
    train_and_eval("Kuramoto-Sivashinsky", ks_train, ks_test)

if __name__ == "__main__":
    main()
