import concurrent.futures
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
from tsfm.chronos_lite import MeanScaleQuantizer, make_token_windows, train_tslm, forecast_channel

def _lqg_worker(args):
    T, params, seed = args
    _, _, obs = simulate_lqg(T, params, key_seed=seed)
    return obs

def generate_lqg_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the LQG system."""
    params = create_lqg_params(key_seed=seed)
    args_list = [(T, params, seed + i) for i in range(n_traj)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        trajectories = list(executor.map(_lqg_worker, args_list))
    return trajectories

def _kuramoto_worker(args):
    T, params, dt, save_every, t_span, seed = args
    integrate = build_integrator(kuramoto_derivative, dt, save_every=save_every)
    np.random.seed(seed)
    y0 = get_initial_conditions(200, key_seed=seed)
    _, traj = integrate(y0, t_span, params)
    return traj[:T] % (2*np.pi)

def generate_kuramoto_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Kuramoto system."""
    params = create_kuramoto_params(key_seed=seed)
    dt = 0.05
    save_every = 2
    t_span = (0.0, dt * T * save_every)
    
    args_list = [(T, params, dt, save_every, t_span, seed + i) for i in range(n_traj)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        trajectories = list(executor.map(_kuramoto_worker, args_list))
    return trajectories

def _sl_worker(args):
    T, params, dt, save_every, t_span, seed = args
    integrate = build_integrator(stuart_landau_derivative, dt, save_every=save_every)
    np.random.seed(seed)
    y0 = get_initial_conditions_sl(200, key_seed=seed)
    _, traj = integrate(y0, t_span, params)
    return np.real(traj[:T])

def generate_stuart_landau_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Stuart-Landau system."""
    params = create_stuart_landau_params(key_seed=seed)
    dt = 0.05
    save_every = 2
    t_span = (0.0, dt * T * save_every)
    
    args_list = [(T, params, dt, save_every, t_span, seed + i) for i in range(n_traj)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        trajectories = list(executor.map(_sl_worker, args_list))
    return trajectories

def _ks_worker(args):
    T, params, dt, save_every, t_span, seed = args
    y0_hat = get_initial_conditions_ks(200, key_seed=seed)
    _, traj_hat = integrate_ks(y0_hat, t_span, params, dt=dt, save_every=save_every)
    traj = np.real(np.fft.ifft(traj_hat, axis=1))
    return traj[:T]

def generate_ks_corpus(n_traj, T, seed=42):
    """Generate n_traj trajectories of length T from the Kuramoto-Sivashinsky system."""
    params = create_ks_params(N=200, L=22.0)
    dt = 0.001
    save_every = 50
    t_span = (0.0, dt * save_every * T)
    
    args_list = [(T, params, dt, save_every, t_span, seed + i) for i in range(n_traj)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        trajectories = list(executor.map(_ks_worker, args_list))
    return trajectories

def evaluate_model(name, model, quant, test_traj):
    print(f"\n--- Evaluating TSFM on {name} ---")
    
    # Evaluate by forecasting one channel
    O = test_traj[0]
    ctxlen = model.ctx
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

def load_or_generate(name, N, T, seed, generate_fn):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{name}_{N}_{T}_{seed}.npy"
    if os.path.exists(file_path):
        print(f"  Loading {name} from {file_path}...")
        return list(np.load(file_path, allow_pickle=True))
    else:
        print(f"  Generating {name}...")
        data = generate_fn(N, T, seed=seed)
        np.save(file_path, np.array(data))
        return data

def main():
    N = 1000
    M = 50
    print("\n--- Processing LQG data ---")
    lqg_train = load_or_generate("lqg_train", N, 700, 10, generate_lqg_corpus)
    lqg_test = load_or_generate("lqg_test", M, 700, 99, generate_lqg_corpus)
    
    print("\n--- Processing Kuramoto data ---")
    kur_train = load_or_generate("kur_train", N, 700, 20, generate_kuramoto_corpus)
    kur_test = load_or_generate("kur_test", M, 700, 199, generate_kuramoto_corpus)
    
    print("\n--- Processing Stuart-Landau data ---")
    sl_train = load_or_generate("sl_train", N, 700, 30, generate_stuart_landau_corpus)
    sl_test = load_or_generate("sl_test", M, 700, 299, generate_stuart_landau_corpus)
    
    print("\n--- Processing Kuramoto-Sivashinsky data ---")
    ks_train = load_or_generate("ks_train", N, 700, 40, generate_ks_corpus)
    ks_test = load_or_generate("ks_test", M, 700, 399, generate_ks_corpus)

    print("\n--- Combining datasets for Unified Training ---")
    B = 4096
    ctx = 512
    stride = 8
    quant = MeanScaleQuantizer(B=B)
    
    tokenized_path = f"data/Wtr_unified_N{N}_B{B}_ctx{ctx}_stride{stride}.npy"
    if os.path.exists(tokenized_path):
        print(f"Loading tokenized trajectories from {tokenized_path}...")
        Wtr = np.load(tokenized_path)
        print(f"\nTotal Combined Training Windows: {len(Wtr)}")
    else:
        print("Tokenizing training trajectories...")
        w_lqg = make_token_windows(lqg_train, quant, ctx, stride=stride)
        print(f"  LQG tokens: {len(w_lqg)}")
        w_kur = make_token_windows(kur_train, quant, ctx, stride=stride)
        print(f"  Kuramoto tokens: {len(w_kur)}")
        w_sl = make_token_windows(sl_train, quant, ctx, stride=stride)
        print(f"  Stuart-Landau tokens: {len(w_sl)}")
        w_ks = make_token_windows(ks_train, quant, ctx, stride=stride)
        print(f"  Kuramoto-Sivashinsky tokens: {len(w_ks)}")
        
        Wtr = np.concatenate([w_lqg, w_kur, w_sl, w_ks], axis=0)
        np.random.shuffle(Wtr)
        print(f"\nTotal Combined Training Windows: {len(Wtr)}")
        print(f"Saving tokenized trajectories to {tokenized_path}...")
        np.save(tokenized_path, Wtr)
    
    print("\n--- Training Unified Foundation Model ---")
    model = train_tslm(Wtr, B=B, ctx=ctx, epochs=30, d_model=256, n_layer=10, verbose=True, seed=0)
    
    evaluate_model("LQG", model, quant, lqg_test)
    evaluate_model("Kuramoto", model, quant, kur_test)
    evaluate_model("Stuart-Landau", model, quant, sl_test)
    evaluate_model("Kuramoto-Sivashinsky", model, quant, ks_test)

if __name__ == "__main__":
    main()
