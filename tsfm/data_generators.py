import concurrent.futures
import numpy as np
import os
import sys

# Add the wam directory to the path so we can import from environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.lqg import create_lqg_params, simulate_lqg
from environment.kuramoto import create_kuramoto_params, kuramoto_derivative, get_initial_conditions as get_ic_kuramoto
from environment.stuart_landau import create_stuart_landau_params, stuart_landau_derivative, get_initial_conditions as get_ic_sl
from environment.kuramoto_sivashinsky import create_ks_params, get_initial_conditions_ks, integrate_ks
from environment.lorenz96 import create_lorenz96_params, lorenz96_derivative, get_initial_conditions as get_ic_lor
from environment.integrators import build_integrator

# ==========================================
# 1. Standardized Worker Wrappers
# ==========================================
def _lqg_worker(args):
    T, params, _, _, _, seed = args  # Ignore standard ODE params
    _, _, obs = simulate_lqg(T, params, key_seed=seed)
    return obs

def _kuramoto_worker(args):
    T, params, dt, save_every, t_span, seed = args
    integrate = build_integrator(kuramoto_derivative, dt, save_every=save_every)
    np.random.seed(seed)
    y0 = get_ic_kuramoto(200, key_seed=seed)
    _, traj = integrate(y0, t_span, params)
    return traj[:T] % (2*np.pi)

def _sl_worker(args):
    T, params, dt, save_every, t_span, seed = args
    integrate = build_integrator(stuart_landau_derivative, dt, save_every=save_every)
    np.random.seed(seed)
    y0 = get_ic_sl(200, key_seed=seed)
    _, traj = integrate(y0, t_span, params)
    return np.real(traj[:T])

def _ks_worker(args):
    T, params, dt, save_every, t_span, seed = args
    y0_hat = get_initial_conditions_ks(200, key_seed=seed)
    _, traj_hat = integrate_ks(y0_hat, t_span, params, dt=dt, save_every=save_every)
    traj = np.real(np.fft.ifft(traj_hat, axis=1))
    return traj[:T]

def _lorenz96_worker(args):
    T, params, dt, save_every, t_span, seed = args
    integrate = build_integrator(lorenz96_derivative, dt, save_every=save_every)
    np.random.seed(seed)
    y0 = get_ic_lor(params[0], key_seed=seed)
    _, traj = integrate(y0, t_span, params)
    return traj[:T]

# ==========================================
# 2. Parameter Creators
# ==========================================
def _get_lqg_params(seed): return create_lqg_params(key_seed=seed)
def _get_kuramoto_params(seed): return create_kuramoto_params(key_seed=seed)
def _get_sl_params(seed): return create_stuart_landau_params(key_seed=seed)
def _get_ks_params(seed): return create_ks_params(N=200, L=22.0)
def _get_lor96_params(seed): return create_lorenz96_params(N=40, F=8.0, key_seed=seed)


# ==========================================
# 3. Systems Registry
# ==========================================
SYSTEMS_REGISTRY = {
    "LQG": {
        "prefix": "lqg",
        "worker": _lqg_worker,
        "param_fn": _get_lqg_params,
        "dt": 1.0,           
        "save_every": 1
    },
    "Kuramoto": {
        "prefix": "kur",
        "worker": _kuramoto_worker,
        "param_fn": _get_kuramoto_params,
        "dt": 0.05,
        "save_every": 2
    },
    "Stuart-Landau": {
        "prefix": "sl",
        "worker": _sl_worker,
        "param_fn": _get_sl_params,
        "dt": 0.05,
        "save_every": 2
    },
    "Kuramoto-Sivashinsky": {
        "prefix": "ks",
        "worker": _ks_worker,
        "param_fn": _get_ks_params,
        "dt": 0.001,
        "save_every": 50
    },
    "Lorenz-96": {
        "prefix": "lor96",
        "worker": _lorenz96_worker,
        "param_fn": _get_lor96_params,
        "dt": 0.01,
        "save_every": 5
    }
}

# ==========================================
# 4. Universal Data Generator
# ==========================================
def generate_corpus(system_name, n_traj, T, seed=42):
    """
    Universal function to generate trajectories for any system defined in the registry.
    """
    if system_name not in SYSTEMS_REGISTRY:
        raise ValueError(f"System '{system_name}' not found in registry.")
        
    sys_config = SYSTEMS_REGISTRY[system_name]
    worker_fn = sys_config["worker"]
    params = sys_config["param_fn"](seed)
    dt = sys_config["dt"]
    save_every = sys_config["save_every"]
    
    t_span = (0.0, dt * T * save_every)
    
    args_list = [(T, params, dt, save_every, t_span, seed + i) for i in range(n_traj)]
    
    # Limit max_workers so we leave CPU cores available for the GPU to use (e.g., PyTorch dispatching)
    workers = max(1, (os.cpu_count() or 4) // 2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        trajectories = list(executor.map(worker_fn, args_list))
        
    return trajectories

def load_or_generate(sys_name, file_prefix, N, T, seed):
    """
    Helper function to load data from disk if it exists, or generate and save it if not.
    """
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file_prefix}_{N}_{T}_{seed}.npy"
    if os.path.exists(file_path):
        print(f"  Loading {sys_name} from {file_path}...")
        return list(np.load(file_path, allow_pickle=True))
    else:
        print(f"  Generating {sys_name}...")
        data = generate_corpus(sys_name, N, T, seed=seed)
        np.save(file_path, np.array(data))
        return data

if __name__ == "__main__":
    import json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    N = config["N"]
    M = config["M"]
    T = config["T"]
    seeds = config["seeds"]
    
    print("Generating all registered datasets...")
    for sys_name, sys_info in SYSTEMS_REGISTRY.items():
        prefix = sys_info["prefix"]
        
        print(f"\n--- Processing {sys_name} data ---")
        train_key = f"{prefix}_train"
        test_key = f"{prefix}_test"
        
        load_or_generate(sys_name, train_key, N, T, seeds.get(train_key, 42))
        load_or_generate(sys_name, test_key, M, T, seeds.get(test_key, 42))
        
    print("\nData generation complete!")
