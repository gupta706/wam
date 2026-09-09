import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess

# Add the wam directory to the path so we can import from environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsfm.chronos_lite import MeanScaleQuantizer, make_token_windows, train_tslm, forecast_channel
from tsfm.data_generators import SYSTEMS_REGISTRY, load_or_generate

from tsfm.evaluate_checkpoint import evaluate_model

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    N = config["N"]
    M = config["M"]
    T = config["T"]
    seeds = config["seeds"]
    
    train_data = {}
    test_data = {}
    
    for sys_name, sys_info in SYSTEMS_REGISTRY.items():
        prefix = sys_info["prefix"]
        
        print(f"\n--- Processing {sys_name} data ---")
        train_key = f"{prefix}_train"
        test_key = f"{prefix}_test"
        
        train_data[sys_name] = load_or_generate(sys_name, train_key, N, T, seeds.get(train_key, 42))
        test_data[sys_name] = load_or_generate(sys_name, test_key, M, T, seeds.get(test_key, 42))
        
    print("\n--- Combining datasets for Unified Training ---")
    B = config["B"]
    ctx = config["ctx"]
    stride = config["stride"]
    quant = MeanScaleQuantizer(B=B)
    
    tokenized_path = os.path.join(os.path.dirname(__file__), f"data/Wtr_unified_N{N}_B{B}_ctx{ctx}_stride{stride}.npy")
    if os.path.exists(tokenized_path):
        print(f"Loading tokenized trajectories from {tokenized_path}...")
        Wtr = np.load(tokenized_path)
        print(f"\nTotal Combined Training Windows: {len(Wtr)}")
    else:
        print("Tokenizing training trajectories...")
        windows = []
        for sys_name in SYSTEMS_REGISTRY.keys():
            w = make_token_windows(train_data[sys_name], quant, ctx, stride=stride)
            print(f"  {sys_name} tokens: {len(w)}")
            windows.append(w)
            
        Wtr = np.concatenate(windows, axis=0)
        np.random.shuffle(Wtr)
        print(f"\nTotal Combined Training Windows: {len(Wtr)}")
        print(f"Saving tokenized trajectories to {tokenized_path}...")
        np.save(tokenized_path, Wtr)
    
    print("\n--- Training Unified Foundation Model ---")
    model = train_tslm(Wtr, B=B, ctx=ctx, epochs=config["epochs"], d_model=config["d_model"], n_layer=config["n_layer"], batch=config["batch"], max_batches=config.get("max_batches"), verbose=True, seed=seeds.get("model_train", 0))
    
    H = config["H"]
    out_dir = os.path.dirname(__file__)
    for sys_name in SYSTEMS_REGISTRY.keys():
        evaluate_model(sys_name, model, quant, test_data[sys_name], H, out_dir)

    print("\n--- Pushing to GitHub ---")
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    try:
        subprocess.run(['git', 'add', '.'], cwd=repo_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Update foundation model and evaluation plots'], cwd=repo_dir, check=True)
        subprocess.run(['git', 'push'], cwd=repo_dir, check=True)
        print("Successfully pushed to GitHub!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub. Error: {e}")

if __name__ == "__main__":
    main()
