import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

# Add the wam directory to the path so we can import from environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tsfm.chronos_lite import MeanScaleQuantizer, TinyTSLM, forecast_channel
from tsfm.data_generators import SYSTEMS_REGISTRY, load_or_generate

def evaluate_model(name, model, quant, test_traj, H, out_dir):
    print(f"\n--- Evaluating TSFM on {name} ---")
    
    # Evaluate by forecasting one channel
    O = test_traj[0]
    ctxlen = model.ctx
    test_channel = 0
    context = O[:ctxlen, test_channel]
    # Adjust H to not exceed the available trajectory length
    H_actual = min(H, len(O) - ctxlen)
    if H_actual <= 0:
        print("Error: Context length is equal to or larger than trajectory length.")
        return
        
    truth = O[ctxlen:ctxlen + H_actual, test_channel]
    
    print(f"Forecasting {H_actual} steps ahead...")
    # Use a low temperature (0.1) to make the categorical sampling nearly deterministic,
    # preventing harsh random jumps that make the physical trajectories look disconnected.
    fc = forecast_channel(model, quant, context, H=H_actual, n_samples=30, temperature=0.7)
    pred_mean = fc.mean(0)
    
    rmse = np.sqrt(np.mean((pred_mean - truth)**2))
    print(f"Forecast RMSE: {rmse:.5f}")
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(8, 4))
    
    time_ctx = np.arange(ctxlen)
    time_pred = np.arange(ctxlen - 1, ctxlen + H_actual)
    
    ax.plot(time_ctx, context, color='black', label='Context')
    
    truth_plot = np.concatenate([[context[-1]], truth])
    ax.plot(time_pred, truth_plot, color='blue', label='Ground Truth')
    
    # Plot samples
    for i in range(min(10, fc.shape[0])):
        fc_plot = np.concatenate([[context[-1]], fc[i]])
        ax.plot(time_pred, fc_plot, color='red', alpha=0.1)
    
    pred_mean_plot = np.concatenate([[context[-1]], pred_mean])
    ax.plot(time_pred, pred_mean_plot, color='red', label='TSFM Mean Forecast')
    
    ax.set_title(f'TSFM Forecast on {name} (Checkpoint)')
    ax.legend()
    
    out_path = os.path.join(out_dir, f'tsfm_forecast_{name}_eval.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved forecast plot to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific TSFM checkpoint.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Name of the checkpoint file. If omitted, uses the most recent checkpoint.")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    B = config["B"]
    ctx = config["ctx"]
    d_model = config["d_model"]
    n_layer = config["n_layer"]
    H = config["H"]
    M = config["M"]
    T = config["T"]
    seeds = config["seeds"]

    # Load Model (Force CPU to prevent interfering with GPU training)
    device = torch.device('cpu')
    model = TinyTSLM(B=B, d_model=d_model, n_layer=n_layer, ctx=ctx).to(device)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    if args.checkpoint:
        ckpt_path = os.path.join(ckpt_dir, args.checkpoint)
    else:
        if not os.path.exists(ckpt_dir):
            print(f"Error: Checkpoints directory not found at {ckpt_dir}")
            sys.exit(1)
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
        if not ckpts:
            print(f"Error: No checkpoints found in {ckpt_dir}")
            sys.exit(1)
        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
        latest = ckpts[-1]
        ckpt_path = os.path.join(ckpt_dir, latest)
        print(f"No checkpoint specified. Automatically using latest: {latest}")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    ckpt_filename = os.path.basename(ckpt_path)
    ckpt_name = os.path.splitext(ckpt_filename)[0]
    out_dir = os.path.join(os.path.dirname(__file__), 'evals', ckpt_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading checkpoint {ckpt_path} onto {device}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    quant = MeanScaleQuantizer(B=B)
    
    test_data = {}
    for sys_name, sys_info in SYSTEMS_REGISTRY.items():
        prefix = sys_info["prefix"]
        test_key = f"{prefix}_test"
        # Load the test trajectories using the cached files
        test_data[sys_name] = load_or_generate(sys_name, test_key, M, T, seeds.get(test_key, 42))

    for sys_name in SYSTEMS_REGISTRY.keys():
        evaluate_model(sys_name, model, quant, test_data[sys_name], H, out_dir)

if __name__ == "__main__":
    main()
