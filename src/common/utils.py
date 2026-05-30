import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def save_checkpoint(state: dict, filepath: str):
    """
    Save PyTorch model checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath: str, device: str = "cpu") -> dict:
    """
    Load PyTorch model checkpoint.
    """
    return torch.load(filepath, map_location=device)

class CSVLogger:
    """
    Simple CSV logger to record training progress metrics.
    """
    def __init__(self, filepath: str, headers: list):
        self.filepath = filepath
        self.headers = headers
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write headers if file is new
        if not os.path.exists(filepath):
            with open(filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, data: dict):
        """
        Append a row of metrics to the CSV file.
        """
        row = [data.get(h, "") for h in self.headers]
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

def plot_comparison_curves(log_files: dict, save_path: str, window_size: int = 100):
    """
    Generate a professional comparative plot of training curves from CSV log files.
    
    Args:
        log_files (dict): Dictionary mapping label names (e.g. 'DQN Scratch')
                          to their corresponding CSV file paths.
        save_path (str): File path to save the generated PNG plot.
        window_size (int): Rolling average window size for smoothing curve.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Custom curated high-quality aesthetic colors
    colors = {
        "DQN (Scratch)": "#1f77b4",            # Muted Blue
        "Dueling DDQN (Scratch)": "#ff7f0e",   # Vibrant Orange
        "DQN (SB3)": "#2ca02c",                # Soft Green
        "Dueling DQN (SB3)": "#d62728",        # Deep Red
        "PPO (SB3)": "#9467bd",                # Amethyst Purple
        "PPO (Scratch)": "#e377c2"             # Pink
    }
    
    for label, filepath in log_files.items():
        if not os.path.exists(filepath):
            continue
            
        try:
            df = pd.read_csv(filepath)
            if "episode" not in df.columns or "reward" not in df.columns:
                continue
                
            episodes = df["episode"].values
            rewards = df["reward"].values
            
            if len(rewards) == 0:
                continue
                
            # Choose color
            color = colors.get(label, None)
            
            # Plot raw rewards in faint background
            ax.plot(episodes, rewards, alpha=0.15, color=color)
            
            # Compute and plot rolling mean
            rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            ax.plot(episodes, rolling_mean, label=label, linewidth=2.0, color=color)
            
        except Exception as e:
            print(f"Error plotting data for {label}: {e}")
            
    ax.set_title("Atari Pong Training Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Episode", fontsize=12, labelpad=10)
    ax.set_ylabel("Reward", fontsize=12, labelpad=10)
    ax.set_ylim(-21.5, 21.5)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
