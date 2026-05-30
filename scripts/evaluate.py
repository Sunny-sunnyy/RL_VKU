import os
import time
import argparse
import numpy as np
import torch
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.common.wrappers import make_atari_env
from src.common.utils import plot_comparison_curves, load_checkpoint
import importlib.util
# Dynamically load DQN, DuelingDQN, ActorCriticCNN directly from models.py to avoid broken imports in src.scratch.__init__
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_models_path = os.path.join(_project_root, "src", "scratch", "models.py")
_spec = importlib.util.spec_from_file_location("standalone_models", _models_path)
_models_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models_module)
DQN = _models_module.DQN
DuelingDQN = _models_module.DuelingDQN
ActorCriticCNN = _models_module.ActorCriticCNN
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3 import PPO as SB3_PPO

def find_ppo_scratch_log(log_dir: str) -> str:
    default_path = os.path.join(log_dir, "ppo_scratch.csv")
    if os.path.exists(default_path):
        return default_path
    import glob
    candidates = glob.glob(os.path.join(log_dir, "ppo_scratch*.csv"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return default_path

def find_ppo_scratch_checkpoint(model_dir: str) -> str:
    for ext in [".pt", ".pth"]:
        path = os.path.join(model_dir, f"ppo_scratch_best{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(model_dir, "ppo_scratch_best.pt")

def evaluate_ppo_scratch(model_path: str, num_episodes: int, device: str = "cpu", video_dir: str = None) -> tuple:
    env = make_atari_env("PongNoFrameskip-v4", render_mode="rgb_array" if video_dir else None)
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="ppo_scratch_eval")
        
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    model = ActorCriticCNN(input_shape=input_shape, num_actions=num_actions)
    checkpoint = load_checkpoint(model_path, device=device)
    
    # Handle nested or direct state dicts
    state_dict = checkpoint.get("state_dict", checkpoint.get("ac_net_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        
        while not (done or truncated):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=1).item()
                
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            
        episode_rewards.append(ep_reward)
        
    duration = time.time() - start_time
    fps = total_steps / duration if duration > 0 else 0
    env.close()
    
    return np.mean(episode_rewards), fps

def evaluate_scratch(model_path: str, dueling: bool, num_episodes: int, device: str = "cpu", video_dir: str = None) -> tuple:
    """
    Evaluate a custom PyTorch Scratch agent.
    """
    env = make_atari_env("PongNoFrameskip-v4", render_mode="rgb_array" if video_dir else None)
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="scratch_eval")
        
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Instantiate custom model
    if dueling:
        model = DuelingDQN(input_shape=input_shape, num_actions=num_actions)
    else:
        model = DQN(input_shape=input_shape, num_actions=num_actions)
        
    # Load state dict
    checkpoint = load_checkpoint(model_path, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        
        while not (done or truncated):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=1).item()
                
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            
        episode_rewards.append(ep_reward)
        
    duration = time.time() - start_time
    fps = total_steps / duration if duration > 0 else 0
    env.close()
    
    return np.mean(episode_rewards), fps

def evaluate_sb3(algo: str, model_path: str, num_episodes: int, video_dir: str = None) -> tuple:
    """
    Evaluate a Stable-Baselines3 agent.
    """
    env = make_atari_env("PongNoFrameskip-v4", render_mode="rgb_array" if video_dir else None)
    if video_dir:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix=f"{algo}_sb3_eval")
        
    # Load corresponding SB3 class
    if algo in ["dqn", "dueling_dqn"]:
        model = SB3_DQN.load(model_path, env=env)
    elif algo == "ppo":
        model = SB3_PPO.load(model_path, env=env)
    else:
        env.close()
        raise ValueError(f"Unsupported algorithm: {algo}")
        
    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            ep_reward += reward
            total_steps += 1
            
        episode_rewards.append(ep_reward)
        
    duration = time.time() - start_time
    fps = total_steps / duration if duration > 0 else 0
    env.close()
    
    return np.mean(episode_rewards), fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Agents and Generate Comparisons")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for PyTorch")
    parser.add_argument("--video-dir", type=str, default="data/videos", help="Directory to save gameplay videos")
    parser.add_argument("--log-dir", type=str, default="data/logs", help="Directory where CSV logs are located")
    parser.add_argument("--plot-only", action="store_true", help="Skip evaluation and only update comparison plot")
    
    args = parser.parse_args()
    
    # 1. Update Comparative Plot from CSV Logs
    print("Generating comparative training curves from CSV logs...")
    log_files = {
        "DQN (Scratch)": os.path.join(args.log_dir, "dqn_scratch.csv"),
        "Dueling DDQN (Scratch)": os.path.join(args.log_dir, "dueling_ddqn_scratch.csv"),
        "DQN (SB3)": os.path.join(args.log_dir, "dqn_sb3.csv"),
        "Dueling DQN (SB3)": os.path.join(args.log_dir, "dueling_dqn_sb3.csv"),
        "PPO (SB3)": os.path.join(args.log_dir, "ppo_sb3.csv"),
        "PPO (Scratch)": find_ppo_scratch_log(args.log_dir)
    }
    
    plot_path = os.path.join(args.log_dir, "comparison_plot.png")
    plot_comparison_curves(log_files, plot_path, window_size=50)
    print(f"Comparison plot saved successfully to {plot_path}")
    
    if args.plot_only:
        print("Plot updated. Skipping evaluation as requested.")
        exit(0)
        
    # 2. Run Evaluations on existing checkpoints
    print("\nRunning Evaluation Benchmark on available model checkpoints...")
    checkpoints = {
        "DQN (Scratch)": {"type": "scratch", "dueling": False, "path": "data/models/dqn_scratch.pth"},
        "Dueling DDQN (Scratch)": {"type": "scratch", "dueling": True, "path": "data/models/dueling_ddqn_scratch.pth"},
        "DQN (SB3)": {"type": "sb3", "algo": "dqn", "path": "data/models/dqn_sb3.zip"},
        "Dueling DQN (SB3)": {"type": "sb3", "algo": "dueling_dqn", "path": "data/models/dueling_dqn_sb3.zip"},
        "PPO (SB3)": {"type": "sb3", "algo": "ppo", "path": "data/models/ppo_sb3.zip"},
        "PPO (Scratch)": {"type": "scratch_ppo", "path": find_ppo_scratch_checkpoint("data/models")}
    }
    
    results = []
    
    for name, cfg in checkpoints.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"Skipping evaluation for {name}: Checkpoint '{path}' does not exist.")
            continue
            
        print(f"Evaluating {name}...")
        try:
            if cfg["type"] == "scratch":
                score, fps = evaluate_scratch(path, cfg["dueling"], args.episodes, device=args.device, video_dir=args.video_dir)
            elif cfg["type"] == "scratch_ppo":
                score, fps = evaluate_ppo_scratch(path, args.episodes, device=args.device, video_dir=args.video_dir)
            else:
                score, fps = evaluate_sb3(cfg["algo"], path, args.episodes, video_dir=args.video_dir)
                
            results.append({
                "Algorithm": name,
                "Average Reward": f"{score:+.2f}",
                "Inference Speed": f"{fps:.1f} frames/sec",
                "Status": "Passed" if score >= 15 else "Converging"
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    # 3. Print Performance Matrix
    if results:
        print("\n" + "="*70)
        print("               PONG RL AGENT COMPARATIVE PERFORMANCE MATRIX")
        print("="*70)
        print(f"{'Algorithm':<25} | {'Avg Reward':<12} | {'Inference Speed':<20} | {'Status':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['Algorithm']:<25} | {r['Average Reward']:<12} | {r['Inference Speed']:<20} | {r['Status']:<10}")
        print("="*70 + "\n")
    else:
        print("\nNo checkpoints were found to evaluate. Place trained models in 'data/models/' to run benchmark evaluations.")
