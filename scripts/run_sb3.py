import argparse
from src.sb3.train import train_sb3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable-Baselines3 Benchmark Training on Pong")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="ppo", 
        choices=["dqn", "dueling_dqn", "ppo"],
        help="Algorithm to train: dqn, dueling_dqn, or ppo"
    )
    parser.add_argument("--steps", type=int, default=1000000, help="Total environment steps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Launching Stable-Baselines3 benchmark training: {args.algo}")
    train_sb3(
        algo=args.algo,
        env_id="PongNoFrameskip-v4",
        total_timesteps=args.steps,
        seed=args.seed
    )
