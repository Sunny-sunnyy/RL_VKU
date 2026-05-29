import argparse
from src.scratch.train import train_scratch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Custom PyTorch Scratch RL Training on Pong")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="dueling_ddqn", 
        choices=["dqn", "dueling_ddqn"],
        help="Algorithm to run: dqn (Standard DQN) or dueling_ddqn (Dueling Double DQN)"
    )
    parser.add_argument("--steps", type=int, default=1000000, help="Total environment steps to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    
    args = parser.parse_args()
    
    dueling = (args.algo == "dueling_ddqn")
    
    print(f"Launching Custom PyTorch Scratch training: {args.algo}")
    train_scratch(
        env_id="PongNoFrameskip-v4",
        dueling=dueling,
        double=True,
        total_steps=args.steps,
        lr=args.lr,
        seed=args.seed
    )
