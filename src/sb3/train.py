import os
import random
import inspect
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.common.wrappers import make_atari_env
from src.common.utils import CSVLogger

class SB3CSVLoggerCallback(BaseCallback):
    """
    Custom callback for logging Stable-Baselines3 training metrics to a CSV file.
    Matches the custom logger from src.common.utils.
    """
    def __init__(self, csv_filepath: str, verbose: int = 0):
        """
        Initialize the callback.

        Args:
            csv_filepath (str): Path to save the CSV log file.
            verbose (int): Verbosity level.
        """
        super().__init__(verbose)
        self.csv_filepath = csv_filepath
        self.csv_logger = None
        self.episode_count = 0
        self.recent_rewards = []

    def _on_training_start(self) -> None:
        """
        Initialize CSVLogger with the required headers.
        """
        headers = ["episode", "reward", "steps", "length"]
        self.csv_logger = CSVLogger(self.csv_filepath, headers)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Extract and log metrics at the end of each episode.
        """
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if "episode" in info:
                    self.episode_count += 1
                    ep_reward = info["episode"]["r"]
                    self.recent_rewards.append(ep_reward)
                    if len(self.recent_rewards) > 10:
                        self.recent_rewards.pop(0)

                    self.csv_logger.log({
                        "episode": self.episode_count,
                        "reward": ep_reward,
                        "steps": self.num_timesteps,
                        "length": info["episode"]["l"]
                    })

                    # Early stopping when average reward over last 10 episodes >= 18.0
                    if len(self.recent_rewards) >= 10 and np.mean(self.recent_rewards) >= 18.0:
                        print(f"Early stopping triggered! Mean reward over last 10 episodes is {np.mean(self.recent_rewards):.1f} >= 18.0. Target solved!")
                        return False
        return True

def train_sb3(
    algo: str,
    env_id: str = "PongNoFrameskip-v4",
    total_timesteps: int = 500000,
    seed: int = 42,
    log_dir: str = "data/logs",
    save_dir: str = "data/models"
):
    """
    Train a Stable-Baselines3 agent (DQN, Dueling DQN, or PPO) on Atari Pong.

    Args:
        algo (str): Algorithm to train ("dqn", "dueling_dqn", or "ppo").
        env_id (str): Gymnasium environment ID.
        total_timesteps (int): Total environment steps to train for.
        seed (int): Random seed for reproducibility.
        log_dir (str): Directory for TensorBoard and CSV logs.
        save_dir (str): Directory to save final model checkpoints.
    """
    # Create necessary folders
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create and wrap the Atari environment
    env = make_atari_env(env_id)
    # Wrap in Monitor to record episode metrics for SB3 callback
    env = Monitor(env)

    # Configure TensorBoard logging path
    tb_log_dir = os.path.join(log_dir, "tb")

    # Instantiate model based on the selected algorithm
    if algo in ["dqn", "dueling_dqn"]:
        dqn_kwargs = {
            "policy": "CnnPolicy",
            "env": env,
            "learning_rate": 1e-4,
            "buffer_size": 30000,
            "learning_starts": 5000,
            "batch_size": 32,
            "target_update_interval": 2000,
            "train_freq": 4,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
            "exploration_initial_eps": 1.0,
            "seed": seed,
            "tensorboard_log": tb_log_dir,
            "policy_kwargs": {"normalize_images": False},
            "optimize_memory_usage": True,
            "replay_buffer_kwargs": {"handle_timeout_termination": False},
            "verbose": 1
        }

        # Inspect if 'dueling' is a supported parameter in the DQN constructor.
        # This handles standard Stable-Baselines3 (which is vanilla-only) as well as custom forks/wrappers gracefully.
        dqn_params = inspect.signature(DQN.__init__).parameters
        if "dueling" in dqn_params:
            dqn_kwargs["dueling"] = (algo == "dueling_dqn")
        else:
            if algo == "dueling_dqn":
                print("Warning: Dueling DQN requested, but standard Stable-Baselines3 DQN does not natively support 'dueling'. Training vanilla DQN.")

        model = DQN(**dqn_kwargs)

    elif algo == "ppo":
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            clip_range=0.2,
            ent_coef=0.01,
            seed=seed,
            tensorboard_log=tb_log_dir,
            policy_kwargs=dict(normalize_images=False),
            verbose=1
        )
    else:
        env.close()
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Set up custom callback
    csv_filepath = os.path.join(log_dir, f"{algo}_sb3.csv")
    callback = SB3CSVLoggerCallback(csv_filepath=csv_filepath)

    # Train the model
    print(f"Starting training: {algo} on {env_id} with seed {seed}")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    save_path = os.path.join(save_dir, f"{algo}_sb3.zip")
    model.save(save_path)
    print(f"Model saved successfully to {save_path}")

    # Clean up resources
    env.close()
    print("Training finished. Environment successfully closed.")
