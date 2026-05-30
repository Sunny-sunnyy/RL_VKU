import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.common.wrappers import make_atari_env
from src.common.utils import CSVLogger, save_checkpoint
from src.scratch.agent import DQNAgent

def train_scratch(
    env_id: str = "PongNoFrameskip-v4",
    dueling: bool = False,
    double: bool = True,
    total_steps: int = 500000,
    lr: float = 1e-4,
    buffer_size: int = 30000,
    batch_size: int = 32,
    target_update_freq: int = 2000,
    learning_starts: int = 5000,
    train_freq: int = 4,
    save_dir: str = "data/models",
    log_dir: str = "data/logs",
    seed: int = 42
):
    """
    Main training pipeline for the scratch PyTorch DQN/Dueling DQN agent on Atari Pong.

    Args:
        env_id (str): Gymnasium environment ID.
        dueling (bool): If True, use Dueling DQN network.
        double (bool): If True, use Double DQN updates.
        total_steps (int): Total environment steps to train for.
        lr (float): Learning rate.
        buffer_size (int): Max capacity of the replay buffer.
        batch_size (int): Size of training minibatches.
        target_update_freq (int): Target network sync frequency (in environment steps).
        learning_starts (int): Steps before training begins to fill the buffer.
        train_freq (int): Number of environment steps between agent updates.
        save_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory for TensorBoard and CSV logs.
        seed (int): Random seed for reproducibility.
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Environment
    env = make_atari_env(env_id)
    env.action_space.seed(seed)
    
    # Setup Loggers
    # Determine experiment name
    algo_name = f"{'Dueling' if dueling else 'DQN'}{'DDQN' if dueling and double else ('DQN' if dueling else ('DDQN' if double else ''))}"
    if dueling:
        algo_label = "Dueling DDQN (Scratch)" if double else "Dueling DQN (Scratch)"
    else:
        algo_label = "DQN (Scratch)" if double else "Standard DQN (Scratch)"
        
    run_name = f"{algo_name.lower()}_scratch_seed{seed}_{int(time.time())}"
    
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb", run_name))
    csv_file = os.path.join(log_dir, f"{run_name}.csv")
    csv_logger = CSVLogger(
        filepath=csv_file,
        headers=["step", "episode", "reward", "epsilon", "loss", "duration"]
    )
    
    # Initialize Agent
    state_shape = env.observation_space.shape  # Expecting (4, 84, 84)
    num_actions = env.action_space.n
    
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        learning_rate=lr,
        gamma=0.99,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device,
        dueling=dueling,
        double=double
    )
    
    # Epsilon decay configuration
    epsilon_start = 1.0
    epsilon_end = 0.01
    # Decay over 10% of total steps, capped at 100,000 steps minimum or custom logic
    exploration_fraction = 0.1
    epsilon_decay_steps = int(total_steps * exploration_fraction)
    
    # Training Loop variables
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    episode_start_time = time.time()
    
    # Track training losses per episode to log average loss
    episode_losses = []
    
    # Best reward tracking for checkpoints
    best_mean_reward = -float("inf")
    recent_rewards = []
    
    print(f"Starting training: {algo_label} on {env_id} using {device}")
    print(f"Total steps: {total_steps} | Warmup: {learning_starts} | Target sync: {target_update_freq}")
    
    for step in range(1, total_steps + 1):
        # Calculate linear epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (step / epsilon_decay_steps)
        )
        
        # Select action
        action = agent.select_action(state, epsilon)
        
        # Step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store in replay buffer
        # ReplayBuffer class internally converts float state to uint8
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # Train the model
        current_loss = 0.0
        if step > learning_starts and step % train_freq == 0:
            loss_val = agent.update()
            episode_losses.append(loss_val)
            current_loss = loss_val
            
            # Log loss to TensorBoard
            if len(episode_losses) % 100 == 0:
                tb_writer.add_scalar("losses/td_loss", loss_val, step)
                
        # Sync target network
        if step % target_update_freq == 0:
            agent.update_target_network()
            print(f"Step {step}/{total_steps} | Synced target network weights.")
            
        # Handle episode completion
        if done:
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            episode_count += 1
            
            # Compute average loss for this episode
            avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
            
            # Log metrics to console periodically
            if episode_count % 10 == 0 or episode_reward > 0:
                print(
                    f"Episode {episode_count:4d} | Step {step:6d} | "
                    f"Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f} | "
                    f"Avg Loss: {avg_loss:.5f} | Time: {episode_duration:.1f}s"
                )
                
            # Log metrics to TensorBoard
            tb_writer.add_scalar("rollout/ep_rew_mean", episode_reward, step)
            tb_writer.add_scalar("rollout/ep_len_mean", episode_steps, step)
            tb_writer.add_scalar("rollout/exploration_rate", epsilon, step)
            
            # Log metrics to CSV Logger
            csv_logger.log({
                "step": step,
                "episode": episode_count,
                "reward": episode_reward,
                "epsilon": epsilon,
                "loss": avg_loss,
                "duration": episode_duration
            })
            
            # Checkpoint saving: Save best models and periodically
            if episode_reward > best_mean_reward and step > learning_starts:
                best_mean_reward = episode_reward
                save_checkpoint(
                    state={
                        "step": step,
                        "episode": episode_count,
                        "online_net_state_dict": agent.online_net.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "reward": episode_reward,
                        "epsilon": epsilon
                    },
                    filepath=os.path.join(save_dir, f"{algo_name.lower()}_best.pt")
                )
                
            # Check early stopping (Mean reward over last 10 episodes >= 18.0)
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            if len(recent_rewards) >= 10 and np.mean(recent_rewards) >= 18.0:
                print(f"Early stopping triggered at step {step}! Mean reward over last 10 episodes is {np.mean(recent_rewards):.1f} >= 18.0. Target solved!")
                break

            # Reset episode variables
            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_start_time = time.time()
            episode_losses = []
            
        # Periodic checkpoint
        if step % 200000 == 0:
            save_checkpoint(
                state={
                    "step": step,
                    "episode": episode_count,
                    "online_net_state_dict": agent.online_net.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "reward": episode_reward,
                    "epsilon": epsilon
                },
                filepath=os.path.join(save_dir, f"{algo_name.lower()}_step_{step}.pt")
            )
            
    # Save final model
    save_checkpoint(
        state={
            "step": total_steps,
            "episode": episode_count,
            "online_net_state_dict": agent.online_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "reward": episode_reward,
            "epsilon": epsilon
        },
        filepath=os.path.join(save_dir, f"{algo_name.lower()}_final.pt")
    )
    
    # Close resources
    env.close()
    tb_writer.close()
    print("Training completed. Logs and models are successfully saved.")
