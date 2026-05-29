import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.scratch.models import ActorCriticCNN

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent with GAE.
    """
    def __init__(
        self,
        state_shape: tuple,
        num_actions: int,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        val_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize the PPOAgent.

        Args:
            state_shape (tuple): Observation space shape.
            num_actions (int): Number of actions.
            learning_rate (float): Learning rate for Adam optimizer.
            gamma (float): Discount factor.
            gae_lambda (float): GAE factor lambda.
            clip_ratio (float): PPO policy clipping ratio.
            val_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy bonus coefficient.
            epochs (int): Number of training epochs per update.
            batch_size (int): Minibatch size for updating.
            max_grad_norm (float): Maximum norm for gradient clipping.
            device (str): Device to run network computations on.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.val_coef = val_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize Actor-Critic Network
        self.ac_net = ActorCriticCNN(state_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate, eps=1e-5)
        
        # Track current state for sequential rollout collection
        self.current_state = None

    def select_action(self, state: np.ndarray) -> tuple:
        """
        Select an action using the current policy network.

        Args:
            state (np.ndarray): The current observation state.

        Returns:
            tuple: (action, log_prob, state_value)
                - action (int): The sampled action.
                - log_prob (float): The log probability of the selected action.
                - state_value (float): The estimated value of the current state.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.ac_net(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return int(action.item()), float(log_prob.item()), float(value.item())

    def collect_rollout(self, env, rollout_length: int) -> tuple:
        """
        Collect trajectories sequentially from a single environment.

        Args:
            env: The Gym environment to step through.
            rollout_length (int): Number of steps to collect.

        Returns:
            tuple: (states, actions, rewards, dones, log_probs, values)
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        if self.current_state is None:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                self.current_state = reset_result[0]
            else:
                self.current_state = reset_result

        for _ in range(rollout_length):
            action, log_prob, value = self.select_action(self.current_state)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            states.append(self.current_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            if done:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    self.current_state = reset_result[0]
                else:
                    self.current_state = reset_result
            else:
                self.current_state = next_state

        return states, actions, rewards, dones, log_probs, values

    def compute_gae(self, rewards: list, values: list, next_value: float, dones: list) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards (list): Collected rewards.
            values (list): Estimated state values.
            next_value (float): Estimated value of the final state in rollout.
            dones (list): Done flags.

        Returns:
            tuple: (advantages, returns)
                - advantages (np.ndarray): Standardized advantages.
                - returns (np.ndarray): Returns target.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            v_next = values[t + 1] if t < T - 1 else next_value
            done_next = dones[t]
            delta = rewards[t] + self.gamma * v_next * (1.0 - done_next) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done_next) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)

        # Standardize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        old_values: np.ndarray = None
    ) -> float:
        """
        Perform policy and value update using PPO loss objectives.

        Args:
            states (np.ndarray): Rollout states.
            actions (np.ndarray): Taken actions.
            old_log_probs (np.ndarray): Log probabilities under the old policy.
            returns (np.ndarray): Target returns.
            advantages (np.ndarray): Standardized advantages.
            old_values (np.ndarray, optional): State values estimated under old policy.

        Returns:
            float: Total loss value from the update steps.
        """
        # Convert all to PyTorch tensors
        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.array(actions)).long().to(self.device)
        old_log_probs_t = torch.from_numpy(np.array(old_log_probs)).float().to(self.device)
        returns_t = torch.from_numpy(np.array(returns)).float().to(self.device)
        advantages_t = torch.from_numpy(np.array(advantages)).float().to(self.device)
        
        if old_values is not None:
            old_values_t = torch.from_numpy(np.array(old_values)).float().to(self.device)
        else:
            old_values_t = None

        dataset_size = len(states)
        total_loss_accum = 0.0
        num_batches = 0

        for _ in range(self.epochs):
            # Shuffle indices
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                # Extract mini-batch tensors
                b_states = states_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_returns = returns_t[batch_indices]
                b_advantages = advantages_t[batch_indices]

                # Policy and Value predictions
                policy_logits, state_value = self.ac_net(b_states)
                v_pred = state_value.squeeze(-1)

                # Policy Loss (Clipped Policy Objective)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (Clipped/Unclipped Value Objective)
                if old_values_t is not None:
                    b_old_values = old_values_t[batch_indices]
                    v_loss_unclipped = (v_pred - b_returns) ** 2
                    v_pred_clipped = b_old_values + torch.clamp(v_pred - b_old_values, -self.clip_ratio, self.clip_ratio)
                    v_loss_clipped = (v_pred_clipped - b_returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((v_pred - b_returns) ** 2).mean()

                # Joint Objective loss calculation
                loss = policy_loss + self.val_coef * value_loss - self.entropy_coef * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += loss.item()
                num_batches += 1

        return total_loss_accum / max(num_batches, 1)
