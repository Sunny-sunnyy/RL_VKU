import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.scratch.replay_buffer import ReplayBuffer
from src.scratch.models import DQN

class DQNAgent:
    """
    Standard Deep Q-Network (DQN) Agent.
    """
    def __init__(
        self,
        state_shape: tuple,
        num_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 10000,
        device: str = "cpu"
    ):
        """
        Initialize the DQNAgent.

        Args:
            state_shape (tuple): Observation space shape.
            num_actions (int): Number of actions.
            learning_rate (float): Learning rate for Adam optimizer.
            gamma (float): Discount factor.
            buffer_size (int): Replay buffer size.
            batch_size (int): Minibatch size.
            target_update_freq (int): Target network update frequency (steps).
            device (str): Device to run network computations on.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Setup networks
        self.online_net = DQN(state_shape, num_actions).to(device)
        self.target_net = DQN(state_shape, num_actions).to(device)
        
        # Synchronize weights initially
        self.update_target_network()
        self.target_net.eval()
        
        # Configure target network to not compute gradients
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # Instantiate the replay buffer
        self.replay_buffer = ReplayBuffer(size=buffer_size, state_shape=state_shape, device=device)
        
        self.loss_fn = nn.MSELoss()

    def update_target_network(self) -> None:
        """
        Copy weights from online_net to target_net.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select an action using epsilon-greedy exploration.

        Args:
            state (np.ndarray): The current observation state.
            epsilon (float): The current exploration parameter.

        Returns:
            int: The selected action index.
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.num_actions))
        
        # Non-random action selection
        # Convert state to tensor and add batch dimension
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            action = q_values.argmax(dim=1).item()
            
        return int(action)

    def update(self) -> float:
        """
        Perform a single gradient descent update step on a sampled batch of experiences.

        Returns:
            float: Loss value from the gradient step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get Q-values for current states
        q_values = self.online_net(states)
        # Select Q-values of actions taken
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute targets for next states
        with torch.no_grad():
            # Standard DQN: Choose action and evaluate with target_net
            next_state_values = self.target_net(next_states).max(dim=1)[0]
                
            # Bellman equation: target = reward + gamma * next_q * (1 - done)
            expected_state_action_values = rewards + self.gamma * next_state_values * (1.0 - dones)
            
        # Compute MSE loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        # Optimize online network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability in deep RL training
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return loss.item()
