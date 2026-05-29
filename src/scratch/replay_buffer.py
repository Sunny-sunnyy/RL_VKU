import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    """
    Experience Replay Buffer for training reinforcement learning agents on Atari.
    
    To conserve RAM/VRAM, states and next_states are stored as np.uint8.
    During sampling, states are converted to float32 and normalized by 255.0.
    """
    def __init__(self, size: int, state_shape: tuple = (4, 84, 84), device: str = "cpu"):
        """
        Initialize the ReplayBuffer.

        Args:
            size (int): Maximum capacity of the buffer.
            state_shape (tuple): Dimension of the observation state.
            device (str): Device to move the sampled PyTorch tensors to.
        """
        self.size = size
        self.state_shape = state_shape
        self.device = device
        
        # Pre-allocate numpy arrays for performance and memory optimization
        self.states = np.zeros((size, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((size, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        
        self.idx = 0
        self.current_size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): The current observation.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The subsequent observation.
            done (bool): Whether the episode ended.
        """
        # Convert state/next_state to uint8 if they are in float [0, 1] range
        if np.issubdtype(state.dtype, np.floating):
            state_uint8 = (state * 255.0).round().astype(np.uint8)
        else:
            state_uint8 = state.astype(np.uint8)
            
        if np.issubdtype(next_state.dtype, np.floating):
            next_state_uint8 = (next_state * 255.0).round().astype(np.uint8)
        else:
            next_state_uint8 = next_state.astype(np.uint8)

        self.states[self.idx] = state_uint8
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state_uint8
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            Tuple of torch.Tensors:
                - states: Shape (batch_size, 4, 84, 84), dtype float32, range [0, 1]
                - actions: Shape (batch_size,), dtype int64
                - rewards: Shape (batch_size,), dtype float32
                - next_states: Shape (batch_size, 4, 84, 84), dtype float32, range [0, 1]
                - dones: Shape (batch_size,), dtype float32 (or bool, typically float32 for RL calculations)
        """
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        
        # Extract and cast observations to float32 and scale to [0, 1]
        states_tensor = torch.from_numpy(self.states[idxs]).to(self.device).float() / 255.0
        next_states_tensor = torch.from_numpy(self.next_states[idxs]).to(self.device).float() / 255.0
        
        actions_tensor = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards_tensor = torch.from_numpy(self.rewards[idxs]).to(self.device)
        dones_tensor = torch.from_numpy(self.dones[idxs].astype(np.float32)).to(self.device)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        """
        Return the current size of the replay buffer.
        """
        return self.current_size
