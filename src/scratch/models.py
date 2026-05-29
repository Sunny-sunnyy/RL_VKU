import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Standard Deep Q-Network (DQN) architecture for Atari games.
    """
    def __init__(self, input_shape: tuple = (4, 84, 84), num_actions: int = 6):
        """
        Initialize DQN.

        Args:
            input_shape (tuple): Shape of the input observation tensor (channels, height, width).
            num_actions (int): Number of actions in the action space.
        """
        super(DQN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute flat feature size after convolutional layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.feature_size = self.cnn(dummy).numel()
            
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input observations of shape (batch, channels, height, width)
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        conv_out = self.cnn(x)
        flat_out = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(flat_out)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network (DuelingDQN) architecture.
    Splits feature representation into a state-value stream V(s) and an advantage stream A(s, a).
    """
    def __init__(self, input_shape: tuple = (4, 84, 84), num_actions: int = 6):
        """
        Initialize DuelingDQN.

        Args:
            input_shape (tuple): Shape of the input observation tensor (channels, height, width).
            num_actions (int): Number of actions in the action space.
        """
        super(DuelingDQN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute flat feature size after convolutional layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.feature_size = self.cnn(dummy).numel()
            
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input observations of shape (batch, channels, height, width)
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        conv_out = self.cnn(x)
        flat_out = conv_out.reshape(conv_out.size(0), -1)
        
        values = self.value_stream(flat_out)
        advantages = self.advantage_stream(flat_out)
        
        # Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic CNN architecture for Atari games (Nature CNN trunk with shared weights).
    """
    def __init__(self, input_shape: tuple = (4, 84, 84), num_actions: int = 6):
        """
        Initialize ActorCriticCNN.

        Args:
            input_shape (tuple): Shape of the input observation tensor (channels, height, width).
            num_actions (int): Number of actions in the action space.
        """
        super(ActorCriticCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute flat feature size after convolutional layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.feature_size = self.cnn(dummy).numel()
            
        # Actor head: policy logits
        self.actor = nn.Linear(self.feature_size, num_actions)
        
        # Critic head: state value
        self.critic = nn.Linear(self.feature_size, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input observations of shape (batch, channels, height, width)
            
        Returns:
            tuple: (policy_logits, state_value)
                - policy_logits (torch.Tensor): Shape (batch, num_actions)
                - state_value (torch.Tensor): Shape (batch, 1)
        """
        conv_out = self.cnn(x)
        flat_out = conv_out.reshape(conv_out.size(0), -1)
        
        policy_logits = self.actor(flat_out)
        state_value = self.critic(flat_out)
        
        return policy_logits, state_value

