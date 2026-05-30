"""
Scratch PyTorch implementations of DQN and Dueling DQN for Atari Pong.
"""

from src.scratch.replay_buffer import ReplayBuffer
from src.scratch.models import DQN, DuelingDQN
from src.scratch.dqn_agent import DQNAgent
from src.scratch.train import train_scratch

__all__ = ["ReplayBuffer", "DQN", "DuelingDQN", "DQNAgent", "train_scratch"]

