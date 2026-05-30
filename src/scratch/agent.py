import numpy as np
import torch
from src.scratch.dqn_agent import DQNAgent as VanillaDQNAgent
from src.scratch.dueling_ddqn_agent import DuelingDDQNAgent

class DQNAgent:
    """
    Unifying DQNAgent adapter.
    Delegates to VanillaDQNAgent or DuelingDDQNAgent based on parameters
    to preserve backwards compatibility with old unified interfaces.
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
        device: str = "cpu",
        dueling: bool = False,
        double: bool = False
    ):
        self.dueling = dueling
        self.double = double
        
        if dueling:
            # Dueling DDQN Agent (which implements Double updates)
            self.underlying_agent = DuelingDDQNAgent(
                state_shape=state_shape,
                num_actions=num_actions,
                learning_rate=learning_rate,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update_freq=target_update_freq,
                device=device
            )
        else:
            # Standard DQN Agent (vanilla non-dueling Q-learning)
            self.underlying_agent = VanillaDQNAgent(
                state_shape=state_shape,
                num_actions=num_actions,
                learning_rate=learning_rate,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update_freq=target_update_freq,
                device=device
            )

    @property
    def online_net(self):
        return self.underlying_agent.online_net

    @property
    def target_net(self):
        return self.underlying_agent.target_net

    @property
    def optimizer(self):
        return self.underlying_agent.optimizer

    @property
    def replay_buffer(self):
        return self.underlying_agent.replay_buffer

    def update_target_network(self) -> None:
        """
        Synchronize target network weights.
        """
        self.underlying_agent.update_target_network()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        return self.underlying_agent.select_action(state, epsilon)

    def update(self) -> float:
        """
        Perform a gradient descent step.
        """
        return self.underlying_agent.update()
