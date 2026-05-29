import torch
import numpy as np
from torch.distributions import Categorical

from src.scratch.models import ActorCriticCNN
from src.scratch.ppo_agent import PPOAgent

def test_actor_critic_shapes():
    """
    Test that the forward pass of ActorCriticCNN processes a batch of observations (B, 4, 84, 84)
    and outputs the correct shape for policy logits (B, num_actions) and state value (B, 1).
    """
    input_shape = (4, 84, 84)
    num_actions = 6
    batch_size = 4
    
    model = ActorCriticCNN(input_shape=input_shape, num_actions=num_actions)
    
    # Create a random dummy input tensor mimicking normalized Atari observations
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # Run forward pass
    policy_logits, state_value = model(dummy_input)
    
    # Assert shapes are mathematically correct
    assert policy_logits.shape == (batch_size, num_actions)
    assert state_value.shape == (batch_size, 1)
    
    assert policy_logits.dtype == torch.float32
    assert state_value.dtype == torch.float32


def test_action_distribution():
    """
    Verify that probabilities computed from a Categorical distribution based on raw policy logits
    sum to exactly 1.0 (within computational tolerances using torch.allclose) and are non-negative.
    """
    batch_size = 4
    num_actions = 6
    logits = torch.randn(batch_size, num_actions)
    
    dist = Categorical(logits=logits)
    probs = dist.probs
    
    # 1. Verify all probabilities are non-negative
    assert torch.all(probs >= 0.0), "Probabilities should all be non-negative"
    
    # 2. Verify that they sum to 1.0 across actions for each batch element
    sum_probs = probs.sum(dim=-1)
    expected_sum = torch.ones_like(sum_probs)
    assert torch.allclose(sum_probs, expected_sum, atol=1e-6), "Action probabilities must sum to 1.0"


def test_gae_computation():
    """
    Validate the mathematical correctness of Generalized Advantage Estimation (GAE) computation.
    We set up dummy values for rewards, state values, next state value, and done flags,
    and compare the output of agent.compute_gae against a manually pre-calculated advantage array.
    """
    # Initialize PPOAgent with dummy parameters (state_shape and num_actions)
    agent = PPOAgent(
        state_shape=(4, 84, 84),
        num_actions=6,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    rewards = [1.0, 0.0, -1.0]
    values = [0.5, -0.2, 0.1]
    next_value = 0.8
    dones = [False, False, True]
    
    # --- Manual GAE & Returns Calculation ---
    # T = len(rewards) = 3
    # gamma = 0.99, lambda = 0.95, gamma_lambda = 0.99 * 0.95 = 0.9405
    #
    # t = 2:
    #   v_next_2 = next_value = 0.8
    #   done_next_2 = dones[2] = True (1.0)
    #   delta_2 = rewards[2] + gamma * v_next_2 * (1.0 - done_next_2) - values[2]
    #           = -1.0 + 0.99 * 0.8 * 0.0 - 0.1
    #           = -1.1
    #   gae_2   = delta_2 + gamma * lambda * (1.0 - done_next_2) * gae_prev (0.0)
    #           = -1.1
    #   advantages[2] = -1.1
    #
    # t = 1:
    #   v_next_1 = values[2] = 0.1
    #   done_next_1 = dones[1] = False (0.0)
    #   delta_1 = rewards[1] + gamma * v_next_1 * (1.0 - done_next_1) - values[1]
    #           = 0.0 + 0.99 * 0.1 * 1.0 - (-0.2)
    #           = 0.099 + 0.2 = 0.299
    #   gae_1   = delta_1 + gamma * lambda * (1.0 - done_next_1) * gae_2
    #           = 0.299 + 0.9405 * 1.0 * (-1.1)
    #           = 0.299 - 1.03455 = -0.73555
    #   advantages[1] = -0.73555
    #
    # t = 0:
    #   v_next_0 = values[1] = -0.2
    #   done_next_0 = dones[0] = False (0.0)
    #   delta_0 = rewards[0] + gamma * v_next_0 * (1.0 - done_next_0) - values[0]
    #           = 1.0 + 0.99 * (-0.2) * 1.0 - 0.5
    #           = 1.0 - 0.198 - 0.5 = 0.302
    #   gae_0   = delta_0 + gamma * lambda * (1.0 - done_next_0) * gae_1
    #           = 0.302 + 0.9405 * 1.0 * (-0.73555)
    #           = 0.302 - 0.691784775 = -0.389784775
    #   advantages[0] = -0.389784775
    
    expected_advantages_raw = np.array([-0.389784775, -0.73555, -1.1], dtype=np.float32)
    expected_returns = expected_advantages_raw + np.array(values, dtype=np.float32)
    
    # Compute standardized advantages manual formula
    mean_adv = expected_advantages_raw.mean()
    std_adv = expected_advantages_raw.std()
    expected_advantages = (expected_advantages_raw - mean_adv) / (std_adv + 1e-8)
    
    # Compute GAE via PPOAgent method
    advantages, returns = agent.compute_gae(rewards, values, next_value, dones)
    
    # Assert correctness within floating point tolerances
    assert np.allclose(advantages, expected_advantages, atol=1e-6), "Standardized advantages do not match manual GAE calculation"
    assert np.allclose(returns, expected_returns, atol=1e-6), "Target returns do not match manual calculation"


def test_entropy_loss():
    """
    Test that a distribution with high entropy (more uniform probability distribution)
    has a larger entropy value compared to a skewed distribution with low entropy.
    This validates the mathematical behavior of the entropy bonus loss component.
    """
    # 1. Uniform logits (highest uncertainty / uniform probability distribution)
    logits_high = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    dist_high = Categorical(logits=logits_high)
    entropy_high = dist_high.entropy().item()
    
    # 2. Skewed logits (high certainty / low entropy distribution)
    logits_low = torch.tensor([[15.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
    dist_low = Categorical(logits=logits_low)
    entropy_low = dist_low.entropy().item()
    
    # Assert that uniform distribution has higher entropy than skewed distribution
    assert entropy_high > entropy_low, f"Uniform entropy ({entropy_high}) should be greater than skewed entropy ({entropy_low})"
    
    # 3. Confirm mathematical properties
    # In PPO's joint loss, the entropy term is added as -entropy_coef * entropy (to maximize entropy).
    # Thus, a larger entropy value translates to a larger (more negative) entropy bonus,
    # which reduces the overall loss, encouraging exploration.
    assert entropy_low >= 0.0, "Entropy of any distribution must be non-negative"
