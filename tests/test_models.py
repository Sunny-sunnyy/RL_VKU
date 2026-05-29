import torch
import numpy as np
from src.scratch.models import DQN, DuelingDQN

def test_dqn_shape():
    """
    Test standard DQN processes a batch of observations and outputs action Q-values of correct shape.
    """
    input_shape = (4, 84, 84)
    num_actions = 6
    batch_size = 4
    
    model = DQN(input_shape=input_shape, num_actions=num_actions)
    
    # Create a random dummy input tensor mimicking normalized Atari observations
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # Run forward pass
    q_values = model(dummy_input)
    
    # Assert output shape is (batch_size, num_actions)
    assert q_values.shape == (batch_size, num_actions)
    assert q_values.dtype == torch.float32

def test_dueling_dqn_shape_and_arithmetic():
    """
    Test DuelingDQN processes a batch of observations, returns action Q-values of correct shape,
    and adheres strictly to the dueling Q-value arithmetic: Q = V + A - mean(A).
    """
    input_shape = (4, 84, 84)
    num_actions = 6
    batch_size = 4
    
    model = DuelingDQN(input_shape=input_shape, num_actions=num_actions)
    
    # Create random dummy input
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # 1. Verify output shape
    q_values = model(dummy_input)
    assert q_values.shape == (batch_size, num_actions)
    assert q_values.dtype == torch.float32
    
    # 2. Extract intermediate streams and verify mathematical arithmetic of Dueling DQN
    with torch.no_grad():
        conv_out = model.cnn(dummy_input)
        flat_out = conv_out.reshape(conv_out.size(0), -1)
        
        # Manually compute values and advantages using model's sub-streams
        values = model.value_stream(flat_out)
        advantages = model.advantage_stream(flat_out)
        
        # Q = V + (A - mean(A))
        expected_q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        # Verify that the forward pass matches the manual dueling arithmetic calculation
        assert torch.allclose(q_values, expected_q_values, atol=1e-6)
        
        # Verify that the mean over the action dimension of Q - A + mean(A) is equal to V,
        # or equivalently, the mean of the Q-values over actions is exactly equal to the value stream V(s)
        mean_q_values = q_values.mean(dim=1, keepdim=True)
        assert torch.allclose(mean_q_values, values, atol=1e-6)
        
        # Verify that the advantages minus their mean equals Q-values minus the state values V(s)
        mean_subtracted_advantages = advantages - advantages.mean(dim=1, keepdim=True)
        q_minus_v = q_values - values
        assert torch.allclose(q_minus_v, mean_subtracted_advantages, atol=1e-6)
