import numpy as np
import torch
import pytest
from src.scratch.replay_buffer import ReplayBuffer

def test_replay_buffer_init():
    """
    Test that ReplayBuffer initializes arrays with correct shapes, dtypes, and attributes.
    """
    size = 100
    state_shape = (4, 84, 84)
    device = "cpu"
    
    buffer = ReplayBuffer(size=size, state_shape=state_shape, device=device)
    
    assert buffer.size == size
    assert buffer.state_shape == state_shape
    assert buffer.device == device
    assert buffer.idx == 0
    assert buffer.current_size == 0
    assert len(buffer) == 0
    
    # Check underlying pre-allocated numpy arrays
    assert buffer.states.shape == (size, 4, 84, 84)
    assert buffer.states.dtype == np.uint8
    assert buffer.next_states.shape == (size, 4, 84, 84)
    assert buffer.next_states.dtype == np.uint8
    assert buffer.actions.shape == (size,)
    assert buffer.actions.dtype == np.int64
    assert buffer.rewards.shape == (size,)
    assert buffer.rewards.dtype == np.float32
    assert buffer.dones.shape == (size,)
    assert buffer.dones.dtype == np.bool_

def test_replay_buffer_add_and_rollover():
    """
    Test adding transitions to the buffer and verifying the modular index rollover when exceeding max capacity.
    """
    size = 5
    state_shape = (4, 84, 84)
    buffer = ReplayBuffer(size=size, state_shape=state_shape)
    
    # 1. Add steps up to max capacity
    for i in range(size):
        state = np.ones(state_shape, dtype=np.float32) * (i / 10.0)
        next_state = np.ones(state_shape, dtype=np.float32) * ((i + 1) / 10.0)
        action = i
        reward = float(i)
        done = (i % 2 == 0)
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == i + 1
        assert buffer.idx == (i + 1) % size
        
    # Check that it's full
    assert len(buffer) == size
    assert buffer.idx == 0
    
    # 2. Add extra step to trigger rollover (step 6, index 0 is overwritten)
    extra_state = np.ones(state_shape, dtype=np.float32) * 0.9
    extra_next_state = np.ones(state_shape, dtype=np.float32) * 1.0
    buffer.add(extra_state, 9, 99.0, extra_next_state, True)
    
    assert len(buffer) == size
    assert buffer.idx == 1  # idx incremented from 0 to 1
    
    # Verify index 0 was overwritten
    # float input 0.9 scaled to uint8: 0.9 * 255 = 229.5 -> rounded to 230
    assert np.all(buffer.states[0] == 230)
    assert buffer.actions[0] == 9
    assert buffer.rewards[0] == 99.0
    assert np.all(buffer.next_states[0] == 255)
    assert bool(buffer.dones[0]) is True

def test_replay_buffer_sample():
    """
    Test sampling a batch of transitions from the buffer and verify tensor types, shapes, and normalization.
    """
    size = 10
    state_shape = (4, 84, 84)
    device = "cpu"
    buffer = ReplayBuffer(size=size, state_shape=state_shape, device=device)
    
    # Add dummy experiences
    for i in range(8):
        state = np.random.rand(*state_shape).astype(np.float32)
        next_state = np.random.rand(*state_shape).astype(np.float32)
        action = np.random.randint(0, 6)
        reward = float(i)
        done = False
        buffer.add(state, action, reward, next_state, done)
        
    batch_size = 4
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    # Verify we got 5 PyTorch Tensors
    assert isinstance(states, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(next_states, torch.Tensor)
    assert isinstance(dones, torch.Tensor)
    
    # Verify shapes
    assert states.shape == (batch_size, *state_shape)
    assert next_states.shape == (batch_size, *state_shape)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)
    
    # Verify device and dtypes
    assert states.device.type == device
    assert states.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32
    
    # Verify normalization of observations [0.0, 1.0]
    assert torch.all(states >= 0.0)
    assert torch.all(states <= 1.0)
    assert torch.all(next_states >= 0.0)
    assert torch.all(next_states <= 1.0)
