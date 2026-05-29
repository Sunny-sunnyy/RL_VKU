import numpy as np
import torch

from src.scratch.models import DQN, DuelingDQN, ActorCriticCNN
from src.scratch.dqn_agent import DQNAgent
from src.scratch.dueling_ddqn_agent import DuelingDDQNAgent
from src.scratch.ppo_agent import PPOAgent

def test_models():
    print("Testing models initialization and forward pass...")
    state_shape = (4, 84, 84)
    num_actions = 6
    batch_size = 2
    dummy_input = torch.zeros(batch_size, *state_shape)
    
    # 1. DQN
    dqn = DQN(state_shape, num_actions)
    q_vals = dqn(dummy_input)
    assert q_vals.shape == (batch_size, num_actions), f"DQN output shape is {q_vals.shape}"
    print("DQN model OK.")
    
    # 2. DuelingDQN
    dueling_dqn = DuelingDQN(state_shape, num_actions)
    q_vals_d = dueling_dqn(dummy_input)
    assert q_vals_d.shape == (batch_size, num_actions), f"DuelingDQN output shape is {q_vals_d.shape}"
    print("DuelingDQN model OK.")
    
    # 3. ActorCriticCNN
    ac_cnn = ActorCriticCNN(state_shape, num_actions)
    logits, val = ac_cnn(dummy_input)
    assert logits.shape == (batch_size, num_actions), f"ActorCriticCNN logits shape is {logits.shape}"
    assert val.shape == (batch_size, 1), f"ActorCriticCNN value shape is {val.shape}"
    print("ActorCriticCNN model OK.")

class MockEnv:
    def __init__(self):
        self.state_shape = (4, 84, 84)
        
    def reset(self):
        return np.zeros(self.state_shape, dtype=np.uint8), {}
        
    def step(self, action):
        state = np.zeros(self.state_shape, dtype=np.uint8)
        reward = 1.0
        terminated = True
        truncated = False
        info = {}
        return state, reward, terminated, truncated, info

def test_agents():
    print("Testing agents initialization...")
    state_shape = (4, 84, 84)
    num_actions = 6
    
    # 1. DQNAgent
    dqn_agent = DQNAgent(state_shape, num_actions, device="cpu")
    print("DQNAgent initialized OK.")
    
    # 2. DuelingDDQNAgent
    dueling_agent = DuelingDDQNAgent(state_shape, num_actions, device="cpu")
    print("DuelingDDQNAgent initialized OK.")
    
    # 3. PPOAgent
    ppo_agent = PPOAgent(state_shape, num_actions, device="cpu")
    print("PPOAgent initialized OK.")
    
    # Test PPO sequential rollout & compute GAE & update
    print("Testing PPOAgent rollout collection & update cycle...")
    env = MockEnv()
    states, actions, rewards, dones, log_probs, values = ppo_agent.collect_rollout(env, rollout_length=10)
    assert len(states) == 10
    assert len(actions) == 10
    assert len(rewards) == 10
    assert len(dones) == 10
    assert len(log_probs) == 10
    assert len(values) == 10
    print("PPOAgent collect_rollout OK.")
    
    next_value = 0.5
    advantages, returns = ppo_agent.compute_gae(rewards, values, next_value, dones)
    assert len(advantages) == 10
    assert len(returns) == 10
    print("PPOAgent compute_gae OK.")
    
    loss = ppo_agent.update(states, actions, log_probs, returns, advantages, values)
    print(f"PPOAgent update completed with loss: {loss:.4f}")
    
    # Test DQN and DuelingDDQN buffer updates
    print("Testing DQN and DuelingDDQN buffer addition and update...")
    state = np.zeros(state_shape, dtype=np.uint8)
    action = 2
    reward = 1.0
    next_state = np.zeros(state_shape, dtype=np.uint8)
    done = False
    
    dqn_agent.replay_buffer.add(state, action, reward, next_state, done)
    loss_dqn = dqn_agent.update()
    print(f"DQNAgent update output (empty/small buffer): {loss_dqn}")
    
    # Add enough samples to trigger an actual update
    for _ in range(40):
        dqn_agent.replay_buffer.add(state, action, reward, next_state, done)
        dueling_agent.replay_buffer.add(state, action, reward, next_state, done)
        
    loss_dqn_active = dqn_agent.update()
    loss_dueling_active = dueling_agent.update()
    print(f"DQNAgent active update loss: {loss_dqn_active:.4f}")
    print(f"DuelingDDQNAgent active update loss: {loss_dueling_active:.4f}")

if __name__ == "__main__":
    test_models()
    test_agents()
    print("All tests passed successfully!")
