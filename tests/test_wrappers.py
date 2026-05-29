import gymnasium as gym
import numpy as np
from src.common.wrappers import make_atari_env

def test_make_atari_env():
    """
    Test that make_atari_env successfully initializes an Atari env,
    the observation space is (4, 84, 84), and env.reset() returns
    a float32 numpy array with values scaled to [0.0, 1.0].
    """
    env_id = "PongNoFrameskip-v4"
    env = make_atari_env(env_id)
    
    try:
        # 1. Test the environment successfully initialized
        assert env is not None
        
        # 2. Test that the environment's observation space shape is exactly (4, 84, 84)
        expected_shape = (4, 84, 84)
        assert env.observation_space.shape == expected_shape
        
        # 3. Test that running env.reset() yields observations of type float32 with values within [0.0, 1.0]
        obs, info = env.reset(seed=42)
        
        # Check that observation is a numpy array (Gymnasium 1.0+ FrameStackObservation returns a numpy array)
        assert isinstance(obs, np.ndarray)
        obs_array = obs
        
        assert obs_array.shape == expected_shape
        assert obs_array.dtype == np.float32
        
        # All values should be normalized within [0.0, 1.0]
        assert np.all(obs_array >= 0.0)
        assert np.all(obs_array <= 1.0)
        
    finally:
        env.close()
