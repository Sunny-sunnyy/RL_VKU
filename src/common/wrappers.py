import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

def make_atari_env(env_id: str, render_mode: str = None) -> gym.Env:
    """
    Create and wrap an Atari environment with standard preprocessing.
    
    Preprocessing steps:
    1. Noop Reset: Takes a random number (up to 30) of no-op actions on reset.
    2. Frame Skip: Repeats each action for 4 frames.
    3. Max Pool: Takes max pixel values over the last 2 frames of the skip.
    4. Grayscale: Converts RGB images to Grayscale.
    5. Screen Resize: Resizes the screen from 210x160 to 84x84.
    6. Normalize: Scales pixel values from [0, 255] to [0.0, 1.0].
    7. Frame Stack: Stacks the 4 most recent frames to capture motion.
    
    Returns:
        gym.Env: The wrapped Gymnasium environment.
    """
    env = gym.make(env_id, render_mode=render_mode)
    
    # Standard Atari Preprocessing wrapper
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True,
    )
    
    # Stack 4 consecutive frames (resulting in an observation shape of 4x84x84)
    env = FrameStackObservation(env, stack_size=4)
    
    return env
