# Pong Reinforcement Learning Comparison

A benchmark suite to compare value-based (scratch PyTorch DQN, Dueling Double DQN) and policy-based (PPO via Stable-Baselines3) Reinforcement Learning algorithms on Atari Pong.

---

## Getting Started

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Run custom PyTorch scratch training:
   ```bash
   uv run scripts/run_scratch.py --algo dueling_ddqn
   ```
3. Run Stable-Baselines3 baseline training:
   ```bash
   uv run scripts/run_sb3.py --algo ppo
   ```
