# CLAUDE.md: Pong RL Comparison Handoff & Guide

This file acts as a master entry point for new AI assistants or human developers to immediately grasp the project context, technical stack, available commands, and active next steps.

---

## 1. Project Overview & Context
This project is a scientific comparative benchmark that compares value-based (Custom PyTorch DQN & Dueling Double DQN from scratch) against policy-based (Custom PyTorch PPO from scratch and SB3 baselines) Reinforcement Learning algorithms on Atari Pong (`PongNoFrameskip-v4`).
*   **Target Device:** Trained on rented high-performance GPUs (Vast.ai / RunPod RTX 3090 Ti or RTX 5090) utilizing Jupyter Server.
*   **Current State:** All custom scratch architectures (DQN, Dueling Double DQN, PPO), Stable-Baselines3 baseline scripts, unified evaluation notebooks, and unit tests are fully written and verified. PPO (Scratch) is now fully integrated into the evaluation suite (`scripts/evaluate.py`). All 12 unit tests compile and pass successfully.

---

## 2. Command Reference

### Environment Sync & Setup
```bash
# Install package and virtual environment dependencies
uv sync

# Auto-download and register Atari game ROMs
uv run AutoRom --accept-license
```

### Executing Unit Tests
```bash
# Run all verified unit tests covering wrappers, replay buffer, networks, and GAE math
uv run pytest
```

### Training & Notebook Workspace
All training, logging, and performance visualization is executed via the independent Jupyter Notebooks under `notebooks/`:
*   `notebooks/dqn_scratch.ipynb`: Custom Standard DQN Agent training.
*   `notebooks/dueling_ddqn_scratch.ipynb`: Custom Dueling Double DQN Agent training.
*   `notebooks/ppo_scratch.ipynb`: Custom PPO Agent (shared trunk, GAE, clipped loss) training.
*   `notebooks/dqn_sb3.ipynb`: Stable-Baselines3 DQN baseline.
*   `notebooks/dueling_dqn_sb3.ipynb`: Stable-Baselines3 Dueling DQN baseline.
*   `notebooks/ppo_sb3.ipynb`: Stable-Baselines3 PPO baseline.

### Comparative Evaluation & Plotting
```bash
# Run full evaluation on existing model checkpoints and record gameplay videos
uv run scripts/evaluate.py --episodes 5 --video-dir data/videos

# Regenerate training curves comparison plot only (skips evaluation)
uv run scripts/evaluate.py --plot-only
```

---

## 3. Key File Map
*   `notebooks/`: Six independent `.ipynb` notebooks running headless sequential rollouts, training models, logging metrics in real-time, and rendering reward charts.
*   `src/common/wrappers.py`: Standard Gymnasium frame preprocessor and stacker wrapper (84x84 grayscale, 4 frame stacks).
*   `src/common/utils.py`: Logging to CSV, saving/loading PyTorch checkpoints, Matplotlib comparative curve plotter.
*   `src/scratch/replay_buffer.py`: Pre-allocated numpy `uint8` experience replay buffer.
*   `src/scratch/models.py`: PyTorch Nature CNN architectures for DQN, Dueling DQN, and ActorCriticCNN.
*   `src/scratch/dqn_agent.py`: Custom Standard DQN agent using standard DQN targets.
*   `src/scratch/dueling_ddqn_agent.py`: Custom Dueling Double DQN agent implementing Double Q-learning updates.
*   `src/scratch/ppo_agent.py`: Custom PPO agent implementing Actor-Critic rollouts, GAE calculations, and clipped PPO updates.
*   `tests/`: Unit test suite covering all core deep learning modules, wrappers, replay buffer, and custom PPO math.
*   `scripts/evaluate.py`: The evaluation suite supporting all 6 algorithms (value-based scratch, PPO scratch, and SB3 baselines).

---

## 4. Next Priorities
1.  Verify GPU card resources (CUDA support) on the target training environment.
2.  Open Jupyter Server on Vast.ai / RunPod, launch the desired notebooks under `notebooks/`, and execute them for high-timestep training runs (1,500,000 steps).
3.  Run a final evaluation comparative analysis combining the unified logged CSV outputs.
