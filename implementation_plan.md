# Implementation Plan: Pong Reinforcement Learning Comparison

This document details the engineering execution plan to benchmark value-based and policy-based Reinforcement Learning (RL) models on Atari Pong (`PongNoFrameskip-v4`), utilizing both custom PyTorch implementations and Stable-Baselines3.

---

## Proposed Changes

We have built the codebase under a new modular layout `g:/RL_VKU/ck/pong-comparison/`.

### 1. Project Scaffolding
*   **`pyproject.toml`:** Setup modern `uv` Python package workspace with PyTorch (CUDA supported), Gymnasium, ALE-Py, Stable-Baselines3, and dependencies.

### 2. Common Infrastructure
*   **`src/common/wrappers.py`:** Gymnasium environment wrapper for frame skipping, max pooling, grayscale conversion, 84x84 resizing, normalization, and 4-frame stacking (Dynamic Atari environments registered with `ale_py`).
*   **`src/common/utils.py`:** Helpers for loading/saving configurations, moving averages calculation, and generating comparison graphs.

### 3. Version A: Custom PyTorch Scratch
*   **`src/scratch/replay_buffer.py`:** Memory-efficient NumPy replay buffer storing transitions `(s, a, r, s_next, done)` as `uint8` to optimize RAM.
*   **`src/scratch/models.py`:** PyTorch network architectures for Standard DQN and Dueling DQN.
*   **`src/scratch/agent.py`:** Custom Agent class implementing the Double DQN update equation, Adam optimization, and gradient norm clipping.
*   **`src/scratch/train.py`:** Headless custom training loop logging performance metrics to TensorBoard and CSV at regular intervals.

### 4. Version B: Stable-Baselines3
*   **`src/sb3/train.py`:** Clean script utilizing SB3 DQN (standard and Dueling modes) and PPO, integrating progress logging callbacks.

### 5. Automation & Evaluation
*   **`scripts/run_scratch.py`:** Script to launch custom PyTorch training runs with hyperparameter inputs.
*   **`scripts/run_sb3.py`:** Script to launch SB3 training runs.
*   **`scripts/evaluate.py`:** Evaluation runner that loads trained checkpoints, runs gameplay test episodes, exports rolling average curves, builds comparative tables, and renders MP4/GIF gameplays.

### 6. Verification
*   **`tests/test_wrappers.py`:** Gymnasium wrappers testing.
*   **`tests/test_replay_buffer.py`:** pre-allocation and indexing rollover testing.
*   **`tests/test_models.py`:** DQN shapes and Dueling Q-value math checks.
