# Walkthrough: Pong RL Comparison Benchmark

This document summarizes the successfully completed implementation of the Pong Reinforcement Learning Comparative Benchmarking Suite. It provides a technical breakdown of all files, mathematical formulations, and verification results.

---

## 1. Project Directory Structure
All deliverables are created in a highly modular and structured directory tree under `g:/RL_VKU/ck/pong-comparison/`:

```
g:/RL_VKU/ck/pong-comparison/
├── data/                          # Outputs
│   ├── models/                    # Saved models (.pth checkpoints)
│   ├── logs/                      # CSV metrics (episodic rewards & losses)
│   └── videos/                    # Gameplay recordings (MP4/GIF)
├── notebooks/                     # Jupyter Notebooks running on Vast.ai / local
│   ├── dqn_scratch.ipynb          # Custom Standard DQN training and plots
│   ├── dueling_ddqn_scratch.ipynb  # Custom Dueling Double DQN training and plots
│   ├── ppo_scratch.ipynb          # Custom PPO training and plots
│   ├── dqn_sb3.ipynb              # SB3 DQN baseline training
│   ├── dueling_dqn_sb3.ipynb      # SB3 Dueling DQN baseline training
│   └── ppo_sb3.ipynb              # SB3 PPO baseline training
├── src/
│   ├── __init__.py
│   ├── common/                    # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── wrappers.py            # Gymnasium AtariPreprocessing & FrameStackObservation
│   │   └── utils.py               # CSVLogger, plotters, PyTorch checkpointing
│   └── scratch/                   # Version A: Custom PyTorch Scratch
│       ├── __init__.py
│       ├── replay_buffer.py       # Pre-allocated np.uint8 Replay Buffer
│       ├── models.py              # CNN DQN, DuelingDQN, and ActorCriticCNN architectures
│       ├── dqn_agent.py           # Standard DQN Agent (non-dueling, Standard Q-learning target)
│       ├── dueling_ddqn_agent.py  # Dueling Double DQN Agent (Double Q-learning updates)
│       └── ppo_agent.py           # PPO Agent (shared trunk, Categorical policy, GAE, clip loss)
├── tests/                         # Unit test suite verifying mathematical correctness
│   ├── __init__.py
│   ├── test_models.py             # DQN shapes & dueling arithmetic test
│   ├── test_replay_buffer.py      # Pre-allocation & ring buffer rollover test
│   ├── test_wrappers.py           # Gymnasium wrapper correctness test
│   └── test_ppo.py                # PPO shapes, probability check, GAE calculation, and entropy test
├── pyproject.toml                 # uv workspace packages config
└── uv.lock                        # uv environment lock file
```

---

## 2. Core Technical Components

### A. Shareable Frame Preprocessing Wrapper (`src/common/wrappers.py`)
To ensure perfectly fair comparative benchmarking, both custom training and Stable-Baselines3 utilize the exact same frame preprocessing and stacking pipeline:
*   **AtariPreprocessing:** Crops static scoreboards, converts to Grayscale, resizes observations to 84x84, max-pools over 4 skipped frames, and scales pixels to float32 range `[0.0, 1.0]`.
*   **FrameStackObservation:** Stacks the 4 most recent grayscale frames together to form a state observation tensor of shape `(4, 84, 84)`, capturing motion direction and velocity.

### B. Custom PyTorch Scratch DQN, Dueling Double DQN, and PPO (`src/scratch/`)
*   **`replay_buffer.py`:** Pre-allocates fixed NumPy arrays. Observations are stored as `np.uint8` arrays (taking up ~2.8GB RAM for 100,000 steps instead of ~11.2GB if stored as `float32`). Upon sampling, states are moved to the target device, converted to `float32`, and normalized by 255.0 in a single GPU operation.
*   **`models.py`:**
    *   `DQN`: Nature CNN mapping observations to action Q-values.
    *   `DuelingDQN`: Splits flat features into Value $V(s)$ and Advantage $A(s, a)$, combined mathematically as:
        $$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$
    *   `ActorCriticCNN`: Shared CNN trunk with linear `actor` head (mapping to action logits) and linear `critic` head (mapping to state-value $V(s)$).
*   **`dqn_agent.py`:** Implements standard DQN. Updates use standard Bellman backups:
    $$Y_t = R_t + \gamma \max_a Q_{\text{target}}(S_{t+1}, a)$$
*   **`dueling_ddqn_agent.py`:** Integrates the Double DQN update equation:
    $$Y_t^{\text{DoubleQ}} = R_t + \gamma Q_{\text{target}}(S_{t+1}, \operatorname{argmax}_a Q_{\text{online}}(S_{t+1}, a))$$
*   **`ppo_agent.py`:** Implements PPO from scratch.
    *   Collects rollout trajectories of size $N=2048$ steps sequentially.
    *   Calculates GAE advantages:
        $$\delta_t^V = r_t + \gamma V(s_{t+1})(1 - \text{done}_t) - V(s_t)$$
        $$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
    *   Updates network weights using stochastic gradient descent over 4 epochs (batch size 64) with clipped policy loss, clipped value loss, and entropy bonuses to stabilize learning.

---

## 3. Unit Test Verification Results
All unit tests are executed using `pytest` inside the virtual environment:
```bash
uv run pytest
```
### Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.12, pytest-9.0.3, pluggy-1.6.0
rootdir: G:\RL_VKU\ck\pong-comparison
configfile: pyproject.toml
collected 12 items

src\scratch\test_imports.py ..                                           [ 16%]
tests\test_models.py ..                                                  [ 33%]
tests\test_ppo.py ....                                                   [ 66%]
tests\test_replay_buffer.py ...                                          [ 91%]
tests\test_wrappers.py .                                                 [100%]

============================= 12 passed in 5.34s ==============================
```
*   **`test_wrappers.py`:** Verified Atari env initialization, observation space shape of `(4, 84, 84)`, and correct scaling in `[0.0, 1.0]`.
*   **`test_replay_buffer.py`:** Verified memory pre-allocation, ring buffer index rollover, sampling shapes, and cast normalizations.
*   **`test_models.py`:** Verified DQN output shapes and mathematically validated Dueling DQN arithmetic $Q = V + A - \text{mean}(A)$.
*   **`test_ppo.py`:** Verified ActorCriticCNN outputs, probability bounds, GAE calculations matching manual dummy computation, and entropy bonus validation.

---

## 4. Guidelines for Vast.ai Training
When renting GPUs on Vast.ai to run training runs, follow these steps:
1.  **Start Jupyter Instance:** Rent an RTX 3090 Ti or RTX 5090 instance and open the provided Jupyter Lab interface.
2.  **Clone / Copy Code:** Copy the `pong-comparison` folder to the Jupyter workspace.
3.  **Sync Workspace & Register ROMs:**
    Open a terminal inside Jupyter and run:
    ```bash
    uv sync
    uv run AutoRom --accept-license
    ```
4.  **Run Jupyter Notebooks:**
    Navigate to the `notebooks/` directory and run any notebook (e.g. `ppo_scratch.ipynb` or `ppo_sb3.ipynb`) cell-by-cell to execute, visualize real-time training plots, and automatically export CSV metrics and models.
