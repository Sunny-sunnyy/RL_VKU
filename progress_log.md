# Progress Log: Pong RL Comparison Project

This file serves as a persistent record of project progress, developmental decisions, and milestones.

---

## Project Status Overview
*   **Current State:** Phase 9 Completed. All custom scratch architectures (DQN, Dueling Double DQN, PPO), Stable-Baselines3 baseline scripts, unified evaluation notebooks, and unit tests are fully written and verified!
*   **Goal:** Implement PyTorch custom scratch implementations of DQN, Dueling Double DQN, and PPO and compare them against Stable-Baselines3 (DQN, Dueling DQN, PPO).
*   **Target Device:** High-Performance GPU Cluster (RTX 3090 Ti / RTX 5090) executing via Jupyter Server.

---

## Development Milestones

### 1. Planning & Design 
*   **Status:** Completed
*   **Deliverables:**
    *   Technical Specification: `implementation_plan.md`

### 2. Workspace Scaffolding
*   **Status:** Completed
*   **Tasks:**
    *   Initialize package workspace with `pyproject.toml` and lock dependencies.

### 3. Common Infrastructure
*   **Status:** Completed
*   **Tasks:**
    *   Atari Gymnasium environment wrappers (`src/common/wrappers.py`).
    *   Plotting and data logging utilities (`src/common/utils.py`).

### 4. Custom PyTorch Scratch Models and Refactoring
*   **Status:** Completed
*   **Tasks:**
    *   Update `models.py` with `ActorCriticCNN` shared trunk backbone.
    *   Refactor Standard DQN Agent into `dqn_agent.py` using Standard DQN targets.
    *   Refactor Dueling Double DQN Agent into `dueling_ddqn_agent.py` using Dueling structures and Double Q-targets.
    *   Implement PPO from scratch in `ppo_agent.py` utilizing Categorical policy distributions and GAE computation.

### 5. Automated Unit Testing & Technical Verification
*   **Status:** Completed
*   **Tasks:**
    *   Implement `tests/test_ppo.py` verifying ActorCriticCNN output shapes, probability checks, GAE calculations, and entropy values.
    *   Run all verified unit tests (12/12 passed!).

### 6. Training Jupyter Notebooks Design
*   **Status:** Completed
*   **Tasks:**
    *   Create `notebooks/` workspace directory.
    *   Build 6 independent `.ipynb` notebooks for modular training (3 custom scratch + 3 library baselines).
    *   Integrate unified loggers and callback monitors tracking episodic rewards and losses into CSV format.
