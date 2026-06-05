# GameBoyRL

**A framework for training and evaluating VLM agents on GameBoy games.**

GameBoyRL combines curiosity-driven RL exploration with Vision-Language Models to automatically discover tasks, generate training data, and benchmark agents across a suite of GameBoy titles. It is built on top of [GameBoyWorlds](GameBoyWorlds/README.md) for environment simulation.

---

# Core Features

**Curiosity-Driven Trajectory Collection:**
RL agents explore GameBoy environments driven by curiosity rewards, building replay buffers of diverse gameplay. Trajectories are clustered by observation similarity to surface distinct game states and behaviours.

**VLM Task Discovery:**
A VLM annotates clustered trajectories to infer what tasks the agent was implicitly solving, producing labelled task stores without any human annotation.

**Zero-Shot Task Proposal:**
VLMs can also propose candidate tasks for a game state directly, with optional augmentation from previously inferred curiosity tasks to improve proposal quality.

**Guided Practice and Attempt:**
Agents are given step-by-step guidance inferred by a VLM and run through structured practice sessions. Scored trajectories from practice and task-attempt runs are collected as fine-tuning data.

**VLM Fine-Tuning:**
LoRA fine-tuning of VLMs on the collected scored trajectories, producing specialised game-playing agents.

**Benchmarking:**
A unified benchmark suite evaluates any VLM agent across games, recording video and logging per-task results.

---

# Table of Contents

- [Setup](#setup)
- [Running Code](#running-code)
- [Developer Reference](#developer-reference)

---

# Setup

## 1. Clone

```bash
git clone --recursive https://github.com/DhananjayAshok/GameBoyRL
cd GameBoyRL
```

The `--recursive` flag is required to pull the [GameBoyWorlds](GameBoyWorlds/README.md) submodule.

## 2. Create Environments

```bash
bash setup/create_env.sh
```

If you intend to run VLM fine-tuning, also create the llm-utils environment:

```bash
bash setup/create_llm_utils_env.sh
```

## 3. Configure

Edit `configs/private_vars.yaml` to set your storage directory and HuggingFace details:

```yaml
storage_dir: "/path/to/your/storage"
huggingface_repo_namespace: "your-username"
huggingface_repo_name: "your-repo"
```

Also set the storage directory in `GameBoyWorlds/configs/private_vars.yaml` to the same path.

## 4. Sync Data

Pull the project data from the HuggingFace Hub:

```bash
python sync_data.py main setup_sync
```

## 5. ROMs

Legally acquire ROMs and place them in the GameBoyWorlds storage directory. See [GameBoyWorlds/README.md](GameBoyWorlds/README.md) for the expected paths per game.

## 6. Test

```bash
bash scripts/core_rl/default_rl.sh test
bash runs/benchmark_openrouter.sh test
```

---

# Running Code

All workflows are driven by shell scripts in `scripts/`. The three main categories are:

**RL** (`scripts/rl/`) — collect and cluster trajectories via curiosity-driven RL.

**VLM** (`scripts/vlm/`) — infer tasks, propose tasks, run practice and attempt sessions, fine-tune.

**Pipeline** (`scripts/pipeline/`) — end-to-end compositions of the above (e.g. curiosity exploration → task inference → attempt in a single call).

A typical end-to-end run looks like:

```bash
# 1. Collect and cluster trajectories for a game state
bash scripts/pipeline/curiosity_tasks.sh --game pokemon_red --run_name my_run ...

# 2. Propose and attempt tasks zero-shot (augmented with curiosity prior)
bash scripts/pipeline/propose_after_curiosity.sh --game pokemon_red --model_name gpt-4o ...

# 3. Benchmark the resulting agent
bash scripts/benchmark.sh --game pokemon_red ...
```

For a full breakdown of every script and what it does, see [README_dev.md](README_dev.md).

---

# Developer Reference

See [README_dev.md](README_dev.md) for a table of all scripts and their core functionality.
