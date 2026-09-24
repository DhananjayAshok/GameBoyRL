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

**Task Attempt:**
Agents attempt the proposed tasks under a two-stage VLM judge that describes what happened and then rules on completion, retrying failures with a derived hint. Successful trajectories are saved as a `<stem>.json` + `<stem>.pkl` pair.

**Info Documents:**
Successful (task, trajectory) pairs are distilled into per-game info documents that benchmark agents can retrieve from at test time (the context-engineering arm).

**VLM Fine-Tuning:**
Trajectories are annotated with step-by-step guidance, replayed as practice sessions, filtered and paraphrased into train/validation CSVs, and used for LoRA fine-tuning of VLMs.

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

The `--recursive` flag is required to pull the `GameBoyWorlds`, `cleanrl` and `llm-utils` submodules.

## 2. Create Environments

### Main environment

```bash
bash setup/create_env.sh --env_dir /path/to/env
```

This script creates a Python 3.12 virtual environment at `env_dir` using `uv`, symlinks it to `setup/.venv`, installs all dependencies via `uv sync`, and generates the `config.env` files for both this repo and GameBoyWorlds.

<details>
<summary>Manual alternative</summary>

```bash
uv venv /path/to/env --python 3.12          # create the virtual environment
ln -s /path/to/env setup/.venv              # symlink so scripts can find it at setup/.venv
cd setup && uv sync && cd ..                # install dependencies from setup/pyproject.toml
source setup/.venv/bin/activate             # activate the environment
python configs/create_env_file.py           # generate config.env for this repo
python GameBoyWorlds/configs/create_env_file.py  # generate config.env for GameBoyWorlds
```
</details>

### llm-utils environment (VLM fine-tuning only)

```bash
bash setup/create_llm_utils_env.sh --env_dir /path/to/llm-env
```

This script creates a separate Python 3.12 environment for the `llm-utils` submodule, symlinks it to `llm-utils/setup/.venv`, and installs its dependencies via `uv sync`.

<details>
<summary>Manual alternative</summary>

```bash
uv venv /path/to/llm-env --python 3.12          # create a separate environment for llm-utils
ln -s /path/to/llm-env llm-utils/setup/.venv    # symlink so llm-utils scripts can find it
cd llm-utils/setup && uv sync                   # install llm-utils dependencies
```
</details>

## 3. Configure

Edit `configs/private_vars.yaml` to set your storage directory and HuggingFace details:

```yaml
storage_dir: "/path/to/your/storage"
huggingface_repo_namespace: "your-username"
huggingface_repo_name: "your-repo"
```

Also set a storage directory in `GameBoyWorlds/configs/private_vars.yaml`. This does **not** have to be the same path as above — GameBoyWorlds storage is independent of this repo's storage.

### ROMs

GameBoyWorlds ROMs must be downloaded and placed at the correct path. All ROMs follow the pattern:

```
<GameBoyWorlds_storage_dir>/rom_data/<game_series>/<game_name>/ROM_NAME.extension
```

See [GameBoyWorlds/README.md](GameBoyWorlds/README.md) for the full list of supported games, how to legally obtain ROMs, and the exact expected path for each.

> **If you skip this step you will not be able to run any games.**

## 4. Sync Data

Pull the project data from the HuggingFace Hub:

```bash
python sync_data.py pull               # dry run: reports what would be written
python sync_data.py pull --no_dry_run  # actually download
```

Use `--set <name>` to pull a single file set instead of all of them.

---

# Running Code

All workflows are driven by shell scripts in `scripts/`:

**RL** (`scripts/rl/`, `scripts/core_rl/`) — curiosity-driven RL training, trajectory collection and clustering.

**VLM** (`scripts/vlm/`) — infer/propose/attempt tasks, build info documents, guidance/practice, dataset creation, fine-tuning.

**Pipeline** (`scripts/pipeline/`) — end-to-end compositions of the above.

**Benchmark** (`scripts/benchmark/`) and **Debug** (`scripts/debug/`) — evaluation and read-only diagnostics.

A typical end-to-end run looks like:

```bash
# 1. Curiosity + zero-shot data collection, then info documents
bash scripts/pipeline/collect_and_info_all.sh --game pokemon_red --model_name gpt-4o ...

# 2. Benchmark an agent
bash scripts/benchmark/run_benchmark.sh --game pokemon_red --executor_vlm_model gpt-4o ...
```

For a full breakdown of every script and what it does, see [README_dev.md](README_dev.md).

---

# Developer Reference

See [README_dev.md](README_dev.md) for a table of all scripts and their core functionality.
