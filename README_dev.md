# Developer Reference

A guide to the scripts in `scripts/` and what each one does. Use `--help` on the corresponding Python file for CLI options.

---

## RL Scripts (`scripts/rl/`)

| Script | What it does |
|--------|-------------|
| `create_traj.sh` | End-to-end trajectory creation for a single init_state: runs iterative RL to fill a replay buffer, then clusters observations into groups via `group_trajectories.sh`. Output feeds the VLM task-discovery pipeline. |
| `create_all_traj.sh` | Batch wrapper of `create_traj.sh` over every registered init_state for a game. Regenerates the state dictionary from GameBoyWorlds before iterating. |
| `iterative_training.sh` | Multi-round RL where each agent's curiosity buffer seeds the next round. Optionally retrains the world model between rounds. Called by `create_traj.sh`; can also be run directly for training-only workflows. |
| `sweep.sh` | Hyperparameter sweep over seeds, gammas, and algorithms. After the first pass it prunes to the best-k models by test reward, then re-runs only the winners to collect replay buffers. Called by `iterative_training.sh` when `--sweep true`. |
| `group_trajectories.sh` | Clusters a collected replay buffer by observation similarity (z-score gating) into grouped trajectory files used for downstream VLM task inference. Can be run standalone to re-cluster without retraining. |

---

## VLM Scripts (`scripts/vlm/`)

| Script | What it does |
|--------|-------------|
| `propose_zeroshot.sh` | Calls the VLM to propose candidate tasks for a single game/init_state with no prior trajectory data. Results are saved to the task store. |
| `propose_all_zeroshot.sh` | Batch version of `propose_zeroshot.sh`: runs across every registered init_state for a game. Regenerates the state dictionary before starting. |
| `infer_tasks.sh` | Uses the VLM to infer task labels from grouped trajectories produced by `group_trajectories.sh`. Populates the task store for a given run. |
| `attempt_tasks.sh` | Runs the VLM agent to attempt tasks loaded from a task file, scoring each attempt. Terminal stage of the zeroshot vertical: writes `success_trajectories.{json,pkl}`. |
| `build_info.sh` | Distils (task, trajectory) pairs into an info document beside its input stem — the context-engineering arm. |
| `train_vlm.sh` | Fine-tunes a VLM with LoRA via `train.py`. Saves the adapter to storage under the run name; optionally pushes to the HuggingFace Hub. Takes `--train_file`/`--validation_file` directly; nothing in this repo builds them. |

---

## Pipeline Scripts (`scripts/pipeline/`)

These compose the RL and VLM scripts above into higher-level end-to-end workflows.

| Script | What it does |
|--------|-------------|
| `curiosity_tasks.sh` | Single init_state: runs `create_traj.sh` then `infer_tasks.sh` to go from raw environment to labelled tasks. |
| `curiosity_all_tasks.sh` | Batch version of `curiosity_tasks.sh` over all registered init_states. |
| `propose_and_attempt.sh` | Single init_state: zero-shot task proposal followed immediately by an attempt run. |
| `propose_and_attempt_all.sh` | Batch version of `propose_and_attempt.sh`: proposes across all init_states then runs a single combined attempt. |
| `curiosity_and_zeroshot.sh` | Single init_state: runs both data-collection verticals. |
| `curiosity_and_zeroshot_all.sh` | Batch version of `curiosity_and_zeroshot.sh`. Called by `full.sh --mode both`. |

---

## Benchmark (`scripts/benchmark.sh`)

| Script | What it does |
|--------|-------------|
| `benchmark.sh` | Runs the full benchmark suite on a game with a specified executor and VLM model. Records video for each task attempt and logs all results to `results/benchmark/<game>/`. |
