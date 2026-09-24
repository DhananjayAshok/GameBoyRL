# Developer Reference

A guide to the scripts in `scripts/` and what each one does. Run a script with no arguments to see its usage; defaults live in `scripts/core/utils.sh`.

---

## RL Scripts (`scripts/rl/`)

| Script | What it does |
|--------|-------------|
| `create_traj.sh` | End-to-end trajectory creation for a single init_state: runs iterative RL to fill a replay buffer, then clusters observations into groups via `group_trajectories.sh`. Output feeds the VLM task-discovery pipeline. |
| `create_all_traj.sh` | Batch wrapper of `create_traj.sh` over every registered init_state for a game. Regenerates the state dictionary from GameBoyWorlds before iterating. |
| `iterative_training.sh` | Multi-round RL where each agent's curiosity buffer seeds the next round. Optionally retrains the world model between rounds. Called by `create_traj.sh`; can also be run directly for training-only workflows. |
| `sweep.sh` | Hyperparameter sweep over seeds, gammas, and algorithms. After the first pass it prunes to the best-k models by test reward, then re-runs only the winners to collect replay buffers. Called by `iterative_training.sh` when `--sweep true`. |
| `group_trajectories.sh` | Clusters a collected replay buffer by observation similarity (z-score gating) into grouped trajectory files used for downstream VLM task inference. Can be run standalone to re-cluster without retraining. |
| `collect_wm_buffers.sh` | Collects random-action and curiosity replay buffers for every train game/init_state in `--game`'s series, for the world-model baseline. No grouping. |

## Core RL Scripts (`scripts/core_rl/`)

| Script | What it does |
|--------|-------------|
| `train.sh` | Single RL training job with a given curiosity module (default: combinationbuffer PPO). Skips if the checkpoint exists unless `--overwrite_model true`. |
| `enjoy.sh` | Evaluates a trained model on a (possibly different) test game/init_state. |
| `default_rl.sh` | `train.sh` then `enjoy.sh`; supports `--train_only` / `--eval_only`. |
| `train_world_model.sh` | Trains a world model on a replay buffer (used by `curiosity_module=world_model` and the `world_model` executor). |
| `train_observation_embedder.sh` | Trains an observation encoder on replay buffer data for similarity-based curiosity. |

---

## VLM Scripts (`scripts/vlm/`)

| Script | What it does |
|--------|-------------|
| `propose_zeroshot.sh` | Calls the VLM to propose candidate tasks for the given init_states with no prior trajectory data. |
| `propose_all_zeroshot.sh` | Batch version of `propose_zeroshot.sh` across every registered init_state for a game. |
| `infer_tasks.sh` | Infers task labels from grouped trajectories produced by `group_trajectories.sh`. |
| `attempt_tasks.sh` | Runs the VLM agent on tasks from a task file under the two-stage judge. Writes `success_trajectories.{json,pkl}`. |
| `build_info.sh` | Distils (task, trajectory) pairs from a trajectory stem into `info_docs/` beside it. `--stage all` (default) is required before benchmarking. |
| `create_info_doc.sh` | Builds the info document for one `--source` (`curiosity` or `zeroshot`) of a game. |
| `create_all_info_docs.sh` | Builds info documents for every train game of `--game` and every source selected by `--mode`. |
| `infer_guidance.sh` | Annotates a trajectory with step-by-step guidance for practice. |
| `practice_tasks.sh` | Runs guided practice sessions from a guidance file, with a binary judge. |
| `clean_practice.sh` | Paraphrases tasks and VLM-filters every practice call in a practice dir (resumable). |
| `create_dataset.sh` | Builds `train_dataset.csv` / `validation_dataset.csv` from a cleaned practice dir. |
| `merge_practices.sh` | Merges the datasets of several practice dirs into one pair, tagged with a `source` column. |
| `train_vlm.sh` | LoRA fine-tunes a VLM via `llm-utils` (`train.py`); optionally pushes to the HuggingFace Hub. |

---

## Pipeline Scripts (`scripts/pipeline/`)

| Script | What it does |
|--------|-------------|
| `curiosity_all_tasks.sh` | `create_traj.sh` → `infer_tasks.sh` over all registered init_states. |
| `propose_and_attempt_all.sh` | Zero-shot proposal across all init_states, then a single combined attempt run. |
| `curiosity_and_zeroshot_all.sh` | Runs both data-collection verticals for a game. |
| `guidance_and_practice.sh` | `infer_guidance.sh` → `practice_tasks.sh` (→ `clean_practice.sh`) for one trajectory. |
| `collect_and_info_all.sh` | `curiosity_and_zeroshot_all.sh` for every train game of `--game`, then `create_all_info_docs.sh`. |

---

## Benchmark Scripts (`scripts/benchmark/`)

| Script | What it does |
|--------|-------------|
| `run_benchmark.sh` | Runs `run_benchmark.py` on a game with a given executor, supervisor and VLM. Records video and logs results to `results/benchmark/<game>/`. |
| `run_benchmark_info_retrieval.sh` | The `info_subgoal_retrieval` arm of `run_benchmark.sh`, with `--info_docs` resolved from disk via `--docs_*` flags. |

---

## Debug Scripts (`scripts/debug/`)

Read-only diagnostics over saved artifacts via `debug.py`; reports land under `<results_dir>/debug/<game>/`. `debug_all.sh` runs every stage in pipeline order (curiosity → infer → zeroshot → attempt → benchmark) and stops at the first failure.

---

## Core (`scripts/core/`)

| Script | What it does |
|--------|-------------|
| `utils.sh` | Loads config and venv; defines every script's default args and the arg-parsing helpers. |
| `all_train_states.sh` | Maps games to their train init_states. Regenerate with `python python_funcs.py task_dictionary`. |
| `serve_vllm.sh` / `stop_vllm.sh` | Start (blocking until healthy) / stop a vLLM server. |
| `llm-utils.sh` | Runs a command inside the `llm-utils` venv. |
| `clean_cache.sh` | Clears CleanRL artifacts and temporary emulator sessions. |
| `unsafe_clean_all.sh` | Wipes all storage; guarded by an exit you must remove first. |
