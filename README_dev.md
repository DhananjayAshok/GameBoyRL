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
| `infer_guidance.sh` | Uses the VLM to annotate a trajectory with step-by-step guidance. Output file is consumed by `practice_tasks.sh`. |
| `practice_tasks.sh` | Runs the VLM agent through guided practice sessions using guidance from `infer_guidance.sh`. Supports a score mode that produces scored trajectories for VLM fine-tuning data collection. |
| `attempt_tasks.sh` | Runs the VLM agent to attempt tasks loaded from a task file, scoring each attempt. Results are logged for evaluation and potential fine-tuning data collection. |
| `train_vlm.sh` | Fine-tunes a VLM with LoRA via `train.py`. Saves the adapter to storage under the run name; optionally pushes to the HuggingFace Hub. |

---

## Pipeline Scripts (`scripts/pipeline/`)

These compose the RL and VLM scripts above into higher-level end-to-end workflows.

| Script | What it does |
|--------|-------------|
| `curiosity_tasks.sh` | Single init_state: runs `create_traj.sh` then `infer_tasks.sh` to go from raw environment to labelled tasks. |
| `curiosity_all_tasks.sh` | Batch version of `curiosity_tasks.sh` over all registered init_states. |
| `propose_and_attempt.sh` | Single init_state: zero-shot task proposal followed immediately by an attempt run. |
| `propose_and_attempt_all.sh` | Batch version of `propose_and_attempt.sh`: proposes across all init_states then runs a single combined attempt. |
| `propose_after_curiosity.sh` | Single init_state: zero-shot proposal augmented with a curiosity prior, then attempt. Requires `infer_tasks.sh` to have been run first. |
| `propose_after_curiosity_all.sh` | Batch version of `propose_after_curiosity.sh`. |
| `propose_zeroshot_with_curiosity.sh` | Two-stage proposal for a single init_state: first proposes with no prior, then again with the curiosity prior. Both stages are followed by attempt runs. |
| `propose_zeroshot_with_curiosity_all.sh` | Batch version of `propose_zeroshot_with_curiosity.sh`. |
| `guidance_and_practice.sh` | Annotates a trajectory with step-by-step guidance via `infer_guidance.sh`, then immediately runs `practice_tasks.sh` on the result. |

---

## Benchmark (`scripts/benchmark.sh`)

| Script | What it does |
|--------|-------------|
| `benchmark.sh` | Runs the full benchmark suite on a game with a specified executor and VLM model. Records video for each task attempt and logs all results to `results/benchmark/<game>/`. |
