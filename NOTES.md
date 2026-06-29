# GameBoyRL Codebase Map

## Architecture Overview

Two-layer system:

```
GameBoyRL/                        ← this repo (agents, benchmarks, configs)
└── GameBoyWorlds/                ← git submodule (emulation, environments, parsers)
    └── src/gameboy_worlds/       ← installable Python package
```

`run_benchmark.py` is the top-level entry point. It loads tasks, creates environments, instantiates executors, and writes CSV results.

---

## Key Directories

### GameBoyRL-level

| Path | Purpose |
|---|---|
| `run_benchmark.py` | Main CLI; flags: `--game`, `--executor`, `--max_steps`, `--max_tool_calls`, `--task_text`, etc. |
| `execution/executor.py` | `Executor` ABC + all built-in executors (SimpleExecutor, HistoryAwareExecutor, BeliefStateExecutor, ...) |
| `execution/executor_action.py` | `ExecutorAction` ABC (passive tools that don't step the env) |
| `execution/registry.py` | `AVAILABLE_EXECUTORS` dict used by `run_benchmark.py` |
| `execution/pokemon/` | Pokemon-specific tools: `PokemonLocateAction`, `CheckInteractionAction` |
| `execution/pokemon_prism/` | **New** — `PokemonPrismBadgeExecutor` (see below) |
| `utils/lm_inference.py` | `InferenceModel` ABC, `OpenAIModel`, `AnthropicModel`, `OpenRouterModel`, `vLLMModel` |
| `utils/vlm.py` | `ExecutorVLM`, `VLM`, `ocr()`, `object_detection()` |
| `configs/params.yaml` | Default parameters (VLM model, token limits, etc.) |
| `configs/private_vars.yaml` | Machine-specific: `storage_dir`, API keys. **Set before running.** |

### GameBoyWorlds submodule

| Path | Purpose |
|---|---|
| `src/gameboy_worlds/emulation/pokemon/parsers.py` | `PokemonPrismStateParser`, `BasePokemonCrystalStateParser`, etc. `read_m(addr)` for memory reads |
| `src/gameboy_worlds/emulation/pokemon/trackers.py` | `StateTracker` subclasses. `PokemonPrismFirstBadgeTestTracker` added here |
| `src/gameboy_worlds/emulation/pokemon/test_metrics.py` | `TerminationMetric` subclasses. `PokemonPrismFirstBadgeTerminateMetric` added here |
| `src/gameboy_worlds/emulation/pokemon/registry.py` | Maps game names → parsers, trackers, emulators |
| `src/gameboy_worlds/interface/pokemon/registry.py` | Maps game names → environments (default, basic, train, test) |
| `src/gameboy_worlds/utils/__init__.py` | `get_benchmark_tasks(game)`, `get_test_environment(row, ...)` |
| `benchmark/tests/pokemon.csv` | Task definitions for pokemon_* games (DO NOT MODIFY — see constraints) |
| `configs/private_vars.yaml` | Storage path for the submodule. **Set `storage_dir` to ROM/state data directory.** |

---

## Executor Pattern

```
Executor.__init__(env, task, max_steps, max_tool_calls)
    └── calls _execute() immediately
         └── loop:
              ├── get_state() → current frame
              ├── _build_prompt(...)
              ├── _vlm_call("action", texts=prompt, images=[frame])
              ├── parse "Action: <action>" from response
              ├── if tool: _use_tool(ToolClass, **kwargs)  ← no env step
              └── else: _take_action(ActionClass, **kwargs) ← env step
```

- `available_tools: list = []` — class attribute listing `ExecutorAction` subclasses
- `_vlm_call(tag, ...)` — **only** way to call the VLM; logs to `report.vlm_call_log`
- `MAX_CONSECUTIVE_INVALID = 10` — exits if VLM keeps producing unparseable output

Built-in executors in `execution/executor.py` (registered in `execution/registry.py`):

| Key | Class | Notes |
|---|---|---|
| `simple` | `SimpleExecutor` | Base loop |
| `history` | `HistoryAwareExecutor` | Includes recent action log + [no change] detection |
| `reflective` | `ReflectiveExecutor` | Reflects every N steps, maintains plan summary |
| `belief` | `BeliefStateExecutor` | Structured belief state updated each step |
| `prism_badge` | `PokemonPrismBadgeExecutor` | **New** — see below |

---

## Pokemon Prism Setup

### What was added

1. **`GameBoyWorlds/src/gameboy_worlds/emulation/pokemon/test_metrics.py`**
   - `PokemonPrismFirstBadgeTerminateMetric` — reads memory `0xD57C & 0x01` (Naljo badge byte, bit 0 = Magma Badge)

2. **`GameBoyWorlds/src/gameboy_worlds/emulation/pokemon/trackers.py`**
   - `PokemonPrismFirstBadgeTestTracker` — wraps the metric above; truncates on battle exit without badge

3. **`GameBoyWorlds/src/gameboy_worlds/emulation/pokemon/registry.py`**
   - Registered `"first_badge_test": PokemonPrismFirstBadgeTestTracker` under `AVAILABLE_STATE_TRACKERS["pokemon_prism"]`

4. **`execution/pokemon_prism/executors.py`** — `PokemonPrismBadgeExecutor`
   - Inherits `HistoryAwareExecutor` (rolling action log, [no change] detection)
   - Adds reflection every 8 steps (VLM critiques recent actions, updates plan)
   - Tools: `PokemonLocateAction` + `CheckInteractionAction`
   - Injects Pokemon Prism game-world knowledge into every prompt

5. **`execution/registry.py`** — `"prism_badge"` key added

6. **`run_benchmark.py`** — `--task_text / --init_state / --state_tracker_class` flags for inline task override

### ROM Setup (GPU machine)

Place the ROM at:
```
<storage_dir>/rom_data/pokemon/pokemon_prism/PokemonPrism.gbc
```
`storage_dir` is set in `GameBoyWorlds/configs/private_vars.yaml`.

Available init states (already in the storage):
- `starter` — after choosing starter, beginning of game
- `city` — in a town
- `initial` — very start

### Running

```bash
# First gym badge task
python3.12 run_benchmark.py \
  --game pokemon_prism \
  --executor prism_badge \
  --executor_vlm_model claude-sonnet-4-6 \
  --executor_vlm_kind anthropic \
  --max_steps 500 \
  --max_tool_calls 50 \
  --task_text "Earn the Magma Badge from Gym Leader Tansy in Brimstone City" \
  --init_state starter \
  --state_tracker_class first_badge_test \
  --verbose
```

### Configs to set on GPU machine

**`configs/private_vars.yaml`** (GameBoyRL root):
```yaml
storage_dir: "/data/gameboy_rl/storage"  # adjust to actual path
env_dir: "setup/.venv/"
huggingface_repo_namespace: "your-namespace"
huggingface_repo_name: "your-repo"
```

**`GameBoyWorlds/configs/private_vars.yaml`**:
```yaml
storage_dir: "/data/gameboy_rl/storage/gameboy_worlds"  # where ROM data lives
```

---

## LM Inference

`utils/lm_inference.py` — use `InferenceModel.infer(texts, max_new_tokens, images, ...)`.

Set the executor VLM via `--executor_vlm_model` and `--executor_vlm_kind`:

| kind | class | env var needed |
|---|---|---|
| `anthropic` | `AnthropicModel` | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAIModel` | `OPENAI_API_KEY` |
| `openrouter` | `OpenRouterModel` | `OPENROUTER_API_KEY` |
| `vllm` | `vLLMModel` | local vLLM server running |

---

## Adding a New Game (no CSV modification)

Use the `--task_text` override in `run_benchmark.py` (added in this session):

```bash
python3.12 run_benchmark.py \
  --game <game_name> \
  --task_text "<task description>" \
  --init_state <state_name> \
  --state_tracker_class <tracker_key> \
  --executor <executor_key>
```

The game must be in `AVAILABLE_GAMES` (from `GameBoyWorlds` emulation registry) and have the tracker registered in `AVAILABLE_STATE_TRACKERS`.
