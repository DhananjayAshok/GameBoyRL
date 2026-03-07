# Code Review Notes

This report covers the project's own source files in `execution/`, `utils/`, and `main.py`. The `PokeWorlds` and `cleanrl` submodules are excluded.

---

## Bugs

### 1. `executor_actions_taken` type mismatch — [execution/report.py:49-110](execution/report.py#L49)

`executor_actions_taken` is annotated and described as a `Dict[int, List[...]]` but initialized as `[]` (a list). In `_add_executor_action`, the check `if self.steps_taken not in self.executor_actions_taken` tests list membership rather than dict key lookup, then immediately reassigns `self.executor_actions_taken = []`, wiping out any previously added actions. The result: executor action history is never actually accumulated.

The same broken check reappears in `get_execution_summary` (both `SimpleReport` and `EQAReport`) at `if i in self.executor_actions_taken:` — since the field is a list of tuples, integer `i` will never be found in it.

### 2. `action_success` overwritten with a string — [execution/pokemon/executors.py:71-74](execution/pokemon/executors.py#L71)

In `PokemonExecutor.get_action_message`, the `BattleMenuAction` branch does:
```python
action_success = "Tried to run, but the wild pokemon was too fast..."
```
but `action_success` is an integer parameter and the function returns `action_success_message`. These two lines should assign to `action_success_message`, not `action_success`.

### 3. `ExecutorVLM` called as a static class — [execution/executor_action.py:73-74](execution/executor_action.py#L73)

`object_detection` calls `ExecutorVLM.infer(...)` and `identify_matches` calls `ExecutorVLM.multi_infer(...)` directly on the class without instantiating it first. `infer` and `multi_infer` are instance methods. Both calls will fail at runtime with a `TypeError`.

### 4. `AnthropicVLMEngine.do_infer` passes wrong object to `_get_output` — [execution/vlm.py:420](execution/vlm.py#L420)

`do_infer` passes the full `response` message object to `OpenAIVLMEngine._get_output(response)`, but `_get_output` iterates over `response.output` (OpenAI's response format). Anthropic's response object has a different structure (`response.content`), so this will either error or silently return wrong data.

### 5. Dead code after `raise NotImplementedError` — [execution/retrieval.py:739](execution/retrieval.py#L739)

`DenseTextDatabase.gate_relevance` raises `NotImplementedError("Gating not implemented yet.")` on line 739 but contains ~18 lines of implementation logic below the raise that will never execute.

---

## Code Quality Issues

### 6. `torch` imported twice — [execution/vlm.py:14,16](execution/vlm.py#L14)

```python
import torch          # line 14
from transformers import AutoModelForImageTextToText, AutoProcessor
from openai import OpenAI
import anthropic
import torch          # line 17  <-- duplicate
```

### 7. Heavy imports unconditionally at module level — [execution/vlm.py:14-19](execution/vlm.py#L14) and [execution/retrieval.py:16-18](execution/retrieval.py#L16)

`torch`, `transformers`, `openai`, and `anthropic` are all imported at the top of `vlm.py`, and `torch`, `transformers` again in `retrieval.py`, with no install guard. Importing either file in a context without GPU packages will crash the whole import chain, even for users who only want to use the emulator.

### 8. Redundant `if/elif` pattern — [execution/vlm.py:485-495](execution/vlm.py#L485)

```python
if vlm_kind not in ["qwen3vl"]:
    log_error(...)
if vlm_kind in ["qwen3vl"]:   # always True here; the else is unreachable
    ...
else:
    log_error(...)
```
The second `if/else` is redundant — only one model kind is handled, and the structure implies there could be others.

### 9. `[VISUAL_CONTEXT]` replaced twice in a row — [execution/pokemon/executors.py:128-130](execution/pokemon/executors.py#L128)

```python
prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)
prompt = prompt.replace("[HIGH_LEVEL_ACTIONS]", allowed_actions_str)
prompt = prompt.replace("[VISUAL_CONTEXT]", self._visual_context)  # duplicate
```
The third line is identical to the first and has no effect.

### 10. `self.question = self.question` — [execution/supervisor.py:515](execution/supervisor.py#L515)

In `EQASupervisor._play`:
```python
self.question = self.question  # does nothing
```
Likely a leftover from a refactor.

### 11. Repetitive `[GAME]` prompt substitution in `__init__` — [execution/supervisor.py:209-227](execution/supervisor.py#L209) and [execution/supervisor.py:481-493](execution/supervisor.py#L481)

Both `SimpleSupervisor.__init__` and `EQASupervisor.__init__` individually call `.replace("[GAME]", game)` on five prompt fields. A simple loop over `self.__dict__` or a list of prompt attribute names would eliminate the repetition.

### 12. `HuggingFaceEmbeddingEngine` uses static methods instead of class methods — [execution/retrieval.py:103-141](execution/retrieval.py#L103)

`start`, `embed`, and `is_loaded` are `@staticmethod` but require an `engine_class` argument passed explicitly. These are exactly the use case for `@classmethod` (`cls` is the engine class). The current design forces every call site to redundantly pass the class as an argument.

### 13. `_gate_image` is a 2D array with no channel dimension — [execution/retrieval.py:632-634](execution/retrieval.py#L632)

```python
_gate_image = np.random.randint(low=0, high=255, size=(40, 40))
```
All VLM methods expect images shaped `(H, W, C)`. This placeholder will fail if the gating is ever enabled. Also, class-level mutable state initialized with random values is fragile.

### 14. `check_optional_installs` referenced but not defined locally — [utils/fundamental.py](utils/fundamental.py)

`utils/fundamental.py` exports `check_optional_installs` (imported in `utils/parameter_handling.py`) but the function isn't present in the file. It exists in `PokeWorlds/src/poke_worlds/utils/fundamental.py`. If `utils/` is meant to be independent of the submodule, this will fail.

### 15. Duplicated `parameter_handling` / `log_handling` between `utils/` and the submodule

`utils/parameter_handling.py` and `utils/log_handling.py` are substantially similar to their counterparts inside `PokeWorlds/src/poke_worlds/utils/`. They have different `compute_secondary_parameters` logic (`data_dir`/`model_dir`/`results_dir` vs `rom_data_dir`), which is fine, but the pattern means any fix to shared logic must be applied in two places.

---

## Minor Notes

- `execution/report.py` — `SimpleReport.get_step_frames` re-implements a method that already exists on the parent `ExecutionReport` class (same body, no override needed).
- `execution/retrieval.py` — several `return` statements at the end of `add_entry`, `modify_entry`, `remove_entry` are unnecessary (returning `None` implicitly is Pythonic).
- `execution/retrieval.py` — `DenseTextDatabase` constructor calls `log_warn("Completely untested DenseTextDatabase.", ...)` which will print on every instantiation in production. Should be removed or guarded by a debug flag when the class is considered stable.
- `main.py` — the main command group is wired up but has no commands attached; the example comment block showing how to add commands should either be removed or turned into actual commands.
