"""
Deep executor test — prints the full per-step trajectory for each executor.

For each executor in EXECUTORS, shows:
  - The raw VLM response (reasoning + chosen action) for every iteration
  - The resulting step record (ENV action or TOOL call)
  - Invalid iterations (parse failures / unrecognised actions)

Alignment note
--------------
``report.vlm_responses`` has one entry per *decision loop iteration*.
``report.steps`` has one entry per *successful* iteration (tool call or env
step).  Iterations that failed to parse produce no step record — they are
stored in ``report.invalid_steps`` instead.  ``SequencePlannerExecutor``
additionally executes several env steps per VLM call, so its step count
typically exceeds its vlm_response count.

We handle this by consuming steps from an iterator while using the raw
response text to detect parse-failure iterations (those responses are stored
verbatim in ``report.invalid_steps``).  Unrecognised-action failures are
detected as iterations where no step could be consumed and no parse failure
was found.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator, Union

import click
from gameboy_worlds import AVAILABLE_GAMES, get_benchmark_tasks, get_test_environment
import os
import matplotlib.pyplot as plt
import shutil

from execution.executor import (
    SimpleExecutor,
    HistoryAwareExecutor,
    SequencePlannerExecutor,
    SubgoalDecomposerExecutor,
    ScreenDiffExecutor,
    SelfConsistencyExecutor,
    ReflectiveExecutor,
    SpatialMapExecutor,
    ConfidenceGatedExecutor,
    ActionValueEstimatorExecutor,
    BeliefStateExecutor,
    AdversarialSamplingExecutor,
)
from execution.report import EnvironmentStepRecord, ExecutorReport, ToolCallRecord

MAX_TOOL_CALLS = 0

EXECUTORS = [
    # Baseline
    ("SimpleExecutor",               SimpleExecutor,               {}),
    # Generation 1
    #("HistoryAwareExecutor",         HistoryAwareExecutor,         {"history_k": 5}),
    #("SequencePlannerExecutor",      SequencePlannerExecutor,      {}),
    #("SubgoalDecomposerExecutor",    SubgoalDecomposerExecutor,    {"steps_per_subgoal": 7}),
    #("ScreenDiffExecutor",           ScreenDiffExecutor,           {}),
    #("SelfConsistencyExecutor",      SelfConsistencyExecutor,      {"k": 3, "temperature": 0.7}),
    ("ReflectiveExecutor",           ReflectiveExecutor,           {"reflection_interval": 5}),
    # Generation 2
    #("SpatialMapExecutor",           SpatialMapExecutor,           {}),
    #("ConfidenceGatedExecutor",      ConfidenceGatedExecutor,      {"low_confidence_threshold": 2}),
    # Generation 3
    #("ActionValueEstimatorExecutor", ActionValueEstimatorExecutor, {}),
    #("BeliefStateExecutor",          BeliefStateExecutor,          {}),
    #("AdversarialSamplingExecutor",  AdversarialSamplingExecutor,  {}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _step_summary(step: Union[EnvironmentStepRecord, ToolCallRecord]) -> str:
    if isinstance(step, EnvironmentStepRecord):
        return f"ENV   {step.action_class.__name__}({step.kwargs})"
    return f"TOOL  {step.executor_action_class.__name__}({step.kwargs})  result={step.result}"


# Tags that drive the action-selection loop and therefore correspond 1:1
# with a step record (or an invalid-step entry when parsing fails).
_ACTION_TAGS = {"action", "score", "decide"}


def print_trajectory(report: ExecutorReport, name=None) -> None:
    """Print the full interleaved VLM-call / step trajectory.

    Every entry in ``report.vlm_call_log`` is printed with its tag.
    Calls tagged as action-selection (``"action"``, ``"score"``,
    ``"decide"``) are paired with the next step record where possible;
    auxiliary calls (``"reflection"``, ``"map_update"``, etc.) are shown
    inline without consuming a step.
    """
    if name is None:
        img_save_path = "tmp/vis/"
    else:
        img_save_path = f"results/{name}"
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
    os.makedirs(img_save_path)
    print(f"Saving images to: {img_save_path}")
    
    if not report.vlm_call_log:
        print("  (no VLM calls recorded)")
        return

    # Parse failures store the raw response verbatim in invalid_steps.
    parse_fail_counter: Counter[str] = Counter(
        s for s in getattr(report, "invalid_steps", [])
        if not s.startswith("Unrecognised")
    )

    steps_iter: Iterator = iter(report.steps)
    call_idx = 0

    for entry in report.vlm_call_log:
        call_idx += 1
        tag_label = f"[{entry.tag.upper()}]"
        print(f"\n  ┌─ {tag_label} (call {call_idx})" + "─" * max(0, 48 - len(tag_label)))
        print("")
        print("  | Prompt:")
        print(_indent(entry.prompt , "  │   "))
        print("  │ VLM output:")
        print(_indent(entry.response, "  │   "))
        for i, image in enumerate(entry.images):
            img_path = os.path.join(img_save_path, f"{call_idx}_{i}.png")
            plt.imshow(image)
            plt.savefig(img_path)
            plt.clf()

        if entry.tag in _ACTION_TAGS:
            if parse_fail_counter.get(entry.response, 0) > 0:
                parse_fail_counter[entry.response] -= 1
                print("  │ → INVALID  (parse failure)")
            else:
                step = next(steps_iter, None)
                if step is None:
                    print("  │ → INVALID  (unrecognised action or end of steps)")
                else:
                    print(f"  │ → {_step_summary(step)}")

        print("  └" + "─" * 57)

    # Any leftover steps (SequencePlannerExecutor runs N steps per VLM call)
    remaining = list(steps_iter)
    if remaining:
        print(f"\n  (+ {len(remaining)} env steps from planned sequences:)")
        for j, step in enumerate(remaining):
            print(f"    [{j}] {_step_summary(step)}")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def deep_test(executor_cls, name, row, max_env_steps, extra_kwargs=None):
    if extra_kwargs is None:
        extra_kwargs = {}

    env = get_test_environment(
        row=row,
        controller_variant="low_level",
        save_video=False,
        max_steps=max_env_steps,
        session_name=f"deep_executor_tests/{name}/",
    )

    executor = executor_cls(
        env=env,
        task=row["task"],
        max_steps=max_env_steps,
        max_tool_calls=MAX_TOOL_CALLS,
        **extra_kwargs,
    )
    report = executor.report

    vlm_count = len(report.vlm_call_log)
    step_count = len(report.steps)
    invalid_count = len(report.invalid_steps)

    print(f"  Outcome:            {report.outcome}  "
          f"({'SUCCESS' if report.termination_reason == 'terminated' else 'fail'})")
    print(f"  Termination reason: {report.termination_reason}")
    print(f"  VLM calls:          {vlm_count}")
    print(f"  Step records:       {step_count}  "
          f"(env={sum(1 for s in report.steps if isinstance(s, EnvironmentStepRecord))}, "
          f"tool={sum(1 for s in report.steps if isinstance(s, ToolCallRecord))})")
    print(f"  Invalid steps:      {invalid_count}")

    print("\n  Full trajectory:")
    print_trajectory(report)

    return (name, report.outcome, report.termination_reason, step_count, invalid_count, None)


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--random_sample", default=None, type=int, help="Number of tasks to sample; omit to run all.")
@click.option("--max_env_steps", default=20, type=int)
def main(game, random_sample, max_env_steps):
    benchmark_tasks = get_benchmark_tasks(game=game)
    if random_sample is not None:
        benchmark_tasks = benchmark_tasks.sample(n=random_sample, random_state=42).reset_index(drop=True)

    for task_idx, row in benchmark_tasks.iterrows():
        print(f"\n{'#'*60}")
        print(f"TASK {task_idx}  ({game})")
        for col in row.index:
            print(f"  {col}: {row[col]}")
        print(f"{'#'*60}")

        results = []
        for name, cls, extra_kwargs in EXECUTORS:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            try:
                result = deep_test(cls, name, row, max_env_steps, extra_kwargs)
                results.append(result)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                results.append((name, None, "error", 0, 0, str(e)))

        print(f"\n{'='*60}")
        print(f"SUMMARY — task {task_idx}")
        print(f"{'='*60}")
        print(f"{'Executor':<32} {'Outcome':>7} {'Reason':<12} {'Steps':>6} {'Invalid':>8}")
        print("-" * 68)
        for name, outcome, reason, n_steps, n_invalid, err in results:
            if err:
                print(f"  {name:<30} {'ERR':>7} {str(reason):<12} {'-':>6} {'-':>8}  {err[:40]}")
            else:
                print(f"  {name:<30} {str(outcome):>7} {str(reason):<12} {n_steps:>6} {n_invalid:>8}")


if __name__ == "__main__":
    main()
