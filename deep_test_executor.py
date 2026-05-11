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

import click
from gameboy_worlds import AVAILABLE_GAMES, get_benchmark_tasks, get_test_environment

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
from execution.report import EnvironmentStepRecord, ToolCallRecord

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
    print(report)

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
