from gameboy_worlds import get_test_environment, get_benchmark_tasks

from execution.executor import (
    SimpleExecutor,
    # Generation 1
    HistoryAwareExecutor,
    SequencePlannerExecutor,
    SubgoalDecomposerExecutor,
    ScreenDiffExecutor,
    SelfConsistencyExecutor,
    ReflectiveExecutor,
    # Generation 2
    SpatialMapExecutor,
    ConfidenceGatedExecutor,
    # Generation 3
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
    ("HistoryAwareExecutor",         HistoryAwareExecutor,         {"history_k": 5}),
    ("SequencePlannerExecutor",      SequencePlannerExecutor,      {}),
    ("SubgoalDecomposerExecutor",    SubgoalDecomposerExecutor,    {"steps_per_subgoal": 7}),
    ("ScreenDiffExecutor",           ScreenDiffExecutor,           {}),
    ("SelfConsistencyExecutor",      SelfConsistencyExecutor,      {"k": 3, "temperature": 0.7}),
    ("ReflectiveExecutor",           ReflectiveExecutor,           {"reflection_interval": 5}),
    # Generation 2
    ("SpatialMapExecutor",           SpatialMapExecutor,           {}),
    ("ConfidenceGatedExecutor",      ConfidenceGatedExecutor,      {"low_confidence_threshold": 2}),
    # Generation 3
    ("ActionValueEstimatorExecutor", ActionValueEstimatorExecutor, {}),
    ("BeliefStateExecutor",          BeliefStateExecutor,          {}),
    ("AdversarialSamplingExecutor",  AdversarialSamplingExecutor,  {}),
]


def test(executor_cls, name, row, max_env_steps, extra_kwargs=None):
    """Run a single executor on a benchmark task row and return its results."""
    if extra_kwargs is None:
        extra_kwargs = {}

    env = get_test_environment(
        row=row,
        controller_variant="low_level",
        save_video=True,
        max_steps=max_env_steps,
        session_name=f"executor_tests/{name}/",
    )

    executor = executor_cls(
        env=env,
        task=row["task"],
        max_steps=max_env_steps,
        max_tool_calls=MAX_TOOL_CALLS,
        **extra_kwargs,
    )
    report = executor.report

    n_env = sum(1 for s in report.steps if isinstance(s, EnvironmentStepRecord))
    n_tool = sum(1 for s in report.steps if isinstance(s, ToolCallRecord))
    n_invalid = len(report.invalid_steps)
    success = report.termination_reason == "terminated"

    print(f"  Outcome:            {report.outcome}  ({'SUCCESS' if success else 'fail'})")
    print(f"  Termination reason: {report.termination_reason}")
    print(f"  Env steps:          {n_env} / {max_env_steps}")
    print(f"  Tool calls:         {n_tool} / {MAX_TOOL_CALLS}")
    print(f"  Invalid steps:      {n_invalid}")

    print("  Step log:")
    for i, step in enumerate(report.steps):
        if isinstance(step, EnvironmentStepRecord):
            print(f"    [{i:>3}] ENV   {step.action_class.__name__}({step.kwargs})  success={step.action_success}")
        elif isinstance(step, ToolCallRecord):
            print(f"    [{i:>3}] TOOL  {step.executor_action_class.__name__}({step.kwargs})")

    if report.invalid_steps:
        print(f"  Invalid responses ({n_invalid}):")
        for j, bad in enumerate(report.invalid_steps[:3]):
            preview = bad[:100].replace("\n", " ")
            print(f"    [{j}] {preview!r}")

    return (name, report.outcome, report.termination_reason, n_env, n_invalid, None)


if __name__ == "__main__":
    row = get_benchmark_tasks(game="pokemon_red").iloc[0]
    max_env_steps = 20

    results = []
    for name, cls, extra_kwargs in EXECUTORS:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            result = test(cls, name, row, max_env_steps, extra_kwargs)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append((name, None, "error", 0, 0, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Executor':<32} {'Outcome':>7} {'Reason':<12} {'EnvSteps':>9} {'Invalid':>8}")
    print("-" * 72)
    for name, outcome, reason, n_env, n_invalid, err in results:
        if err:
            print(f"  {name:<30} {'ERR':>7} {str(reason):<12} {'-':>9} {'-':>8}  {err[:40]}")
        else:
            print(f"  {name:<30} {str(outcome):>7} {str(reason):<12} {n_env:>9} {n_invalid:>8}")
