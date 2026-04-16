from execution.executor import (
    Executor,
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


AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    "simple": SimpleExecutor,
    "history_aware": HistoryAwareExecutor,
    "sequence_planner": SequencePlannerExecutor,
    "subgoal_decomposer": SubgoalDecomposerExecutor,
    "screen_diff": ScreenDiffExecutor,
    "self_consistency": SelfConsistencyExecutor,
    "reflective": ReflectiveExecutor,
    "spatial_map": SpatialMapExecutor,
    "confidence_gated": ConfidenceGatedExecutor,
    "action_value_estimator": ActionValueEstimatorExecutor,
    "belief_state": BeliefStateExecutor,
    "adversarial_sampling": AdversarialSamplingExecutor,
}
""" Registry of available executors. Keys are short names, values are the corresponding classes. """
