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
    "history": HistoryAwareExecutor,
    "sequence": SequencePlannerExecutor,
    "subgoal": SubgoalDecomposerExecutor,
    "screendiff": ScreenDiffExecutor,
    "selfconsistency": SelfConsistencyExecutor,
    "reflective": ReflectiveExecutor,
    "spatialmap": SpatialMapExecutor,
    "confidence": ConfidenceGatedExecutor,
    "value": ActionValueEstimatorExecutor,
    "belief": BeliefStateExecutor,
    "adversarial": AdversarialSamplingExecutor,
}
""" Registry of available executors. Keys are short names, values are the corresponding classes. """
