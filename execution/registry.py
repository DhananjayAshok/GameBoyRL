from execution.executors import (
    Executor,
    SimpleExecutor,
    HistoryAwareExecutor,
    SequencePlannerExecutor,
    SubgoalDecomposerExecutor,
    ScreenDiffExecutor,
    ReflectiveExecutor,
    SpatialMapExecutor,
    ActionValueEstimatorExecutor,
    BeliefStateExecutor,
    AdversarialSamplingExecutor,
)


AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    "simple": SimpleExecutor,
    "history": HistoryAwareExecutor,
    "sequence": SequencePlannerExecutor,
    "subgoal": SubgoalDecomposerExecutor, # Should be a Supervisor
    "screendiff": ScreenDiffExecutor,
    "reflective": ReflectiveExecutor, # Should be a Supervisor 
    "spatialmap": SpatialMapExecutor,
    "value": ActionValueEstimatorExecutor,
    "belief": BeliefStateExecutor,
    "adversarial": AdversarialSamplingExecutor,
}
""" Registry of available executors. Keys are short names, values are the corresponding classes. """
