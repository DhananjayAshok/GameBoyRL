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
)


AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    "simple": SimpleExecutor,
    "history": HistoryAwareExecutor, # should instead be a togglable feature for all executors. 
    "sequence": SequencePlannerExecutor,
    "subgoal": SubgoalDecomposerExecutor, # Should be a Supervisor
    "screendiff": ScreenDiffExecutor, # Should instead be a togglable feature for the history executor. Essentially after a step, if screendiff mode is on, the executor will be given both the previous and current frame to reason about what changed and store that along with its past actions record. Then when rendering its action history, it lists both the last actions taken AND the change of those steps. 
    "reflective": ReflectiveExecutor, # Should be a Supervisor 
    "spatialmap": SpatialMapExecutor, # Explain this to me. 
    "value": ActionValueEstimatorExecutor,
}
""" Registry of available executors. Keys are short names, values are the corresponding classes. """
