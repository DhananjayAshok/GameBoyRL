"""
Executor agents, grouped by where each variant intervenes in the loop.

The base class defines a template method; the variants differ in which hooks they
override.  That is the grouping used by the modules here:

- :mod:`.base`          the loop and its hooks
- :mod:`.simple`        the reference implementation, plus a history block
- :mod:`.planning`      commit to a multi-step plan up front
- :mod:`.stateful`      maintain a running artifact and inject it into the prompt
- :mod:`.deliberative`  change how a single decision is made

Every name below is re-exported so that ``from execution.executors import X`` works
regardless of which module X happens to live in.  :mod:`execution.registry` is the
public surface for consumers outside this package.
"""

from execution.executors.base import (
    DEBUG_ON_INVALID,
    MAX_CONSECUTIVE_INVALID,
    Executor,
)
from execution.executors.simple import (
    HistoryAwareExecutor,
    SimpleExecutor,
)
from execution.executors.planning import (
    SequencePlannerExecutor,
    SubgoalDecomposerExecutor,
)
from execution.executors.stateful import (
    BeliefStateExecutor,
    ScreenDiffExecutor,
    SpatialMapExecutor,
)
from execution.executors.deliberative import (
    ActionValueEstimatorExecutor,
    AdversarialSamplingExecutor,
    ReflectiveExecutor,
)

__all__ = [
    "Executor",
    "MAX_CONSECUTIVE_INVALID",
    "DEBUG_ON_INVALID",
    "SimpleExecutor",
    "HistoryAwareExecutor",
    "SequencePlannerExecutor",
    "SubgoalDecomposerExecutor",
    "ScreenDiffExecutor",
    "SpatialMapExecutor",
    "BeliefStateExecutor",
    "ReflectiveExecutor",
    "ActionValueEstimatorExecutor",
    "AdversarialSamplingExecutor",
]
