"""
The benchmark arms: which executor plays, and which supervisor wraps it.
"""

from execution.executors import Executor
from execution.executors.executor import make_executor_class
from execution.executors.policies import (AVAILABLE_ACTION_POLICIES,
                                          AVAILABLE_HISTORY_POLICIES)
from execution.supervisors import (DummySupervisor, InfoSubgoalSupervisor,
                                   RevisingSupervisor, SubgoalSupervisor, Supervisor)

AVAILABLE_EXECUTORS: dict[str, type[Executor]] = {
    f"{action}_{history}": make_executor_class(action, history)
    for action in AVAILABLE_ACTION_POLICIES
    for history in AVAILABLE_HISTORY_POLICIES
}
""" Registry of available executors. Keys are ``<action>_<history>``, values the classes.
"""

AVAILABLE_SUPERVISORS: dict[str, type[Supervisor]] = {
    "dummy": DummySupervisor,
    "revision": RevisingSupervisor,
    "subgoal": SubgoalSupervisor,
    "info_subgoal": InfoSubgoalSupervisor,
}
""" Registry of available supervisors.
"""
