"""
Executor agents: one loop, composed from two independent policies.

- :mod:`.base`      the abstract contract — reporting, VLM-call logging, step ownership,
                    the completion check. Everything that is the same for every arm.
- :mod:`.executor`  the single concrete executor: the loop, the prompt, the environment.
- :mod:`.policies`  the two axes an arm varies on — how a decision is made
                    (``single`` / ``scored`` / ``sequence``) and what is remembered
                    (``none`` / ``actions`` / ``visual``).

This used to be a class per variant, grouped by which hook each one overrode. That grouping
described the *implementation* rather than the behaviour, and it could not express the
thing that actually matters: the two axes are independent, so a variant that changed how
an action was requested could not also change what was remembered. Nine arms now come from
three plus three plus one loop.

:mod:`execution.registry` is the public surface for consumers outside this package.
"""

from execution.executors.base import (
    DEBUG_ON_INVALID,
    MAX_CONSECUTIVE_INVALID,
    Executor,
)
from execution.executors.executor import PolicyExecutor, make_executor_class
from execution.executors.policies import (
    AVAILABLE_ACTION_POLICIES,
    AVAILABLE_HISTORY_POLICIES,
    ActionPolicy,
    Decision,
    HistoryPolicy,
)

__all__ = [
    "Executor",
    "PolicyExecutor",
    "make_executor_class",
    "MAX_CONSECUTIVE_INVALID",
    "DEBUG_ON_INVALID",
    "ActionPolicy",
    "HistoryPolicy",
    "Decision",
    "AVAILABLE_ACTION_POLICIES",
    "AVAILABLE_HISTORY_POLICIES",
]
