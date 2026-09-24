"""
The two axes an executor varies on, as strategy objects rather than subclasses.

- :mod:`.action`   how one decision produces action(s): ``single``, ``scored``, ``sequence``
- :mod:`.history`  what is remembered between decisions: ``none``, ``actions``, ``visual``

Nine arms come from three plus three plus one loop. The axes are independent: any action
policy composes with any history policy.
"""

from execution.executors.policies.action import (
    AVAILABLE_ACTION_POLICIES,
    ActionPolicy,
    Decision,
    ScoredActionPolicy,
    SequenceActionPolicy,
    SingleActionPolicy,
)
from execution.executors.policies.history import (
    AVAILABLE_HISTORY_POLICIES,
    ActionHistoryPolicy,
    HistoryPolicy,
    NoHistoryPolicy,
    VisualHistoryPolicy,
)

__all__ = [
    "ActionPolicy",
    "Decision",
    "SingleActionPolicy",
    "ScoredActionPolicy",
    "SequenceActionPolicy",
    "AVAILABLE_ACTION_POLICIES",
    "HistoryPolicy",
    "NoHistoryPolicy",
    "ActionHistoryPolicy",
    "VisualHistoryPolicy",
    "AVAILABLE_HISTORY_POLICIES",
]
