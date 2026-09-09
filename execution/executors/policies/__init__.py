"""
The two axes an executor varies on, as strategy objects rather than subclasses.

- :mod:`.action`   how one decision produces action(s): ``single``, ``scored``, ``sequence``
- :mod:`.history`  what is remembered between decisions: ``none``, ``actions``,
  ``visual``, ``escape``

Nine arms come from three plus three plus one loop, instead of nine classes. The axes are
independent — any action policy composes with any history policy — which is exactly the
property a class hierarchy cannot express without multiplying out.
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
    EscapeHistoryPolicy,
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
    "EscapeHistoryPolicy",
    "VisualHistoryPolicy",
    "AVAILABLE_HISTORY_POLICIES",
]
