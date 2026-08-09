"""
Supervisors — agents that wrap executor runs and do something with the result.

The three concrete supervisors share :class:`~execution.supervisors.base.Supervisor`
and little else, so each lives in its own module:

- :mod:`.checker`      judge a finished trajectory
- :mod:`.info_hint`    turn retrieved knowledge into a hint
- :mod:`.info_plan`    drive a task one plan step at a time

:mod:`.prompts` holds every supervisor prompt (see that module for why).

The names re-exported below are the package's public surface — those imported by
callers outside ``execution/``, plus ``PLAN_SEPARATOR``, which lives in
:mod:`utils.parsing` and is forwarded through here so that
``from execution.supervisors import PLAN_SEPARATOR`` keeps working.
"""

from utils import PLAN_SEPARATOR

from execution.supervisors.base import Supervisor
from execution.supervisors.checker import (
    SimpleCheckerSupervisor,
    derive_critique_hint,
    summarise_trajectory_segments,
)
from execution.supervisors.info_hint import InfoHintSupervisor
from execution.supervisors.info_plan import InfoPlanSupervisor

__all__ = [
    "Supervisor",
    "SimpleCheckerSupervisor",
    "summarise_trajectory_segments",
    "derive_critique_hint",
    "InfoHintSupervisor",
    "InfoPlanSupervisor",
    "PLAN_SEPARATOR",
]
