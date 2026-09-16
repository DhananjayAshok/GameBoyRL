"""
The supervisor a strategist runs its tasks through.

Almost nothing, deliberately. The strategist's reasoning happens *above* the supervisor, so
the supervisor beneath it must add none of its own — otherwise an episode carries two
planners disagreeing about the same task, and no result can be attributed to either.

:class:`~execution.supervisors.dummy.DummySupervisor` is that supervisor already, with one
gap: it calls ``call_executor(self._task)`` bare, so the ``hint`` parameter that
:meth:`~execution.supervisors.base.Supervisor.call_executor` already accepts goes unused.
This subclass exists only to pass it through.
"""

from __future__ import annotations

from typing import Any, Optional

from execution.report import ExecutorReport
from execution.supervisors.dummy import DummySupervisor


class HintedSupervisor(DummySupervisor):
    """
    One executor leg, run with a hint the strategist wrote.

    :param hint: Advice for this attempt, or ``None`` for an unhinted run. Accepted as a
        constructor argument rather than reaching the executor through ``executor_kwargs``
        because :meth:`~execution.supervisors.base.Supervisor.call_executor` overwrites
        ``run_kwargs["hint"]`` with its own parameter on every leg — so a hint passed the
        other way is silently replaced by ``None`` and the strategist's advice never arrives.
    """

    def __init__(self, *args: Any, hint: Optional[str] = None, **kwargs: Any) -> None:
        # Set before super().__init__() so _run_config() can record it: the base class calls
        # that during construction to build the report's init_kwargs.
        self._hint = hint
        super().__init__(*args, **kwargs)

    def _run_config(self) -> dict:
        """Record the hint alongside the rest of the arm's knobs."""
        config = super()._run_config()
        config["hint"] = self._hint
        return config

    def _evaluate(self) -> Optional[dict]:
        """Run the single leg, with the strategist's hint attached.

        ``allow_self_termination=True``, unlike every benchmark arm. Those run in a *test*
        environment whose tracker supplies a ground-truth verdict on one fixed task, so
        letting the executor grade itself would replace a real signal with an opinion. A
        strategist run has no such verdict available: its tasks are written at runtime by
        the planner, no tracker exists for them, and it runs in the ``default`` environment
        precisely to avoid a termination metric belonging to some other benchmark task.

        Left at the default, ``agent_done`` is unreachable, so every task reads as failed
        however well it went, the strategist can never learn that a subtask worked, and each
        leg burns its whole step budget after already achieving its aim. A self-report is a
        weaker signal than a tracker, and it is the only one this layer can have.
        """
        self.call_executor(self._task, hint=self._hint, allow_self_termination=True)
        return None

    def process_executor_return(self, report: ExecutorReport) -> Any:
        """Hand the report back unchanged, as the control arm does."""
        return report
