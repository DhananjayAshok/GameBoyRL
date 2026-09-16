"""
A strategist whose goal is a benchmark task, graded by that task's tracker.

The measurement variant. The default :class:`~execution.strategists.strategist.Strategist`
pursues an open-ended goal in the ``default`` environment, where no tracker exists for a
task the planner invented at runtime — so success is the executor's self-report, and two
arms that chose different sub-tasks cannot be compared on it: the arm that set itself
easier work banks more completions without getting closer to anything.

Here the goal is a *benchmark* task, run in the ``test`` environment, so the task's own
tracker decides. That verdict is identical for every arm and is not a model's opinion,
which is what makes a control-versus-ablation number mean something. It also puts the
strategist on the same scale as the bare executor arms in ``run_benchmark.py``, which are
graded by exactly this signal.

The trade is that the goal must be one of the benchmark's tasks. That is a narrower
question than "obtain a Pokemon", and a measurable one.
"""

from __future__ import annotations

from typing import Optional

from execution.report import SupervisorReport
from execution.strategists.strategist import Strategist


class TrackerGoalStrategist(Strategist):
    """
    Pursue a benchmark task, letting its tracker decide when it is done.

    Everything else — planning, the notebook, reflection, compression — is inherited
    unchanged. Only how the goal is judged differs, which is the point: any difference in
    results between this and the base class is attributable to the grading, not the agent.
    """

    #: The executor's ``termination_reason`` when the ENVIRONMENT ended the episode. Set by
    #: the test environment's tracker, not by the model. ``agent_done`` is the self-report
    #: and is deliberately not accepted here.
    ENV_TERMINATED = "terminated"

    @staticmethod
    def _env_terminated(report: Optional[SupervisorReport]) -> bool:
        """Whether the tracker fired on any leg of this task.

        Any leg rather than the last: a task may reach the tracker's condition part-way
        through and keep pressing buttons afterwards, and the episode still succeeded.
        """
        if report is None:
            return False
        return any(leg.termination_reason == TrackerGoalStrategist.ENV_TERMINATED
                   for leg in report.executor_reports)

    def _succeeded(self, report: Optional[SupervisorReport]) -> bool:  # type: ignore[override]
        """Ground truth from the tracker, with the self-report as a fallback.

        The tracker's verdict is what counts. ``agent_done`` is still accepted so the
        notebook can record that a *sub*-task the planner invented was finished — that is
        useful context for the next plan — but only the tracker can end the run.
        """
        if self._env_terminated(report):
            return True
        return super()._succeeded(report)

    def _goal_looks_reached(self) -> bool:
        """
        Whether the tracker has fired at any point in this episode.

        Replaces the ledger heuristic entirely. The ledger is written by the same model that
        wants the goal met and was observed claiming `has_pokemon=true` on a screen with no
        Pokemon; with a tracker available there is no reason to consult an opinion.
        """
        return any(self._env_terminated(task.report) for task in self.report.tasks)

    def _confirm_goal(self) -> bool:
        """
        No confirmation step: the tracker already IS the confirmation.

        The base class spends a probe task opening the party menu and then judges the frame,
        because on an open-ended goal there is nothing else to ask. Doing that here would
        replace a ground-truth verdict with a model's reading of a screenshot, and would
        also charge the run a probe it does not need.
        """
        return True
