"""
The supervisor that does nothing.

:class:`DummySupervisor` runs the executor on the task it was given and adds no reasoning
of its own. It exists so **nothing is ever benchmarked with a bare executor**: the baseline
arm goes through a supervisor like every other arm, so all three produce a
:class:`~execution.report.SupervisorReport` and every reader — the archive, the debug
report, the comparators — has one shape to handle instead of two.

It is the control in the experimental sense: any arm's advantage over the baseline has to
come from the supervisor's reasoning, and this one has none, so the difference is the
measurement. Adding anything here — a judge, a hint, a retry — would stop that being true.
"""

from __future__ import annotations

from typing import Any, Optional

from execution.report import ExecutorReport
from execution.supervisors.base import Supervisor


class DummySupervisor(Supervisor):
    """Pass the task straight to one executor run and surface nothing extra.

    Success is decided by the executor's own ``termination_reason`` — the environment's
    verdict — not by any judgement made here. That is deliberate: it is the only
    ground-truth success signal in the pipeline, and the baseline must not substitute a VLM's
    opinion for it.
    """

    def _evaluate(self) -> Optional[dict]:
        """Run the executor once. No extras: the report is the whole result."""
        self.call_executor(self._task)
        return None

    def process_executor_return(self, report: ExecutorReport) -> Any:
        """Hand the report back unchanged — ``call_executor`` has already filed it."""
        return report
