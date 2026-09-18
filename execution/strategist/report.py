"""
The record of one strategist run.

A strategist run is a sequence of supervised episodes, so its record interleaves the
strategist's own calls with the :class:`~execution.report.SupervisorReport` of each
episode. Kept out of ``execution.report`` because nothing below this layer knows it
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from execution.report import SupervisorReport
from utils import sum_optional


@dataclass
class StrategistVLMCallRecord:
    stage: str
    images: List[np.ndarray]
    prompt: str
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class EpisodeRecord:
    n: int
    goal: str
    task: str
    guidance: Optional[str] = None
    location_before: Optional[str] = None
    location_after: Optional[str] = None
    supervisor_report: Optional[SupervisorReport] = None
    status: str = "incomplete"
    summary: str = ""
    goal_complete: bool = False
    facts_added: List[str] = field(default_factory=list)
    subgoals_added: List[str] = field(default_factory=list)


@dataclass
class StrategistReport:
    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_ladder: List[str] = field(default_factory=list)
    episodes: List[EpisodeRecord] = field(default_factory=list)
    knowledge_snapshots: List[List[Dict[str, Any]]] = field(default_factory=list)
    final_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    event_log: List[Union[StrategistVLMCallRecord, SupervisorReport]] = field(default_factory=list)
    stop_reason: str = "incomplete"

    @property
    def strategist_calls(self) -> List[StrategistVLMCallRecord]:
        return [e for e in self.event_log if isinstance(e, StrategistVLMCallRecord)]

    @property
    def supervisor_reports(self) -> List[SupervisorReport]:
        return [e for e in self.event_log if isinstance(e, SupervisorReport)]

    @property
    def strategist_input_tokens(self) -> Optional[int]:
        return sum_optional([call.input_tokens for call in self.strategist_calls])

    @property
    def strategist_output_tokens(self) -> Optional[int]:
        return sum_optional([call.output_tokens for call in self.strategist_calls])

    @property
    def supervisor_input_tokens(self) -> Optional[int]:
        return sum_optional([report.total_input_tokens for report in self.supervisor_reports])

    @property
    def supervisor_output_tokens(self) -> Optional[int]:
        return sum_optional([report.total_output_tokens for report in self.supervisor_reports])

    @property
    def total_input_tokens(self) -> Optional[int]:
        return sum_optional([self.strategist_input_tokens, self.supervisor_input_tokens])

    @property
    def total_output_tokens(self) -> Optional[int]:
        return sum_optional([self.strategist_output_tokens, self.supervisor_output_tokens])

    @property
    def n_invalid(self) -> int:
        return sum(report.n_invalid for report in self.supervisor_reports)

    def summary_table(self) -> str:
        lines = [f"{'#':>3}  {'status':<24} {'location':<24} {'facts':>5}  goal / task"]
        for episode in self.episodes:
            lines.append(
                f"{episode.n:>3}  {episode.status:<24} {(episode.location_after or '?'):<24} "
                f"{len(episode.facts_added):>5}  {episode.goal}"
            )
            lines.append(f"{'':>3}  {'':<24} {'':<24} {'':>5}  -> {episode.task}")
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = [
            f"Strategist: {self.strategist_name} on {self.game}",
            f"Stopped: {self.stop_reason} after {len(self.episodes)} episodes",
            f"Locations seen: {', '.join(self.locations) or '(none)'}",
            f"Knowledge: {len(self.final_knowledge)} facts",
            "",
            self.summary_table(),
        ]
        return "\n".join(lines)
