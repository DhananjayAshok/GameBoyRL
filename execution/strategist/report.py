"""
The record of one strategist run.

A strategist run is a sequence of supervised episodes, so its record interleaves the
strategist's own calls with the :class:`~execution.report.SupervisorReport` of each
episode. Kept out of ``execution.report`` because nothing below this layer knows it
exists.
"""

from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import numpy as np

from execution.artifact import episode_dirs
from execution.report import ExecutorReport, SupervisorReport
from utils import file_makedir, log_warn, sum_optional

#: How much of an episode is archived. The executor legs carry every frame the run saw and
#: dwarf everything else, so keeping them is opt-in.
REPORT_DETAILS = ("strategist", "supervisor", "executor")

EPISODE_REPORT_FILENAME = "report.pkl.gz"


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
    task: str
    guidance: Optional[str] = None
    supervisor_report: Optional[SupervisorReport] = None
    status: str = "incomplete"
    summary: str = ""


@dataclass
class StrategistReport:
    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    episodes: List[EpisodeRecord] = field(default_factory=list)
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
        lines = [f"{'#':>3}  {'status':<24} task"]
        for episode in self.episodes:
            lines.append(f"{episode.n:>3}  {episode.status:<24} {episode.task}")
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = [
            f"Strategist: {self.strategist_name} on {self.game}",
            f"Stopped: {self.stop_reason} after {len(self.episodes)} episodes",
            "",
            self.summary_table(),
        ]
        return "\n".join(lines)


@dataclass
class EpisodeArchive:
    """One episode of a run, as saved beside that episode's artifacts.

    Self-describing like the benchmark's ``report.pkl.gz``: a reader gets the game and the
    settings without having to find the provenance file that sits a directory up.
    """

    game: str
    strategist_name: str
    init_kwargs: Dict[str, Any]
    detail: str
    episode: EpisodeRecord
    strategist_calls: List[StrategistVLMCallRecord] = field(default_factory=list)


def trim_episode(record: EpisodeRecord, detail: str) -> EpisodeRecord:
    """``record`` with the parts ``detail`` excludes removed, as a copy.

    The live report keeps everything: the strategist is still reading it while the run
    continues, so trimming happens on the way to disk and never in place.
    """
    if detail == "executor" or record.supervisor_report is None:
        return record
    kept = ([e for e in record.supervisor_report.event_log if not isinstance(e, ExecutorReport)]
            if detail == "supervisor" else [])
    return replace(record, supervisor_report=replace(record.supervisor_report, event_log=kept))


def episode_report_path(run_dir: str, episode_number: int) -> str:
    return os.path.join(run_dir, f"episode_{episode_number}", EPISODE_REPORT_FILENAME)


def save_episode_report(run_dir: str, episode_number: int, archive: EpisodeArchive,
                        parameters: Optional[dict] = None) -> Optional[str]:
    """Write one episode's archive, gzipped.

    Failure is logged and swallowed, as in the benchmark's archiver: an episode already
    paid for in VLM calls must not be lost to an unwritable file.
    """
    path = episode_report_path(run_dir, episode_number)
    try:
        file_makedir(path)
        with gzip.open(path, "wb", compresslevel=6) as handle:
            pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path
    except Exception as error:  # noqa: BLE001
        log_warn(f"Could not write the episode report to {path}: {error}", parameters)
        return None


def load_episode_report(run_dir: str, episode_number: int) -> EpisodeArchive:
    with gzip.open(episode_report_path(run_dir, episode_number), "rb") as handle:
        return pickle.load(handle)


def saved_episode_reports(run_dir: str) -> List[int]:
    return sorted(n for n, path in episode_dirs(run_dir).items()
                  if os.path.exists(os.path.join(path, EPISODE_REPORT_FILENAME)))


def load_run(run_dir: str) -> List[EpisodeArchive]:
    """Every archived episode of a run, oldest first, across however many processes wrote them."""
    return [load_episode_report(run_dir, n) for n in saved_episode_reports(run_dir)]
