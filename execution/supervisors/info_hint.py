"""
Turning past attempts into a hint for the next one.

:class:`InfoHintSupervisor` judges whether retrieved knowledge is relevant to the
task at hand and writes a hint from what survives.  :class:`InfoPlanSupervisor`
subclasses it, so anything added here is inherited by the plan arm.

Both route every VLM call through :meth:`~execution.supervisors.base.Supervisor._vlm_call`,
so the supervisor's own reasoning lands on its
:class:`~execution.report.SupervisorReport` beside the executor runs it drove.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import ExecutorReport
from execution.supervisors.base import Supervisor
from execution.supervisors.prompts import RELEVANCE_PROMPT, WRITE_HINT_PROMPT
from utils import log_warn, parse_key_value, parse_yes_no, VLM


class InfoHintSupervisor(Supervisor):
    """
    Reads a prebuilt info document, writes one hint for the task at hand, then runs the
    executor with it.

    This is the test-time half of the context-engineering vertical. Two modes decide *which*
    knowledge reaches the hint writer; the synthesis call is identical in both, so any
    difference in results is attributable purely to selection:

    ``retrieval``
        Iterate over every entry of every loaded document and ask, one call each, whether it
        fits this task and this screen. Entries are judged on ``Description`` + ``Examples`` +
        their representative frame — never on their ``Insights``, so relevance is decided on
        whether the context fits rather than on whether the advice sounds appealing.

    ``init_state``
        Skip the document entirely and read the stage-A ``insights.jsonl``, keeping rows whose
        ``init_state`` matches the episode's. The init state is *given* by the benchmark row
        rather than inferred, which makes this the retrieval-free upper bound. Rows are still
        task-filtered by the same relevance call — one init state can carry many unrelated
        tasks, so the init state narrows the candidate pool and the task filter picks from it.

    No executor is modified or subclassed: the hint travels through ``Executor.__init__``'s
    existing ``hint`` argument, which ``Supervisor.call_executor`` already forwards. The
    executor's hint block presents a hint as reliable, so calibration lives in the hint text —
    :attr:`WRITE_HINT_PROMPT` requires hints to be conditional and self-limiting, and to emit
    ``NO HINT`` rather than guess.

    :param task: The benchmark task string.
    :param executor_class: :class:`~execution.executors.Executor` subclass to run.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Env-step budget forwarded to the executor.
    :param max_tool_calls: Tool-call budget forwarded to the executor.
    :param documents: Parsed :class:`~execution.info_doc.InfoDocument` objects (retrieval mode).
    :param insight_rows: Stage-A rows from ``insights.jsonl`` (init_state mode).
    :param mode: ``"retrieval"`` or ``"init_state"``.
    :param init_state: The episode's init state; required for ``init_state`` mode.
    :param hint_vlm_model: Model name for the relevance and hint-writing calls.
    :param hint_vlm_kind: VLM kind for those calls.
    :param hint_max_new_tokens: Token budget per hint-pipeline VLM call.
    :param max_concurrency: Parallel relevance calls (they are independent).
    :param parameters: Optional parameter overrides.
    :param executor_kwargs: Extra keyword arguments forwarded to the executor constructor.
    """

    def __init__(
        self,
        task: str,
        executor_class: Type[Executor],
        env: Environment,
        game: str,
        max_steps: int,
        max_tool_calls: int,
        documents: Optional[List[Any]] = None,
        insight_rows: Optional[List[dict]] = None,
        mode: str = "retrieval",
        init_state: Optional[str] = None,
        hint_vlm_model: str = None,
        hint_vlm_kind: str = None,
        hint_max_new_tokens: int = 1000,
        max_concurrency: int = 8,
        parameters: Optional[dict] = None,
        **executor_kwargs: Any,
    ) -> None:
        self._documents = documents or []
        self._insight_rows = insight_rows or []
        self._mode = mode
        self._init_state = init_state
        self._max_concurrency = max_concurrency
        self._hint_max_new_tokens = hint_max_new_tokens
        super().__init__(task, executor_class, env, game, max_steps, max_tool_calls,
                         parameters, **executor_kwargs)
        self._hint_vlm = VLM(hint_vlm_model, hint_vlm_kind)
        # Populated by write_hint() so the caller (and debug.py info_hint) can inspect why a
        # hint came out the way it did without re-running the pipeline. ``selected_ids`` is
        # the durable part: the benchmark CSV stores it per episode, so a hint can be traced
        # back to the exact entries it was synthesised from long after the run.
        self.selection_log: List[dict] = []
        self.selected_ids: List[str] = []
        self.hint: Optional[str] = None

    @staticmethod
    def entry_id(entry) -> str:
        """
        Stable identifier for one selected entry, for the benchmark CSV and debug reports.

        ``source`` is the provenance label the loader attached: the document's vertical in
        retrieval mode (``zeroshot``), or vertical/group_idx in init_state mode
        (``zeroshot/12_0``) — which pins the exact stage-A row in insights.jsonl. The
        category disambiguates entries within a source.
        """
        return f"{entry.source}#{entry.category}" if entry.source else entry.category

    # -- Hint pipeline ---------------------------------------------------------

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    @staticmethod
    def _entry_frame(entry):
        from PIL import Image

        # Stored frame paths are relative to their document's frames_root, so they are not
        # openable on their own. Whoever produced the entry — load_document, or
        # _init_state_candidates for rows read straight off insights.jsonl — has already
        # resolved it.
        path = entry.resolved_frame
        if not path or not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")

    def _judge_relevance(self, entry, kind: str, screen) -> tuple:
        """One yes/no call for a single entry. Returns (is_relevant, reason)."""
        prompt = (
            RELEVANCE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[KIND]", kind)
            .replace("[ENTRY]", entry.evidence_block())
        )
        images = [screen]
        entry_frame = self._entry_frame(entry)
        if entry_frame is not None:
            images.append(entry_frame)

        output = self._vlm_call("filter", self._hint_vlm, texts=prompt, images=images,
                                max_new_tokens=self._hint_max_new_tokens)
        verdict = parse_yes_no(output, "Relevant")
        reason = (parse_key_value(output, "Reasoning") or "").strip()
        return verdict is True, reason

    def _select_entries(self, candidates, screen) -> List[Any]:
        """Run the relevance pass over (entry, kind) candidates, in parallel."""
        if not candidates:
            return []

        selected = []
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = {
                pool.submit(self._judge_relevance, entry, kind, screen): (entry, kind)
                for entry, kind in candidates
            }
            for future in as_completed(futures):
                entry, kind = futures[future]
                try:
                    relevant, reason = future.result()
                except Exception as error:      # a failed judgement must not sink the episode
                    log_warn(f"relevance call failed for '{entry.category}': {error}")
                    continue
                self.selection_log.append({
                    "category": entry.category,
                    "kind": kind,
                    "source": entry.source,
                    "relevant": relevant,
                    "reason": reason,
                })
                if relevant:
                    selected.append(entry)
        return selected

    def _retrieval_candidates(self):
        from execution.info_doc import IMAGE_SECTION, TASK_SECTION

        candidates = []
        for document in self._documents:
            candidates += [(e, "task") for e in document.entries(TASK_SECTION)]
            candidates += [(e, "kind of screen") for e in document.entries(IMAGE_SECTION)]
        return candidates

    def _init_state_candidates(self):
        """Stage-A rows for this episode's init state, as entries for the same relevance pass."""
        from execution.info_doc import TASK_SECTION, InfoDocument, resolve_frame

        available = sorted({row.get("init_state") for row in self._insight_rows
                            if row.get("init_state")})
        matching = [row for row in self._insight_rows
                    if row.get("init_state") == self._init_state]

        if not matching:
            log_warn(
                f"init_state '{self._init_state}' has no records in the loaded insights — "
                f"running with no hint. Available init_states: {available}",
                self._parameters,
            )
            return []

        candidates = []
        for row in matching:
            document = InfoDocument.from_dict(row["document"])
            for entry in document.entries(TASK_SECTION):
                # These rows come straight off insights.jsonl rather than through
                # load_document, so the two load-time fields have to be filled in here.
                #
                # Prefer the document's own recorded provenance; the row's "source" (set by
                # the loader that read the file) is the fallback, and group_idx alone is the
                # last resort — it is not comparable across verticals, since each numbers
                # its groups independently.
                label = document.provenance.label or row.get("source")
                entry.source = f"{label}/{row.get('group_idx')}" if label else row.get("group_idx")
                entry.resolved_frame = resolve_frame(document, entry.frame, self._parameters)
                candidates.append((entry, "task"))
        return candidates

    def write_hint(self) -> Optional[str]:
        """Select relevant knowledge for the current screen and synthesise one hint."""
        self.selection_log = []
        self.selected_ids = []
        screen = self._current_frame()

        candidates = (self._init_state_candidates() if self._mode == "init_state"
                      else self._retrieval_candidates())
        selected = self._select_entries(candidates, screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]

        if not selected:
            self.hint = None
            return None

        blocks = []
        for entry in selected:
            label = f" (source: {entry.source})" if entry.source else ""
            blocks.append(f"From '{entry.category}'{label}:\n{entry.insights_block()}")

        prompt = (
            WRITE_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", "\n\n".join(blocks))
        )
        output = self._vlm_call("hint", self._hint_vlm, texts=prompt, images=[screen],
                                max_new_tokens=self._hint_max_new_tokens)
        hint = (parse_key_value(output, "Hint") or "").strip()

        if not hint or hint.upper().startswith("NO HINT"):
            self.hint = None
            return None
        self.hint = hint
        return hint

    # -- Supervisor API --------------------------------------------------------

    def _evaluate(self) -> dict:
        """Write a hint from the current screen, then run the executor with it."""
        hint = self.write_hint()
        if hint is not None:
            self._executor_kwargs["hint"] = hint
        else:
            self._executor_kwargs.pop("hint", None)
        self.call_executor(self._task)
        return {"hint": self.hint, "selected_ids": list(self.selected_ids)}

    def process_executor_return(self, report: ExecutorReport) -> Any:
        """Hand the report back unchanged — ``call_executor`` has already filed it."""
        return report


