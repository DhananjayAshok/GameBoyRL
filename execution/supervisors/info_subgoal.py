"""
Planning from retrieved knowledge.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

from execution.supervisors.prompts import (DISTILL_INSIGHTS_PROMPT, FILTER_INSIGHTS_PROMPT,
                                           RELEVANCE_PROMPT)
from execution.supervisors.subgoal import SubgoalSupervisor
from utils import log_warn, parse_key_value, parse_list, parse_yes_no


class InfoSubgoalSupervisor(SubgoalSupervisor):
    """Select relevant knowledge, then plan and hint from it.

    :param documents: Parsed :class:`~execution.info_doc.InfoDocument` objects, however the
        arm obtained them.
    :param max_concurrency: Parallel relevance calls (they are independent).

    :ivar selected_ids: Ids of the entries the relevance pass kept. The durable part: the
        benchmark CSV stores it per episode, so a plan can be traced back to the exact
        entries it was synthesised from long after the run.
    :ivar insights_block: What :meth:`_knowledge` returns — filtered once per episode and
        reused by the planner, the hint writer and the replanner.
    """

    def __init__(self, *args, documents: Optional[List[Any]] = None,
                 max_concurrency: int = 8, **kwargs) -> None:
        self._documents = documents or []
        self._max_concurrency = max_concurrency
        self.selection_log: List[dict] = []
        self.selected_ids: List[str] = []
        self.insights_block: str = ""
        self.n_insights_candidate = 0
        self.n_insights_kept = 0
        self.n_insights_distilled = 0
        super().__init__(*args, **kwargs)
        # Built here rather than on first use. The relevance pass fans out across threads,
        # and the base class's lazy property is a check-then-set with no lock, so several
        # workers could pass the `is None` test at once and construct several VLMs.
        _ = self._vlm

    def _run_config(self) -> dict:
        return {**super()._run_config(),
                "max_concurrency": self._max_concurrency,
                "n_documents": len(self._documents)}

    def _knowledge(self) -> str:
        """The distilled insights, or nothing if selection found none.

        Filtered once in :meth:`_resolve_targets` and reused verbatim by the planner, the
        reviser and the hint writer — deliberately, because re-filtering per call site
        would multiply the arm's VLM cost for a judgement that rarely changes within one
        task.
        """
        return self.insights_block

    def _resolve_targets(self) -> List[Optional[str]]:
        """Select knowledge, then plan from it.

        With nothing selected there is no knowledge to plan *from*, but there is still a task
        to plan for, so this falls through to the ordinary subgoal planner with an empty
        insights block — :meth:`SubgoalSupervisor.write_plan` already renders that as
        ``(nothing recorded)``. The episode is then a knowledge-free planned run, which is
        exactly the ``subgoal`` arm, and ``selected_entry_ids`` is empty on that row to say so.
        """
        self.selection_log = []
        self.selected_ids = []
        self.insights_block = ""
        screen = self._current_frame()

        selected = self._select_entries(self._candidates(), screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]
        if not selected:
            log_warn("[info] nothing selected from the documents; planning without "
                     "insights (equivalent to the subgoal arm for this episode).",
                     self._parameters)
        else:
            self.insights_block = self.filter_insights(selected, screen)
        return super()._resolve_targets()

    def _extras(self) -> dict:
        return {**super()._extras(),
                "selected_ids": list(self.selected_ids),
                "insights_block": self.insights_block,
                "n_insights_candidate": self.n_insights_candidate,
                "n_insights_kept": self.n_insights_kept,
                "n_insights_distilled": self.n_insights_distilled}

    # -- Knowledge selection --------------------------------------------------

    @staticmethod
    def entry_id(entry) -> str:
        """
        Stable identifier for one selected entry, for the benchmark CSV and debug reports.

        ``source`` is the provenance label the loader attached — the document's vertical
        (``zeroshot``, ``curiosity``) for a built document, or ``parametric`` for one written
        from the model's priors. The category disambiguates entries within a source.
        """
        return f"{entry.source}#{entry.category}" if entry.source else entry.category

    @staticmethod
    def _entry_frame(entry):
        from PIL import Image

        # Stored frame paths are relative to their document's frames_root, so they are not
        # openable on their own. load_document has already resolved it. A parametric entry
        # has no frame at all, which lands on the None branch below.
        path = entry.resolved_frame
        if not path or not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")

    def _judge_relevance(self, entry, kind: str, screen) -> tuple:
        """One yes/no call for a single entry. Returns (is_relevant, reason, records).

        Runs on a worker thread, so it **files nothing**: it goes through
        :meth:`~execution.supervisors.base.Supervisor._vlm_infer` and hands its call records
        back for :meth:`_select_entries` to append once the pool has joined. Appending from
        the worker put the records into ``event_log`` in completion order, which made the
        supervisor's own record of an episode differ between two runs of the same episode.
        """
        images = [screen]
        entry_frame = self._entry_frame(entry)
        if entry_frame is not None:
            images.append(entry_frame)

        # The prompt must describe the images it actually receives. A parametric entry has
        # no frame, so both slots drop the second-image language rather than referring to a
        # picture that was never attached.
        if entry_frame is not None:
            frame_note = ("The images are: first the CURRENT screen the player is looking "
                          "at, then the representative frame recorded with this entry.")
            evidence_note = ("The frames are your primary evidence: compare what is "
                             "actually visible in them.")
        else:
            frame_note = ("The image is the CURRENT screen the player is looking at. This "
                          "entry has no recorded frame of its own — it was written from "
                          "general knowledge of the game rather than from a playthrough, so "
                          "judge it against its description and the current screen alone, "
                          "and be correspondingly more willing to answer no.")
            evidence_note = ("The current screen is your primary evidence: the entry's "
                             "description must fit what is actually visible in it.")

        prompt = (
            RELEVANCE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[KIND]", kind)
            .replace("[ENTRY]", entry.evidence_block())
            .replace("[FRAME_NOTE]", frame_note)
            .replace("[EVIDENCE_NOTE]", evidence_note)
        )

        output, records = self._vlm_infer("filter", texts=prompt, images=images)
        verdict = parse_yes_no(output, "Relevant")
        reason = (parse_key_value(output, "Reasoning") or "").strip()
        return verdict is True, reason, records

    def _select_entries(self, candidates, screen) -> List[Any]:
        """Run the relevance pass over (entry, kind) candidates, in parallel.
        """
        if not candidates:
            return []

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = [pool.submit(self._judge_relevance, entry, kind, screen)
                       for entry, kind in candidates]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:  # a failed judgement must not sink the episode
                    results.append(error)

        selected = []
        for (entry, kind), result in zip(candidates, results):
            if isinstance(result, Exception):
                log_warn(f"relevance call failed for '{entry.category}': {result}",
                         self._parameters)
                continue
            relevant, reason, records = result
            self.report.event_log.extend(records)
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

    def _candidates(self):
        """Every entry of every document, tagged with the kind of thing it describes.
        """
        from execution.info_doc import IMAGE_SECTION, TASK_SECTION

        candidates = []
        for document in self._documents:
            candidates += [(e, "task") for e in document.entries(TASK_SECTION)]
            candidates += [(e, "kind of screen") for e in document.entries(IMAGE_SECTION)]
        return candidates

    def filter_insights(self, selected, screen) -> str:
        """Prune the selected entries' insights to what could bear on this task, once.

        :return: The insight block to interpolate, grouped by entry. Falls back to the
            unfiltered block when the reply cannot be parsed — a filter that silently
            returns nothing is indistinguishable from a document with nothing in it.
        """
        pairs = [(entry, insight) for entry in selected for insight in entry.insights]
        self.n_insights_candidate = len(pairs)
        self.n_insights_kept = len(pairs)
        if not pairs:
            return "(no insights recorded)"

        numbered = "\n".join(
            f"  {i + 1}. [{entry.source or 'unknown'} / {entry.category}] {insight}"
            for i, (entry, insight) in enumerate(pairs)
        )
        prompt = (
            FILTER_INSIGHTS_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", numbered)
        )
        output = self._vlm_call("filter_insights", texts=prompt, images=[screen])
        answer = (parse_key_value(output, "Keep") or "").strip()

        if not answer or answer.upper().startswith("ALL"):
            kept = pairs
        else:
            wanted = {int(n) for n in re.findall(r"\d+", answer)}
            kept = [pair for i, pair in enumerate(pairs) if i + 1 in wanted]
            if not kept:
                # Either the model dropped everything or the reply did not parse. Both are
                # more likely to be a filter failure than a document with nothing useful in
                # it, so fail open rather than plan from an empty page.
                log_warn(f"[plan] insight filter kept nothing from {len(pairs)} candidates "
                         f"(reply: {answer[:80]!r}); keeping all.", self._parameters)
                kept = pairs

        self.n_insights_kept = len(kept)
        self._say(f"  insight filter: kept {len(kept)}/{len(pairs)}")

        grouped: dict = {}
        for entry, insight in kept:
            grouped.setdefault((entry.category, entry.source), []).append(insight)
        blocks = []
        for (category, source), insights in grouped.items():
            label = f" (source: {source})" if source else ""
            body = "\n".join(f"- {i}" for i in insights)
            blocks.append(f"From '{category}'{label}:\n{body}")
        grouped_block = "\n\n".join(blocks)

        return self.distil_insights(kept, grouped_block, screen)

    def distil_insights(self, kept: list, grouped_block: str, screen) -> str:
        """Consolidate the surviving insights into one concrete, non-redundant list.

        :return: The distilled block, or *grouped_block* unchanged if the reply cannot be
            parsed — a distillation that silently returns nothing is worse than a verbose
            but complete briefing.
        """
        if not kept:
            return grouped_block

        prompt = (
            DISTILL_INSIGHTS_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", grouped_block)
        )
        output = self._vlm_call("distil_insights", texts=prompt, images=[screen])

        lines = parse_list(output, "Insights")

        if not lines:
            log_warn(f"[plan] insight distillation produced nothing from {len(kept)} "
                     f"insights; keeping the unconsolidated block.", self._parameters)
            self.n_insights_distilled = len(kept)
            return grouped_block

        self.n_insights_distilled = len(lines)
        self._say(f"  insight distillation: {len(kept)} -> {len(lines)} statements")
        # parse_list returns bare items; the bullet is re-added here because this block goes
        # straight into a prompt and the "- " is part of how that prompt reads.
        return "\n".join(f"- {line}" for line in lines)
