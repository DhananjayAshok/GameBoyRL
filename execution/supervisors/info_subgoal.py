"""
Planning from retrieved knowledge.

:class:`InfoSubgoalSupervisor` is
:class:`~execution.supervisors.subgoal.SubgoalSupervisor` plus one thing: the plan and the
hints are written from a document rather than from the task alone. It overrides
``_knowledge()`` and nothing else about the loop.

Knowledge reaches the planner through a relevance pass over the documents it was given:
every entry of every document is judged, one call each, on whether it fits this task and
this screen. Entries are judged on ``Description`` + ``Examples`` + their representative
frame — never on their ``Insights``, so relevance is decided on whether the context fits
rather than on whether the advice sounds appealing.

This class does not care where the documents came from. The benchmark arm's ``--mode``
decides that — a document distilled from real trajectories (``retrieval``) or one written
from the model's own priors (``parametric``) — and both arrive here as
:class:`~execution.info_doc.InfoDocument` objects and are treated identically. That is what
makes the two modes comparable: they differ only in the document.

**Why knowledge is only available with a plan.** A document is spent writing a plan; with
no plan there is nothing to spend it on but a single hint written once at the opening
frame. That arm existed — ``InfoHintSupervisor`` — and has been retired along with its
benchmark arm, which is why this class sits above the subgoal one rather than beside it.

The selection also used to offer a second path, ``init_state``, which skipped the document
and read stage-A rows matching the episode's starting state. It has been removed: the arm
now always judges document entries, and *which* document is the benchmark's choice rather
than the supervisor's.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _knowledge(self) -> str:
        """The distilled insights, or nothing if selection found none.

        Filtered once in :meth:`_before_targets` and reused verbatim by the planner, the
        reviser and the hint writer — deliberately, because re-filtering per call site
        would multiply the arm's VLM cost for a judgement that rarely changes within one
        task.
        """
        return self.insights_block

    def _before_targets(self) -> List[Optional[str]]:
        """Select knowledge, then plan from it.

        With nothing selected there is no knowledge to plan from, so the arm degrades to
        running the task unplanned rather than planning from an empty page — which would
        make it an unlabelled copy of the subgoal arm.
        """
        self.selection_log = []
        self.selected_ids = []
        self.insights_block = ""
        screen = self._current_frame()

        selected = self._select_entries(self._candidates(), screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]
        if not selected:
            log_warn("[info] nothing selected from the documents; running the task "
                     "unplanned.", self._parameters)
            self.plan = []
            self.completed_steps = []
            self.n_replans = 0
            return [None]

        self.insights_block = self.filter_insights(selected, screen)
        return super()._before_targets()

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
        """One yes/no call for a single entry. Returns (is_relevant, reason)."""
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

        output = self._vlm_call("filter", texts=prompt, images=images)
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

    def _candidates(self):
        """Every entry of every document, tagged with the kind of thing it describes.

        A parametric document contributes only task entries — the model has seen no screens,
        so it has no image categories to offer — but that needs no special case here: the
        image section is simply empty and the loop yields nothing for it.
        """
        from execution.info_doc import IMAGE_SECTION, TASK_SECTION

        candidates = []
        for document in self._documents:
            candidates += [(e, "task") for e in document.entries(TASK_SECTION)]
            candidates += [(e, "kind of screen") for e in document.entries(IMAGE_SECTION)]
        return candidates

    def filter_insights(self, selected, screen) -> str:
        """Prune the selected entries' insights to what could bear on this task, once.

        The relevance pass that chose these entries is deliberately blind to their insights
        — ``evidence_block()`` is Description and Examples only, so identity is judged on
        whether the *context* fits rather than on whether the advice sounds appealing. The
        consequence is that a correctly-chosen entry drags its whole bundle along, including
        insights about a character who is not here or a menu this task never opens.

        This is the only pass that reads insights as insights. It runs **once per episode**,
        and the surviving text is reused verbatim by the planner, the reviser and the hint
        writer — deliberately, because re-filtering per call site would multiply the arm's
        VLM cost for a judgement that rarely changes within one task.

        Being a one-shot filter makes it asymmetric on purpose: what it drops is gone for
        the whole episode, including for screens not yet reached, so
        :data:`FILTER_INSIGHTS_PROMPT` is written to keep anything that might matter later
        and to keep when unsure. Over-keeping costs prompt tokens; over-dropping costs
        knowledge that was expensive to build and cannot be recovered.

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

        The filter decides *what* survives; this decides *how it reads*. They are separate
        calls on purpose: a single prompt asked to both drop and rewrite can quietly lose
        content inside a rewrite, and there would be no way to tell that from legitimate
        filtering. Keeping them apart means :attr:`n_insights_kept` is a real count of
        surviving knowledge, and this pass is judged only on whether it says the same thing
        more usefully.

        The merge tree that builds the documents produces near-duplicates at several levels
        of detail — that is what stage B does — so by the time several entries are selected
        the same fact can appear three times, vaguest version included. Aggregating raises
        the specificity of what the planner reads.

        Provenance is deliberately dropped here: a distilled statement may draw on entries
        from several sources, so a per-entry label would be a lie. ``selected_ids`` still
        records what was retrieved, which is what a reader needs to trace it back.

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
