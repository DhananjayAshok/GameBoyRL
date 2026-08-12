"""
Driving a task one plan step at a time.

:class:`InfoPlanSupervisor` selects relevant knowledge, writes a plan from it, hands the
executor one step at a time, judges each step, and decides whether to retry it, resume
with a hint, or abandon the plan and rewrite it.

This is the largest module in the package and still interleaves two concerns: plan
policy (when to retry, when to replan, when to give up) and VLM mechanics (build the
prompt, call, parse, log). Separating them is the obvious next move and is
deliberately not attempted here — it is the newest and least-validated code in the
project, and restructuring it before knowing which of its intervention mechanisms
earns its VLM calls would be sculpting the wrong shape.

The knowledge-selection half used to live on a separate ``InfoHintSupervisor``, which this
class subclassed. That class was the test-time arm that spent its retrieved knowledge on a
single hint written once at the opening frame; it has been retired along with its benchmark
arm, and its selection machinery — :meth:`~InfoPlanSupervisor._candidates` and
:meth:`~InfoPlanSupervisor._select_entries` — was folded in here.

The selection also used to offer a second path, ``init_state``, which skipped the document
and read stage-A rows matching the episode's starting state. It has been removed: the arm
now always judges document entries, and *which* document is the benchmark's choice rather
than the supervisor's.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Type

from gameboy_worlds.interface import Environment

from execution.executors import Executor
from execution.report import EnvironmentStepRecord, ExecutorReport, parse_completion
from execution.supervisors._format import action_trace, attempt_history_line
from execution.supervisors.base import Supervisor
from execution.supervisors.checker import summarise_trajectory_segments
from execution.supervisors.prompts import (
    DISTILL_INSIGHTS_PROMPT,
    FILTER_INSIGHTS_PROMPT,
    JUDGE_CONSOLIDATE_PROMPT,
    JUDGE_SLICE_PROMPT,
    PLAN_FLAW_PROMPT,
    PLAN_PROMPT,
    REGRESSION_CHECK_PROMPT,
    RELEVANCE_PROMPT,
    RESUME_HINT_PROMPT,
)
from utils import log_warn, parse_key_value, parse_list, parse_steps, parse_yes_no


class InfoPlanSupervisor(Supervisor):
    """
    Plans from the info document, then supervises the executor through the plan step by step.

    Knowledge reaches the planner through a relevance pass over the documents it was given:
    every entry of every document is judged, one call each, on whether it fits this task and
    this screen. Entries are judged on ``Description`` + ``Examples`` + their representative
    frame — never on their ``Insights``, so relevance is decided on whether the context fits
    rather than on whether the advice sounds appealing.

    This class does not care where the documents came from. The benchmark arm's ``--mode``
    decides that — a document distilled from real trajectories (``retrieval``) or one written
    from the model's own priors (``parametric``) — and both arrive here as
    :class:`~execution.info_doc.InfoDocument` objects and are treated identically. That is
    what makes the two modes comparable: they differ only in the document.

    The loop, per episode:

    1. Select relevant entries for the opening screen (inherited, both modes supported).
    2. Write a plan from them, as steps separated by ``[STEP]``. :data:`PLAN_PROMPT`
       constrains the planner to visible artefacts and forbids naming buttons, because a
       button sequence recorded in one playthrough is wrong from any other screen, and
       requires every step to end in a visually checkable condition — without one there is
       nothing for step 4 to judge against.
    3. Run the step, capped at :attr:`max_leg_steps` env steps per attempt.
    4. Judge completion from the trajectory and the final screen. If incomplete, check
       whether progress was undone, then ask whether the **plan itself** is at fault
       (:meth:`check_plan_flaw`). A flawed plan is replaced from the current step onwards;
       a sound plan gets a hint that assumes the player is *resuming from where the failed
       attempt left them*, and the step is retried.
    5. Move on after :attr:`max_attempts_per_step` failures. The final step is exempt from
       both the judge and that cap — only the environment ending clears it — but it is
       replanned like any other, which is where a plan that was wrong from the start shows
       up most clearly.

    Repair is at plan level, not step level. An earlier design rewrote the failing step in
    place; that can only fix wording, which assumes the route is right and only the phrasing
    is wrong. Most failures worth catching are not like that — the route does not exist, the
    object named is not on this screen, the game does not work as assumed — and a
    better-worded version of the same wrong step cannot fix any of them.

    Two budgets bound all of this. ``max_leg_steps`` (5) caps a single executor call, and
    is purely internal — it decides how often the supervisor gets to look. ``max_steps``
    (75, the benchmark's) caps the emulator steps across the supervisor's entire lifetime,
    every attempt and retry included, so this arm and the baseline play the same game with
    the same allowance and their CSVs stay comparable.

    No executor is modified or subclassed: a step's hint travels through
    ``Executor.__init__``'s existing ``hint`` argument, which ``Supervisor.call_executor``
    already forwards.

    :param task: The benchmark task string.
    :param executor_class: :class:`~execution.executors.Executor` subclass to run.
    :param env: The game environment.
    :param game: Game name string.
    :param max_steps: Env-step budget for the supervisor's whole lifetime.
    :param max_tool_calls: Tool-call budget forwarded to the executor.
    :param documents: Parsed :class:`~execution.info_doc.InfoDocument` objects, however the
        arm obtained them.
    :param supervisor_vlm_model: The one model this arm reasons with — selection, planning,
        insight filtering, judging, hinting and replanning all use it. Under
        ``--mode parametric`` it also writes the document, which is why the document's
        provenance records it and no separate generator model is passed in.
    :param supervisor_vlm_kind: VLM kind for that model.
    :param max_new_tokens: Token budget for every one of those calls.
    :param max_concurrency: Parallel relevance calls (they are independent).
    :param parameters: Optional parameter overrides.
    :param max_leg_steps: Env-step cap for one executor attempt.
    :param max_attempts_per_step: Failed attempts at one step before the supervisor gives up
        on it and moves to the next. With per-step rewriting gone this is the only exit from
        the retry loop besides the episode budget, so it is load-bearing rather than a
        safety net. The final step ignores it.
    :param max_replans: How many times the plan may be rewritten in one episode. Bounds both
        cost and the risk of thrashing between two readings of the same screen.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param verbose: Print the per-attempt narration — the step, the hint, the executor's
        stop reason, the judge's verdict, any revision. Off by default.

    :ivar plan: The current plan. Mutated in place when a step is revised.
    :ivar step_log: One dict per step attempted — the step text, every attempt's
        termination reason, the judge's verdict and reasoning, the hints tried, and any
        revision. The whole record of what the supervisor did and why.
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
        supervisor_vlm_model: Optional[str] = None,
        supervisor_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        max_concurrency: int = 8,
        parameters: Optional[dict] = None,
        max_leg_steps: int = 5,
        max_attempts_per_step: int = 3,
        max_replans: int = 2,
        max_frames_per_slice: int = 8,
        verbose: bool = False,
        **executor_kwargs: Any,
    ) -> None:
        self._documents = documents or []
        self._max_concurrency = max_concurrency
        super().__init__(task, executor_class, env, game, max_steps, max_tool_calls,
                         supervisor_vlm_model, supervisor_vlm_kind, max_new_tokens,
                         parameters, **executor_kwargs)
        # Populated by the relevance pass so the caller can inspect why a plan came out the
        # way it did without re-running the pipeline. ``selected_ids`` is the durable part:
        # the benchmark CSV stores it per episode, so a plan can be traced back to the exact
        # entries it was synthesised from long after the run.
        self.selection_log: List[dict] = []
        self.selected_ids: List[str] = []
        self.verbose = verbose
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_step = max_attempts_per_step
        self.max_replans = max_replans
        self.n_replans = 0
        self.last_diagnosis: Optional[str] = None
        self._max_frames_per_slice = max_frames_per_slice
        self.plan: List[str] = []
        self.step_log: List[dict] = []
        # Steps already cleared, each with the frame that proved it, so a later step can be
        # checked for having undone one of them.
        self.completed_steps: List[dict] = []
        # Filtered once in write_plan and reused by revise_step and write_resume_hint.
        self.insights_block: str = ""
        self.n_insights_candidate = 0
        self.n_insights_kept = 0
        self.n_insights_distilled = 0

    def _say(self, message: str) -> None:
        """Per-attempt commentary. Silent unless the caller asked for it.

        Off by default: a supervisor that narrates whether or not anyone wired up a flag
        makes a 35-episode sweep unreadable.
        """
        if self.verbose:
            print(message)

    # -- Knowledge selection ---------------------------------------------------

    @staticmethod
    def entry_id(entry) -> str:
        """
        Stable identifier for one selected entry, for the benchmark CSV and debug reports.

        ``source`` is the provenance label the loader attached — the document's vertical
        (``zeroshot``, ``curiosity``) for a built document, or ``parametric`` for one written
        from the model's priors. The category disambiguates entries within a source.
        """
        return f"{entry.source}#{entry.category}" if entry.source else entry.category

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

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

    # -- Planning --------------------------------------------------------------

    def write_plan(self) -> List[str]:
        """Select relevant entries for the opening screen and turn them into a plan.

        Returns ``[]`` when nothing was selected — with no knowledge there is nothing to
        plan from, and :meth:`_evaluate` falls back to running the task unplanned rather
        than inventing a plan out of the prompt alone.
        """
        self.selection_log = []
        self.selected_ids = []
        screen = self._current_frame()

        selected = self._select_entries(self._candidates(), screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]
        if not selected:
            self.plan = []
            return []

        self.insights_block = self.filter_insights(selected, screen)
        prompt = (
            PLAN_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[INSIGHTS]", self.insights_block)
        )
        output = self._vlm_call("plan", texts=prompt, images=[screen])
        raw = (parse_key_value(output, "Plan") or "").strip()
        self.plan = parse_steps(raw)
        return self.plan

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

    # -- Judging and repair ----------------------------------------------------

    def _segment_summaries(self, env_steps: list, step: str) -> List[str]:
        # The infer happens inside summarise_trajectory_segments, which batches the windows
        # itself, so it takes a recording caller rather than a VLM. This used to be the one
        # site that could not use _call and had to set a stage on a wrapper object first.
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, step,
            self._vlm_caller("judge_slice"),
            self._max_new_tokens, self._max_frames_per_slice,
        )

    def judge_step(self, env_steps: list, step: str, stop_reason: str) -> tuple:
        """Decide whether *step* was actually completed.

        Windowed exactly like the hint critique (:func:`summarise_trajectory_segments`),
        then consolidated against the **current** screen rather than the summaries alone —
        the step's termination condition is a statement about what the screen shows, so the
        screen is what settles it. The executor's own stop reason is passed in as context
        and explicitly framed as a claim to be checked, since a leg that self-terminated is
        exactly the case where the agent's opinion and the screen may disagree.

        :return: ``(complete, reasoning, segment_summaries)``. An attempt with no env steps
            is never complete: nothing happened, so nothing can have been achieved.
        """
        if not env_steps:
            return False, "No environment steps were taken in this attempt.", []

        summaries = self._segment_summaries(env_steps, step)
        prompt = (
            JUDGE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", step)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries))
            .replace("[STOP_REASON]", stop_reason or "unknown")
        )
        output = self._vlm_call("judge", texts=prompt, images=[self._current_frame()])
        verdict = parse_completion(output)
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if verdict is None:
            # Truncation lands here, and it is not a harmless parse miss: `Complete:` is the
            # last line of the reply, so a response cut short loses the verdict while keeping
            # the reasoning, and an unparsed verdict reads as NOT complete. Every step then
            # fails its judgement no matter what the screen shows. Loud, because the symptom
            # — nothing ever clears — looks exactly like a bad plan.
            log_warn(f"[plan] completion check returned no 'Complete:' line within "
                     f"{self._max_new_tokens} tokens; treating as not complete. "
                     f"Raise --supervisor_max_new_tokens if this repeats.", self._parameters)
        return verdict is True, reasoning, summaries

    def check_regression(self, summaries: List[str]) -> Optional[str]:
        """Ask whether the failed attempt undid a step that had already been completed.

        The judge cannot see this. It is asked only whether *this* step's termination
        condition is met, so a run that quits the menu it had opened, or puts back the item
        it had taken, reads as a plain failure of the current step — and the hint that
        follows tells the player to keep pushing forward from a state that has silently
        moved backwards.

        The comparison is between two frames: the screen when the earlier step was judged
        complete, and the screen now. Two images rather than a description, because the
        thing being asked about is exactly a difference between two screens, and the
        segment summaries were written without any knowledge of the earlier state.

        Runs only when a step has actually been completed, so an episode whose first step
        never lands never pays for it.

        :return: A sentence naming what was lost, or ``None`` if nothing was.
        """
        if not self.completed_steps:
            return None
        last = self.completed_steps[-1]
        earlier = self.completed_steps[:-1]
        earlier_block = ""
        if earlier:
            earlier_block = ("Steps completed before that one, which must also still hold:\n"
                             + "\n".join(f"- {s['step']}" for s in earlier) + "\n\n")

        prompt = (
            REGRESSION_CHECK_PROMPT
            .replace("[GAME]", self._game)
            .replace("[PREVIOUS_STEP]", last["step"])
            .replace("[EARLIER_STEPS_BLOCK]", earlier_block)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries) or "  (no actions taken)")
        )
        output = self._vlm_call(
            "regression_check",
            texts=prompt, images=[last["frame"], self._current_frame()],
        )
        if parse_yes_no(output, "Undone") is not True:
            return None
        lost = (parse_key_value(output, "What was lost") or "").strip()
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if lost.upper().startswith("NONE") or not lost:
            lost = reasoning or "progress from an earlier step is no longer visible"
        note = (f"The attempt undid earlier progress: {lost} "
                f"(the step '{last['step']}' was completed and no longer holds).")
        self._say(f"      REGRESSION: {lost}")
        return note

    def write_resume_hint(self, summaries: List[str], step: str, judgement: str,
                          previous_hint: str = "", report=None,
                          regression: Optional[str] = None) -> Optional[str]:
        """Write the hint for the next attempt at *step*, anchored on the current screen.

        *report* is the attempt that just failed. When given, the executor's own reasoning
        goes in beside the summaries: the summaries say what changed on screen, the
        reasoning says what the executor believed while it was failing to change it. A hint
        written against the belief can correct it directly — telling a player who thinks
        they are already on the icon that they are two tiles left of it beats telling them
        again to go to the icon.

        Returns ``None`` when no hint could be written — a truncated reply, or the writer
        declining. **Both callers keep the previous attempt's hint in that case**
        (``write_resume_hint(...) or hint``) rather than retrying unaided. A stale hint is
        advice about an earlier failure and may no longer describe where the player is,
        which is a real cost; it is accepted because the alternative throws away the only
        guidance available at the exact moment the writer is already struggling, and a
        retry with nothing is the weaker of the two. The two call sites disagreed on this
        for a while — if that reasoning is ever revisited, change both.
        """
        prior_block = (f'Previous hint, which did not work (do not simply repeat it):\n'
                       f'"{previous_hint}"\n\n' if previous_hint else "")
        trace_block = (f"\n\nWhat they pressed, and the reason they gave for each:\n"
                       f"{action_trace(report)}" if report is not None else "")
        prompt = (
            RESUME_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", step)
            .replace("[SEGMENT_SUMMARIES]",
                     ("\n".join(summaries) or "  (no actions taken)") + trace_block)
            .replace("[JUDGEMENT]", judgement or "the screen does not show the step complete")
            .replace("[REGRESSION_BLOCK]",
                     f"**{regression} Say what to do about that first — the player cannot "
                     f"finish this step from a state they have gone backwards into.**\n\n"
                     if regression else "")
            .replace("[INSIGHTS]", self.insights_block or "(nothing recorded)")
            .replace("[PRIOR_HINT_BLOCK]", prior_block)
        )
        output = self._vlm_call("hint", texts=prompt, images=[self._current_frame()])
        # The diagnosis is not passed to the executor — it is the hint writer's working, and
        # the executor gets instructions, not analysis. It is kept for the debug panel,
        # where a correct diagnosis followed by a useless instruction is a different failure
        # from a wrong diagnosis, and the two need telling apart.
        self.last_diagnosis = (parse_key_value(output, "Diagnosis") or "").strip() or None
        hint = (parse_key_value(output, "Hint") or "").strip()
        if self.last_diagnosis:
            self._say(f"      diagnosis: {self.last_diagnosis}")
        return hint or None

    def check_plan_flaw(self, index: int, steps: List[Optional[str]], history: List[str],
                        judgement: str, regression: Optional[str]) -> Optional[List[str]]:
        """Ask whether the failure is the plan's fault, and replan the remainder if so.

        This replaces per-step rewriting. A rewrite could only ever fix the wording of one
        step, which assumes the route is right and only the phrasing is wrong. Most of the
        failures worth catching are not like that: the route itself does not exist, an object
        the plan names is not on this screen, or the game does not work the way the plan
        assumed. Those need a new plan, and a better-worded version of the same wrong step
        cannot produce one.

        Only the **remaining** steps are replanned. Steps already cleared are shown as DONE
        and explicitly excluded, because re-planning finished work spends budget redoing what
        the screen already shows is done — and, worse, invites the executor to undo it.

        :data:`PLAN_FLAW_PROMPT` is written to answer *no* by default. Repeated failure is
        not evidence of a bad plan — a fumbled step fails repeatedly too — so it demands
        something visible that the plan is incompatible with, and a replan that fires on
        ordinary fumbling would throw away a sound route on the first hard step.

        :return: The replacement steps for ``index`` onwards, or ``None`` to carry on
            hinting the current step.
        """
        if self.n_replans >= self.max_replans:
            return None

        plan_lines = []
        for i, text in enumerate(steps):
            if text is None:
                continue
            if i < index:
                marker = "  [DONE]"
            elif i == index:
                marker = "  <-- CURRENT, just failed"
            else:
                marker = "  [not yet attempted]"
            plan_lines.append(f"  {i + 1}. {text}{marker}")

        prompt = (
            PLAN_FLAW_PROMPT
            .replace("[GAME]", self._game)
            .replace("[OVERALL_TASK]", self._task)
            .replace("[PLAN_BLOCK]", "\n".join(plan_lines))
            .replace("[FAILURE_HISTORY]", "\n\n".join(f"  attempt {i + 1}: {h}"
                                                      for i, h in enumerate(history)))
            .replace("[JUDGEMENT]", judgement or "(not judged)")
            .replace("[REGRESSION_LINE]", f"\n{regression}\n" if regression else "")
            .replace("[INSIGHTS]", self.insights_block or "(nothing recorded)")
        )
        output = self._vlm_call("plan_flaw", texts=prompt, images=[self._current_frame()])
        if parse_yes_no(output, "Flawed") is not True:
            return None

        raw = (parse_key_value(output, "Plan") or "").strip()
        replacement = parse_steps(raw)
        if not replacement or raw.upper().startswith("NONE"):
            # Said flawed and produced nothing to replace it with. Hinting the existing step
            # is a worse plan than no plan, but it is a plan.
            log_warn("[plan] plan judged flawed but no replacement was produced; "
                     "continuing with the current step.", self._parameters)
            return None

        self.n_replans += 1
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        self._say(f"      REPLAN ({self.n_replans}/{self.max_replans}): {reasoning}")
        for text in replacement:
            self._say(f"        - {text}")
        return replacement

    # -- Supervisor API --------------------------------------------------------

    def _run_leg(self, leg_task: str, hint: Optional[str], self_terminate: bool,
                 budget: int) -> ExecutorReport:
        """One executor attempt, capped at the smaller of the leg cap and what is left.

        ``self._max_steps`` is what :meth:`Supervisor.call_executor` hands the executor, so
        it has to be narrowed to this leg's cap and put back afterwards. Restored in a
        ``finally`` here, beside the write, rather than once after the loop in
        :meth:`evaluate`: an executor that raises would otherwise leave the supervisor with
        a single leg's budget standing in for the whole episode's.
        """
        if hint is None:
            self._executor_kwargs.pop("hint", None)
        else:
            self._executor_kwargs["hint"] = hint
        self._executor_kwargs["allow_self_termination"] = self_terminate
        episode_budget = self._max_steps
        self._max_steps = min(self.max_leg_steps, budget)
        try:
            return self.call_executor(leg_task)
        finally:
            self._max_steps = episode_budget

    def _evaluate(self) -> dict:
        """Plan, then supervise the executor through the plan until it lands or the budget ends.

        Intermediate steps are given to the executor as its **task** with
        ``allow_self_termination=True``, rather than as a hint: a hint is advice about a goal
        the executor already holds, so a step delivered as a hint while the task still reads
        the benchmark's own would ask the executor to declare victory on a means rather than
        an end. The final step reverts to the real task with the step as a hint and no
        self-termination, so success stays the environment's call.

        The retry loop is bounded three ways, in increasing order of bluntness:
        :attr:`max_attempts_per_step` failures move the plan on, :attr:`max_replans`
        rewrites exhaust the episode's tolerance for replanning, and the env-step budget
        stops everything. The first is load-bearing: with per-step rewriting removed there
        is no other way out of a step that never clears.

        The attempt cap used to carry a second job — guarding against an attempt that
        consumed no budget at all, back when the executor could self-terminate on the
        opening frame before acting. It cannot any more: the completion check runs *after*
        an action, so every attempt spends at least one env step and the budget always
        moves. The cap is now purely about step quality.

        :return: This arm's extra values. ``Supervisor.evaluate`` attaches the
            :class:`~execution.report.SupervisorReport`, whose last executor leg carries the
            ``termination_reason`` that decides the episode.
        """
        self.write_plan()
        self.step_log = []
        self.completed_steps = []
        self.n_replans = 0
        budget = self._max_steps

        # No plan (nothing retrieved, or an unparseable planner reply) degrades to the
        # unplanned baseline rather than to a fabricated plan.
        #
        # A copy, not self.plan itself: the replan paths below splice into `steps` in place
        # and then rebind self.plan to a freshly filtered list, so aliasing them would mean
        # the two names refer to the same object before the first replan and different
        # objects after it. `steps` is the working list; self.plan is derived from it.
        steps = list(self.plan) if self.plan else [None]
        if not self.plan:
            log_warn("[plan] no plan produced; running the task unplanned.", self._parameters)

        index = 0
        while index < len(steps):
            if budget <= 0:
                self._say(f"  step budget exhausted at step {index + 1}/{len(steps)}")
                break

            step = steps[index]
            is_last = index == len(steps) - 1
            record = {"step": step, "attempts": [], "replans": [], "cleared": False}
            hint: Optional[str] = None
            history: List[str] = []

            while budget > 0:
                if step is None:
                    leg_task, leg_hint, self_terminate = self._task, None, False
                elif is_last:
                    leg_task, leg_hint, self_terminate = self._task, hint or step, False
                else:
                    leg_task, leg_hint, self_terminate = step, hint, True

                self._say(f"\n  --- step {index + 1}/{len(steps)}"
                      f" attempt {len(record['attempts']) + 1}"
                      f"{' (final)' if is_last else ''}, {budget} steps left")
                if step is not None:
                    self._say(f"      {step}")
                if leg_hint and leg_hint != step:
                    self._say(f"      hint: {leg_hint}")

                # _run_leg -> call_executor files the report into report.event_log, after the
                # supervisor calls that produced this leg's hint and before the ones that
                # judge it. That ordering is the record; no per-leg tagging is needed, since
                # each ExecutorReport carries its own task and hint.
                report = self._run_leg(leg_task, leg_hint, self_terminate, budget)
                env_steps = [s for s in report.steps if isinstance(s, EnvironmentStepRecord)]
                budget -= len(env_steps)
                attempt = {"termination_reason": report.termination_reason,
                           "n_steps": len(env_steps), "hint": leg_hint}
                self._say(f"      -> {report.termination_reason} after {len(env_steps)} step(s)")

                # The environment ending outranks every judgement: terminated is the
                # benchmark's own success signal and truncated/max_invalid are states no
                # further hint can act on.
                if report.termination_reason in ("terminated", "truncated", "max_invalid"):
                    attempt["cleared"] = report.termination_reason == "terminated"
                    record["attempts"].append(attempt)
                    record["cleared"] = attempt["cleared"]
                    index = len(steps)          # leave the outer loop too
                    break

                if is_last:
                    # Not judged: the only thing that clears the final step is the
                    # environment terminating, which the branch above already caught.
                    attempt["cleared"] = False
                    record["attempts"].append(attempt)
                    summaries = self._segment_summaries(env_steps, self._task) if env_steps else []
                    # The final step is not judged, but it can still undo what came before —
                    # and on the final step that is the whole task going backwards.
                    regression = self.check_regression(summaries)
                    if regression:
                        attempt["regression"] = regression
                    verdict = "the environment did not signal the task complete"
                    history.append(attempt_history_line(
                        report, env_steps, summaries, leg_hint, verdict, regression))

                    # The final step is replanned like any other. This is where a plan that
                    # was wrong from the start shows up most clearly — the route ran out and
                    # the task is still not done — and a replacement that splits the last
                    # step turns its front half into an ordinary judged step.
                    replacement = self.check_plan_flaw(index, steps, history, verdict,
                                                       regression)
                    if replacement:
                        record["replans"].append(replacement)
                        steps[index:] = replacement
                        self.plan = [s for s in steps if s is not None]
                        step = steps[index]
                        is_last = index == len(steps) - 1
                        record["step"] = step
                        hint = None
                        continue

                    hint = self.write_resume_hint(
                        summaries, self._task, verdict, hint or "",
                        report=report, regression=regression,
                    ) or hint
                    continue

                complete, reasoning, summaries = self.judge_step(
                    env_steps, step, report.termination_reason
                )
                attempt["cleared"] = complete
                attempt["judgement"] = reasoning
                record["attempts"].append(attempt)
                self._say(f"      judge: {'COMPLETE' if complete else 'NOT COMPLETE'} — {reasoning}")

                if complete:
                    record["cleared"] = True
                    # Remembered with the frame that proved it, so a later step undoing it
                    # can be spotted by comparing that frame against the screen then.
                    self.completed_steps.append(
                        {"step": step, "frame": self._current_frame()})
                    break

                # Before the hint is written, not after: a hint composed without knowing
                # progress was lost will push the player forward from a state that has
                # moved backwards.
                regression = self.check_regression(summaries)
                if regression:
                    attempt["regression"] = regression

                history.append(attempt_history_line(
                    report, env_steps, summaries, leg_hint,
                    f"judged incomplete — {reasoning}", regression))

                # Is the plan itself wrong? Asked before the hint, because a hint written for
                # a step that should not be attempted at all is wasted either way.
                replacement = self.check_plan_flaw(index, steps, history,
                                                   f"judged incomplete — {reasoning}",
                                                   regression)
                if replacement:
                    record["replans"].append(replacement)
                    # Everything from here on is replaced; cleared steps before it are not
                    # touched, so progress already made survives the replan.
                    steps[index:] = replacement
                    self.plan = [s for s in steps if s is not None]
                    step = steps[index]
                    is_last = index == len(steps) - 1
                    record["step"] = step
                    hint = None
                    continue

                # Without per-step rewriting there is no other exit from this loop, so the
                # attempt cap is what stops one unclearable step from eating the episode.
                # The final step is exempt: only the environment ends it.
                if len(record["attempts"]) >= self.max_attempts_per_step:
                    self._say(f"      step not cleared after {len(record['attempts'])} "
                              f"attempts; moving on.")
                    break

                # ``or hint`` keeps the previous attempt's hint when the writer produces
                # none — see the note on :meth:`write_resume_hint`. The final-step path
                # above does the same; the two used to disagree, and an unaided retry is
                # not what either of them meant.
                hint = self.write_resume_hint(summaries, step, reasoning, hint or "",
                                              report=report, regression=regression) or hint
                if self.last_diagnosis:
                    attempt["diagnosis"] = self.last_diagnosis

            self.step_log.append(record)
            index += 1

        # Only this arm's extras. Supervisor.evaluate attaches the report, and the last
        # executor leg in its event_log is what used to be returned as `last_report`.
        return {
            "plan": list(self.plan),
            "selected_ids": list(self.selected_ids),
            "step_log": self.step_log,
            "insights_block": self.insights_block,
            "n_replans": self.n_replans,
            "n_insights_candidate": self.n_insights_candidate,
            "n_insights_kept": self.n_insights_kept,
            "n_insights_distilled": self.n_insights_distilled,
        }

    def process_executor_return(self, report: ExecutorReport) -> ExecutorReport:
        """Hand the report back unchanged — ``call_executor`` has already filed it.

        This arm reads each leg in :meth:`_evaluate`, where the plan step it belongs to is
        known; there is nothing useful to do with a report in isolation.
        """
        return report
