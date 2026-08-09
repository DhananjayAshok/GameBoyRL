"""
Driving a task one plan step at a time.

:class:`InfoPlanSupervisor` writes a plan, hands the executor one step at a time,
judges each step, and decides whether to retry it, resume with a hint, or abandon the
plan and rewrite it.

This is the largest module in the package and still interleaves two concerns: plan
policy (when to retry, when to replan, when to give up) and VLM mechanics (build the
prompt, call, parse, log). Separating them is the obvious next move and is
deliberately not attempted here — it is the newest and least-validated code in the
project, and restructuring it before knowing which of its intervention mechanisms
earns its VLM calls would be sculpting the wrong shape.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from execution.report import EnvironmentStepRecord, ExecutorReport, parse_completion
from execution.supervisors._format import action_trace, attempt_history_line
from execution.supervisors.checker import summarise_trajectory_segments
from execution.supervisors.info_hint import InfoHintSupervisor, RecordingVLM
from execution.supervisors.prompts import (
    DISTILL_INSIGHTS_PROMPT,
    FILTER_INSIGHTS_PROMPT,
    JUDGE_CONSOLIDATE_PROMPT,
    JUDGE_SLICE_PROMPT,
    PLAN_FLAW_PROMPT,
    PLAN_PROMPT,
    REGRESSION_CHECK_PROMPT,
    RESUME_HINT_PROMPT,
)
from utils import log_warn, parse_key_value, parse_list, parse_steps, parse_yes_no, VLM


class InfoPlanSupervisor(InfoHintSupervisor):
    """
    Plans from the info document, then supervises the executor through the plan step by step.

    Where :class:`InfoHintSupervisor` spends its retrieved knowledge on a single hint written
    once at the opening frame, this spends it on a *plan* and then stays in the loop: it
    watches each step, judges whether it actually landed, and intervenes when it did not.
    The retrieval is deliberately identical — this subclasses it and reuses
    ``_retrieval_candidates`` / ``_init_state_candidates`` / ``_select_entries`` unchanged —
    so a difference between the two arms is attributable to what is done with the knowledge,
    not to which knowledge was found.

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
    every attempt and retry included, so this arm and the info arm play the same game with
    the same allowance and their CSVs stay comparable.

    :param plan_vlm_model: Model for the planning, judging, hinting and revision calls.
        Defaults to ``hint_vlm_model``.
    :param plan_vlm_kind: VLM kind for those calls.
    :param max_leg_steps: Env-step cap for one executor attempt.
    :param max_attempts_per_step: Failed attempts at one step before the supervisor gives up
        on it and moves to the next. With per-step rewriting gone this is the only exit from
        the retry loop besides the episode budget, so it is load-bearing rather than a
        safety net. The final step ignores it.
    :param max_replans: How many times the plan may be rewritten in one episode. Bounds both
        cost and the risk of thrashing between two readings of the same screen.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param plan_max_new_tokens: Token budget for the planning, replanning and
        segment-summary calls, which are longer than a hint.
    :param judge_max_new_tokens: Token budget for the completion check. Largest of the
        three: its ``Complete:`` verdict is the last line of the reply, so truncation
        costs the answer while keeping the reasoning, and an unparsed verdict reads as
        not complete — every step then fails regardless of the screen.
    :param verbose: Print the per-attempt narration — the step, the hint, the executor's
        stop reason, the judge's verdict, any revision. Off by default.

    :ivar plan: The current plan. Mutated in place when a step is revised.
    :ivar step_log: One dict per step attempted — the step text, every attempt's
        termination reason, the judge's verdict and reasoning, the hints tried, and any
        revision. The whole record of what the supervisor did and why.
    """

    def __init__(
        self,
        *args: Any,
        plan_vlm_model: str = None,
        plan_vlm_kind: str = None,
        max_leg_steps: int = 5,
        max_attempts_per_step: int = 3,
        max_replans: int = 2,
        max_frames_per_slice: int = 8,
        plan_max_new_tokens: int = 2400,
        judge_max_new_tokens: int = 4800,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_step = max_attempts_per_step
        self.max_replans = max_replans
        self.n_replans = 0
        self.last_diagnosis: Optional[str] = None
        self._max_frames_per_slice = max_frames_per_slice
        self._plan_max_new_tokens = plan_max_new_tokens
        # The completion check gets the largest budget of the three. Its verdict line comes
        # last, after the reasoning, so it is the call that loses its answer first when a
        # reply is cut short — and losing it silently means nothing ever clears.
        self._judge_max_new_tokens = judge_max_new_tokens
        # Every supervisor VLM call, in order, as {stage, prompt, response}. The runner
        # writes it to the CSV so the supervisor's reasoning is as inspectable after the
        # fact as the executor's already is.
        self.supervisor_calls: List[dict] = []
        self._plan_vlm = RecordingVLM(
            self._hint_vlm if plan_vlm_model is None else VLM(plan_vlm_model, plan_vlm_kind),
            self.supervisor_calls,
        )
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
        # Every attempt's report, tagged with which step and attempt produced it, so the
        # runner can render one trajectory cell covering the whole episode. step_log carries
        # the supervisor's decisions; this carries what the executor actually did.
        self.leg_reports: List[dict] = []

    def _say(self, message: str) -> None:
        """Per-attempt commentary. Silent unless the caller asked for it.

        Off by default: a supervisor that narrates whether or not anyone wired up a flag
        makes a 35-episode sweep unreadable.
        """
        if self.verbose:
            print(message)

    def _call(self, stage: str, **kwargs):
        """Tag the recorded call with *stage*, then make it.

        ``RecordingVLM`` labels whatever it is asked to infer with whatever ``stage`` was
        last assigned, so setting the stage and calling were two separate statements at
        every one of these sites — and a site that forgot to set it would be logged under
        the *previous* stage, silently. ``supervisor_calls`` is the only record of this
        arm's reasoning, so a mislabelled call is a quietly corrupted artifact rather than
        a cosmetic problem. Binding the two together makes the mistake unavailable.
        """
        self._plan_vlm.stage = stage
        return self._plan_vlm.infer(**kwargs)

    # -- Planning --------------------------------------------------------------

    def write_plan(self) -> List[str]:
        """Select relevant entries for the opening screen and turn them into a plan.

        Reuses :meth:`InfoHintSupervisor.write_hint`'s selection half verbatim, then swaps
        the synthesis prompt. Returns ``[]`` when nothing was selected — with no knowledge
        there is nothing to plan from, and the caller falls back to running the task
        unplanned rather than inventing a plan out of the prompt alone.
        """
        self.selection_log = []
        self.selected_ids = []
        screen = self._current_frame()

        candidates = (self._init_state_candidates() if self._mode == "init_state"
                      else self._retrieval_candidates())
        selected = self._select_entries(candidates, screen)
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
        output = self._call("plan", texts=prompt, images=[screen],
                                      max_new_tokens=self._plan_max_new_tokens)
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
        output = self._call("filter_insights", texts=prompt, images=[screen],
                                      max_new_tokens=self._hint_max_new_tokens)
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
        output = self._call("distil_insights", texts=prompt, images=[screen],
                                      max_new_tokens=self._plan_max_new_tokens)

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
        # The one site that cannot use _call: the infer happens inside
        # summarise_trajectory_segments, which is handed the recording VLM and calls it
        # itself, so the stage has to be set on the object beforehand.
        self._plan_vlm.stage = "judge_slice"
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, step, self._plan_vlm,
            self._plan_max_new_tokens, self._max_frames_per_slice,
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
        output = self._call("judge", texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._judge_max_new_tokens)
        verdict = parse_completion(output)
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if verdict is None:
            # Truncation lands here, and it is not a harmless parse miss: `Complete:` is the
            # last line of the reply, so a response cut short loses the verdict while keeping
            # the reasoning, and an unparsed verdict reads as NOT complete. Every step then
            # fails its judgement no matter what the screen shows. Loud, because the symptom
            # — nothing ever clears — looks exactly like a bad plan.
            log_warn(f"[plan] completion check returned no 'Complete:' line within "
                     f"{self._judge_max_new_tokens} tokens; treating as not complete. "
                     f"Raise --judge_max_new_tokens if this repeats.", self._parameters)
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
        output = self._call("regression_check", 
            texts=prompt, images=[last["frame"], self._current_frame()],
            max_new_tokens=self._hint_max_new_tokens,
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
        output = self._call("hint", texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._hint_max_new_tokens)
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
        output = self._call("plan_flaw", texts=prompt, images=[self._current_frame()],
                                      max_new_tokens=self._plan_max_new_tokens)
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

    def evaluate(self) -> Any:
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

        :return: The report of the last attempt that ran, whose ``termination_reason``
            decides the episode.
        """
        # Cleared before write_plan so the log covers exactly this evaluate() call.
        self.supervisor_calls.clear()
        self._plan_vlm.context = {"phase": "planning"}
        self.write_plan()
        self.step_log = []
        self.leg_reports = []
        self.completed_steps = []
        self.n_replans = 0
        budget = self._max_steps
        last_report = None

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

                # Every supervisor call from here until the next leg — judging, hinting,
                # revising — belongs to this attempt, and is tagged so the replay can slot
                # them in after the executor calls they respond to.
                self._plan_vlm.context = {"phase": "leg", "step_index": index,
                                          "attempt": len(record["attempts"]) + 1}
                report = self._run_leg(leg_task, leg_hint, self_terminate, budget)
                last_report = report
                self.leg_reports.append({
                    "step_index": index, "attempt": len(record["attempts"]) + 1,
                    "step": step, "hint": leg_hint, "report": report,
                })
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

        return last_report
