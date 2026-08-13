"""
The retry engine: attempt a target, critique what happened, revise the hint, try again.

:class:`RevisingSupervisor` is the whole of the ``revision`` arm and the base of the two
that follow it. Its loop is built on one rule:

    **A list of targets, attempted in order. Each target is judged and retried until it
    clears or the budget ends — except the last, which only the environment can clear.**

Every dependency between the arms follows from that instead of being enforced anywhere:

- With ``_targets()`` returning ``[None]`` — this class — there is exactly one target, it
  is therefore the last, and nothing is VLM-judged. The environment decides success and
  the critique exists only to write a better hint.
- :class:`~execution.supervisors.subgoal.SubgoalSupervisor` overrides ``_targets()`` to
  return plan steps. Intermediate steps have no environment signal, so they are judged.
  *Subgoal cannot exist without revision* because it only supplies a target list; this is
  the loop that consumes one.
- :class:`~execution.supervisors.info_subgoal.InfoSubgoalSupervisor` overrides
  ``_knowledge()``. *It cannot exist without subgoal* because knowledge is spent writing a
  plan; with no plan there is nothing to spend it on but a single opening hint, which is
  what the retired ``InfoHintSupervisor`` did.

**Success is the environment's verdict in every arm.** The judge advances a plan; it never
decides the episode. That is what keeps all four arms comparable against the baseline.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from execution.report import EnvironmentStepRecord, ExecutorReport, parse_completion
from execution.supervisors._format import action_trace, attempt_history_line
from execution.supervisors.base import Supervisor
from execution.supervisors.checker import summarise_trajectory_segments
from execution.supervisors.prompts import (JUDGE_CONSOLIDATE_PROMPT, JUDGE_SLICE_PROMPT,
                                           RESUME_HINT_PROMPT)
from utils import parse_key_value


class RevisingSupervisor(Supervisor):
    """Run the task in short legs, critiquing and re-hinting after each failure.

    :param max_leg_steps: Env-step cap for one executor attempt. Internal to the
        supervisor: it decides how often the supervisor gets to look, not the episode
        budget.
    :param max_attempts_per_target: Failed attempts at one target before moving on. Applies
        to intermediate targets only — its purpose is to stop one unclearable step eating
        the episode by moving *on*, and the last target has nowhere to move on to, so it is
        bounded by the budget alone.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param verbose: Print the per-attempt narration. Off by default.

    :ivar step_log: One dict per target attempted — its text, every attempt's termination
        reason, the judge's verdict and reasoning, the hints tried, and any revision.
    """

    def __init__(self, *args, max_leg_steps: int = 5, max_attempts_per_target: int = 3,
                 max_frames_per_slice: int = 8, verbose: bool = False, **kwargs) -> None:
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_target = max_attempts_per_target
        self._max_frames_per_slice = max_frames_per_slice
        self.verbose = verbose
        self.step_log: List[dict] = []
        self.last_diagnosis: Optional[str] = None
        super().__init__(*args, **kwargs)

    # -- Hooks the arms above override ----------------------------------------

    def _targets(self) -> List[Optional[str]]:
        """What to attempt, in order. ``None`` means the benchmark task itself.

        One target here, so it is the last one, so it is never judged — which is exactly
        the revision arm.
        """
        return [None]

    def _knowledge(self) -> str:
        """The insights block interpolated into the hint and plan prompts.

        Empty for every arm without a document. The prompts take ``[INSIGHTS]`` and degrade
        to "(nothing recorded)", so the knowledge-free arms use the *same* prompts rather
        than a second family of them.
        """
        return ""

    def _judge_target(self, target: str, env_steps: list, stop_reason: str) -> Tuple:
        """Decide whether a non-final target was reached.

        Never called here: this class has one target and the last one is the environment's
        to clear. :class:`SubgoalSupervisor` supplies it.
        """
        raise NotImplementedError(
            "A supervisor with more than one target must implement _judge_target."
        )

    def _diagnose_failure(self, index: int, targets: List[Optional[str]],
                          history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        """Optionally replace the remaining targets after a failure. ``None`` to keep going."""
        return None

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        """Whether the attempt undid earlier progress. Only meaningful with several targets."""
        return None

    def _on_target_cleared(self, target: str) -> None:
        """Note that a target was reached. Only meaningful with several targets."""

    def _on_targets_replaced(self, targets: List[Optional[str]]) -> None:
        """The target list was spliced. Only meaningful for an arm that tracks a plan.

        Called with the list *after* splicing, not with the replacement, because the
        replacement is only the tail — an arm that stored it directly would lose every
        target already cleared.
        """

    # -- Shared machinery -----------------------------------------------------

    def _say(self, message: str) -> None:
        """Per-attempt commentary. Silent unless the caller asked for it."""
        if self.verbose:
            print(message)

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    def _segment_summaries(self, env_steps: list, target: str) -> List[str]:
        # The infer happens inside summarise_trajectory_segments, which batches the windows
        # itself, so it takes a recording caller rather than a VLM.
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, target,
            self._vlm_caller("judge_slice"),
            self._max_new_tokens, self._max_frames_per_slice,
        )

    def _run_leg(self, leg_task: str, hint: Optional[str], self_terminate: bool,
                 budget: int) -> ExecutorReport:
        """One executor attempt, capped at the smaller of the leg cap and what is left.

        ``self._max_steps`` is what :meth:`Supervisor.call_executor` hands the executor, so
        it has to be narrowed to this leg's cap and put back afterwards. Restored in a
        ``finally`` beside the write rather than once after the loop: an executor that
        raises would otherwise leave the supervisor with a single leg's budget standing in
        for the whole episode's.
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

    def _leg_spec(self, target: Optional[str], is_last: bool,
                  hint: Optional[str]) -> Tuple[str, Optional[str], bool]:
        """``(task, hint, allow_self_termination)`` for one attempt at *target*.

        An intermediate target becomes the executor's **task**, not its hint: a hint is
        advice about a goal the executor already holds, so a step delivered as a hint while
        the task still reads the benchmark's own asks the executor to declare victory on a
        means rather than an end. The last target reverts to the real task with
        self-termination off, so success stays the environment's call.
        """
        if target is None:
            return self._task, hint, False
        if is_last:
            return self._task, hint or target, False
        return target, hint, True

    def judge_step(self, env_steps: list, target: str, stop_reason: str) -> Tuple:
        """Windowed judgement of whether *target* was reached, against the current screen.

        The step's termination condition is a statement about what the screen shows, so the
        screen is what settles it. The executor's own stop reason is passed in as context
        and framed as a claim to be checked, since a leg that self-terminated is exactly
        the case where the agent's opinion and the screen may disagree.

        :return: ``(complete, reasoning, segment_summaries)``. An attempt with no env steps
            is never complete: nothing happened, so nothing can have been achieved.
        """
        if not env_steps:
            return False, "No environment steps were taken in this attempt.", []

        summaries = self._segment_summaries(env_steps, target)
        prompt = (
            JUDGE_CONSOLIDATE_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", target)
            .replace("[SEGMENT_SUMMARIES]", "\n".join(summaries))
            .replace("[STOP_REASON]", stop_reason or "unknown")
        )
        output = self._vlm_call("judge", texts=prompt, images=[self._current_frame()])
        verdict = parse_completion(output)
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        if verdict is None:
            # Truncation lands here, and it is not a harmless parse miss: `Complete:` is the
            # last line of the reply, so a response cut short loses the verdict while
            # keeping the reasoning, and an unparsed verdict reads as NOT complete. Every
            # step then fails its judgement no matter what the screen shows.
            self._log_truncated_verdict()
        return verdict is True, reasoning, summaries

    def _log_truncated_verdict(self) -> None:
        from utils import log_warn
        log_warn(f"[supervisor] completion check returned no 'Complete:' line within "
                 f"{self._max_new_tokens} tokens; treating as not complete. "
                 f"Raise --supervisor_max_new_tokens if this repeats.", self._parameters)

    def write_resume_hint(self, summaries: List[str], target: str, judgement: str,
                          previous_hint: str = "", report=None,
                          regression: Optional[str] = None) -> Optional[str]:
        """Write the hint for the next attempt, anchored on the current screen.

        *report* is the attempt that just failed. The executor's own reasoning goes in
        beside the summaries: the summaries say what changed on screen, the reasoning says
        what the executor believed while it was failing to change it. A hint written
        against the belief can correct it directly.

        Returns ``None`` when no hint could be written — a truncated reply, or the writer
        declining. **Callers keep the previous attempt's hint in that case**
        (``write_resume_hint(...) or hint``) rather than retrying unaided. A stale hint may
        no longer describe where the player is, which is a real cost; it is accepted
        because the alternative throws away the only guidance available at the exact moment
        the writer is already struggling.
        """
        prior_block = (f'Previous hint, which did not work (do not simply repeat it):\n'
                       f'"{previous_hint}"\n\n' if previous_hint else "")
        trace_block = (f"\n\nWhat they pressed, and the reason they gave for each:\n"
                       f"{action_trace(report)}" if report is not None else "")
        prompt = (
            RESUME_HINT_PROMPT
            .replace("[GAME]", self._game)
            .replace("[TASK]", target)
            .replace("[SEGMENT_SUMMARIES]",
                     ("\n".join(summaries) or "  (no actions taken)") + trace_block)
            .replace("[JUDGEMENT]", judgement or "the screen does not show the step complete")
            .replace("[REGRESSION_BLOCK]",
                     f"**{regression} Say what to do about that first — the player cannot "
                     f"finish this step from a state they have gone backwards into.**\n\n"
                     if regression else "")
            .replace("[INSIGHTS]", self._knowledge() or "(nothing recorded)")
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

    # -- The loop -------------------------------------------------------------

    def _evaluate(self) -> dict:
        """Attempt each target in turn until one lands the episode or the budget ends.

        The retry loop is bounded three ways, in increasing order of bluntness:
        :attr:`max_attempts_per_target` failures move on to the next target,
        :meth:`_diagnose_failure` may replace the remaining targets, and the env-step
        budget stops everything.
        """
        self.step_log = []
        budget = self._max_steps
        targets = self._before_targets()

        index = 0
        while index < len(targets):
            if budget <= 0:
                self._say(f"  step budget exhausted at target {index + 1}/{len(targets)}")
                break

            target = targets[index]
            is_last = index == len(targets) - 1
            record = {"step": target, "attempts": [], "replans": [], "cleared": False}
            hint: Optional[str] = None
            history: List[str] = []

            while budget > 0:
                leg_task, leg_hint, self_terminate = self._leg_spec(target, is_last, hint)

                self._say(f"\n  --- target {index + 1}/{len(targets)}"
                          f" attempt {len(record['attempts']) + 1}"
                          f"{' (final)' if is_last else ''}, {budget} steps left")
                if target is not None:
                    self._say(f"      {target}")
                if leg_hint and leg_hint != target:
                    self._say(f"      hint: {leg_hint}")

                # call_executor files the report into report.event_log, after the supervisor
                # calls that produced this leg's hint and before the ones that judge it.
                # That ordering is the record; no per-leg tagging is needed.
                report = self._run_leg(leg_task, leg_hint, self_terminate, budget)
                env_steps = [s for s in report.steps
                             if isinstance(s, EnvironmentStepRecord)]
                budget -= len(env_steps)
                attempt = {"termination_reason": report.termination_reason,
                           "n_steps": len(env_steps), "hint": leg_hint}
                self._say(f"      -> {report.termination_reason} after "
                          f"{len(env_steps)} step(s)")

                # The environment ending outranks every judgement: terminated is the
                # benchmark's own success signal, and truncated/max_invalid are states no
                # further hint can act on.
                if report.termination_reason in ("terminated", "truncated", "max_invalid"):
                    attempt["cleared"] = report.termination_reason == "terminated"
                    record["attempts"].append(attempt)
                    record["cleared"] = attempt["cleared"]
                    index = len(targets)          # leave the outer loop too
                    break

                if is_last:
                    # Not judged: the only thing that clears the last target is the
                    # environment terminating, which the branch above already caught.
                    attempt["cleared"] = False
                    record["attempts"].append(attempt)
                    summaries = (self._segment_summaries(env_steps, self._task)
                                 if env_steps else [])
                    regression = self._check_regression(summaries)
                    if regression:
                        attempt["regression"] = regression
                    verdict = "the environment did not signal the task complete"
                    history.append(attempt_history_line(
                        report, env_steps, summaries, leg_hint, verdict, regression))

                    replacement = self._diagnose_failure(index, targets, history, verdict,
                                                         regression)
                    if replacement:
                        record["replans"].append(replacement)
                        targets[index:] = replacement
                        self._on_targets_replaced(targets)
                        target = targets[index]
                        is_last = index == len(targets) - 1
                        record["step"] = target
                        hint = None
                        continue

                    hint = self.write_resume_hint(
                        summaries, self._task, verdict, hint or "",
                        report=report, regression=regression,
                    ) or hint
                    continue

                complete, reasoning, summaries = self._judge_target(
                    target, env_steps, report.termination_reason
                )
                attempt["cleared"] = complete
                attempt["judgement"] = reasoning
                record["attempts"].append(attempt)
                self._say(f"      judge: {'COMPLETE' if complete else 'NOT COMPLETE'} "
                          f"— {reasoning}")

                if complete:
                    record["cleared"] = True
                    self._on_target_cleared(target)
                    break

                # Before the hint is written, not after: a hint composed without knowing
                # progress was lost will push the player forward from a state that has
                # moved backwards.
                regression = self._check_regression(summaries)
                if regression:
                    attempt["regression"] = regression

                history.append(attempt_history_line(
                    report, env_steps, summaries, leg_hint,
                    f"judged incomplete — {reasoning}", regression))

                replacement = self._diagnose_failure(
                    index, targets, history, f"judged incomplete — {reasoning}", regression)
                if replacement:
                    record["replans"].append(replacement)
                    # Everything from here on is replaced; targets already cleared are not
                    # touched, so progress already made survives.
                    targets[index:] = replacement
                    self._on_targets_replaced(targets)
                    target = targets[index]
                    is_last = index == len(targets) - 1
                    record["step"] = target
                    hint = None
                    continue

                if len(record["attempts"]) >= self.max_attempts_per_target:
                    self._say(f"      target not cleared after {len(record['attempts'])} "
                              f"attempts; moving on.")
                    break

                hint = self.write_resume_hint(summaries, target, reasoning, hint or "",
                                              report=report, regression=regression) or hint
                if self.last_diagnosis:
                    attempt["diagnosis"] = self.last_diagnosis

            self.step_log.append(record)
            index += 1

        return self._extras()

    def _before_targets(self) -> List[Optional[str]]:
        """Resolve the target list once, before the loop. Hook for per-episode setup."""
        return list(self._targets())

    def _extras(self) -> dict:
        """Arm-specific values for the CSV. ``Supervisor.evaluate`` attaches the report."""
        return {"step_log": self.step_log}

    def process_executor_return(self, report: ExecutorReport) -> ExecutorReport:
        """Hand the report back unchanged — ``call_executor`` has already filed it.

        This arm reads each leg in :meth:`_evaluate`, where the target it belongs to is
        known, so nothing needs to be derived here.
        """
        return report
