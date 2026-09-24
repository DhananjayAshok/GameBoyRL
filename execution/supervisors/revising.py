"""
The retry engine: attempt a target, critique what happened, revise the hint, try again.

:class:`RevisingSupervisor` is the whole of the ``revision`` arm and the base of the two
that follow it. Its loop: a list of targets, attempted in order; each target is judged and
retried until it clears, runs out of attempts, or the budget ends — except the last, which
only the environment can clear.

- ``_resolve_targets()`` returning ``[None]`` — this class — gives one target, which is
  therefore the last, and nothing is VLM-judged.
- :class:`~execution.supervisors.subgoal.SubgoalSupervisor` overrides
  ``_resolve_targets()`` to return plan steps, which are judged.
- :class:`~execution.supervisors.info_subgoal.InfoSubgoalSupervisor` overrides
  ``_knowledge()``.

Success is the environment's verdict in every arm; the judge only advances a plan.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from execution.report import EnvironmentStepRecord, ExecutorReport, parse_completion
from execution.supervisors._format import action_trace, attempt_history_line
from execution.supervisors.base import Supervisor
from execution.supervisors.checker import summarise_trajectory_segments
from execution.supervisors.prompts import (JUDGE_CONSOLIDATE_PROMPT, JUDGE_SLICE_PROMPT,
                                           RESUME_HINT_PROMPT)
from utils import log_info, parse_key_value


class RevisingSupervisor(Supervisor):
    """Run the task in short legs, critiquing and re-hinting after each failure.

    :param max_leg_steps: Env-step cap for one executor attempt.
    :param max_attempts_per_target: Failed attempts at one *intermediate* target before
        moving on. Counted against the target as it currently reads, so a replan starts the
        count fresh.
    :param final_attempt_multiplier: The last target's cap, as a multiple of
        *max_attempts_per_target*.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param max_history_attempts: How many past attempts at the current target are shown to
        :meth:`_diagnose_failure`. Defaults to *max_attempts_per_target*.
    :param verbose: Log the per-attempt narration. Off by default.

    :ivar step_log: One dict per target attempted — its text, every attempt's termination
        reason, the judge's verdict and reasoning, the hints tried, and any revision.
    """

    def __init__(self, *args, max_leg_steps: int = 5, max_attempts_per_target: int = 3,
                 final_attempt_multiplier: int = 3, max_frames_per_slice: int = 8,
                 max_history_attempts: Optional[int] = None, verbose: bool = False,
                 **kwargs) -> None:
        self.max_leg_steps = max_leg_steps
        self.max_attempts_per_target = max_attempts_per_target
        self.final_attempt_multiplier = final_attempt_multiplier
        self._max_frames_per_slice = max_frames_per_slice
        self.max_history_attempts = (max_attempts_per_target if max_history_attempts is None
                                     else max_history_attempts)
        self.verbose = verbose
        self.step_log: List[dict] = []
        self.last_diagnosis: Optional[str] = None
        super().__init__(*args, **kwargs)

    @property
    def max_final_attempts(self) -> int:
        """Failed attempts at the *last* target before the episode gives up.

        ``final_attempt_multiplier * max_attempts_per_target`` is the floor; the cap is
        never allowed below the episode's own step budget.

        :return: The attempt cap for the last target.
        :rtype: int
        """
        floor = self.max_attempts_per_target * self.final_attempt_multiplier
        return max(floor, self._max_steps)

    def _run_config(self) -> dict:
        return {**super()._run_config(),
                "max_leg_steps": self.max_leg_steps,
                "max_attempts_per_target": self.max_attempts_per_target,
                "max_final_attempts": self.max_final_attempts,
                "max_frames_per_slice": self._max_frames_per_slice,
                "max_history_attempts": self.max_history_attempts}

    # -- Hooks the arms above override ----------------------------------------

    def _resolve_targets(self) -> List[Optional[str]]:
        """What to attempt, in order, plus any per-episode setup that decides it.

        ``None`` means the benchmark task itself.

        :return: The targets to attempt, in order.
        :rtype: List[Optional[str]]
        """
        return [None]

    def _knowledge(self) -> str:
        """The insights block interpolated into the hint and plan prompts. Empty for every
        arm without a document.

        :return: The insights block, or ``""``.
        :rtype: str
        """
        return ""

    def _diagnose_failure(self, index: int, targets: List[Optional[str]],
                          history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        """Optionally replace the remaining targets after a failure. ``None`` to keep going.

        *history* holds the failed attempts at the target as it currently reads.

        :return: Replacement targets, or ``None`` to keep the current list.
        :rtype: Optional[List[str]]
        """
        return None

    def _check_regression(self, summaries: List[str]) -> Optional[str]:
        """Whether the attempt undid earlier progress. Only meaningful with several targets."""
        return None

    def _on_target_cleared(self, target: str) -> None:
        """Note that a target was reached. Only meaningful with several targets."""

    def _on_targets_replaced(self, targets: List[Optional[str]]) -> None:
        """The target list was spliced. Only meaningful for an arm that tracks a plan.

        Called with the list *after* splicing, not with the replacement tail.
        """

    # -- Shared machinery -----------------------------------------------------

    def _say(self, message: str) -> None:
        """Per-attempt commentary. Silent unless the caller asked for it."""
        if self.verbose:
            log_info(message, self._parameters)

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    def _segment_summaries(self, env_steps: list, target: str) -> List[str]:
        return summarise_trajectory_segments(
            env_steps, JUDGE_SLICE_PROMPT, self._game, target,
            self._vlm_caller("judge_slice"),
            self._max_new_tokens, self._max_frames_per_slice,
        )

    def _run_leg(self, leg_task: str, hint: Optional[str], self_terminate: bool,
                 budget: int) -> ExecutorReport:
        """One executor attempt, capped at the smaller of the leg cap and what is left.

        :return: The attempt's executor report.
        :rtype: ExecutorReport
        """
        return self.call_executor(
            leg_task,
            hint=hint,
            allow_self_termination=self_terminate,
            max_steps=min(self.max_leg_steps, budget),
        )

    @staticmethod
    def _budget_spent(report: ExecutorReport) -> int:
        """What one leg cost the episode's step budget: every recorded step, not just the
        ones that reached the environment. Floored at 1 so the budget strictly decreases.

        :return: Steps to charge the budget for this leg.
        :rtype: int
        """
        return max(1, len(report.steps))

    def _leg_spec(self, target: Optional[str], is_last: bool,
                  hint: Optional[str]) -> Tuple[str, Optional[str], bool]:
        """``(task, hint, allow_self_termination)`` for one attempt at *target*.

        An intermediate target becomes the executor's task, not its hint. The last target
        reverts to the real task with self-termination off.

        :return: ``(task, hint, allow_self_termination)``.
        :rtype: Tuple[str, Optional[str], bool]
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

        *report* is the attempt that just failed. Callers keep the previous attempt's hint
        when this returns ``None`` (``write_resume_hint(...) or hint``).

        :return: The hint for the next attempt, or ``None`` if none could be written.
        :rtype: Optional[str]
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
        self.last_diagnosis = (parse_key_value(output, "Diagnosis") or "").strip() or None
        hint = (parse_key_value(output, "Hint") or "").strip()
        if self.last_diagnosis:
            self._say(f"      diagnosis: {self.last_diagnosis}")
        return hint or None

    # -- The loop -------------------------------------------------------------

    def _handle_failed_attempt(self, *, index: int, targets: List[Optional[str]],
                               history: List[str], attempt: dict, report,
                               env_steps: list, summaries: List[str], verdict: str,
                               hint_target: str, hint_reason: str, hint: Optional[str],
                               attempts_at_target: int, attempt_cap: int) -> Tuple:
        """Everything that happens after an attempt fails, whichever target it was.

        Regression-check, record the attempt in the failure history, ask whether the plan
        itself is at fault, and — if it is not and there are attempts left — write the hint
        for the next try. The cap is tested after the replan check and before the hint is
        written.

        :param verdict: How this attempt is described in the failure history and to the
            replanner.
        :param hint_target: What the next attempt is being advised about. Not the same as
            the target: the last target reverts to the benchmark task.
        :param hint_reason: The reason given to the hint writer, which is the judge's
            reasoning where there is one.
        :return: ``(replacement, hint, exhausted)`` — replacement targets if the plan was
            judged flawed, the hint to run the next attempt under, and whether this target
            has run out of attempts.
        """
        regression = self._check_regression(summaries)
        if regression:
            attempt["regression"] = regression

        history.append(attempt_history_line(
            report, env_steps, summaries, attempt["hint"], verdict, regression))
        if self.max_history_attempts > 0:
            del history[:-self.max_history_attempts]
        else:
            history.clear()

        replacement = self._diagnose_failure(index, targets, history, verdict, regression)
        if replacement:
            return replacement, hint, False

        if attempts_at_target >= attempt_cap:
            return None, hint, True

        hint = self.write_resume_hint(summaries, hint_target, hint_reason, hint or "",
                                      report=report, regression=regression) or hint
        if self.last_diagnosis:
            attempt["diagnosis"] = self.last_diagnosis
        return None, hint, False

    def _evaluate(self) -> dict:
        """Attempt each target in turn until one lands the episode or the budget ends.

        The retry loop is bounded three ways, in increasing order of bluntness: an attempt
        cap moves on to the next target (:attr:`max_attempts_per_target`, or
        :attr:`max_final_attempts` for the last one), :meth:`_diagnose_failure` may replace
        the remaining targets, and the step budget stops everything.

        :return: The episode record.
        :rtype: dict
        """
        self.step_log = []
        budget = self._max_steps
        targets = self._resolve_targets()

        index = 0
        while index < len(targets):
            if budget <= 0:
                self._say(f"  step budget exhausted at target {index + 1}/{len(targets)}")
                break

            target = targets[index]
            is_last = index == len(targets) - 1
            record = {"step": target, "attempts": [], "replans": [], "cleared": False}
            # All three describe the target *as it currently reads*, so all three are reset
            # when a replan changes what that is. ``record`` is not: it is the audit trail
            # for this slot and has to hold everything attempted in it, replan included.
            hint: Optional[str] = None
            history: List[str] = []
            attempts_at_target = 0

            while budget > 0:
                leg_task, leg_hint, self_terminate = self._leg_spec(target, is_last, hint)
                attempts_at_target += 1

                self._say(f"\n  --- target {index + 1}/{len(targets)}"
                          f" attempt {attempts_at_target}"
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
                budget -= self._budget_spent(report)
                attempt = {"termination_reason": report.termination_reason,
                           "n_steps": len(env_steps), "hint": leg_hint}
                self._say(f"      -> {report.termination_reason} after "
                          f"{len(env_steps)} step(s)")

                # The environment ending outranks every judgement. `max_invalid` is NOT
                # here: it is re-hintable, not terminal.
                if report.termination_reason in ("terminated", "truncated"):
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
                    verdict = "the environment did not signal the task complete"
                    hint_target, hint_reason = self._task, verdict
                    attempt_cap = self.max_final_attempts
                else:
                    complete, reasoning, summaries = self.judge_step(
                        env_steps, target, report.termination_reason
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
                    verdict = f"judged incomplete — {reasoning}"
                    hint_target, hint_reason = target, reasoning
                    attempt_cap = self.max_attempts_per_target

                replacement, hint, exhausted = self._handle_failed_attempt(
                    index=index, targets=targets, history=history, attempt=attempt,
                    report=report, env_steps=env_steps, summaries=summaries,
                    verdict=verdict, hint_target=hint_target, hint_reason=hint_reason,
                    hint=hint, attempts_at_target=attempts_at_target,
                    attempt_cap=attempt_cap,
                )

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
                    history = []
                    attempts_at_target = 0
                    continue

                if exhausted:
                    self._say(f"      target not cleared after {attempts_at_target} "
                              f"attempts; {'giving up' if is_last else 'moving on'}.")
                    break

            self.step_log.append(record)
            index += 1

        return self._extras()

    def _extras(self) -> dict:
        """Arm-specific values for the CSV. ``Supervisor.evaluate`` attaches the report."""
        return {"step_log": self.step_log}

    def process_executor_return(self, report: ExecutorReport) -> ExecutorReport:
        """Hand the report back unchanged — ``call_executor`` has already filed it.

        This arm reads each leg in :meth:`_evaluate`, where the target it belongs to is
        known, so nothing needs to be derived here.
        """
        return report
