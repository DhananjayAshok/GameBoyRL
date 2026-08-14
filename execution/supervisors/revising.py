"""
The retry engine: attempt a target, critique what happened, revise the hint, try again.

:class:`RevisingSupervisor` is the whole of the ``revision`` arm and the base of the two
that follow it. Its loop is built on one rule:

    **A list of targets, attempted in order. Each target is judged and retried until it
    clears, runs out of attempts, or the budget ends — except the last, which only the
    environment can clear.**

Every dependency between the arms follows from that instead of being enforced anywhere:

- With ``_resolve_targets()`` returning ``[None]`` — this class — there is exactly one
  target, it is therefore the last, and nothing is VLM-judged. The environment decides
  success and the critique exists only to write a better hint.
- :class:`~execution.supervisors.subgoal.SubgoalSupervisor` overrides
  ``_resolve_targets()`` to return plan steps. Intermediate steps have no environment
  signal, so they are judged. *Subgoal cannot exist without revision* because it only
  supplies a target list; this is the loop that consumes one. That hook both writes the
  plan and returns it, because the list depends on per-episode work that has to happen
  before there is anything to return.
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
    :param max_attempts_per_target: Failed attempts at one *intermediate* target before
        moving on. Its purpose is to stop one unclearable step eating the episode by moving
        *on*. Counted against the target *as it currently reads*: a replan makes this a
        different step, so it starts its attempts fresh rather than inheriting the count
        that triggered the replan.
    :param final_attempt_multiplier: The last target's cap, as a multiple of
        *max_attempts_per_target*. The last target has nowhere to move on to, so it used to
        be documented as "bounded by the budget alone" — and was not bounded at all when a
        leg spent its whole allowance on replies that never reached the environment, since
        the budget then did not move. It is generous rather than equal because moving on
        from the last target ends the episode, which is a much worse outcome than one wasted
        retry.
    :param max_frames_per_slice: Trajectory frames per judging call.
    :param max_history_attempts: How many past attempts at the current target are shown to
        :meth:`_diagnose_failure`. The last target can now be attempted many times over, and
        the history was previously unbounded — one ``attempt_history_line`` per attempt,
        each carrying an uncapped action trace — so the replanner's prompt grew linearly for
        the whole back half of an episode. Kept at the intermediate cap by default, since
        that is how many attempts the prompt's "attempt 1…n" rendering was written for.
    :param verbose: Print the per-attempt narration. Off by default.

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

        **A backstop, not a second budget.** ``final_attempt_multiplier *
        max_attempts_per_target`` is the floor, but the cap is never allowed below the
        episode's own step budget, because :meth:`_budget_spent` charges at least 1 per leg
        and so the budget can never permit more legs than that.

        Without the ``max`` this bound *binds first on a healthy run* and quietly truncates
        the episode: at the defaults (200 steps, 5-step legs, 3 attempts per target) the
        floor is 9, so the arm would stop after 9 legs having spent 45 of its 200 steps,
        while the baseline arm spends all 200. That is the same arms-are-not-comparable
        problem :meth:`_budget_spent` exists to fix, in the opposite direction — and much
        harder to notice, because every episode still looks like it ran normally.
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

        ``None`` means the benchmark task itself. One target here, so it is the last one, so
        it is never judged — which is exactly the revision arm.

        This is the *single* hook for the question. There used to be two — ``_targets()``
        for the list and ``_before_targets()`` for the setup — and no subclass ever
        overrode the first, because the list always depends on work (writing a plan,
        selecting knowledge) that has to happen in the same call.
        """
        return [None]

    def _knowledge(self) -> str:
        """The insights block interpolated into the hint and plan prompts.

        Empty for every arm without a document. The prompts take ``[INSIGHTS]`` and degrade
        to "(nothing recorded)", so the knowledge-free arms use the *same* prompts rather
        than a second family of them.
        """
        return ""

    def _diagnose_failure(self, index: int, targets: List[Optional[str]],
                          history: List[str], judgement: str,
                          regression: Optional[str]) -> Optional[List[str]]:
        """Optionally replace the remaining targets after a failure. ``None`` to keep going.

        *history* holds the failed attempts at the target **as it currently reads**, so a
        replaced target is diagnosed from its own record rather than from the one that
        provoked the replan. Otherwise the first failure of a fresh step arrives carrying
        the previous step's attempts, and a caller that reads "this has failed three times"
        would be reading three failures of something else.
        """
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

        All three per-leg settings go through :meth:`Supervisor.call_executor`'s parameters.
        They used to be written onto the supervisor first — ``_executor_kwargs["hint"]``,
        and a temporary overwrite of ``self._max_steps`` restored in a ``finally`` — which
        meant the episode's budget attribute briefly held a single leg's, and nothing the
        executor recorded said what it had run under.
        """
        return self.call_executor(
            leg_task,
            hint=hint,
            allow_self_termination=self_terminate,
            max_steps=min(self.max_leg_steps, budget),
        )

    @staticmethod
    def _budget_spent(report: ExecutorReport) -> int:
        """What one leg cost the episode's step budget.

        **Every recorded step, not just the ones that reached the environment.** This is
        the same quantity the executor's own ``n_env_steps`` counter tracks — it increments
        on a parse failure, an unrecognised action and a passive tool call as well as on a
        real action — so an episode's ``--max_steps`` now means the same thing here as it
        does in the baseline arm, where one executor is handed the whole budget directly.

        Counting only ``EnvironmentStepRecord``s, as this used to, had two consequences. The
        supervised arms silently got more emulator interaction than the baseline for the
        same ``--max_steps``, so the arms were not comparable on a run where the model
        emitted unparseable actions. And a leg that spent its entire allowance on invalid
        replies returned *zero* env steps, so the budget did not move — which, on the last
        target, meant a loop with nothing left to bound it.

        The floor of 1 is belt and braces for that second failure mode: the executor's loop
        records at least one step per iteration and cannot return an empty report today, but
        a leg that somehow cost nothing must still not be free, or the caller's "budget
        strictly decreases" guarantee rests on the executor's internals.
        """
        return max(1, len(report.steps))

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

    def _handle_failed_attempt(self, *, index: int, targets: List[Optional[str]],
                               history: List[str], attempt: dict, report,
                               env_steps: list, summaries: List[str], verdict: str,
                               hint_target: str, hint_reason: str, hint: Optional[str],
                               attempts_at_target: int, attempt_cap: int) -> Tuple:
        """Everything that happens after an attempt fails, whichever target it was.

        Regression-check, record the attempt in the failure history, ask whether the plan
        itself is at fault, and — if it is not and there are attempts left — write the hint
        for the next try.

        This was written out twice, once per branch of :meth:`_evaluate`, and the two copies
        had already drifted: only the intermediate one copied :attr:`last_diagnosis` onto the
        attempt, so the revision arm (whose every attempt is a final-target attempt) never
        recorded a diagnosis at all despite going to some trouble to produce one. The two
        callers now differ only in what they *produce* — a judged verdict or a fixed one —
        and in the value of *attempt_cap*.

        The cap is tested **after** the replan check and **before** the hint is written.
        Both halves of that matter: a failure may mean the plan is wrong rather than the
        attempt, which is worth asking even on the last try at a target; and a hint for an
        attempt that will never be made is a VLM call spent on nothing.

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
        # Before the hint is written, not after: a hint composed without knowing progress
        # was lost will push the player forward from a state that has moved backwards.
        regression = self._check_regression(summaries)
        if regression:
            attempt["regression"] = regression

        history.append(attempt_history_line(
            report, env_steps, summaries, attempt["hint"], verdict, regression))
        # Bounded, because the last target can now be attempted many times over and each
        # line carries a full uncapped action trace. The replanner's prompt renders these as
        # "attempt 1…n" and is written for a handful, not for the whole back half of an
        # episode.
        #
        # Not `del history[:-n]`: at n == 0 that is `del history[:0]`, which deletes
        # *nothing* rather than everything, so the one setting that reads as "show no
        # history" would silently restore the unbounded growth this line exists to stop.
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

        **Every one of those bounds has to be able to fire.** The last target used to have
        no attempt cap at all, on the reasoning that the budget alone would stop it — but
        the budget only moved when a leg reached the environment, so a leg that spent its
        whole allowance on unparseable replies left the loop with nothing decreasing.
        :meth:`_budget_spent` fixes the accounting and the cap is the second line of
        defence.
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

                # The environment ending outranks every judgement: terminated is the
                # benchmark's own success signal, and truncated is a state no further hint
                # can act on. `max_invalid` is deliberately NOT here — that is the acting
                # model failing to format a reply, which is the ordinary failure this whole
                # class exists to re-hint its way out of, and ending the episode on it threw
                # away the remaining budget for a reason a new hint can address.
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
