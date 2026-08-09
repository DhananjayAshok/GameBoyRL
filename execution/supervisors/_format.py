"""
Rendering an executor attempt as text for a supervisor prompt.

Pure functions of ``(report, env_steps)`` — no policy, no VLM, no supervisor state. They
were methods on :class:`~execution.supervisors.info_plan.InfoPlanSupervisor` purely because
that is where they were written; none of them read ``self`` for anything but a call to
another function in this module.

Splitting them out leaves ``info_plan`` holding the three concerns that actually interact
— planning, judging, and the loop — rather than four.
"""

from __future__ import annotations

from typing import List, Optional

from execution.report import (ACTION_TAGS, DONE_CHECK_TAG, EnvironmentStepRecord,
                              says_complete)
from utils import parse_key_value


def action_trace(report, max_chars: int = 0) -> str:
    """Each action the executor took, beside the reasoning it gave for taking it.

    Already on the report — ``vlm_call_log`` holds every response verbatim, and each call
    holds the steps it produced — so this costs nothing and has simply never been read.

    Pairing the two is what makes a failure diagnosable. The button alone shows a run
    of identical presses; the reasoning beside it shows *why*, and the usual answer is
    that the executor believes something about the screen that is not true ("the cursor
    is now over the hand icon" while it is not). That is a perception failure, and no
    rewording of the step will fix it — which is precisely the judgement the reviser is
    being asked to make.

    *max_chars* is 0 (uncapped) by default. The belief that matters usually arrives
    mid-sentence — "the cursor is now over the hand icon, so pressing A selects it" —
    so a cap tends to remove exactly the clause the reader needs while leaving the part
    that says nothing. Longer prompts are the cheaper problem.

    Completion checks are shown too, indented under the action they judged. A check
    answering "no" three times in a row, with its reasoning, is a different failure from
    an executor that thinks it has already finished, and the reviser can only tell them
    apart if it sees both. They are not numbered — only actions are — so the numbering
    still counts steps taken.
    """
    def compress(text: str) -> str:
        text = " ".join((text or "").strip().split())
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return text

    lines, n_actions = [], 0
    for entry in report.vlm_call_log:
        if entry.tag == DONE_CHECK_TAG:
            verdict = "yes" if says_complete(entry.response) else "no"
            reason = compress(parse_key_value(entry.response, "Reasoning"))
            lines.append(f"            ↳ finished? {verdict}"
                         + (f" — {reason}" if reason else ""))
            continue
        if entry.tag not in ACTION_TAGS:
            continue
        reason = compress(parse_key_value(entry.response, "Reasoning"))
        # A call usually owns one step, but a planned sequence owns several. Each gets its
        # own numbered line so the count still reads as "actions taken", while the reasoning
        # is shown once, on the first — it was given once, for the whole sequence.
        for i, step in enumerate(entry.steps or [None]):
            if isinstance(step, EnvironmentStepRecord):
                try:
                    name = step.action_class.get_action_name(**step.kwargs)
                except Exception:
                    name = step.action_class.__name__
            elif step is None:
                name = "(no action)"
            else:
                name = "INVALID"
            n_actions += 1
            if i == 0:
                lines.append(f"         {n_actions}. {name} — {reason or '(no reasoning given)'}")
            else:
                lines.append(f"         {n_actions}. {name} — (same planned sequence)")
    return "\n".join(lines) if lines else "         (no actions taken)"


def attempt_history_line(report, env_steps: list, summaries: List[str],
                         hint: Optional[str], verdict: str,
                         regression: Optional[str] = None) -> str:
    """One attempt, described richly enough for the plan reviser to diagnose it.

    The summaries are already paid for — the completion check builds them from the frames
    and they were previously used once and dropped — and the button list costs nothing
    at all. Together they are what separates the two failures that a bare termination
    reason cannot: a step that is badly worded, and a step that is fine but that the
    executor cannot act on. The second looks like a run of identical buttons with
    nothing changing on screen, which is invisible unless the buttons are shown.
    """
    return (
        f"{report.termination_reason} after {len(env_steps)} step(s)\n"
        f"       hint given: {hint or '(none — the step text was the only instruction)'}\n"
        f"       what the player pressed, and why they said they pressed it:\n"
        f"{action_trace(report)}\n"
        f"       what visibly happened: "
        f"{' '.join(summaries) if summaries else '(nothing summarised)'}\n"
        + (f"       REGRESSION: {regression}\n" if regression else "")
        + f"       verdict: {verdict}"
    )
