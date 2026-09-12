"""
What the executor remembers between decisions, and how it reads in the prompt.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol

from gameboy_worlds.interface.action import LowLevelAction

from execution.report import (EnvironmentStepRecord, ExecutorToolCallRecord,
                              StepRecord, tool_call_string)

#: Recent steps rendered into the prompt. Applies to every policy that keeps a list: a
#: visual description is much longer per entry than an action name, so an untruncated
#: visual history blows the prompt long before an action history would.
DEFAULT_HISTORY_K = 5


class HistoryPolicy(Protocol):
    """The memory half of an executor."""

    #: Registry name, and half of the arm's ``<action>_<history>`` identity.
    name: str

    def reset(self) -> None:
        """Forget everything. Called once before the loop starts."""

    def observe(self, steps: List[StepRecord]) -> None:
        """Fold in the environment steps and tool calls one decision produced, in order."""

    def render(self) -> str:
        """The block to splice into ``[CONTEXT_SECTION]``, or ``""`` for nothing.

        Ends with a blank line when non-empty; it sits directly before the instruction.
        """


class NoHistoryPolicy:
    """Remember nothing. The reference behaviour."""

    name = "none"

    def __init__(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def observe(self, steps: List[StepRecord]) -> None:
        pass

    def render(self) -> str:
        return ""


def _action_line(record: StepRecord) -> str:
    """One past action, tagged with whether it did anything.
    """
    if isinstance(record, ExecutorToolCallRecord):
        return f"  {tool_call_string(record)} -> {record.result}"
    action_str = record.action_class.get_action_name(**record.kwargs)
    changed = record.frame_changed
    if issubclass(record.action_class, LowLevelAction):
        # LowLevelActions report success=0 by convention rather than as a failure signal,
        # so their status is never rendered — only whether the screen moved.
        return f"  {action_str}{'' if changed else ' [no change]'}"
    status = ("ok" if record.action_success == 1
              else "failed" if record.action_success == 0 else "unknown")
    return f"  {action_str}  [{status}{'' if changed else ', no change'}]"


STUCK_HINT = """
If you have been trying to execute the same action repeatedly 
(specifically A or B) and especially if you get the [no change] message on your recent actions, consider that you may be stuck in a loop, and should try something else. 
Look at the screen deeply and use the visual cues to guide your decision making. 
If you are trying to interact with something, you likely have the incorrect orientation and need to slightly adjust your positioning
"""


class ActionHistoryPolicy:
    """List the last *k* actions, tagged with whether the screen changed."""

    name = "actions"

    def __init__(self, history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        self._history_k = history_k
        self._records: List[StepRecord] = []

    def reset(self) -> None:
        self._records = []

    def observe(self, steps: List[StepRecord]) -> None:
        self._records.extend(steps)

    def render(self) -> str:
        if not self._records:
            return ""
        recent = self._records[-self._history_k:]
        lines = ["Recent actions (oldest first):"]
        lines += [_action_line(record) for record in recent]
        stuck = STUCK_HINT if any(isinstance(r, EnvironmentStepRecord)
                                  and not r.frame_changed for r in recent) else ""
        return "\n".join(lines) + stuck + "\n\n"


class VisualHistoryPolicy:
    """
    Describe what each action actually changed on screen, and remember that.
    """

    name = "visual"

    DIFF_PROMPT = """
You are watching someone play the GameBoy game [GAME].

Image 1 is the screen BEFORE they pressed [ACTION]. Image 2 is the screen AFTER.

In one short sentence, say what changed between the two screens as a result of that action. If nothing changed, say exactly: nothing changed.
[STOP]"""

    def __init__(self, call: Callable = None, game: str = "",
                 history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        self._call = call
        self._game = game
        self._history_k = history_k
        # (action string, what it changed)
        self._entries: List[tuple] = []

    def reset(self) -> None:
        self._entries = []

    def observe(self, steps: List[StepRecord]) -> None:
        if not steps:
            return

        # (action string, description) with description None for the entries still waiting
        # on a frame diff. Built in step order so a tool call keeps its place in the
        # sequence even though it does not go out to the VLM.
        pending: List[list] = []
        prompts, images = [], []
        for record in steps:
            if isinstance(record, ExecutorToolCallRecord):
                pending.append([tool_call_string(record), str(record.result)])
                continue
            # frame_after is the screen the *next* step starts from, so a pair is always
            # (before, after) of one action rather than two consecutive observations.
            if (self._call is None or record.frame_before is None
                    or record.frame_after is None):
                continue
            action_str = record.action_class.get_action_name(**record.kwargs)
            pending.append([action_str, None])
            prompts.append(self.DIFF_PROMPT
                           .replace("[GAME]", self._game)
                           .replace("[ACTION]", action_str))
            images.append([record.frame_before, record.frame_after])

        if not pending:
            return

        responses = iter(self._call("frame_diff", texts=prompts, images=images,
                                    max_new_tokens=60) if prompts else [])
        for entry in pending:
            if entry[1] is None:
                entry[1] = _first_sentence(next(responses, ""))
            self._entries.append((entry[0], entry[1]))

    def render(self) -> str:
        if not self._entries:
            return ""
        lines = ["What your recent actions did (oldest first):"]
        for action_str, change in self._entries[-self._history_k:]:
            lines.append(f"  {action_str} -> {change}")
        stuck = STUCK_HINT if any("nothing changed" in change.lower()
                                  for _, change in self._entries[-self._history_k:]) else ""
        return "\n".join(lines) + stuck + "\n\n"


def _first_sentence(response: str) -> str:
    """
    The description, without the model's preamble.
    """
    text = (response or "").strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "(no description)"


#: Pixel-difference floor below which a frame is treated as unchanged no matter what the
#: adaptive estimate says. Calibrated on hardware: on Red a genuine dialogue-text advance
#: measures ~1.1 mean absolute difference while its idle flicker measures ~0.26, so the
#: constant sits between them.
ABSOLUTE_FLOOR = 0.6

#: How far above the measured idle floor a difference must sit to count as a real change.
#: Prism's sprites animate every frame at ~0.2-1.5; genuine movement there measures 7-17,
#: so a 5x margin separates them. Measured on Prism after real movement, where the
#: quietest animation is 0.178 and the loudest 1.686.
FLOOR_MULTIPLE = 5.0

#: The estimated floor is never allowed above this. Two reasons, both found by test:
#: a stretch where every action works would otherwise pull the floor up until it masked
#: the very changes that raised it, and before any idle frame has been seen there is
#: nothing to estimate from. Sits above Prism's loudest animation (1.512) and below its
#: quietest genuine movement (6.685).
MAX_FLOOR = 4.0

#: Differences needed before the floor estimate is trusted. Below this the threshold is
#: :data:`ABSOLUTE_FLOOR` and **no action is ever called dead**. Pinning the warm-up high
#: instead looks safer and is not: Red advances dialogue in steps of ~0.5-1.6, so a high
#: warm-up threshold reads a normal conversation as three dead presses of A and tells the
#: model to stop pressing it. Being permissive early costs nothing, because the ban -- the
#: only part with teeth -- stays switched off until the floor is known.
MIN_SAMPLES = 6

#: Percentile of recent differences taken as the idle floor. The low end of the
#: distribution is what animation looks like; the median drifts upward as soon as the
#: agent starts making real progress.
FLOOR_PERCENTILE = 20

#: Consecutive dead repeats of one action before it is called out by name.
REPEATS_BEFORE_ESCAPE = 3

#: The movement actions.
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

#: Dead steps in a row before movement is judged blocked outright. Counted since the last
#: step that changed anything, not over a fixed window: a model that favours one or two
#: directions (a real run pressed UP 203 times and LEFT 9) may never try all four inside a
#: window, and waiting for it to means the message never fires at all.
BLOCKED_AFTER = 6

#: Recent differences kept for the idle-floor estimate.
FLOOR_WINDOW = 30


def _mean_abs_diff(before, after) -> Optional[float]:
    """Mean absolute pixel difference, or ``None`` if it cannot be computed.

    Cast to a signed type first. The frames are ``uint8``, and ``numpy`` subtraction on
    ``uint8`` wraps modulo 256, so a difference of -10 reads as 246 — which inflates small
    differences into large ones and biases every comparison toward "something changed".
    """
    try:
        import numpy as np
        return float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
    except Exception:  # noqa: BLE001 - a diagnostic must never end a run
        return None


class EscapeHistoryPolicy(ActionHistoryPolicy):
    """
    Action history that measures "no change" itself, and names a dead action out loud.

    Two defects in :class:`ActionHistoryPolicy` show up only on hardware, and only the
    first is a bug in the tag:

    **The environment's flag is not usable on every game.** ``core.frame_changed`` fires
    when any frame since the last step differs by more than 0.001 on a ``uint8`` image —
    roughly one fully-flipped pixel. Pokemon Red has no idle animation, so the flag is
    accurate there. Prism runs a Gen-2 engine whose sprites animate in place, so *every*
    frame differs and the flag is ``True`` unconditionally: across 2,170 history blocks in
    one run it never once reported no change, while the agent pressed a direction into a
    wall for its whole budget. The animation floor is game-specific — Red's is ~0.26 and
    Prism's is ~1.5, and Red's smallest *genuine* change is ~1.1, inside Prism's floor — so
    no fixed threshold separates them and the floor is estimated per run instead.

    **Being told does not stop the repetition.** On Red, where the flag *is* accurate, 87%
    of steps followed an action that did nothing and the model repeated that same action
    86% of the time, despite a paragraph telling it not to. So the dead action is named
    explicitly and singled out, rather than left to a general warning the model reads past.

    This is a separate arm (``*_escape``) rather than a change to ``actions``: the existing
    policy is what every published benchmark number was measured with.
    """

    name = "escape"

    def __init__(self, history_k: int = DEFAULT_HISTORY_K, **kwargs) -> None:
        super().__init__(history_k=history_k, **kwargs)
        self._diffs: List[float] = []
        self._changed: dict = {}

    def reset(self) -> None:
        super().reset()
        self._diffs = []
        self._changed = {}

    def _threshold(self) -> float:
        """How large a difference must be, given the idle animation seen so far."""
        if len(self._diffs) < MIN_SAMPLES:
            return ABSOLUTE_FLOOR
        ordered = sorted(self._diffs[-FLOOR_WINDOW:])
        floor = ordered[min(len(ordered) - 1, (len(ordered) * FLOOR_PERCENTILE) // 100)]
        return min(MAX_FLOOR, max(ABSOLUTE_FLOOR, FLOOR_MULTIPLE * floor))

    def observe(self, steps: List[StepRecord]) -> None:
        super().observe(steps)
        for record in steps:
            if not isinstance(record, EnvironmentStepRecord):
                continue
            diff = _mean_abs_diff(record.frame_before, record.frame_after)
            if diff is None:
                # Fall back to the environment's own flag rather than guessing.
                self._changed[id(record)] = record.frame_changed
                continue
            # The threshold is taken BEFORE this difference joins the window.
            self._changed[id(record)] = diff > self._threshold()
            self._diffs.append(diff)

    def changed(self, record: StepRecord) -> bool:
        """Whether this step actually moved the screen, by our own measurement."""
        return self._changed.get(id(record), getattr(record, "frame_changed", True))

    def _stuck_span(self) -> List[StepRecord]:
        """The run of environment steps since the last one that changed the screen."""
        span = []
        for record in reversed(self._records):
            if not isinstance(record, EnvironmentStepRecord):
                continue
            if self.changed(record):
                break
            span.append(record)
        return span

    def _movement_blocked(self) -> bool:
        """Whether the pad is inert: a long dead run of directions, with A and B untried.

        A wall blocks one direction. When several directions are dead and nothing has
        changed for a while, the cause is usually not spatial: a dialogue box, sign or menu
        is open and movement is disabled until it is dismissed. Telling the model to "go
        around the obstacle" then is worse than silence, because the only actions that help
        -- A and B -- are the two it is being steered away from.

        Found on hardware: a Gemma 3 run opened a town sign on Pokemon Brown and spent its
        whole 2,263-step budget cycling directions against the open box, taking 464
        single-direction bans and pressing A exactly once. One screen, zero effective steps.

        Deliberately does NOT require all four directions to have been tried -- the first
        attempt at this fix did, and never fired, because that run pressed UP 203 times and
        LEFT 9. Two distinct dead directions separate a blocked pad from a single wall.
        """
        if len(self._diffs) < MIN_SAMPLES:
            return False
        span = self._stuck_span()
        if len(span) < BLOCKED_AFTER:
            return False
        names = {r.action_class.get_action_name(**r.kwargs) for r in span}
        if not names <= set(DIRECTIONS):
            # A, B or START already tried in this dead run: a text box is not the story.
            return False
        return len(names) >= 2

    def _dead_action(self) -> Optional[str]:
        """The action repeated to no effect, if there is one to call out."""
        # Never accuse an action while the floor is still a guess -- see MIN_SAMPLES.
        if len(self._diffs) < MIN_SAMPLES:
            return None
        # When nothing moves at all, banning one direction just rotates the model onto the
        # next dead one. :meth:`_movement_blocked` speaks to that case instead.
        if self._movement_blocked():
            return None
        steps = [r for r in self._records if isinstance(r, EnvironmentStepRecord)]
        if len(steps) < REPEATS_BEFORE_ESCAPE:
            return None
        name = steps[-1].action_class.get_action_name(**steps[-1].kwargs)
        streak = 0
        for record in reversed(steps):
            if record.action_class.get_action_name(**record.kwargs) != name:
                break
            if self.changed(record):
                break
            streak += 1
        return name if streak >= REPEATS_BEFORE_ESCAPE else None

    def render(self) -> str:
        if not self._records:
            return ""
        recent = self._records[-self._history_k:]
        lines = ["Recent actions (oldest first):"]
        for record in recent:
            if isinstance(record, ExecutorToolCallRecord):
                lines.append(f"  {tool_call_string(record)} -> {record.result}")
            else:
                action = record.action_class.get_action_name(**record.kwargs)
                lines.append(f"  {action}" + ("" if self.changed(record) else " [no change]"))

        if self._movement_blocked():
            return "\n".join(lines) + (
                "\n\nSTOP. Several different directions have done nothing at all. This is "
                "NOT a wall - when movement itself is dead, a dialogue box, a sign, a menu "
                "or a battle prompt is open on screen and the arrow keys are disabled until "
                "it is cleared.\n"
                "Do NOT answer a direction this step. Press A to advance or confirm what is "
                "on screen, or B to close it. Look at the bottom of the screen for a text box."
            ) + "\n\n"

        dead = self._dead_action()
        if dead is not None:
            streak = sum(1 for r in self._records
                         if isinstance(r, EnvironmentStepRecord)
                         and r.action_class.get_action_name(**r.kwargs) == dead
                         and not self.changed(r))
            tail = (f"\n\nSTOP. You have pressed {dead} {streak} times without the screen "
                    f"changing at all. {dead} is BLOCKED - a wall, a counter or a ledge is "
                    f"in that direction, and pressing it again will do nothing.\n"
                    f"Do NOT answer {dead} this step. Pick a DIFFERENT action and go around "
                    f"the obstacle. If you are trying to talk to somebody or use something, "
                    f"you are facing the wrong way: step to one side and approach it from "
                    f"another direction.")
        else:
            tail = STUCK_HINT if any(isinstance(r, EnvironmentStepRecord)
                                     and not self.changed(r) for r in recent) else ""
        return "\n".join(lines) + tail + "\n\n"


AVAILABLE_HISTORY_POLICIES = {
    policy.name: policy
    for policy in (NoHistoryPolicy, ActionHistoryPolicy, VisualHistoryPolicy,
                   EscapeHistoryPolicy)
}
""" History policies by registry name — the second half of an arm's identity. """
