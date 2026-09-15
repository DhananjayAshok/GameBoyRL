"""
Repairing the ways a model mis-renders an advertised action, without touching the executor.

A controller advertises its actions as a format string with placeholders --
``move(<up, down, right or left> <steps: int>)`` on the Pokemon state_wise controller --
and a model filling that in keeps the brackets, drops the parentheses, or capitalises the
verb. Every one of these was produced on hardware by a model that had chosen the RIGHT
action and written it in the wrong shape:

    move(<up> 1)        brackets copied out of the advertised format
    move(<right>, 1)    brackets, plus a comma between arguments
    Move right 1        prose: capitalised, no parentheses
    Interact            prose: no parentheses and no arguments
    openmenu<trainer>   brackets used *as* the parentheses

One 407-step run lost 19.4% of its steps to them, which at four consecutive failures ends
a leg on ``max_invalid``. It is the same failure that made ``ScoredActionPolicy`` unusable
(it wrote ``Arrow Keys (DOWN)`` for ``DOWN``): whatever syntax appears in the prompt ends
up in the action field.

The repair belongs beside the parser that owns the format -- the controller's own
``string_to_high_level_action`` -- and not in the shared executor, which should not know
about any particular controller's placeholder syntax. Until it lives there, this wraps the
environment the strategist hands to its executor, so the strategist's runs are not
throwing away a fifth of their steps and no shared code changes. The wrapper delegates
everything else untouched, so removing it later is a one-line change at the call site.
"""

from __future__ import annotations

import re
from typing import Any

#: Angle-bracket placeholders as they appear in an advertised action format.
_PLACEHOLDER_BRACKETS = re.compile(r"[<>]")


def advertised_verbs(action_strings) -> set:
    """The call names a controller advertises, e.g. ``{"move", "interact", "openmenu"}``.

    Read off the advertised strings rather than hardcoded, so this follows whatever the
    controller offers for the current game and state -- the Pokemon state_wise controller
    advertises different actions in free roam, in dialogue and in battle.
    """
    verbs = set()
    for text in (action_strings or {}).values():
        head = str(text).split("(", 1)[0].strip().lower()
        if head and head.isidentifier():
            verbs.add(head)
    return verbs


def normalise_action_string(action_str: str, verbs=()) -> str:
    """Repair a mis-rendered call, and change nothing else.

    An action whose verb the controller does not advertise is returned untouched, so a
    genuinely unrecognised action still fails and is reported rather than silently
    rewritten into something the agent did not ask for. Low-level controllers advertise no
    verbs at all, so their bare tokens (``UP``, ``A``) pass straight through.
    """
    text = _PLACEHOLDER_BRACKETS.sub(" ", action_str or "").strip()
    if not text:
        return ""
    verbs = set(verbs)

    if "(" in text:
        head, _, rest = text.partition("(")
        head = head.strip().lower()
        if head not in verbs:
            return (action_str or "").strip()
        args = " ".join(rest.rsplit(")", 1)[0].replace(",", " ").split())
        return f"{head}({args})"

    parts = text.split()
    head = parts[0].strip().lower()
    if head not in verbs:
        return (action_str or "").strip()
    args = " ".join(" ".join(parts[1:]).replace(",", " ").split())
    return f"{head}({args})"


class ActionRepairingEnvironment:
    """An environment that repairs mis-rendered action strings before parsing them.

    Wraps rather than subclasses: the environment classes are built by a registry with
    game-specific parts, so a proxy is the only way to add this without either editing
    them or reimplementing their construction. Every attribute except
    ``string_to_high_level_action`` is delegated verbatim, including the ones the executor
    reaches through (``get_info``, ``step_high_level_action``, ``get_action_strings``).
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def string_to_high_level_action(self, action_str: str):
        """Parse *action_str*, repairing the renderings listed in this module's docstring."""
        try:
            verbs = advertised_verbs(self._environment.get_action_strings())
        except Exception:  # noqa: BLE001 - a repair that cannot read the format must not
            verbs = set()  # break parsing; an empty verb set makes this a passthrough
        return self._environment.string_to_high_level_action(
            normalise_action_string(action_str, verbs))

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class does not define, so the method above is
        # never shadowed by it.
        return getattr(self._environment, name)
