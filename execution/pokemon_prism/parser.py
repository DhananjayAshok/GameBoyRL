"""
PrismActionParser — converts VLM action strings to (ActionClass, kwargs).

Extends the state_wise semantic vocabulary with low-level single-step
button aliases so the VLM can emit precise, one-frame-at-a-time inputs
alongside the existing semantic shortcuts.

Supported surface forms
-----------------------
Low-level single-step (no parens required):
  up / down / left / right            → MoveStepsAction(direction=X, steps=1)
  n / s / e / w / north / south ...   → same cardinal aliases
  a / press_a                         → A_BUTTON sentinel (executor resolves)
  b / press_b                         → B_BUTTON sentinel (executor resolves)

Semantic multi-step (parens required):
  move(down 3)                        → MoveStepsAction(direction="down", steps=3)
  interact()                          → InteractAction
  passdialogue()                      → PassDialogueAction
  openmenu(pokemon|bag|trainer)       → OpenMenuAction
  battlemenu(fight|pokemon|bag|run|progress) → BattleMenuAction
  pickattack(1-4)                     → PickAttackAction
  menu(up|down|left|right|confirm|back) → MenuAction
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from gameboy_worlds.interface.pokemon.actions import (
    BattleMenuAction,
    InteractAction,
    MenuAction,
    MoveStepsAction,
    OpenMenuAction,
    PassDialogueAction,
    PickAttackAction,
)


class PrismActionParser:
    """
    Maps VLM action strings to ``(ActionClass, kwargs)`` for Pokemon Prism.

    Returns ``None`` when the string cannot be parsed; callers should fall
    back to ``env.string_to_high_level_action``.

    A/B button results are returned as the :attr:`A_BUTTON` and
    :attr:`B_BUTTON` string sentinels rather than concrete action classes
    because the correct action depends on the current game state (FREE_ROAM
    vs IN_DIALOGUE vs IN_MENU vs IN_BATTLE).  The executor resolves them
    via :meth:`PrismHarnessExecutor._resolve_button`.
    """

    A_BUTTON: str = "__A__"
    B_BUTTON: str = "__B__"
    START_BUTTON: str = "__START__"

    _STEP_MAP: Dict[str, Tuple[str, int]] = {
        "up": ("up", 1),    "u": ("up", 1),
        "north": ("up", 1), "n": ("up", 1),
        "down": ("down", 1), "d": ("down", 1),
        "south": ("down", 1), "s": ("down", 1),
        "left": ("left", 1),  "l": ("left", 1),
        "west": ("left", 1),  "w": ("left", 1),
        "right": ("right", 1), "r": ("right", 1),
        "east": ("right", 1),  "e": ("right", 1),
    }

    _A_TOKENS = frozenset({"a", "press_a", "button_a", "press a"})
    _B_TOKENS = frozenset({"b", "press_b", "button_b", "press b"})
    _START_TOKENS = frozenset({"start", "press_start", "start_button", "press start"})

    def parse(self, action_str: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Parse *action_str* and return ``(ActionClass, kwargs)`` or ``None``.

        Returns the string sentinels :attr:`A_BUTTON` / :attr:`B_BUTTON` for
        context-dependent button presses.
        """
        s = action_str.strip()
        sl = s.lower()

        # Single-step directional shorthands
        if sl in self._STEP_MAP:
            direction, steps = self._STEP_MAP[sl]
            return MoveStepsAction, {"direction": direction, "steps": steps}

        # Button sentinels
        if sl in self._A_TOKENS:
            return self.A_BUTTON, {}
        if sl in self._B_TOKENS:
            return self.B_BUTTON, {}
        if sl in self._START_TOKENS:
            return self.START_BUTTON, {}

        # Semantic actions (require parens)
        if "(" not in sl or ")" not in sl:
            return None

        name = sl.split("(")[0].strip()
        args = sl.split("(")[1].split(")")[0].strip()

        if name == "interact":
            return InteractAction, {}

        if name == "passdialogue":
            return PassDialogueAction, {}

        if name == "battlemenu":
            if args in ("fight", "pokemon", "bag", "run", "progress"):
                return BattleMenuAction, {"option": args}
            return None

        if name == "pickattack":
            if args.isnumeric() and 1 <= int(args) <= 4:
                return PickAttackAction, {"option": int(args)}
            return None

        if name == "menu":
            if args in ("up", "down", "left", "right", "confirm", "back"):
                return MenuAction, {"menu_action": args}
            return None

        if name == "openmenu":
            if args in ("pokemon", "bag", "trainer"):
                return OpenMenuAction, {"option": args}
            return None

        if name == "move":
            alias_map = {
                "north": "up", "south": "down", "east": "right", "west": "left",
                "up": "up", "down": "down", "left": "left", "right": "right",
            }
            for word, cardinal in alias_map.items():
                if word in args:
                    steps_str = args.replace(word, "").strip()
                    if not steps_str:
                        return MoveStepsAction, {"direction": cardinal, "steps": 1}
                    if steps_str.isnumeric():
                        steps = int(steps_str)
                        if 1 <= steps <= 5:
                            return MoveStepsAction, {"direction": cardinal, "steps": steps}
            return None

        return None
