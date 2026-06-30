"""
Pokemon Prism-specific ExecutorAction tools.

PrismLocateAction extends LocateAction with:
- Pre-described sprite options for common Prism overworld objects
- Concrete verbalize() and string_to_kwargs() implementations required by the
  ExecutorAction ABC (LocateAction leaves these abstract)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from execution.executor_action import LocateAction


class PrismLocateAction(LocateAction):
    """
    Locates a named object on the current screen using recursive quadrant search.
    Valid only during FREE_ROAM (overworld navigation).
    """

    pre_described_options: Dict[str, str] = {
        "item": "a pixelated, greyscale Poke Ball sprite or item sprite on the ground",
        "pokeball": "a pixelated, greyscale Poke Ball sprite, circular with a white center",
        "npc": "a pixelated human-like character sprite standing still",
        "grass": "a pixelated patch of tall grass with wavy dark lines",
        "sign": "a pixelated white signpost",
        "gym": "a pixelated building entrance with a badge symbol or 'GYM' label",
        "door": "a pixelated building entrance or doorway",
        "pokemon center": "a pixelated red-roofed building entrance with a nurse icon or 'CENTER' label",
        "pokemart": "a Poke Mart shop building",
    }

    @classmethod
    def verbalize(cls) -> str:
        options = ", ".join(f"'{k}'" for k in cls.pre_described_options)
        return (
            "locate(target=<str>) — find where a named object is on the current screen.\n"
            f"  target: object to search for. Pre-described shortcuts: {options}.\n"
            "           Any other free-form description also works.\n"
            "  Example: locate(target=gym)"
        )

    @classmethod
    def string_to_kwargs(cls, action_str: str) -> Optional[Dict[str, Any]]:
        """Parse 'locate(target=X)' → {'target': 'X'}, or return None on no match."""
        s = action_str.strip()
        if not s.lower().startswith("locate("):
            return None
        inner = s[len("locate("):].rstrip(")")
        match = re.match(r"target\s*=\s*['\"]?(.+?)['\"]?\s*$", inner, re.IGNORECASE)
        if match:
            return {"target": match.group(1).strip()}
        return None

    def is_valid(self, target: str = None, **kwargs) -> bool:
        return super().is_valid(target=target)
