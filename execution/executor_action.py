from utils import object_detection
from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod
from gameboy_worlds.emulation.parser import StateParser


def _get_quadrants(
    grid_cells: Dict[Tuple[int, int], np.ndarray],
) -> Dict[str, Dict]:
    """Split grid_cells into four quadrants and stitch each into a screen image."""
    coords = grid_cells.keys()
    xs = sorted(set(c[0] for c in coords))
    ys = sorted(set(c[1] for c in coords))
    mid_x = xs[len(xs) // 2]
    mid_y = ys[len(ys) // 2]
    lower_x = [x for x in xs if x < mid_x]
    higher_x = [x for x in xs if x >= mid_x]
    lower_y = [y for y in ys if y < mid_y]
    higher_y = [y for y in ys if y >= mid_y]
    raw = {
        "tl": {(x, y): grid_cells[(x, y)] for x in lower_x for y in higher_y},
        "tr": {(x, y): grid_cells[(x, y)] for x in higher_x for y in higher_y},
        "bl": {(x, y): grid_cells[(x, y)] for x in lower_x for y in lower_y},
        "br": {(x, y): grid_cells[(x, y)] for x in higher_x for y in lower_y},
    }
    return {
        q: {"cells": cells, "screen": StateParser.reform_image(cells)}
        for q, cells in raw.items()
    }


def coord_to_string(coord: Tuple[int, int]) -> str:
    x, y = coord
    parts = []
    if x > 0:
        parts.append(f"{x} steps to right from you")
    elif x < 0:
        parts.append(f"{-x} steps to left from you")
    if y > 0:
        parts.append(f"{y} steps up from you")
    elif y < 0:
        parts.append(f"{-y} steps down from you")
    return "(" + ", ".join(parts) + ")"


def coords_to_string(coords: List[Tuple[int, int]]) -> str:
    return "[" + ", ".join(coord_to_string(c) for c in coords) + "]"


class ExecutorAction(ABC):
    """
    Passive (non-interactive) actions that can be called on by an Executor to understand the game state.
    Will often include VLM inference to understand the game screen in some manner.

    Development Note: Unlike HighLevelAction, ExecutorAction does NOT call emulator.step.
    If you want to implement an action that does that, you probably want to be doing that in an Executor class instead.

    :param vlm_call: The owning executor's recording VLM entry point
        (:meth:`~execution.executors.Executor._vlm_call`). Tools must infer through this
        so their calls land in the executor's ``vlm_call_log``.
    :param parameters: The owning executor's resolved parameters.
    """

    def __init__(self, vlm_call: Callable[..., Any], parameters: Dict[str, Any]) -> None:
        self._vlm_call = vlm_call
        self._parameters = parameters

    @abstractmethod
    def _execute(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[Dict[str, Any], int]:
        """
        Executes the specified executor action.

        :param info: Full state information from the environment, as returned by
            :meth:`~gameboy_worlds.interface.Environment.get_info`.
        :type info: Dict[str, Dict[str, Any]]
        :param kwargs: Additional arguments required for the specific executor action.
        :return: A tuple containing:

            - A dictionary with execution return information.
            - An integer representing the action success code.
        :rtype: Tuple[Dict[str, Any], int]
        """
        raise NotImplementedError

    def is_valid(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Validates the game state and arguments before execution. Valid by default.
        Subclasses may override to reject invalid calls early.

        :param info: Full state information from the environment.
        :type info: Dict[str, Dict[str, Any]]
        :return: ``(valid, reason)``, where ``reason`` explains a rejection and is
            ``None`` when the call is valid.
        :rtype: Tuple[bool, Optional[str]]
        """
        return True, None

    @classmethod
    @abstractmethod
    def verbalize(cls) -> str:
        """
        Return a human-readable string that fully describes this tool call for
        inclusion in a VLM prompt.

        .. warning:: **Format contract**

            The string returned here is the **exact format** the VLM is expected
            to reproduce when it decides to invoke this tool.  :meth:`string_to_kwargs`
            must be able to parse every string that ``verbalize`` declares as valid.
            If the two are out of sync the executor will silently drop tool calls.

        .. warning:: **No colons in the call syntax**

            The invocation string a tool declares — the ``Example:`` line, and any
            signature the model might copy verbatim — must not contain ``:``.
            :class:`~execution.executors.policies.action.ScoredActionPolicy` delimits its
            reply as ``<action>: <reasoning>: <score>`` and reads the action as everything
            before the *first* colon, so a colon inside the call syntax truncates the tool
            name and the call is recorded as an unrecognised action.  Use ``=`` for
            arguments (``locate(target=door)``) and keep type annotations and prose out of
            the part the model is asked to reproduce.

        A good verbalization includes:

        - The tool name (use a stable, unambiguous identifier).
        - Every parameter name, its type, and a brief description.
        - One concrete example in the exact output format.

        Example return value::

            "locate(target=<str>) — find a named object on the current screen.\\n"
            "  target — the object to search for (e.g. 'pokémon center door').\\n"
            "  Example: locate(target=pokémon center door)"

        :return: Prompt-ready description of the tool and its call syntax.
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def string_to_kwargs(cls, action_str: str) -> Dict[str, Any]:
        """
        Parse a tool-call string (in the format declared by :meth:`verbalize`)
        into a ``kwargs`` dictionary suitable for passing to :meth:`execute`.

        .. warning:: **Must mirror** :meth:`verbalize`

            This method is the inverse of the call syntax shown in
            :meth:`verbalize`.  Any format change in ``verbalize`` must be
            reflected here, and vice versa.  The executor relies on this
            round-trip to dispatch tool calls produced by the VLM.

        :param action_str: Raw action string as output by the VLM, matching the
            format advertised in :meth:`verbalize`.
        :type action_str: str
        :return: Keyword arguments to forward to :meth:`execute`.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError

    def execute(self, info: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Public entry point. Validates arguments then delegates to :meth:`_execute`.

        :param info: Full state information from the environment.
        :type info: Dict[str, Dict[str, Any]]
        :param kwargs: Additional arguments forwarded to :meth:`_execute`.
        :return: A tuple containing:

            - A dictionary with execution return information, or ``{"invalid": reason}``
              if the call was rejected by :meth:`is_valid`.
            - An integer success code, or ``None`` if the call was rejected.
        :rtype: Tuple[Dict[str, Any], Optional[int]]
        """
        valid, reason = self.is_valid(info, **kwargs)
        if not valid:
            return {"invalid": reason or "invalid call"}, None
        return self._execute(info, **kwargs)


class LocateAction(ExecutorAction):
    """
    Locates a target in the current screen by recursively searching quadrants.

    Divides the screen into grid cells, then recursively narrows down which
    quadrant contains the target using VLM-based object detection. Returns both
    high-confidence (definitive) and lower-confidence (potential) cell coordinates.

    The target can be a key in ``pre_described_options`` (uses a curated description)
    or any free-form string (used directly as the VLM description).

    Action Success Codes:
    - -1: Object not found
    -  0: Exactly one definitive match
    -  1: Multiple definitive matches
    -  2: Exactly one potential match (no definitives)
    -  3: Multiple potential matches (no definitives)

    Action Returns:
    - ``found`` (bool): whether the target was found at any level.
    - ``potential_cells`` (List[Tuple[int, int]]): grid coords that may contain the target.
    - ``definitive_cells`` (List[Tuple[int, int]]): grid coords that with high confidence contain the target.
    - ``potential_cells_str`` (str): human-readable form of potential_cells.
    - ``definitive_cells_str`` (str): human-readable form of definitive_cells.
    """

    pre_described_options: Dict[str, str] = {}
    """Maps known target names to VLM-ready descriptions. Subclasses override to add entries."""

    def coord_to_string(self, coord: Tuple[int, int]) -> str:
        return coord_to_string(coord)

    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        return coords_to_string(coords)

    def is_valid(self, info: Dict[str, Dict[str, Any]], target: str = None,
                 **kwargs) -> Tuple[bool, Optional[str]]:
        if target is None or (isinstance(target, str) and target.strip()):
            return True, None
        return False, f"target must be a non-empty string, got {target!r}"

    def check_for_target(self, description: str, screens: List[np.ndarray]) -> List[bool]:
        return object_detection(description=description, images=screens)

    def get_centroid(
        self, cells: Dict[Tuple[int, int], np.ndarray]
    ) -> Tuple[float, float]:
        xs = [coord[0] for coord in cells.keys()]
        ys = [coord[1] for coord in cells.keys()]
        return ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)

    def get_cells_found(
        self,
        grid_cells: Dict[Tuple[int, int], np.ndarray],
        description: str,
    ) -> Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Recursively divides the grid cells into quadrants and checks each quadrant for the target.

        :param grid_cells: The dict of the subset of grid cells to search over.
        :type grid_cells: Dict[Tuple[int, int], np.ndarray]
        :param description: VLM-ready description of the target.
        :type description: str
        :return: ``(found, potential_cells, definitive_cells)``
        :rtype: Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]
        """
        quadrant_keys = ["tl", "tr", "bl", "br"]
        if len(grid_cells) == 1:
            screen = list(grid_cells.values())[0]
            if self.check_for_target(description, [screen])[0]:
                return True, list(grid_cells.keys()), list(grid_cells.keys())
            return False, [], []

        quadrants = _get_quadrants(grid_cells)
        screens = [quadrants[q]["screen"] for q in quadrant_keys]
        quadrant_founds = self.check_for_target(description, screens)

        if not any(quadrant_founds):
            return False, [], []

        potential_cells = []
        quadrant_definites = []
        for i, quadrant in enumerate(quadrant_keys):
            if not quadrant_founds[i]:
                continue
            cells = quadrants[quadrant]["cells"]
            if len(cells) < 4:
                potential_cells.append(self.get_centroid(cells))
                cell_keys = list(cells.keys())
                cell_screens = [cells[k] for k in cell_keys]
                for j, found in enumerate(self.check_for_target(description, cell_screens)):
                    if found:
                        quadrant_definites.append(cell_keys[j])
            else:
                found_in_quadrant, quadrant_potentials, recursive_definites = self.get_cells_found(
                    cells, description
                )
                quadrant_definites.extend(recursive_definites)
                if found_in_quadrant:
                    if quadrant_potentials:
                        potential_cells.extend(quadrant_potentials)
                    else:
                        potential_cells.append(self.get_centroid(cells))

        return True, potential_cells, quadrant_definites

    def do_location(
        self, info: Dict[str, Any], description: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Performs the locate action to find the target described by ``description`` in the current screen.

        :param info: Full state information from the environment.
        :type info: Dict[str, Any]
        :param description: VLM-ready description of the target to locate.
        :type description: str
        :return: Result dict and action success code.
        :rtype: Tuple[Dict[str, Any], int]
        """
        frame = info["core"]["current_frame"]
        grid_cells = StateParser.capture_grid_cells(frame)
        found, potential_cells, definitive_cells = self.get_cells_found(grid_cells, description)
        ret_dict = {
            "found": found,
            "potential_cells": potential_cells,
            "definitive_cells": definitive_cells,
            "potential_cells_str": self.coords_to_string(potential_cells),
            "definitive_cells_str": self.coords_to_string(definitive_cells),
        }
        if not found:
            action_success = -1
        elif not definitive_cells:
            if len(potential_cells) == 1:
                action_success = 2
            elif len(potential_cells) > 1:
                action_success = 3
            else:
                action_success = -1
        else:
            action_success = 0 if len(definitive_cells) == 1 else 1
        return ret_dict, action_success

    def _execute(self, info: Dict[str, Dict[str, Any]], target: str, **kwargs) -> Tuple[Dict[str, Any], int]:
        key = target.lower().strip()
        description = self.pre_described_options.get(key, target)
        return self.do_location(info, description)
