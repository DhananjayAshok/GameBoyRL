from gameboy_worlds.interface import HighLevelAction

from utils import ExecutorVLM
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image
from abc import abstractmethod


class ExecutorAction(HighLevelAction):
    """
    Passive (non-interactive) actions that can be called on by an Executor to understand the game state.
    Will often include VLM inference to understand the game screen in some manner.

    Development Note: Unlike HighLevelAction, ExecutorAction does NOT call emulator.step.
    If you want to implement an action that does that, you probably want to be doing that in an Executor class instead.
    """

    def parameters_to_space(self, **kwargs):
        raise NotImplementedError(
            f"ExecutorAction does not implement parameters_to_space. Ensure that it is not being treated as a typical HighLevelAction."
        )

    def space_to_parameters(self, **kwargs):
        raise NotImplementedError(
            f"ExecutorAction does not implement space_to_parameters. Ensure that it is not being treated as a typical HighLevelAction."
        )

    def get_action_space(self):
        raise NotImplementedError(
            f"ExecutorAction does not implement get_action_space. Ensure that it is not being treated as a typical HighLevelAction."
        )

    @abstractmethod
    def _execute(self, **kwargs) -> Tuple[Dict[str, Any], int]:
        """
        Executes the specified executor action.
        Does not check for validity
        Development Note: Unlike HighLevelAction, ExecutorAction does NOT call emulator.step.
        If you want to implement an action that does that, you probably want to be doing that in an Executor class instead.

        :param kwargs: Additional arguments required for the specific executor action.
        :return: A tuple containing:

            - A dictionary with execution return information.
            - An integer representing the action success code.
        :rtype: Tuple[Dict[str, Any], int]
        """
        raise NotImplementedError


def object_detection(
    description: str, images: List[np.ndarray], text_prompt: str = None
) -> List[bool]:
    """
    Performs object detection on the given images with the given texts.

    :param images: List of images that may contain the object described in texts
    :type images: List[np.ndarray]
    :param text_prompt: A prompt with the textual description of the object to detect, and that requests a Yes/No answer.
    :type text_prompt: str
    :return: List of booleans indicating whether the object was detected in each image.
    :rtype: List[bool]
    """
    if text_prompt is None:
        text_prompt = f"""You are playing a gameboy game and are given a screen capture of the game. 
        Your job is to locate the target that best fits the description `{description}`

        Do you see the target described? Answer with a single sentence and then [YES] or [NO]
        [STOP]
        Output:
        """
    outputs = ExecutorVLM.infer(
        texts=[text_prompt for _ in images], images=images, max_new_tokens=60
    )
    founds = []
    for i, output in enumerate(outputs):
        if "yes" in output.lower():
            founds.append(True)
        else:
            founds.append(False)
    return founds


def identify_matches(
    description: str,
    screens: List[np.ndarray],
    reference: Image.Image,
    text_prompt: str = None,
) -> List[bool]:
    """
    Identifies which screens match the given reference image based on the description.
    Args:
        description: A textual description of the target object.
        screens: A list of screen images in numpy array format (H x W x C).
        reference: A PIL Image of the reference object.
        text_prompt: Optional prompt to guide the VLM.

    Returns:
        A list of booleans indicating whether each screen contains the target object.
    """
    if text_prompt is None:
        text_prompt = f"The target, described as {description} is shown as reference in Picture 1. Does Picture 2 contain the object from Picture 1 in it? Answer in the following format: \nExplanation: <briefly describe what is in Picture 2, with reference to the image in Picture 1>\nAnswer: <Yes or No>[STOP]"
    texts = [text_prompt for _ in screens]
    images = []
    for screen in screens:
        images.append([reference, screen])
    outputs = ExecutorVLM.multi_infer(texts=texts, images=images, max_new_tokens=120)
    results = []
    for output in outputs:
        if "yes" in output.lower():
            results.append(True)
        else:
            results.append(False)
    return results


class LocateAction(ExecutorAction):
    """
    Locates a target in the current screen.
    1. Divides the screen into grid cells.
    2. Recursively divides the grid cells into quadrants and checks each quadrant for the target.
    3. If a quadrant contains the target, further divides it into smaller quadrants until the smallest grid cells are reached.
    4. Returns the grid cell coordinates that may contain the target.

    Uses VLM inference to check each grid cell for the target. There are three kinds of VLM search used:
    - Description matching: if the target is specified as a known object in `pre_described_options`, uses a pre-defined description to search for the target.
    - Image matching: if the target is specified as a known object in the provided state parser's `image_references`, uses a reference image as well as a description to search for the target.
    - Free-form description matching: if the target is specified as a free-form string, uses the string as a description to search for the target.

    Action Success Interpretation:
    - -1: Object not found
    - 0: A single object found and only definitively
    - 1: Multiple objects found, but only definitively
    - 2: A single object found, only potentially
    - 3: Multiple objects found, both definitively and potentially

    Action Returns:
    - `found` (`bool`): whether the target was found in any of the grid cells at any point.
    - `potential_cells` (`List[Tuple[int, int]]`): list of grid cell coordinates that may contain the target.
    - `definitive_cells` (`List[Tuple[int, int]]`): list of grid cell coordinates that, with high confidence, contain the target.
    """

    pre_described_options = {}
    """ Pre-defined descriptions for known objects to locate. Subclasses can override this dictionary to options. """

    image_references = {}
    """ Pre-defined mapping of strings to image_reference keys in the state parser. Subclasses can override this dictionary to options. This is needed because some objects may have image_reference keys that are not the same as their name. """

    def coord_to_string(self, coord: Tuple[int, int]) -> str:
        start = "("
        c1 = coord[0]
        if c1 > 0:
            start += f"{c1} steps to right from you, "
        elif c1 < 0:
            start += f"{-c1} steps to left from you, "
        c2 = coord[1]
        if c2 > 0:
            start += f"{c2} steps up from you)"
        elif c2 < 0:
            start += f"{-c2} steps down from you)"
        return start

    def coords_to_string(self, coords: List[Tuple[int, int]]) -> str:
        coord_strings = [self.coord_to_string(coord) for coord in coords]
        return "[" + ", ".join(coord_strings) + "]"

    def is_valid(self, target: str = None):
        if target is not None:
            if not isinstance(target, str):
                return False
            if len(target.strip()) == 0:
                return False
            target = target.lower().strip()
            if target not in self.image_references.keys():
                if target not in self.pre_described_options.keys():
                    return False  # most permissive case. Return False here if you want to restrict to known options.
                else:
                    pass  # known pre-described option. Return False here if you want to restrict to only image references.
            else:
                pass  # known image reference
        return True

    def check_for_target(self, description, screens, image_reference: str = None):
        if image_reference is None:
            return object_detection(description=description, images=screens)
        else:
            reference_image = self._emulator.state_parser.get_image_reference(
                image_reference
            )
            founds = identify_matches(
                description=description, screens=screens, reference=reference_image
            )
            return founds

    def get_centroid(
        self, cells: Dict[Tuple[int, int], np.ndarray]
    ) -> Tuple[float, float]:
        min_x = min([coord[0] for coord in cells.keys()])
        min_y = min([coord[1] for coord in cells.keys()])
        max_x = max([coord[0] for coord in cells.keys()])
        max_y = max([coord[1] for coord in cells.keys()])
        centroid_x = (min_x + max_x) // 2
        centroid_y = (min_y + max_y) // 2
        return (centroid_x, centroid_y)

    def get_cells_found(
        self,
        grid_cells: Dict[Tuple[int, int], np.ndarray],
        description: str,
        image_reference: str = None,
    ) -> Tuple[bool, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Recursively divides the grid cells into quadrants and checks each quadrant for the target.
        Args:
            grid_cells: the dict of the subset of grid cells to search over
            description: description of target
            image_reference: reference image for the target

        Returns:
            found (bool): whether the target was found in any of the grid cells at any point.
            potential_cells (List[Tuple[int, int]]): list of grid cell coordinates that may contain the target. If found is true, this is almost always populated with something
                                                    The only exception is when the item was found at too high a scan and not found at lower levels (and so too many cells would have been potentials)
            definitive_cells (List[Tuple[int, int]]): list of grid cell coordinates that, with high confidence, contain the target.
        """
        quadrant_keys = ["tl", "tr", "bl", "br"]
        if len(grid_cells) == 1:
            screen = list(grid_cells.values())[0]
            keys = list(grid_cells.keys())[0]
            target_in_grid = self.check_for_target(
                description, [screen], image_reference=image_reference
            )[0]
            if target_in_grid:
                return True, list(grid_cells.keys()), list(grid_cells.keys())
            else:
                return False, [], []
        quadrants = self._emulator.state_parser.get_quadrant_frame(
            grid_cells=grid_cells
        )
        screens = []
        for quadrant in quadrant_keys:
            screen = quadrants[quadrant]["screen"]
            screens.append(screen)
        quadrant_founds = self.check_for_target(
            description, screens, image_reference=image_reference
        )
        if not any(quadrant_founds):
            return False, [], []
        else:
            potential_cells = []
            quadrant_definites = []
            for i in range(len(quadrant_keys)):
                quadrant = quadrant_keys[i]
                if quadrant_founds[i]:
                    cells = quadrants[quadrant]["cells"]
                    if len(cells) < 4:
                        potential_cells.append(self.get_centroid(cells))
                        cell_keys = list(cells.keys())
                        cell_screens = [cells[key] for key in cell_keys]
                        cell_founds = self.check_for_target(
                            description, cell_screens, image_reference=image_reference
                        )
                        for i, found in enumerate(cell_founds):
                            if found:
                                quadrant_definites.append(cell_keys[i])
                            else:
                                pass
                    else:
                        (
                            found_in_quadrant,
                            quadrant_potentials,
                            recursive_quadrant_definites,
                        ) = self.get_cells_found(
                            cells, description, image_reference=image_reference
                        )
                        if len(recursive_quadrant_definites) > 0:
                            quadrant_definites.extend(recursive_quadrant_definites)
                        if (
                            found_in_quadrant
                        ):  # then there is some potential, so add the quadrants potentials.
                            if len(quadrant_potentials) != 0:
                                potential_cells.extend(quadrant_potentials)
                            else:
                                potential_cells.append(self.get_centroid(cells))
            return True, potential_cells, quadrant_definites

    def do_location(
        self, description: str, image_reference: str = None
    ) -> Tuple[Dict[str, Any], int]:
        """
        Performs the locate action to find the target described by `description` in the current screen.

        :param description: Description of the target to locate.
        :type description: str
        :param image_reference: String key for the image reference in the state parser to use for matching.
        :type image_reference: str
        :return: A tuple containing:

            - A dictionary with:

                * `found` (`bool`): whether the target was found in any of the grid cells at any point.
                * `potential_cells` (`List[Tuple[int, int]]`): list of grid cell coordinates that may contain the target.
                * `definitive_cells` (`List[Tuple[int, int]]`): list of grid cell coordinates that, with high confidence, contain the target.
                * `potential_cells_str` (`str`): string representation of potential_cells for logging.
                * `definitive_cells_str` (`str`): string representation of definitive_cells for logging.

            - An integer representing the action success code.
        :rtype: Tuple[Dict[str, Any], int]
        """
        grid_cells = self._emulator.state_parser.capture_grid_cells(
            self._emulator.get_current_frame()
        )
        found, potential_cells, definitive_cells = self.get_cells_found(
            grid_cells, description, image_reference=image_reference
        )
        ret_dict = {
            "found": found,
            "potential_cells": potential_cells,
            "definitive_cells": definitive_cells,
            "potential_cells_str": self.coords_to_string(potential_cells),
            "definitive_cells_str": self.coords_to_string(definitive_cells),
        }
        action_success = None
        if not found:
            action_success = -1
        else:
            if len(definitive_cells) == 0:
                if len(potential_cells) == 1:
                    action_success = 2
                elif len(potential_cells) > 1:
                    action_success = 3
                else:
                    action_success = -1
            else:
                if len(definitive_cells) == 1:
                    action_success = 0
                else:
                    action_success = 1
        return ret_dict, action_success

    def _execute(self, target: str):
        if target in self.image_references:
            return self.do_location(
                description=self.pre_described_options[target],
                image_reference=self.image_references[target],
            )
        elif target in self.pre_described_options:
            return self.do_location(description=self.pre_described_options[target])
        else:
            return self.do_location(description=target)
