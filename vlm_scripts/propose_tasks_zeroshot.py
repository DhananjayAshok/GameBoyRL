"""
Called by scripts/vlm/propose_zeroshot.sh and scripts/vlm/propose_all_zeroshot.sh (via vlm.py propose_tasks_zeroshot).
Use --help for CLI options.

Every path below comes from :mod:`utils.paths`, which owns the directory scheme; this
module names accessors rather than spelling layouts out, so there is nothing here to drift
out of step with the readers.

Input: one or more init_state names (--init_states, comma-separated) and the game environment
    - The first frame and action space are loaded live from the environment via get_first_frame_and_actions.

Output: JSONL file, one JSON object per line, one line per init_state
    Path: Paths.tasks_file(), i.e. <zeroshot_dir>/zeroshot_tasks.jsonl
    Each line: {"init_state": <str>, "tasks": [<task_str>, ...]}
        - init_state: the name of the initial game state
        - tasks: list of imperative task strings proposed by the VLM from the first frame
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import pandas as pd
from tqdm import tqdm

from utils import (log_info, log_warn, VLM, HuggingFaceModel, parse_list)
from utils.paths import Paths
from gameboy_worlds import get_environment

PROPOSE_PROMPT = """You are observing the initial frame of a game of [GAME].

You are an expert game analyst. Your job is to look carefully at this frame and reason about:
- What is visible in the immediate surroundings (objects, NPCs, structures, interactables)
- What areas or items are accessible from this position
- What mechanisms or interactions are available given the action space: [ACTION_SPACE]

Propose an exhaustive list of distinct tasks that a player could meaningfully attempt to achieve starting from this exact state. Focus on tasks that are:
- Grounded in what is actually visible or reachable from this state
- Specific and concrete (not vague like "explore the area")
- Achievable as a single coherent goal
- Varied in scope (include both short and longer-horizon tasks)

Respond in exactly this format:
Reasoning: <brief analysis of what is visible and what interactions are possible>
Tasks:
- <task 1>
- <task 2>
- <task 3>
...
[STOP]"""


def get_first_frame_and_actions(
    init_state: str, game: str, controller_variant: str = "low_level"
):
    env = get_environment(
        game=game,
        environment_variant="default",
        controller_variant=controller_variant,
        init_state=init_state,
        max_steps=2,
        headless=True,
        save_video=False,
    )
    try:
        obs, info = env.reset()
        first_frame = info["core"]["current_frame"]
        action_string = env.get_action_strings(return_all=True)
        verbalized_string = ""
        for high_level_action, string in action_string.items():
            verbalized_string += f"- {string}\n"
        return (first_frame, verbalized_string)
    finally:
        env.close()


def _propose_for_init_state(
    init_state: str,
    game: str,
    vlm: VLM,
    max_new_tokens: int,
    verbose: bool,
) -> list[str]:
    """Propose tasks from the initial frame of a single init_state."""
    first_frame, action_space_str = get_first_frame_and_actions(init_state, game)

    prompt = (
        PROPOSE_PROMPT.replace("[GAME]", game)
        .replace("[ACTION_SPACE]", action_space_str)
    )

    if verbose:
        print(f"PROPOSE prompt for '{init_state}':\n{prompt}\n---")

    output = vlm.infer(
        texts=prompt, images=[first_frame], max_new_tokens=max_new_tokens
    ).lower()

    if verbose:
        print(f"PROPOSE output for '{init_state}':\n{output}\n---")

    # Lowercased here, not in the parser: these strings go on to name directories on disk
    # (proposed_tasks/<game>/.../<task>_attempts), and every artifact already written was
    # keyed on the lowercase form. The old _parse_task_list lowercased internally; dropping
    # that without restoring it here would orphan the existing corpus.
    tasks = [task.lower() for task in parse_list(output, "Tasks")]
    if not tasks:
        log_warn(
            f"no tasks parsed from VLM output for init_state '{init_state}'."
        )
    return tasks


@click.command(name="propose_tasks_zeroshot")
@click.option("--init_states", required=True, help="Comma-separated name(s) of the init state(s) to load")
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent propose pipelines (across init_states). Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def propose_tasks_zeroshot(obj, init_states, max_concurrency):
    """Zero-shot VLM task proposal from the initial frame of each given init_state."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    max_new_tokens = obj["max_new_tokens"]
    verbose = obj["verbose"]
    overwrite = obj["overwrite"]
    # The scheme lives in utils.paths, so this script no longer spells out the directory
    # layout; it used to be duplicated here and re-derived by every reader.
    paths = Paths(parameters=parameters, game=game, model_name=model_name)
    out_path = paths.tasks_file()
    os.makedirs(paths.zeroshot_dir(), exist_ok=True)

    if os.path.exists(out_path):
        df = pd.read_json(out_path, lines=True)
    else:
        df = pd.DataFrame([], columns=["init_state", "tasks"])

    init_state_list = [s.strip() for s in init_states.split(",") if s.strip()]
    done = set(df["init_state"].values)
    jobs = []
    for state in init_state_list:
        if state in done:
            if not overwrite:
                log_info(
                    f"Skipping — output already exists for initial state {state} at {out_path}. Use --overwrite to rerun."
                )
                continue
            log_warn(
                f"Overwriting existing output for initial state {state} at {out_path}."
            )
            df = df[df["init_state"] != state].reset_index(drop=True)
        jobs.append(state)

    if not jobs:
        return

    vlm = VLM(model_name, vlm_kind)

    # propose pipelines are independent across init_states, so submit them all to
    # one shared pool. --verbose interleaves prompt/output prints and
    # HuggingFaceModel isn't safe for concurrent generate() calls — both fall
    # back to max_workers=1, which processes jobs one at a time in submission
    # order (i.e. identical to the old sequential loop).
    effective_workers = (
        1 if (verbose or isinstance(vlm._vlm, HuggingFaceModel)) else max_concurrency
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_state = {
            executor.submit(
                _propose_for_init_state,
                state,
                game,
                vlm,
                max_new_tokens,
                verbose,
            ): state
            for state in jobs
        }

        for future in tqdm(as_completed(future_to_state), total=len(future_to_state), desc="Proposing tasks"):
            state = future_to_state[future]
            tasks = future.result()
            df = pd.concat([df, pd.DataFrame([{"init_state": state, "tasks": tasks}])], ignore_index=True)
            df.to_json(out_path, orient="records", lines=True)
            log_info(f"Saved proposed tasks for init_state '{state}' to {out_path}.")
