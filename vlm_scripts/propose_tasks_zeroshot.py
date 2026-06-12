"""
Called by scripts/vlm/propose_zeroshot.sh and scripts/vlm/propose_all_zeroshot.sh (via vlm.py propose_tasks_zeroshot).
Use --help for CLI options.

Input: a single init_state name (str) and the game environment
    - The first frame and action space are loaded live from the environment via get_first_frame_and_actions.
    - Optional prior tasks (--extra) are loaded from:
        - 'zeroshot':  parameters["storage_dir"]/proposed_tasks/$game/$model/zeroshot/zeroshot_tasks.jsonl
        - 'curiosity': parameters["storage_dir"]/proposed_tasks/$game/$model/zeroshot/curiosity/all/trajectory_annotation.json
                       (dict[int, str] — group_idx → distilled task string; see infer_task.py)

Output: JSONL file, one JSON object per line, one line per init_state
    Path: parameters["storage_dir"]/proposed_tasks/$game/$model/zeroshot/
        - no --extra:          zeroshot_tasks.jsonl
        - --extra zeroshot:    zeroshot_tasks_prior_zeroshot.jsonl
        - --extra curiosity:   zeroshot_tasks_prior_curiosity.jsonl
    Each line: {"init_state": <str>, "tasks": [<task_str>, ...]}
        - init_state: the name of the initial game state
        - tasks: list of imperative task strings proposed by the VLM from the first frame
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_parameters, log_info, log_warn, log_error, VLM, HuggingFaceModel
from gameboy_worlds import get_environment

PROPOSE_PROMPT = """You are observing the initial frame of a game of [GAME].

You are an expert game analyst. Your job is to look carefully at this frame and reason about:
- What is visible in the immediate surroundings (objects, NPCs, structures, interactables)
- What areas or items are accessible from this position
- What mechanisms or interactions are available given the action space: [ACTION_SPACE]

[EXTRA_CONTEXT]

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

EXTRA_CONTEXT_PROMPT = """The following tasks have been identified from related game states and may or may not be relevant here:
[PRIOR_TASKS]
Use these only if they are clearly applicable to the current state. Do not include them blindly."""


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


def get_extra_context(
    outdir: str, init_state: str, extra, run_name: str, parameters: dict
) -> list[str]:
    tasks_dir = outdir
    if extra is None:
        return []
    tasks = []
    if "zeroshot" in extra:
        path = tasks_dir + "zeroshot/zeroshot_tasks.jsonl"
        if not os.path.exists(path):
            log_error(
                f"Zeroshot tasks file not found at {path}. Run propose_tasks_zeroshot without --extra first.",
                parameters,
            )
        df = pd.read_json(path, lines=True)
        if len(df) == 0:
            log_error(f"Zeroshot tasks file at {path} is empty.", parameters)
        if init_state not in df["init_state"].values:
            log_error(
                f"init_state '{init_state}' not found in {path} — file may be incomplete.",
                parameters,
            )
        tasks.extend([task for tasks_list in df["tasks"] for task in tasks_list])

    if "curiosity" in extra:
        path = tasks_dir + f"curiosity/{run_name}/trajectory_annotation.json"
        if not os.path.exists(path):
            log_error(
                f"Curiosity tasks file not found at {path}. Run infer_tasks first.",
                parameters,
            )
        curiosity_data = json.load(open(path, "r"))
        task_list = []
        for group_idx, task_info in curiosity_data.items():
            task_list.append(task_info)
        tasks.extend(task_list)
    return tasks


def _parse_task_list(text: str) -> list[str]:
    tasks = []
    in_tasks = False
    for line in text.lower().splitlines():
        if "[stop]" in line:
            break
        if line.strip().startswith("tasks:"):
            in_tasks = True
            continue
        if in_tasks and line.strip().startswith("- "):
            tasks.append(line.strip()[2:].strip())
    return tasks


def _propose_for_init_state(
    init_state: str,
    game: str,
    outdir: str,
    extra,
    extra_k: int,
    run_name: str,
    vlm: VLM,
    max_new_tokens: int,
    verbose: bool,
    parameters: dict,
) -> list[str]:
    """Propose tasks from the initial frame of a single init_state."""
    first_frame, action_space_str = get_first_frame_and_actions(init_state, game)
    prior_tasks = get_extra_context(outdir, init_state, extra, run_name, parameters)
    if prior_tasks and len(prior_tasks) > extra_k:
        prior_tasks = list(np.random.choice(prior_tasks, size=extra_k, replace=False))

    extra_context_str = ""
    if prior_tasks:
        prior_list = "\n".join(f"- {t}" for t in prior_tasks)
        extra_context_str = EXTRA_CONTEXT_PROMPT.replace("[PRIOR_TASKS]", prior_list)

    prompt = (
        PROPOSE_PROMPT.replace("[GAME]", game)
        .replace("[ACTION_SPACE]", action_space_str)
        .replace("[EXTRA_CONTEXT]", extra_context_str)
    )

    if verbose:
        print(f"PROPOSE prompt for '{init_state}':\n{prompt}\n---")

    output = vlm.infer(
        texts=prompt, images=[first_frame], max_new_tokens=max_new_tokens
    ).lower()

    if verbose:
        print(f"PROPOSE output for '{init_state}':\n{output}\n---")

    tasks = _parse_task_list(output)
    if not tasks:
        print(
            f"Warning: no tasks parsed from VLM output for init_state '{init_state}'."
        )
    return tasks


@click.command(name="propose_tasks_zeroshot")
@click.option("--init_state", required=True, help="Comma-separated name(s) of the init state(s) to load")
@click.option(
    "--extra",
    type=click.Choice([None, "zeroshot", "curiosity", "zeroshot_with_curiosity"], case_sensitive=False),
    default=None,
    help="How to look for prior tasks. None: no prior tasks. 'zeroshot': use all other init states zeroshot tasks. 'curiosity': use all other init states curiosity based exploration tasks.",
)
@click.option(
    "--extra_k",
    default=20,
    type=int,
    help="Maximum number of extra tasks to sample when --extra is set.",
)
@click.option(
    "--run_name",
    default="all",
    help="run_name used when infer_tasks was run, to locate the curiosity annotation.",
)
@click.option(
    "--max_concurrency",
    default=8,
    show_default=True,
    help="Max concurrent propose pipelines (across init_states). Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def propose_tasks_zeroshot(obj, init_state, extra, extra_k, run_name, max_concurrency):
    """Zero-shot VLM task proposal from the initial frame of each given init_state."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    max_new_tokens = obj["max_new_tokens"]
    verbose = obj["verbose"]
    overwrite = obj["overwrite"]
    model_save_name = model_name.split("/")[-1]
    outdir = parameters["storage_dir"] + f"/proposed_tasks/{game}/{model_save_name}/"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + "/zeroshot/", exist_ok=True)
    if extra is None:
        outpath = outdir + "zeroshot/zeroshot_tasks"
    elif extra == "zeroshot":
        outpath = outdir + "zeroshot/zeroshot_tasks_prior_zeroshot"
    elif extra == "curiosity":
        outpath = outdir + "zeroshot/zeroshot_tasks_prior_curiosity"
    elif extra == "zeroshot_with_curiosity":
        outpath = outdir + "zeroshot/zeroshot_tasks_prior_zeroshot_with_curiosity"
    out_path = outpath + ".jsonl"

    if os.path.exists(out_path):
        df = pd.read_json(out_path, lines=True)
    else:
        df = pd.DataFrame([], columns=["init_state", "tasks"])

    init_states = [s.strip() for s in init_state.split(",") if s.strip()]
    done = set(df["init_state"].values)
    jobs = []
    for state in init_states:
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
                outdir,
                extra,
                extra_k,
                run_name,
                vlm,
                max_new_tokens,
                verbose,
                parameters,
            ): state
            for state in jobs
        }

        for future in tqdm(as_completed(future_to_state), total=len(future_to_state), desc="Proposing tasks"):
            state = future_to_state[future]
            tasks = future.result()
            df = pd.concat([df, pd.DataFrame([{"init_state": state, "tasks": tasks}])], ignore_index=True)
            df.to_json(out_path, orient="records", lines=True)
            log_info(f"Saved proposed tasks for init_state '{state}' to {out_path}.")
