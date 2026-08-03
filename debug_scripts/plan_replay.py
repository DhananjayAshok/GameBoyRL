"""
Frame-by-frame replay of an info-plan episode: every VLM call beside the screen it saw.

Why a replay and not saved frames
---------------------------------
Nothing on disk carries the frames. ``ExecutorReport.__str__`` renders prompts and outputs
as text only; the PNGs are written by ``_save_images``, which runs only under ``--verbose``,
is keyed on ``<executor class>/<task>`` and rmtree's that directory first — so within one
episode each of the plan arm's legs would overwrite the last, and the surviving frames would
belong to whichever attempt happened to run last. Unusable.

The emulator is deterministic given an init state and an action sequence, and the action
sequence is recoverable: each call's VLM output ends in the ``Action:`` line the executor
parsed, and ``Environment.step_str`` accepts exactly that string. So the run can be
reconstructed by replaying the recorded outputs into a fresh environment and capturing the
frame *before* each action — which is the frame that call was actually looking at.

The replay is checked against the CSV's own ``n_steps``; a mismatch is reported rather than
silently rendered, because a divergent replay shows frames the model never saw, which is
worse than no frames at all.

Usage
-----
    python -m debug_scripts.plan_replay --game deja_vu_1 \\
        --csv results/benchmark/deja_vu_1/info_plan_retrieval_history_gemma-4-31b-it_first5.csv \\
        --episodes 0,1,2

Output
------
<results_dir>/debug/<game>/info_plan/replay/episode_<i>.md, with frames beside it.
"""

import json
import os
import re
import shutil

import click
import pandas as pd

from debug_scripts.benchmark import parse_report
from debug_scripts.frames import to_pil
from utils import load_parameters, log_info, log_warn

# Written by run_benchmark_info_plan._join_leg_reports.
LEG_HEADER = re.compile(
    r"^===== STEP (\d+) ATTEMPT (\d+) \[([^\]]*)\] (.*?) =====$", re.MULTILINE
)
ACTION_LINE = re.compile(r"^\s*Action:\s*(.+?)\s*$", re.MULTILINE)


def split_legs(report_cell: str) -> list[dict]:
    """Split a joined `report` cell back into its per-attempt legs, in order."""
    if not isinstance(report_cell, str) or not report_cell.strip():
        return []
    marks = list(LEG_HEADER.finditer(report_cell))
    if not marks:
        # A single-leg episode, or a report written before the headers existed.
        return [{"step": None, "attempt": 1, "reason": None, "text": report_cell}]
    legs = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(report_cell)
        body = report_cell[mark.end():end]
        # An optional "HINT: ..." line sits between the header and the trajectory.
        legs.append({
            "step": int(mark.group(1)), "attempt": int(mark.group(2)),
            "reason": mark.group(3), "step_text": mark.group(4), "text": body,
        })
    return legs


def load_supervisor_calls(row) -> tuple[list, dict]:
    """Split the recorded supervisor calls into the planning preamble and per-leg groups.

    The supervisor's calls are the other half of the episode and are recorded separately
    from the executor's, because they never pass through an ``ExecutorReport``. Each carries
    the leg it was made for, so they can be put back in sequence rather than listed apart
    — which is the difference between a log and a readable account of what happened.

    :return: ``(planning_calls, {(step_index, attempt): [calls]})``.
    """
    raw = row.get("supervisor_calls")
    if not isinstance(raw, str) or not raw.strip():
        return [], {}
    try:
        calls = json.loads(raw)
    except json.JSONDecodeError:
        return [], {}

    planning, per_leg = [], {}
    for call in calls:
        if call.get("phase") == "leg":
            key = (call.get("step_index"), call.get("attempt"))
            per_leg.setdefault(key, []).append(call)
        else:
            planning.append(call)
    return planning, per_leg


def _leg_key(header_key):
    """Executor-header (step, attempt) -> supervisor-context (step_index, attempt).

    The report headers number steps from 1 (``===== STEP 1 ATTEMPT 1``) while the
    supervisor records the loop's own 0-based ``index``. Same leg, two conventions.
    """
    step, attempt = header_key
    return (None if step is None else step - 1, attempt)


def render_call(call, index: int, max_prompt_chars: int) -> list[str]:
    """One supervisor call as a collapsible block."""
    prompt = call.get("prompt") or ""
    if max_prompt_chars:
        prompt = prompt[:max_prompt_chars]
    response = (call.get("response") or "").strip()
    return [
        f"<details><summary><b>supervisor {index}. [{call.get('stage')}]</b> — "
        f"{response[:110].replace(chr(10), ' ')}…</summary>",
        "", "**prompt**", "", "```", prompt, "```", "",
        "**response**", "", "```", response, "```", "", "</details>", "",
    ]


def action_from_output(output: str):
    """The action string the executor parsed out of this call, or None."""
    found = ACTION_LINE.findall(output or "")
    return found[-1].strip() if found else None


def replay_episode(row, bench_row, images_dir: str, controller_variant: str = "low_level"):
    """Re-run the recorded actions, capturing the frame each call was looking at.

    :return: ``(calls, n_replayed, note)`` where each call dict gains ``frame_path`` and
        ``replay_action``.
    """
    from gameboy_worlds import get_test_environment

    legs = split_legs(row["report"])
    if not legs:
        return [], 0, "no report text"

    environment = get_test_environment(
        row=bench_row, controller_variant=controller_variant, headless=True,
        save_video=False, session_name=f"debug_plan_replay/{bench_row.name}",
        max_steps=100000, wait_ticks=20,
    )
    calls, n_env = [], 0
    try:
        for leg in legs:
            for call in parse_report(leg["text"]):
                call.update({"leg_step": leg.get("step"), "leg_attempt": leg.get("attempt"),
                             "leg_reason": leg.get("reason"),
                             "leg_step_text": leg.get("step_text")})
                # Captured BEFORE stepping: this is the screen the call reasoned over.
                frame = environment.get_info()["core"]["current_frame"]
                path = os.path.join(
                    images_dir,
                    f"s{leg.get('step') or 0}_a{leg.get('attempt') or 0}_c{call['call']}.jpg",
                )
                to_pil(frame).save(path, "JPEG", quality=88)
                call["frame_path"] = path

                action = action_from_output(call["output"])
                call["replay_action"] = action
                if action and call["tag"] in ("action", "score", "decide"):
                    observation, *_ = environment.step_str(action)
                    if observation is not None:
                        n_env += 1
                calls.append(call)
    finally:
        environment.close()

    # Checked against the report's own ENV records, NOT the CSV's n_steps: n_steps is the
    # emulator's step counter, and a single high-level action may tick it more than once
    # (see emulator.step, "some HighLevelActions may call step() multiple times"), so it
    # runs a fixed amount ahead of the action count and would flag every episode.
    note = ""
    recorded = sum(1 for call in calls if call["outcome"].startswith("ENV"))
    if recorded and abs(recorded - n_env) > 1:
        note = (f"replay took {n_env} env steps, the run recorded {recorded} actions — the "
                f"frames below may have diverged from what the model saw")
    return calls, n_env, note


@click.command()
@click.option("--game", required=True)
@click.option("--csv", "csv_path", required=True, help="An info_plan_*.csv.")
@click.option("--episodes", default="0", help="Comma-separated row indices, or 'all'.")
@click.option("--max_prompt_chars", default=0, show_default=True,
              help="Prompt characters kept per call; 0 (the default) keeps the whole thing. "
                   "Truncating hides the end of a prompt, which is where the judgement, the "
                   "insights and the response format live — a reader then sees a prompt that "
                   "looks like it never carried them.")
@click.option("--controller_variant", default="low_level", show_default=True)
def main(game, csv_path, episodes, max_prompt_chars, controller_variant):
    """Render per-call trajectories with frames for chosen episodes of a plan run."""
    from gameboy_worlds import get_benchmark_tasks

    parameters = load_parameters()
    frame = pd.read_csv(csv_path)
    bench = get_benchmark_tasks(game=game)

    indices = (list(range(len(frame))) if episodes == "all"
               else [int(i) for i in episodes.split(",") if i.strip()])

    out_dir = os.path.join(parameters["results_dir"], "debug", game, "info_plan", "replay")
    # Cleared, not merged into. These files are rebuilt from one CSV, and that CSV is
    # rewritten wholesale by every --regenerate run, so a leftover episode_N.md describes a
    # run that no longer exists — while the report beside it, rebuilt from the new CSV,
    # links to it as if it were current. Two files disagreeing about whether an episode
    # succeeded is worse than a missing file.
    if episodes == "all" and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for index in indices:
        if index >= len(frame):
            log_warn(f"[replay] episode {index} is past the end of {csv_path}")
            continue
        row = frame.iloc[index]
        images_dir = os.path.join(out_dir, f"episode_{index}_frames")
        os.makedirs(images_dir, exist_ok=True)

        log_info(f"[replay] episode {index}: {row['task']}")
        calls, n_env, note = replay_episode(row, bench.iloc[index], images_dir,
                                            controller_variant)

        lines = [
            f"# episode {index} — {row['task']}",
            "",
            f"- success: **{row['success']}**   env steps: **{row['n_steps']}**   "
            f"replayed: **{n_env}**   VLM calls: **{len(calls)}**",
            f"- plan: {row['n_steps_cleared']}/{row['n_plan_steps']} cleared over "
            f"{row['n_attempts']} attempts, {row.get('n_replans', 0)} replans",
            "",
        ]
        if note:
            lines += [f"> **WARNING** {note}", ""]
        if isinstance(row.get("hint"), str):
            lines += ["**Plan**", ""]
            lines += [f"{i + 1}. {s.strip()}"
                      for i, s in enumerate(row["hint"].split("[STEP]"))]
            lines += [""]

        planning, per_leg = load_supervisor_calls(row)
        if planning:
            lines += ["---", "",
                      "## before the episode — retrieval, filtering and planning", ""]
            for i, call in enumerate(planning):
                lines += render_call(call, i + 1, max_prompt_chars)

        # Supervisor calls are emitted AFTER the leg they belong to, because that is when
        # they happened: the executor plays, then the supervisor judges what it did and
        # decides what to say next.
        emitted = set()
        current = None
        for call in calls:
            key = (call["leg_step"], call["leg_attempt"])
            if key != current:
                if current is not None:
                    for i, sup in enumerate(per_leg.get(_leg_key(current), [])):
                        lines += render_call(sup, i + 1, max_prompt_chars)
                    emitted.add(_leg_key(current))
                current = key
                lines += [
                    "---", "",
                    f"## step {call['leg_step']} · attempt {call['leg_attempt']} "
                    f"→ `{call['leg_reason']}`",
                    "",
                    f"> {call['leg_step_text']}",
                    "",
                ]
            rel = os.path.relpath(call["frame_path"], out_dir)
            lines += [
                f"### call {call['call']} [{call['tag']}] → `{call['outcome'] or '—'}`",
                "",
                f"![call {call['call']}]({rel})",
                "",
                f"**Action parsed:** `{call['replay_action'] or '(none)'}`",
                "",
                "<details><summary>prompt</summary>", "",
                "```",
                call["prompt"][:max_prompt_chars] if max_prompt_chars else call["prompt"],
                "```", "", "</details>", "",
                "**VLM output**", "", "```", call["output"], "```", "",
            ]

        # The last leg's supervisor calls, and anything whose leg produced no executor
        # calls at all (an attempt that failed before acting still gets judged).
        if current is not None:
            for i, sup in enumerate(per_leg.get(_leg_key(current), [])):
                lines += render_call(sup, i + 1, max_prompt_chars)
            emitted.add(_leg_key(current))
        leftover = [key for key in per_leg if key not in emitted]
        if leftover:
            lines += ["---", "", "## supervisor calls with no executor calls beside them", ""]
            for key in sorted(leftover, key=lambda k: (k[0] or 0, k[1] or 0)):
                lines += [f"**step {key[0]} · attempt {key[1]}**", ""]
                for i, sup in enumerate(per_leg[key]):
                    lines += render_call(sup, i + 1, max_prompt_chars)

        path = os.path.join(out_dir, f"episode_{index}.md")
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        log_info(f"[replay] wrote {path} ({len(calls)} calls, {len(os.listdir(images_dir))} frames)")


if __name__ == "__main__":
    main()
