"""
Paired comparison of two arbitrary benchmark CSVs, with replayed frames.

Input
-----
Two CSVs written by run_benchmark.py (or its _retry / _info variants):
    game, task, success, n_resets, n_steps, n_invalid, subgoals_reached, all_subgoals, report
Unlike ``debug.py benchmark``, the two CSVs are given as **paths**, so any two runs can be
compared (base vs fine-tuned, two executors, two models, two prompting variants).

Tasks are paired on the ``task`` column and split into the four quadrants
(both pass, both fail, A only, B only); ``--n_examples`` episodes from each are rendered.

Frames
------
The per-call PNGs ExecutorReport writes are keyed on the executor *class* and only written
under ``--verbose``, so they cannot be relied on for a model comparison (see
debug_scripts/benchmark.py). Instead each episode's frames are **replayed**: the CSV's
``report`` column records every step as ``ENV <ActionClass>({kwargs})``, and the benchmark
row supplies ``init_state`` / ``state_tracker_class``, so the episode can be re-executed on
the emulator from the same start with the same actions. CPU + ROM only — no VLM, no GPU.

Two things this does *not* claim:
  - The frames are **environment** frames, not the model's literal visual input. An
    executor may feed annotated screens or several images per call; those are not
    reconstructable from the CSV.
  - Replay fidelity is not verified against the original run. If a game drives RNG from
    timing rather than input alone, a replay can diverge from the recorded episode. The
    report says so in its header rather than implying the frames are authoritative.

INVALID calls (the model emitted something unparseable) advance no step and so produce no
frame; they are kept in the action listing so the misparse rate stays visible.

Output
------
<results_dir>/debug/<game>/compare/<label_a>__vs__<label_b>.md
<results_dir>/debug/<game>/compare/frames/<label>__<task>.png
"""

import ast
import os
import re

import click
import pandas as pd

from execution.report import DONE_CHECK_TAG
from utils import log_info, log_warn, log_error
from debug_scripts import markdown as md
from debug_scripts import frames as frames_mod
from debug_scripts.benchmark import parse_report, _load
from utils.paths import Paths
from debug_scripts.stats import mcnemar_exact, wilson_str

# `ENV   LowLevelAction({'low_level_action': <LowLevelActions.PRESS_BUTTON_A: 5>})`
ENV_STEP = re.compile(r"^ENV\s+(\w+)\((.*)\)$", re.DOTALL)
INVALID_STEP = re.compile(r"^INVALID\s*\((.*)\)$", re.DOTALL)
# Enum members render as `<LowLevelActions.PRESS_BUTTON_A: 5>`, which is not literal-eval'able.
ENUM_REPR = re.compile(r"<(\w+)\.(\w+):\s*[^>]*>")
ENUM_SENTINEL = "__ENUM__"

QUADRANTS = [
    ("both_pass", "Both pass"),
    ("a_only", "{a} passes, {b} fails"),
    ("b_only", "{b} passes, {a} fails"),
    ("both_fail", "Both fail"),
]


def _resolve_enum(qualified: str):
    """``LowLevelActions.PRESS_BUTTON_A`` -> the enum member, searched over the modules
    that define the pipeline's action enums."""
    class_name, _, member = qualified.partition(".")
    from gameboy_worlds.emulation import emulator
    from gameboy_worlds.interface import controller
    for module in (emulator, controller):
        enum_class = getattr(module, class_name, None)
        if enum_class is not None:
            try:
                return getattr(enum_class, member)
            except AttributeError:
                break
    raise ValueError(f"cannot resolve enum '{qualified}'")


def parse_kwargs(text: str) -> dict:
    """Recover the kwargs dict from a rendered step summary.

    ``_step_summary`` f-strings the dict, so enum values arrive as ``<Enum.MEMBER: n>``.
    They are swapped for sentinel strings, literal-eval'd, then swapped back. Raises on
    anything it cannot round-trip — a silently dropped action would misalign every frame
    that follows it.
    """
    if not text.strip():
        return {}
    substituted = ENUM_REPR.sub(lambda m: f"'{ENUM_SENTINEL}{m.group(1)}.{m.group(2)}'", text)
    try:
        parsed = ast.literal_eval(substituted)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"un-parseable step kwargs: {text!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"step kwargs did not parse to a dict: {text!r}")
    return {
        key: _resolve_enum(value[len(ENUM_SENTINEL):])
        if isinstance(value, str) and value.startswith(ENUM_SENTINEL) else value
        for key, value in parsed.items()
    }


def episode_actions(report: str) -> list[dict]:
    """Ordered action records for one episode, from its rendered report.

    :return: list of ``{"call", "kind", "action", "kwargs", "label", "text"}`` where *kind*
        is ``"env"`` (advanced the emulator), ``"invalid"`` (unparseable model output) or
        ``"other"`` (tool calls and anything unrecognised).

    .. todo:: Assumes one outcome per call (``call["outcome"]``, singular). A VLM call can
        now own several steps — see ``VLMCallRecord.steps`` and
        ``SequencePlannerExecutor`` — so an episode from the "sequence" executor silently
        loses every action but one. Depends on the matching fix in
        ``debug_scripts.benchmark.parse_report``, which drops the extra outcome lines
        before they ever reach here.
    """
    records = []
    for call in parse_report(report):
        # The completion check renders an outcome line ("COMPLETE: no"), so it is no longer
        # excluded by the empty-outcome test below. It advanced nothing, and this function's
        # output is replayed as an action sequence, so it is dropped on the tag.
        if call.get("tag") == DONE_CHECK_TAG:
            continue
        outcome = (call.get("outcome") or "").strip()
        if not outcome:
            continue
        env_match = ENV_STEP.match(outcome)
        if env_match:
            action_name, raw_kwargs = env_match.group(1), env_match.group(2)
            kwargs = parse_kwargs(raw_kwargs)
            records.append({
                "call": call["call"], "kind": "env", "action": action_name,
                "kwargs": kwargs, "label": _short_label(action_name, kwargs), "text": outcome,
            })
            continue
        invalid_match = INVALID_STEP.match(outcome)
        if invalid_match:
            records.append({
                "call": call["call"], "kind": "invalid", "action": None, "kwargs": {},
                "label": "INVALID", "text": outcome,
            })
            continue
        records.append({
            "call": call["call"], "kind": "other", "action": None, "kwargs": {},
            "label": "", "text": outcome,
        })
    return records


def _short_label(action_name: str, kwargs: dict) -> str:
    """Compact label for a frame panel — the button, not the wrapper class."""
    low_level = kwargs.get("low_level_action")
    if low_level is not None:
        return (
            str(low_level)
            .replace("LowLevelActions.PRESS_BUTTON_", "")
            .replace("LowLevelActions.PRESS_ARROW_", "")
        )
    return action_name[:12]


def _benchmark_row(game: str, task: str, parameters: dict):
    """The benchmark row (init_state, state_tracker_class, ...) backing one task."""
    from gameboy_worlds import get_benchmark_tasks
    tasks = get_benchmark_tasks(game=game)
    matches = tasks[tasks["task"] == task]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def replay(game: str, task: str, actions: list[dict], parameters: dict,
           controller_variant: str, session_label: str) -> tuple[list, list]:
    """Re-execute one episode's env actions from its init_state.

    :return: ``(frames, labels)`` — the initial frame followed by the frame after each
        successful env step, with the action that produced each. ``([], [])`` when the
        task has no benchmark row or the emulator fails.
    """
    from gameboy_worlds import get_test_environment

    row = _benchmark_row(game, task, parameters)
    if row is None:
        log_warn(f"[compare] no benchmark row for task '{task}' in {game} — skipping replay.")
        return [], []

    env_actions = [record for record in actions if record["kind"] == "env"]
    if not env_actions:
        return [], []

    task_str = task.replace(" ", "_").lower()
    environment = None
    try:
        environment = get_test_environment(
            row=row,
            controller_variant=controller_variant,
            headless=True,
            save_video=False,
            session_name=f"debug_compare_{session_label}/{task_str}/",
            # Headroom over the recorded episode so replay is never truncated early.
            max_steps=len(env_actions) + 10,
            wait_ticks=20,
        )
        # run_benchmark.py hands the freshly constructed env straight to the executor
        # without reset(), so the recorded trajectory starts here. Do the same.
        info = environment.get_info()
        collected = [info["core"]["current_frame"]]
        labels = [""]
        action_classes = {cls.__name__: cls for cls in type(environment._controller).ACTIONS}
        for record in env_actions:
            action_class = action_classes.get(record["action"])
            if action_class is None:
                log_warn(
                    f"[compare] action '{record['action']}' is not in "
                    f"{type(environment._controller).__name__}.ACTIONS — stopping replay of "
                    f"'{task}' at call {record['call']}."
                )
                break
            _obs, _reward, terminated, truncated, state = environment.step_high_level_action(
                action_class, **record["kwargs"]
            )
            collected.append(state["core"]["current_frame"])
            labels.append(record["label"])
            if terminated or truncated:
                break
        return collected, labels
    except Exception as exc:
        log_warn(f"[compare] replay of '{task}' failed: {exc}")
        return [], []
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass


def _episode_block(label: str, row, actions: list[dict], strip_path: str | None,
                   report_dir: str, max_actions: int) -> str:
    n_env = sum(1 for record in actions if record["kind"] == "env")
    n_invalid = sum(1 for record in actions if record["kind"] == "invalid")
    lines = [md.para(
        f"**{label}** — success: `{bool(row['success'])}` · steps: {row['n_steps']} · "
        f"invalid: {row['n_invalid']} · resets: {row['n_resets']} · "
        f"subgoals: {row['subgoals_reached_n']}/{row['all_subgoals_n']} · "
        f"parsed calls: {len(actions)} ({n_env} env, {n_invalid} invalid)"
    )]
    if strip_path:
        lines.append(md.img(f"{label} — {row.name}", strip_path, report_dir))
    else:
        lines.append(md.para("_(no replayed frames)_"))
    shown = actions[:max_actions] if max_actions else actions
    listing = "\n".join(
        f"{i:>3}  [{record['kind']:<7}] {record['text']}"
        for i, record in enumerate(shown, 1)
    )
    if max_actions and len(actions) > max_actions:
        listing += f"\n     … {len(actions) - max_actions} further calls omitted (--max_actions)"
    lines.append(md.details(f"{label} — actions selected ({len(actions)} calls)", md.code(listing)))
    return "\n".join(lines)


@click.command(name="compare")
@click.option("--csv_a", required=True, help="Path to the first benchmark CSV.")
@click.option("--csv_b", required=True, help="Path to the second benchmark CSV.")
@click.option("--label_a", default=None, help="Short name for A. Defaults to the CSV stem.")
@click.option("--label_b", default=None, help="Short name for B. Defaults to the CSV stem.")
@click.option("--n_examples", default=2, show_default=True,
              help="Episodes rendered per quadrant (0 = all).")
@click.option("--n_frames", default=8, show_default=True,
              help="Frames sampled into each episode's strip.")
@click.option("--max_actions", default=60, show_default=True,
              help="Cap actions listed per episode (0 = all).")
@click.option("--controller_variant", default="low_level", show_default=True,
              help="Controller variant used for replay; must match the benchmarked run.")
@click.option("--replay/--no_replay", "do_replay", default=True, show_default=True,
              help="Re-execute episodes on the emulator to produce frames.")
@click.pass_obj
def debug_compare(obj, csv_a, csv_b, label_a, label_b, n_examples, n_frames,
                  max_actions, controller_variant, do_replay):
    """Four-quadrant comparison of two benchmark CSVs, with replayed frames."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], output_dir=obj["output_dir"], mode=obj["mode"],
    )
    report_dir = paths.debug_dir("compare")
    frames_dir = paths.debug_dir("compare", "frames")
    overwrite = obj["overwrite"]

    label_a = label_a or os.path.splitext(os.path.basename(csv_a))[0]
    label_b = label_b or os.path.splitext(os.path.basename(csv_b))[0]
    if label_a == label_b:
        label_a, label_b = f"{label_a} (A)", f"{label_b} (B)"

    a = _load(paths.require(csv_a, "benchmark"))
    b = _load(paths.require(csv_b, "benchmark"))
    log_info(f"[compare] {label_a}: {len(a)} tasks · {label_b}: {len(b)} tasks")

    games = set(a["game"]) | set(b["game"])
    if paths.game not in games:
        log_error(
            f"--game '{paths.game}' is not present in either CSV (found: {sorted(games)}). "
            f"Replay needs the right game to look up each task's init_state.",
            paths.parameters,
        )
    if len(games) > 1:
        log_warn(f"[compare] CSVs span several games {sorted(games)}; "
                 f"only '{paths.game}' rows can be replayed.")

    shared = sorted(set(a["task"]) & set(b["task"]))
    if not shared:
        log_error("The two CSVs share no tasks — nothing to compare.", paths.parameters)
    only_a = sorted(set(a["task"]) - set(shared))
    only_b = sorted(set(b["task"]) - set(shared))
    fa = a[a["task"].isin(shared)].drop_duplicates("task").set_index("task").loc[shared]
    fb = b[b["task"].isin(shared)].drop_duplicates("task").set_index("task").loc[shared]

    quadrant_tasks = {
        "both_pass": [t for t in shared if fa.loc[t, "success"] and fb.loc[t, "success"]],
        "a_only": [t for t in shared if fa.loc[t, "success"] and not fb.loc[t, "success"]],
        "b_only": [t for t in shared if fb.loc[t, "success"] and not fa.loc[t, "success"]],
        "both_fail": [t for t in shared if not fa.loc[t, "success"] and not fb.loc[t, "success"]],
    }

    # ---------------- examples, with replayed frames ----------------
    sections = []
    for key, title in QUADRANTS:
        tasks = quadrant_tasks[key]
        chosen = tasks[:n_examples] if n_examples else tasks
        blocks = [md.h2(f"{title.format(a=label_a, b=label_b)} — {len(tasks)} task(s)")]
        if not tasks:
            blocks.append(md.para("_(none)_"))
            sections.append("\n".join(blocks))
            continue
        blocks.append(md.para("Tasks: " + ", ".join(f"`{t}`" for t in tasks)))
        for task in chosen:
            blocks.append(md.h3(task))
            for label, frame in [(label_a, fa), (label_b, fb)]:
                row = frame.loc[task]
                actions = episode_actions(row["report"])
                strip_path = None
                if do_replay:
                    safe_label = re.sub(r"[^\w.-]", "_", label)
                    safe_task = re.sub(r"[^\w]", "_", task.lower()).strip("_")
                    out_path = os.path.join(frames_dir, f"{safe_label}__{safe_task}.png")
                    if os.path.exists(out_path) and not overwrite:
                        strip_path = out_path
                    else:
                        replayed, labels = replay(
                            game=paths.game, task=task, actions=actions,
                            parameters=paths.parameters,
                            controller_variant=controller_variant, session_label=safe_label,
                        )
                        if replayed:
                            strip_path = frames_mod.strip(
                                replayed, out_path, labels=labels,
                                n=n_frames, overwrite=True,
                            )
                blocks.append(_episode_block(label, row, actions, strip_path,
                                             report_dir, max_actions))
        if n_examples and len(tasks) > len(chosen):
            blocks.append(md.para(
                f"_… {len(tasks) - len(chosen)} further task(s) in this quadrant "
                f"(--n_examples)_"
            ))
        sections.append("\n".join(blocks))

    # ---------------- summary ----------------
    n = len(shared)
    hits_a, hits_b = int(fa["success"].sum()), int(fb["success"].sum())
    summary = pd.DataFrame([
        {"run": label_a, "success": hits_a, "n": n, "success_%": hits_a / n * 100,
         "subgoal_frac_%": fa["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fa["n_invalid"].mean(), "steps_per_ep": fa["n_steps"].mean()},
        {"run": label_b, "success": hits_b, "n": n, "success_%": hits_b / n * 100,
         "subgoal_frac_%": fb["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fb["n_invalid"].mean(), "steps_per_ep": fb["n_steps"].mean()},
    ])

    blocks = [
        md.h1(f"Benchmark comparison — {paths.game}"),
        md.bullets([
            f"**A** = `{label_a}` — `{csv_a}`",
            f"**B** = `{label_b}` — `{csv_b}`",
        ]),
        md.h2("Pairing"),
        md.bullets([
            f"tasks compared: **{n}** (intersection)",
            f"only in A: **{len(only_a)}** {'(' + ', '.join(only_a[:10]) + ')' if only_a else ''}",
            f"only in B: **{len(only_b)}** {'(' + ', '.join(only_b[:10]) + ')' if only_b else ''}",
        ]),
        md.h2("Summary"),
        md.table(summary),
        md.bullets([
            f"A success: {wilson_str(hits_a, n)}",
            f"B success: {wilson_str(hits_b, n)}",
            f"both pass: **{len(quadrant_tasks['both_pass'])}** · "
            f"A only: **{len(quadrant_tasks['a_only'])}** · "
            f"B only: **{len(quadrant_tasks['b_only'])}** · "
            f"both fail: **{len(quadrant_tasks['both_fail'])}**",
            f"McNemar exact p on the "
            f"{len(quadrant_tasks['a_only'])}/{len(quadrant_tasks['b_only'])} discordant "
            f"pairs: **{mcnemar_exact(len(quadrant_tasks['a_only']), len(quadrant_tasks['b_only'])):.3f}**",
        ]),
        md.note(
            "Frames are **replayed**: each episode's recorded actions are re-executed on the "
            "emulator from the task's init_state, because the per-call PNGs the executor "
            "writes are keyed on the executor class and only saved under --verbose. They are "
            "environment frames, not the model's literal visual input, and replay fidelity is "
            "not verified against the original run — if a game drives RNG from timing rather "
            "than input alone, a replay can diverge. Treat them as an illustration of the "
            "recorded action sequence, not as a byte-exact record of it."
        ),
    ] + sections

    stem_a = re.sub(r"[^\w.-]", "_", label_a)
    stem_b = re.sub(r"[^\w.-]", "_", label_b)
    path = md.write_report(os.path.join(report_dir, f"{stem_a}__vs__{stem_b}.md"), blocks)
    log_info(f"[compare] wrote {path}")
    print(path)
