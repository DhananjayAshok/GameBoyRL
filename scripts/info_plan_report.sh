#!/usr/bin/env bash
# Deep diagnostics for the info_subgoal arm: writes results/debug/<game>/info_subgoal/report<suffix>.md
# and regenerates the per-episode frame-by-frame replays it links to.
#
# Reads the two CSVs an info_subgoal run leaves behind — the baseline (dummy) arm and the
# info_subgoal_retrieval arm — so it only makes sense after both have been run for the same
# <stem> (model name tail) and <suffix> (e.g. _first5).
#
# Previously lived in tmp.sh and was called positionally from run.sh; it is here so that
# tmp.sh stays a throwaway scratch file.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["suffix"]="_first5"
ARGS["stem"]="gemma-4-31b-it"
ARGS["executor"]="single_actions"
ARGS["controller_variant"]="low_level"

REQUIRED_ARGS=("game")


# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

GAME="${ARGS["game"]}"
SUFFIX="${ARGS["suffix"]}"
STEM="${ARGS["stem"]}"
EXECUTOR="${ARGS["executor"]}"
CONTROLLER_VARIANT="${ARGS["controller_variant"]}"

GAME=$GAME SUFFIX=$SUFFIX STEM=$STEM EXECUTOR=$EXECUTOR \
    CONTROLLER_VARIANT=$CONTROLLER_VARIANT python - <<'PY'
"""Deep diagnostics for the info_subgoal arm: where the budget goes and why steps do not clear."""
import json
import os
from collections import Counter

import pandas as pd

from python_scripts import paths

GAME = os.environ.get("GAME", "deja_vu_1")
SUFFIX = os.environ.get("SUFFIX", "_first5")
STEM = os.environ.get("STEM", "gemma-4-31b-it")
# The arm name is part of both CSV paths now, so it cannot be hardcoded here.
EXECUTOR = os.environ.get("EXECUTOR", "single_actions")
CONTROLLER_VARIANT = os.environ.get("CONTROLLER_VARIANT", "low_level")
# Recovered from the suffix the caller passes ("" or "_firstN"), because that is the only
# form this script is given. paths.benchmark_csv re-applies it, so the two cannot disagree.
N_TASKS = int(SUFFIX[len("_first"):]) if SUFFIX.startswith("_first") else None

OUT_DIR = paths.debug_dir(game=GAME, stage="info_subgoal")

ARMS = {
    "baseline": paths.benchmark_csv(game=GAME, supervisor="dummy", executor=EXECUTOR,
                                    controller_variant=CONTROLLER_VARIANT,
                                    model=STEM, n_tasks=N_TASKS),
    "info subgoal": paths.benchmark_csv(game=GAME, supervisor="info_subgoal_retrieval",
                                        executor=EXECUTOR,
                                        controller_variant=CONTROLLER_VARIANT,
                                        model=STEM, n_tasks=N_TASKS),
}

lines = []


def emit(text=""):
    print(text)
    lines.append(text)


frames = {k: pd.read_csv(v) for k, v in ARMS.items() if os.path.exists(v)}
for label, path in ARMS.items():
    if label not in frames:
        emit(f"MISSING: {path}")

emit(f"# info-plan diagnostics — {GAME}{SUFFIX}")
emit()

# ---------------------------------------------------------------- 1. headline
emit("## 1. Success rates")
emit()
emit("| arm | n | success | rate | mean steps |")
emit("|---|---:|---:|---:|---:|")
for label, frame in frames.items():
    n, wins = len(frame), int(frame["success"].sum())
    emit(f"| {label} | {n} | {wins} | {100.0 * wins / max(n, 1):.1f}% | "
         f"{frame['n_steps'].mean():.1f} |")
emit()

base, plan = frames.get("baseline"), frames.get("info subgoal")

# ------------------------------------------------------- 2. paired comparison
# Joined on ROW INDEX: both runners walk get_benchmark_tasks() in order, and task strings
# repeat across init_states, so a text join fans out.
emit("## 2. Per task")
emit()
if base is not None and plan is not None:
    # `cleared` counts target SLOTS, so it is shown against `slots`, not against the plan
    # length — those differ after a replan, and reading one over the other is what produced
    # rows like "1/0 cleared".
    emit("| # | task | base | plan | plan steps | slots | cleared | attempts | replans "
         "| steps |")
    emit("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    for i in range(min(len(base), len(plan))):
        b, p = base.iloc[i], plan.iloc[i]
        emit(f"| {i} | {str(b['task'])[:40]} | {'W' if b['success'] else '.'} | "
             f"{'W' if p['success'] else '.'} | {p['n_plan_steps']} | "
             f"{p['n_slots_attempted']} | {p['n_steps_cleared']} | "
             f"{p['n_attempts']} | {p['n_replans']} | {p['n_steps']} |")
    emit()

# ----------------------------------------------- 3. per-attempt decomposition
records, steps_rows = [], []
if plan is not None:
    for episode, row in plan.iterrows():
        try:
            log = json.loads(row["step_log"]) if isinstance(row["step_log"], str) else []
        except (json.JSONDecodeError, TypeError):
            log = []
        for step_idx, record in enumerate(log):
            is_final = step_idx == len(log) - 1
            steps_rows.append({
                "episode": episode, "step_idx": step_idx, "is_final": is_final,
                "cleared": record.get("cleared", False),
                "n_attempts": len(record.get("attempts", [])),
                "n_replans": len(record.get("replans", [])),
                "replans": record.get("replans", []),
                "steps_spent": sum(a.get("n_steps", 0) for a in record.get("attempts", [])),
                "step_text": record.get("step") or "", "success": bool(row["success"]),
            })
            for attempt_idx, attempt in enumerate(record.get("attempts", [])):
                records.append({
                    "episode": episode, "step_idx": step_idx, "is_final": is_final,
                    "attempt_idx": attempt_idx,
                    "reason": attempt.get("termination_reason"),
                    "n_steps": attempt.get("n_steps", 0),
                    "cleared": attempt.get("cleared", False),
                    "hint": attempt.get("hint"), "judgement": attempt.get("judgement"),
                })

attempts = pd.DataFrame(records)
steps = pd.DataFrame(steps_rows)

emit("## 3. Where the step budget goes")
emit()
if len(attempts):
    total = attempts.n_steps.sum()
    final = attempts[attempts.is_final].n_steps.sum()
    emit(f"{total} env steps over {len(attempts)} attempts in {attempts.episode.nunique()} episodes")
    emit()
    emit("| bucket | attempts | env steps | share |")
    emit("|---|---:|---:|---:|")
    emit(f"| intermediate steps | {int((~attempts.is_final).sum())} | {total - final} | "
         f"{100.0 * (total - final) / max(total, 1):.0f}% |")
    emit(f"| FINAL step loop | {int(attempts.is_final.sum())} | {final} | "
         f"{100.0 * final / max(total, 1):.0f}% |")
    emit()
    fin = steps[steps.is_final]
    emit(f"- final-step attempts per episode: mean **{fin.n_attempts.mean():.1f}**, "
         f"max **{fin.n_attempts.max()}**")
    emit(f"- **replans on the FINAL step: {int(fin.n_replans.sum())}** "
         f"(the change under test — 0 means it never fired)")
    emit(f"- replans on intermediate steps: "
         f"{int(steps[~steps.is_final].n_replans.sum())}")
    emit()

# ------------------------------------------------------ 4. step clearing
emit("## 4. Do steps clear?")
emit()
if len(steps):
    inter = steps[~steps.is_final]
    emit(f"- plan steps executed: **{len(steps)}** ({len(inter)} intermediate, "
         f"{int(steps.is_final.sum())} final)")
    if len(inter):
        emit(f"- intermediate cleared: **{int(inter.cleared.sum())}/{len(inter)}** "
             f"({100.0 * inter.cleared.mean():.0f}%)")
        emit(f"- mean attempts on an intermediate step: **{inter.n_attempts.mean():.1f}**")
    cleared_at = attempts[(~attempts.is_final) & attempts.cleared]
    hist = Counter(cleared_at.attempt_idx + 1)
    if hist:
        emit()
        emit("| clears on attempt # | count |")
        emit("|---:|---:|")
        for k in sorted(hist):
            emit(f"| {k} | {hist[k]} |")
    emit()

# ------------------------------------------------- 5. termination reasons
emit("## 5. How each attempt ended")
emit()
if len(attempts):
    table = attempts.groupby(["is_final", "reason"]).size().reset_index(name="n")
    emit("| final step? | termination | attempts |")
    emit("|---|---|---:|")
    for _, r in table.iterrows():
        emit(f"| {'final' if r.is_final else 'intermediate'} | {r.reason} | {r.n} |")
    emit()
    inter = attempts[~attempts.is_final]
    if len(inter):
        ad = inter[inter.reason == "agent_done"]
        emit(f"- intermediate `agent_done`: **{len(ad)}**/{len(inter)}; judge agreed on "
             f"**{int(ad.cleared.sum())}**")
    emit()

# ------------------------------------------------------------- 6. the plans
emit("## 5b. The supervisor's own calls")
emit()
sup_calls = pd.DataFrame()
# The per-stage breakdown used to be built from a `supervisor_calls` JSON column. That column
# is gone: the calls now live on the archived SupervisorReport's event_log, interleaved with
# the executor legs they drove, where they keep the images each call saw. `n_supervisor_calls`
# survives as a scalar column because it is a number worth sorting and plotting on.
if plan is not None and "n_supervisor_calls" in plan.columns:
    emit(f"Total supervisor calls across {len(plan)} episode(s): "
         f"**{int(plan['n_supervisor_calls'].sum())}**. The prompts and replies behind them "
         f"are in `report.pkl.gz` under each episode's `session_dirs` path, in call order.")
    emit()
if plan is not None and "n_insights_candidate" in plan.columns:
    emit("**Insight funnel per episode** — retrieved, kept by the filter, after "
         "distillation. A big drop at either stage is the first thing to suspect when "
         "plans get worse.")
    emit()
    emit("| # | task | retrieved | kept | distilled |")
    emit("|---:|---|---:|---:|---:|")
    for i in range(len(plan)):
        r = plan.iloc[i]
        emit(f"| {i} | {str(r['task'])[:40]} | {r['n_insights_candidate']} | "
             f"{r['n_insights_kept']} | {r['n_insights_distilled']} |")
    emit()

emit("## 6. Plans, in full")
emit()
if plan is not None:
    for ep in range(len(plan)):
        row = plan.iloc[ep]
        emit(f"**{ep}. {row['task'][:70]}** — success={row['success']}, "
             f"{row['n_steps_cleared']}/{row['n_slots_attempted']} cleared, "
             f"{row['n_attempts']} attempts, {row['n_replans']} replans"
             f"{'' if row['planned'] else ', UNPLANNED'}")
        emit()
        if isinstance(row["hint"], str):
            for i, step in enumerate(row["hint"].split("[STEP]")):
                emit(f"  {i + 1}. {step.strip()}")
        else:
            emit("  _no plan produced_")
        emit()

# ------------------------------------------------------------ 7. every episode
emit("## 7. Attempt-by-attempt, every episode")
emit()
emit("Every VLM call and the screen it was looking at are in the archived supervisor report "
     "beside each episode's video, at the `session_dirs` path on its CSV row "
     "(`report.pkl.gz`).")
emit()
if len(steps):
    for ep in sorted(steps.episode.unique()):
        task = plan.iloc[ep]["task"]
        emit(f"### episode {ep} — {task[:70]} (success={plan.iloc[ep]['success']})")
        emit()
        emit(f"[full trajectory with frames](replay/episode_{ep}.md)")
        emit()
        for _, r in steps[steps.episode == ep].iterrows():
            tag = "FINAL" if r.is_final else f"step {r.step_idx + 1}"
            emit(f"**{tag}** — {r.n_attempts} attempts, {r.steps_spent} steps, "
                 f"cleared={r.cleared}, replans={r.n_replans}")
            emit()
            emit(f"> {r.step_text}")
            emit()
            for rev in r.replans:
                emit(f"  - REWRITTEN TO: {rev if rev else '(SKIP)'}")
            rows = attempts[(attempts.episode == ep) & (attempts.step_idx == r.step_idx)]
            for _, a in rows.iterrows():
                emit(f"  - attempt {a.attempt_idx + 1} [{a.reason}, {a.n_steps} steps]")
                if isinstance(a.hint, str):
                    emit(f"      hint : {a.hint[:180]}")
                if isinstance(a.judgement, str):
                    emit(f"      judge: {a.judgement[:180]}")
            emit()

emit("## 8. Every supervisor call, verbatim")
emit()
emit("The prompt and reply behind each decision above — filter, distillation, plan, "
     "judgement, hint, replan. Nothing else records these: they never reach an "
     "ExecutorReport, so this is the only place the supervisor's reasoning survives.")
emit()
if len(sup_calls):
    for ep in sorted(sup_calls.episode.unique()):
        emit(f"### episode {ep} — {plan.iloc[ep]['task'][:70]}")
        emit()
        for i, call in sup_calls[sup_calls.episode == ep].reset_index(drop=True).iterrows():
            emit(f"<details><summary><b>{i + 1}. [{call['stage']}]</b> — "
                 f"{str(call['response'])[:110].strip()}…</summary>")
            emit()
            emit("**prompt**")
            emit()
            emit("```")
            emit(str(call["prompt"]))
            emit("```")
            emit()
            emit("**response**")
            emit()
            emit("```")
            emit(str(call["response"]))
            emit("```")
            emit()
            emit("</details>")
            emit()

path = os.path.join(OUT_DIR, f"report{SUFFIX}.md")
with open(path, "w") as handle:
    handle.write("\n".join(lines) + "\n")
print()
print(f"Wrote {path}")
PY

# The frame-by-frame replay step is gone. It re-executed each episode on a fresh emulator to
# reconstruct the frames, because nothing on disk carried them. Every arm now archives its
# SupervisorReport to <session_dir>/report.pkl.gz, and those records hold the images each
# call actually saw — so the frames are read rather than rebuilt, and a divergent replay can
# no longer misrepresent a run.
