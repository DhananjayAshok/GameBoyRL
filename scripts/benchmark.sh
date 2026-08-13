#!/usr/bin/env bash
# Runs the full benchmark suite on a game with a specified executor and VLM model,
# recording video for each task attempt and logging all results to
# results/benchmark/<game>/. The --regenerate flag forces task regeneration before
# benchmarking.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Covers the three knowledge-free arms via --supervisor:
#   dummy     the control: one executor run per task, no supervisor reasoning
#   revision  short executor legs with a hint revised between them
#   subgoal   a plan written from the task alone, driven step by step
# The fourth arm, info_subgoal, lives in benchmark_plan.sh because it needs the document
# flags, and those must not be accepted by an arm with no plan to spend a document on.
#
# --max_leg_steps / --max_attempts_per_step / --max_replans apply only to the supervised
# arms and are not passed under --supervisor dummy, which takes no options of its own.
# That is what keeps the control arm the control.

# Script-specific defaults and required args
declare -A ARGS
ARGS["supervisor"]="dummy"
ARGS["executor"]="single_none"
ARGS["executor_vlm_model"]="Qwen/Qwen3-VL-8B-Instruct"   # use "none" for absent optionals, never ""
ARGS["executor_vlm_kind"]="huggingface"   # use "none" for absent optionals, never ""
ARGS["supervisor_vlm_model"]="none"   # unset falls back to the executor's model
ARGS["supervisor_vlm_kind"]="none"
ARGS["max_steps"]="50"
ARGS["max_leg_steps"]="5"
ARGS["max_attempts_per_step"]="3"
ARGS["max_replans"]="2"
ARGS["regenerate"]="false"

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

regenerate_flag=""
if [[ "${ARGS["regenerate"]}" == "true" ]]; then regenerate_flag="--regenerate"; fi

# The supervisor name is the registry key. The control arm's subcommand word is `baseline`
# rather than `dummy`, so it is mapped here rather than renamed out from under callers.
case "${ARGS["supervisor"]}" in
    dummy)
        subcommand="baseline"; arm_flags="" ;;
    revision)
        subcommand="revision"
        arm_flags="--max_leg_steps ${ARGS["max_leg_steps"]}" ;;
    subgoal)
        subcommand="subgoal"
        arm_flags="--max_leg_steps ${ARGS["max_leg_steps"]} --max_attempts_per_step ${ARGS["max_attempts_per_step"]} --max_replans ${ARGS["max_replans"]}" ;;
    info_subgoal)
        echo "Error: --supervisor info_subgoal needs --mode/--info_docs; use scripts/benchmark_plan.sh"
        exit 1 ;;
    *)
        echo "Error: unknown --supervisor '${ARGS["supervisor"]}' (dummy, revision, subgoal)"
        exit 1 ;;
esac

# Passed only when set: the group resolves an unset supervisor model to the executor's, and
# forwarding the literal "none" would name a model that does not exist.
supervisor_flags=""
if [[ "${ARGS["supervisor_vlm_model"]}" != "none" ]]; then
    supervisor_flags="--supervisor_vlm_model ${ARGS["supervisor_vlm_model"]}"
fi
if [[ "${ARGS["supervisor_vlm_kind"]}" != "none" ]]; then
    supervisor_flags="$supervisor_flags --supervisor_vlm_kind ${ARGS["supervisor_vlm_kind"]}"
fi

# Group options precede the subcommand word: run_benchmark.py is a click group.
group_flags="--game ${ARGS["game"]} --executor ${ARGS["executor"]} --save_video True --max_steps ${ARGS["max_steps"]} --executor_vlm_model ${ARGS["executor_vlm_model"]} --executor_vlm_kind ${ARGS["executor_vlm_kind"]} $supervisor_flags $regenerate_flag"

common="python run_benchmark.py $group_flags $subcommand $arm_flags"

echo "$common"
$common