# Core shared library sourced by every other script in this project.
# Loads the project config and virtual environment, then defines all default argument
# dictionaries (training, evaluation, VLM, sweep, iterative training, etc.) and the
# utility functions (args_to_flags, populate_dict, etc.) used for argument parsing
# and forwarding across the script hierarchy. Also defines the TRAINING_ARG_KEYS,
# EVALUATION_ARG_KEYS, and related arrays that callers use to subset flags when
# invoking sub-scripts.

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
PROJECT_ROOT=$(pwd)

# args_to_flags <assoc_array_name>
#
# Converts a bash associative array into a flat --key value string suitable
# for passing to a Python click command or another bash script.
# Empty values ("") are emitted as --key none.
#
# Usage (capture-safe — all diagnostics go to stderr):
#   declare -A ARGS=( ["lr"]="0.001" ["dataset"]="" )
#   flags=$(args_to_flags ARGS)
#   python get_strings.py <string_kind> $flags
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then
            val="none"
        fi
        result+="--${key} ${val} "
    done
    echo "${result% }"  # trim trailing space
}

# get_string_from_args <string_kind> <assoc_array_name>
#
# Utility function to get a string from get_strings.py by passing an associative array of args.
# Usage:
#   string=$(get_string_from_args <string_kind> ARGS)
function get_string_from_args() {
    local string_kind="$1"
    shift
    local flags=$(args_to_flags "$1")
    python $PROJECT_ROOT/scripts/python/get_strings.py "$string_kind" $flags
}



# args_to_flags_subset <assoc_array_name> <array_of_keys>
#
# Like args_to_flags, but only emits flags for the specified keys.
# Keys not present in the array are silently skipped.
# Use this when calling a subscript that doesn't accept all of the caller's ARGS.
#
# Usage:
#   subset=$(args_to_flags_subset ARGS REQUESTED_KEYS_ARRAY)
#   bash scripts/a.sh $subset
function args_to_flags_subset() {
    local -n _dict="$1"
    local -n _keys="$2"
    local result=""
    for key in "${_keys[@]}"; do
        if [[ -v _dict["$key"] ]]; then
            local val="${_dict[$key]}"
            if [[ -z "$val" ]]; then val="none"; fi
            result+="--${key} ${val} "
        fi
    done
    echo "${result% }"
}

function populate_dict(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    for key in "${!_source_dict[@]}"; do
        _target_dict["$key"]="${_source_dict[$key]}"
    done
}

function populate_array(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    _target_arr+=("${_source_arr[@]}")
}

function populate_dict_subset(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        if [[ -v _source_dict["$key"] ]]; then
            _target_dict["$key"]="${_source_dict[$key]}"
        fi
    done
}


################################################################################
ESSENTIAL_ARGS=("game") # should be game later. 
declare -A ALL_DEFAULTS=( # make this empty later
)

ENV_ESSENTIALS=("init_state")
declare -A ENV_DEFAULTS=(
    ["env"]="default"
    ["controller"]="low_level"
    ["max_steps"]=30
)


declare -A ALGORITHM_DEFAULTS=(
    ["timesteps"]=100000
    ["algorithm"]="ppo"
    ["gamma"]="0.99"
    ["seed"]=1
)

declare -A CURIOUSITY_DEFAULTS=(
    ["curiosity_module"]="combinationbuffer"
    ["ocr_alpha"]=0.0
    ["buffer_load_path"]="none"
    ["similarity_metric"]="cosine"    
    ["observation_embedder"]="random_patch"
    ["embedder_load_path"]="none"    
)

declare -A SAVE_DEFAULTS=(
    ["buffer_save_path"]="none"
    ["replay_buffer_save_folder"]="none"
    ["model_dir"]="none"
    ["overwrite_model"]=false
    ["log_folder"]="none"
    ["capture_video"]=false
)

declare -A TRAINING_DEFAULTS
TRAINING_ESSENTIALS=()
populate_dict ALL_DEFAULTS TRAINING_DEFAULTS
populate_dict ENV_DEFAULTS TRAINING_DEFAULTS
populate_dict ALGORITHM_DEFAULTS TRAINING_DEFAULTS
populate_dict CURIOUSITY_DEFAULTS TRAINING_DEFAULTS
populate_dict SAVE_DEFAULTS TRAINING_DEFAULTS
populate_array ESSENTIAL_ARGS TRAINING_ESSENTIALS
populate_array ENV_ESSENTIALS TRAINING_ESSENTIALS

TRAINING_ARG_KEYS=("${TRAINING_ESSENTIALS[@]}" "${!TRAINING_DEFAULTS[@]}")

declare -A EVALUATION_DEFAULTS
EVALUATION_ESSENTIALS=()
populate_dict ALL_DEFAULTS EVALUATION_DEFAULTS
populate_dict ENV_DEFAULTS EVALUATION_DEFAULTS
populate_dict SAVE_DEFAULTS EVALUATION_DEFAULTS
populate_dict CURIOUSITY_DEFAULTS EVALUATION_DEFAULTS
populate_array ESSENTIAL_ARGS EVALUATION_ESSENTIALS
populate_array ENV_ESSENTIALS EVALUATION_ESSENTIALS

EVALUATION_ESSENTIALS+=("algorithm" "exp_name")

EVALUATION_ARG_KEYS=("${EVALUATION_ESSENTIALS[@]}" "${!EVALUATION_DEFAULTS[@]}")

declare -A WORLD_MODEL_DEFAULTS
WORLD_MODEL_ESSENTIALS=("latest_replay_buffer_folder" "game")
populate_dict ALL_DEFAULTS WORLD_MODEL_DEFAULTS
populate_array ESSENTIAL_ARGS WORLD_MODEL_ESSENTIALS
SAME_AS_TRAINING=("observation_embedder" "embedder_load_path" "buffer_save_path" "buffer_load_path" "controller")
populate_dict_subset TRAINING_DEFAULTS WORLD_MODEL_DEFAULTS SAME_AS_TRAINING

WORLD_MODEL_ARG_KEYS=("${WORLD_MODEL_ESSENTIALS[@]}" "${!WORLD_MODEL_DEFAULTS[@]}")

declare -A SWEEP_DEFAULTS
SWEEP_ESSENTIALS=()
populate_array ESSENTIAL_ARGS SWEEP_ESSENTIALS
populate_array ENV_ESSENTIALS SWEEP_ESSENTIALS
populate_dict TRAINING_DEFAULTS SWEEP_DEFAULTS
SWEEP_DEFAULTS["best_k"]=6
SWEEP_DEFAULTS["clear_loser_replay_buffer"]=true

SWEEP_ARG_KEYS=("${SWEEP_ESSENTIALS[@]}" "${!SWEEP_DEFAULTS[@]}")

# declare an ITERATIVE_TRAINING_DEFAULTS if needed, and an ITERATIVE_TRAINING_ARG_KEYS array, if there are any args specific to iterative training that aren't already covered by the above categories. Otherwise, just use TRAINING_DEFAULTS and TRAINING_ARG_KEYS for iterative training as well.
ITERATIVE_TRAINING_ESSENTIALS=()
declare -A ITERATIVE_TRAINING_DEFAULTS
populate_array ESSENTIAL_ARGS ITERATIVE_TRAINING_ESSENTIALS
populate_array ENV_ESSENTIALS ITERATIVE_TRAINING_ESSENTIALS
populate_dict ALL_DEFAULTS ITERATIVE_TRAINING_DEFAULTS
populate_dict SWEEP_DEFAULTS ITERATIVE_TRAINING_DEFAULTS
ITERATIVE_TRAINING_DEFAULTS["n_agents"]=3
ITERATIVE_TRAINING_DEFAULTS["sweep"]=false
ITERATIVE_TRAINING_DEFAULTS["combination_buffer_sweep"]=true
ITERATIVE_TRAINING_DEFAULTS["init_state_group"]="none"
ITERATIVE_TRAINING_DEFAULTS["run_name"]="iterative"

ITERATIVE_TRAINING_ARG_KEYS=("${ITERATIVE_TRAINING_ESSENTIALS[@]}" "${!ITERATIVE_TRAINING_DEFAULTS[@]}")


CREATE_TRAJ_ESSENTIALS=()
declare -A CREATE_TRAJ_DEFAULTS
populate_array ITERATIVE_TRAINING_ESSENTIALS CREATE_TRAJ_ESSENTIALS
populate_dict ALL_DEFAULTS CREATE_TRAJ_DEFAULTS
populate_dict ITERATIVE_TRAINING_DEFAULTS CREATE_TRAJ_DEFAULTS
CREATE_TRAJ_DEFAULTS["skip_training"]=false
CREATE_TRAJ_DEFAULTS["z_min"]=6.0
CREATE_TRAJ_DEFAULTS["overwrite"]=false

CREATE_TRAJ_ARG_KEYS=("${CREATE_TRAJ_ESSENTIALS[@]}" "${!CREATE_TRAJ_DEFAULTS[@]}")

VLM_ESSENTIALS=("model_name" "vlm_kind")
populate_array ESSENTIAL_ARGS VLM_ESSENTIALS
declare -A VLM_DEFAULTS=(
    ["overwrite"]=false
    ["verbose"]=false
    ["max_new_tokens"]=1000
)
VLM_ARG_KEYS=("${VLM_ESSENTIALS[@]}" "${!VLM_DEFAULTS[@]}")

PROPOSE_ZEROSHOT_ESSENTIALS=("init_states")
populate_array VLM_ESSENTIALS PROPOSE_ZEROSHOT_ESSENTIALS
declare -A PROPOSE_ZEROSHOT_DEFAULTS
populate_dict VLM_DEFAULTS PROPOSE_ZEROSHOT_DEFAULTS

PROPOSE_ZEROSHOT_ARG_KEYS=("${PROPOSE_ZEROSHOT_ESSENTIALS[@]}" "${!PROPOSE_ZEROSHOT_DEFAULTS[@]}")

ATTEMPT_TASKS_ESSENTIALS=("tasks_path")
populate_array VLM_ESSENTIALS ATTEMPT_TASKS_ESSENTIALS
declare -A ATTEMPT_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS ATTEMPT_TASKS_DEFAULTS
ATTEMPT_TASKS_DEFAULTS["executor"]="history"
ATTEMPT_TASKS_DEFAULTS["max_steps"]=50
ATTEMPT_TASKS_DEFAULTS["max_tool_calls"]=10
ATTEMPT_TASKS_DEFAULTS["lookback"]=8
ATTEMPT_TASKS_DEFAULTS["controller_variant"]="low_level"
ATTEMPT_TASKS_DEFAULTS["max_attempts"]=5
ATTEMPT_TASKS_DEFAULTS["checker_max_new_tokens"]=1000

ATTEMPT_TASKS_ARG_KEYS=("${ATTEMPT_TASKS_ESSENTIALS[@]}" "${!ATTEMPT_TASKS_DEFAULTS[@]}")

INFER_TASKS_ESSENTIALS=("trajectory_path")
populate_array VLM_ESSENTIALS INFER_TASKS_ESSENTIALS
declare -A INFER_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS INFER_TASKS_DEFAULTS
INFER_TASKS_DEFAULTS["run_name"]="all"
INFER_TASKS_DEFAULTS["lookback"]=8
INFER_TASKS_DEFAULTS["max_trajectories_per_group"]=3
INFER_TASKS_DEFAULTS["describe_pairs"]=false

INFER_TASKS_ARG_KEYS=("${INFER_TASKS_ESSENTIALS[@]}" "${!INFER_TASKS_DEFAULTS[@]}")

# build_info: distils (task, trajectory) pairs into an info document, beside its own input
# stem. --stage a is enough for benchmark_info --mode
# init_state; the full run also builds the merge tree.
# source is required, not defaulted: it is the document's recorded provenance, and a
# wrong-but-plausible default would be written into the artifact and believed by every
# reader. build_info_all.sh has it in hand as a loop variable.
BUILD_INFO_ESSENTIALS=("trajectory_path" "source")
populate_array VLM_ESSENTIALS BUILD_INFO_ESSENTIALS
declare -A BUILD_INFO_DEFAULTS
populate_dict VLM_DEFAULTS BUILD_INFO_DEFAULTS
BUILD_INFO_DEFAULTS["n_frames"]=8
BUILD_INFO_DEFAULTS["max_concurrency"]=16
BUILD_INFO_DEFAULTS["stage"]="all"
# Names the output dir (info_<model>_<executor>), so documents distilled from one executor's
# trajectories never overwrite another's. The curiosity stem carries no executor of its own,
# so without this the two verticals would collide on the same path.
BUILD_INFO_DEFAULTS["executor"]="history"
BUILD_INFO_DEFAULTS["overwrite_from_round"]=none
# Override the shared VLM default of 1000: stage A emits six labelled fields plus a bullet
# list of insights, and the stage-B combine emits a merged list that grows with the entry.
# Truncation here does not fail loudly — it silently drops the trailing insights.
BUILD_INFO_DEFAULTS["max_new_tokens"]=3000

BUILD_INFO_ARG_KEYS=("${BUILD_INFO_ESSENTIALS[@]}" "${!BUILD_INFO_DEFAULTS[@]}")

# info_mode_sources <mode>
#
# The source verticals a --mode selects. Shares its vocabulary with full.sh's --mode so the
# context-engineering arm and the fine-tuning arm mean the same thing by the same word.
# Order matters: zeroshot first, because it is the verified-solution source and reads first
# in a comma-joined --insights_paths / --info_docs list.
function info_mode_sources() {
    case "$1" in
        curiosity_only) echo "curiosity" ;;
        zeroshot_only)  echo "zeroshot" ;;
        both)           echo "zeroshot curiosity" ;;
        *)              echo "" ;;
    esac
}

# info_source_stem <game> <model_save_name> <run_name> <executor> <source>
#
# The trajectory stem build_info.py consumes for one source: <stem>.json + <stem>.pkl.
# build_info writes its output beside this stem, so this function also fixes where the
# info dir lands (see info_dir_for_stem).
#
# Defined here because build_info_all.sh writes these
# directories and benchmark_info_all.sh reads them, and the two must not drift — neither
# passes the path to the other, both derive it from this function. utils/paths.py re-derives
# the identical rule on the python side (info_dir / curiosity_info_dir), and
# tests/test_paths.py pins the two against each other.
#
# model_save_name is in the path because a document is built from one model's own output;
# two models sharing a game must not share an info dir.
function info_source_stem() {
    case "$5" in
        zeroshot)
            echo "$storage_dir/proposed_tasks/$1/$2/zeroshot/zeroshot_tasks_$4_attempts/success_trajectories" ;;
        curiosity)
            echo "$storage_dir/proposed_tasks/$1/$2/curiosity/$3/trajectory_annotation" ;;
        *)
            echo "" ;;
    esac
}

# info_available_sources <game> <model_save_name> <run_name> <executor>
#
# Which of {zeroshot, curiosity} actually have inputs on disk for this game. The all-games
# sweep uses this to pick each game's --mode rather than assuming both exist: most games
# have only one vertical, and demanding both would skip them entirely.
function info_available_sources() {
    local found=""
    for source in zeroshot curiosity; do
        local stem
        stem=$(info_source_stem "$1" "$2" "$3" "$4" "$source")
        if [[ -f "$stem.json" && -f "$stem.pkl" ]]; then found+="${found:+ }$source"; fi
    done
    echo "$found"
}

# info_sources_to_mode <sources>
#
# Inverse of info_mode_sources: the --mode that selects exactly this set.
function info_sources_to_mode() {
    case "$1" in
        "zeroshot curiosity"|"curiosity zeroshot") echo "both" ;;
        "zeroshot")                                echo "zeroshot_only" ;;
        "curiosity")                               echo "curiosity_only" ;;
        *)                                         echo "" ;;
    esac
}

# info_dir_for_stem <stem> <model_save_name> <executor>
#
# Where build_info.py puts everything for that stem. Mirrors the one line in build_info.py
# that decides it:
#   os.path.join(os.path.dirname(trajectory_path), f"info_{model_save_name}_{executor}").
#
# The executor is in the name because it is the identity of the trajectories the document was
# distilled from. The zeroshot stem already encodes it (zeroshot_tasks_<executor>_attempts),
# but the curiosity stem does not — so before this, a curiosity document built from one
# executor's annotations silently overwrote another's.
function info_dir_for_stem() {
    echo "$(dirname "$1")/info_$2_$3"
}

# build_info_all: stage A/B for every source a --mode selects, plus the debug report.
BUILD_INFO_ALL_ESSENTIALS=()
populate_array VLM_ESSENTIALS BUILD_INFO_ALL_ESSENTIALS
BUILD_INFO_ALL_ESSENTIALS+=("run_name")
declare -A BUILD_INFO_ALL_DEFAULTS
populate_dict VLM_DEFAULTS BUILD_INFO_ALL_DEFAULTS
BUILD_INFO_ALL_DEFAULTS["max_new_tokens"]=3000
BUILD_INFO_ALL_DEFAULTS["executor"]="history"
BUILD_INFO_ALL_DEFAULTS["mode"]="both"
BUILD_INFO_ALL_DEFAULTS["stage"]="all"
BUILD_INFO_ALL_DEFAULTS["n_frames"]=8
BUILD_INFO_ALL_DEFAULTS["max_concurrency"]=16
BUILD_INFO_ALL_DEFAULTS["do_debug"]=true

BUILD_INFO_ALL_ARG_KEYS=("${BUILD_INFO_ALL_ESSENTIALS[@]}" "${!BUILD_INFO_ALL_DEFAULTS[@]}")

# benchmark_info_all: hinted benchmarks over the documents build_info_all produced, plus the
# no-hint baseline the results are only interpretable against.
BENCHMARK_INFO_ALL_ESSENTIALS=()
populate_array VLM_ESSENTIALS BENCHMARK_INFO_ALL_ESSENTIALS
BENCHMARK_INFO_ALL_ESSENTIALS+=("run_name")
declare -A BENCHMARK_INFO_ALL_DEFAULTS=(
    # One knob for both halves: which attempts/curiosity dirs the documents were built from,
    # AND the executor run at test time. These were separate (build=history, bench=simple)
    # and must not be — the no-hint baseline has to be the same executor as the hinted run,
    # or the delta mixes the hint effect with a scaffold change.
    ["executor"]="history"
    ["mode"]="both"
    ["hint_mode"]="both"
    ["baseline"]=true
    ["max_steps"]=50
    ["max_concurrency"]=8
    ["hint_vlm_model"]=none
    ["hint_vlm_kind"]=none
    ["regenerate"]=false
)

BENCHMARK_INFO_ALL_ARG_KEYS=("${BENCHMARK_INFO_ALL_ESSENTIALS[@]}" "${!BENCHMARK_INFO_ALL_DEFAULTS[@]}")

# info_full: the per-game driver. Its key set is the union of the two stages it calls, so
# the all-games sweep can forward one dict down without knowing the split.
INFO_FULL_ESSENTIALS=()
populate_array VLM_ESSENTIALS INFO_FULL_ESSENTIALS
INFO_FULL_ESSENTIALS+=("run_name")
declare -A INFO_FULL_DEFAULTS
populate_dict BUILD_INFO_ALL_DEFAULTS INFO_FULL_DEFAULTS
populate_dict BENCHMARK_INFO_ALL_DEFAULTS INFO_FULL_DEFAULTS
INFO_FULL_DEFAULTS["do_build"]=true
INFO_FULL_DEFAULTS["do_benchmark"]=true
# Longer episodes than a bare benchmark_info_all run: a hint the executor never gets far
# enough to use scores like the baseline for reasons unrelated to hint quality.
INFO_FULL_DEFAULTS["max_steps"]=150
# The two stages want different concurrency (build 16, benchmark 8) and the flat union above
# holds one value per key — the second populate_dict would win and silently halve the build.
# Split the key so both halves are reachable from here, and drop the merged one; info_full.sh
# assigns the right half to max_concurrency before each forwarding call.
INFO_FULL_DEFAULTS["build_max_concurrency"]="${BUILD_INFO_ALL_DEFAULTS["max_concurrency"]}"
INFO_FULL_DEFAULTS["bench_max_concurrency"]="${BENCHMARK_INFO_ALL_DEFAULTS["max_concurrency"]}"
unset INFO_FULL_DEFAULTS["max_concurrency"]

INFO_FULL_ARG_KEYS=("${INFO_FULL_ESSENTIALS[@]}" "${!INFO_FULL_DEFAULTS[@]}")

# info_full_all_games: the sweep around info_full. Inherits info_full's entire key set so a
# default moves in one place instead of three, then adds its own game-selection flags. The
# extras are deliberately absent from INFO_FULL_ARG_KEYS, which is what makes the
# args_to_flags_subset forward drop them on the way down.
#
# game is NOT essential here (it is in VLM_ESSENTIALS, hence in INFO_FULL_ESSENTIALS): a
# sweep discovers its own games, so requiring one would be requiring a value it ignores.
INFO_FULL_ALL_GAMES_ESSENTIALS=("model_name" "vlm_kind" "run_name")   # VLM_ESSENTIALS - game
declare -A INFO_FULL_ALL_GAMES_DEFAULTS
populate_dict INFO_FULL_DEFAULTS INFO_FULL_ALL_GAMES_DEFAULTS
# mode is discovered per game from what is on disk (info_available_sources ->
# info_sources_to_mode), so it must not be user-settable here. Unsetting it removes it from
# the sweep's ALLOWED_FLAGS; forwarding still emits it, because the loop assigns
# ARGS["mode"] before the subset call and args_to_flags_subset keys on presence.
unset INFO_FULL_ALL_GAMES_DEFAULTS["mode"]
# Accepted and ignored, so the flag set stays uniform with the other pipeline scripts. The
# loop overwrites it per game. Use --games to restrict the sweep.
INFO_FULL_ALL_GAMES_DEFAULTS["game"]=none
INFO_FULL_ALL_GAMES_DEFAULTS["games"]="auto"
INFO_FULL_ALL_GAMES_DEFAULTS["skip_games"]=none
INFO_FULL_ALL_GAMES_DEFAULTS["dry_run"]=false

INFO_FULL_ALL_GAMES_ARG_KEYS=("${INFO_FULL_ALL_GAMES_ESSENTIALS[@]}" "${!INFO_FULL_ALL_GAMES_DEFAULTS[@]}")

# curiosity_tasks + curiosity_all_tasks: create_traj → infer_tasks
# run_name is required (no default) — removed from DEFAULTS after population
CURIOSITY_TASKS_ESSENTIALS=("run_name")
populate_array VLM_ESSENTIALS CURIOSITY_TASKS_ESSENTIALS
populate_array ENV_ESSENTIALS CURIOSITY_TASKS_ESSENTIALS

declare -A CURIOSITY_TASKS_DEFAULTS
populate_dict CREATE_TRAJ_DEFAULTS CURIOSITY_TASKS_DEFAULTS
populate_dict INFER_TASKS_DEFAULTS CURIOSITY_TASKS_DEFAULTS
unset CURIOSITY_TASKS_DEFAULTS["run_name"]

CURIOSITY_TASKS_ARG_KEYS=("${CURIOSITY_TASKS_ESSENTIALS[@]}" "${!CURIOSITY_TASKS_DEFAULTS[@]}")

# propose_and_attempt + propose_and_attempt_all: propose_zeroshot → attempt_tasks
# tasks_path is derived (not a user arg)
PROPOSE_AND_ATTEMPT_ESSENTIALS=()
populate_array PROPOSE_ZEROSHOT_ESSENTIALS PROPOSE_AND_ATTEMPT_ESSENTIALS

declare -A PROPOSE_AND_ATTEMPT_DEFAULTS
populate_dict PROPOSE_ZEROSHOT_DEFAULTS PROPOSE_AND_ATTEMPT_DEFAULTS
populate_dict ATTEMPT_TASKS_DEFAULTS PROPOSE_AND_ATTEMPT_DEFAULTS
# Not inherited from PROPOSE_ZEROSHOT_DEFAULTS: proposal itself has no run_name (its output
# path is keyed on game/model only). The curiosity annotation lookup in
# curiosity_and_zeroshot{,_all}.sh does need one.
PROPOSE_AND_ATTEMPT_DEFAULTS["run_name"]=all
PROPOSE_AND_ATTEMPT_DEFAULTS["propose_only"]=false
PROPOSE_AND_ATTEMPT_DEFAULTS["attempt_only"]=false

PROPOSE_AND_ATTEMPT_ARG_KEYS=("${PROPOSE_AND_ATTEMPT_ESSENTIALS[@]}" "${!PROPOSE_AND_ATTEMPT_DEFAULTS[@]}")

# debug.py: read-only diagnostics over saved artifacts (scripts/debug/*.sh).
# These map to debug.py's *group* options, which every subcommand shares. Per-subcommand
# options (n_samples, n_frames, ...) stay local to their own script.
DEBUG_ESSENTIALS=()
populate_array ESSENTIAL_ARGS DEBUG_ESSENTIALS
declare -A DEBUG_DEFAULTS=(
    ["run_name"]="my_run"
    ["executor"]="history"
    ["output_dir"]="none"
    ["mode"]="both"
)
DEBUG_ARG_KEYS=("${DEBUG_ESSENTIALS[@]}" "${!DEBUG_DEFAULTS[@]}")

# Same, plus model_name — required by every debug subcommand except curiosity, whose
# artifacts are not keyed on a model. model_name is a *subcommand* option in debug.py, so
# debug_group_flags deliberately does not emit it.
DEBUG_MODEL_ESSENTIALS=("model_name")
populate_array DEBUG_ESSENTIALS DEBUG_MODEL_ESSENTIALS
declare -A DEBUG_MODEL_DEFAULTS
populate_dict DEBUG_DEFAULTS DEBUG_MODEL_DEFAULTS

DEBUG_MODEL_ARG_KEYS=("${DEBUG_MODEL_ESSENTIALS[@]}" "${!DEBUG_MODEL_DEFAULTS[@]}")

# debug_group_flags <assoc_array_name>
#
# Build the flag string for debug.py's group options from an ARGS dict. Handles what
# args_to_flags cannot: --output_dir uses the `none` sentinel and must be omitted when
# absent. Debug reports are always regenerated, so --overwrite (a click is_flag) is
# always passed and is not exposed as a script argument.
#
# Usage:
#   group_flags=$(debug_group_flags ARGS)
#   python debug.py $group_flags curiosity --n_frames 5
function debug_group_flags() {
    local -n _dict="$1"
    local result="--game ${_dict["game"]} --run_name ${_dict["run_name"]} --executor ${_dict["executor"]} --mode ${_dict["mode"]} --overwrite"
    if [[ "${_dict["output_dir"]}" != "none" ]]; then
        result+=" --output_dir ${_dict["output_dir"]}"
    fi
    echo "$result"
}
