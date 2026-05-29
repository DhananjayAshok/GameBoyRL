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
    python $PROJECT_ROOT/scripts/get_strings.py "$string_kind" $flags
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

CREATE_TRAJ_ARG_KEYS=("${CREATE_TRAJ_ESSENTIALS[@]}" "${!CREATE_TRAJ_DEFAULTS[@]}")

VLM_ESSENTIALS=("model_name" "vlm_kind")
populate_array ESSENTIAL_ARGS VLM_ESSENTIALS
declare -A VLM_DEFAULTS=(
    ["overwrite"]=false
    ["verbose"]=false
    ["max_new_tokens"]=1000
)
VLM_ARG_KEYS=("${VLM_ESSENTIALS[@]}" "${!VLM_DEFAULTS[@]}")

PROPOSE_ZEROSHOT_ESSENTIALS=("init_state")
populate_array VLM_ESSENTIALS PROPOSE_ZEROSHOT_ESSENTIALS
declare -A PROPOSE_ZEROSHOT_DEFAULTS
populate_dict VLM_DEFAULTS PROPOSE_ZEROSHOT_DEFAULTS
PROPOSE_ZEROSHOT_DEFAULTS["extra"]=none
PROPOSE_ZEROSHOT_DEFAULTS["extra_k"]=20

PROPOSE_ZEROSHOT_ARG_KEYS=("${PROPOSE_ZEROSHOT_ESSENTIALS[@]}" "${!PROPOSE_ZEROSHOT_DEFAULTS[@]}")

ATTEMPT_TASKS_ESSENTIALS=("tasks_path")
populate_array VLM_ESSENTIALS ATTEMPT_TASKS_ESSENTIALS
declare -A ATTEMPT_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS ATTEMPT_TASKS_DEFAULTS
ATTEMPT_TASKS_DEFAULTS["executor"]="simple"
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

INFER_GUIDANCE_ESSENTIALS=("trajectory_path")
populate_array VLM_ESSENTIALS INFER_GUIDANCE_ESSENTIALS
declare -A INFER_GUIDANCE_DEFAULTS
populate_dict VLM_DEFAULTS INFER_GUIDANCE_DEFAULTS
INFER_GUIDANCE_DEFAULTS["max_obs_at_once"]=8

INFER_GUIDANCE_ARG_KEYS=("${INFER_GUIDANCE_ESSENTIALS[@]}" "${!INFER_GUIDANCE_DEFAULTS[@]}")

PRACTICE_TASKS_ESSENTIALS=("guidance_path")
populate_array VLM_ESSENTIALS PRACTICE_TASKS_ESSENTIALS
declare -A PRACTICE_TASKS_DEFAULTS
populate_dict VLM_DEFAULTS PRACTICE_TASKS_DEFAULTS
PRACTICE_TASKS_DEFAULTS["n_attempts"]=3
PRACTICE_TASKS_DEFAULTS["n_random_actions"]=5
PRACTICE_TASKS_DEFAULTS["score_mode"]=false
PRACTICE_TASKS_DEFAULTS["max_steps"]=50
PRACTICE_TASKS_DEFAULTS["max_tool_calls"]=10
PRACTICE_TASKS_DEFAULTS["lookback"]=8
PRACTICE_TASKS_DEFAULTS["executor"]="simple"
PRACTICE_TASKS_DEFAULTS["controller_variant"]="low_level"
PRACTICE_TASKS_DEFAULTS["checker_max_new_tokens"]=1000

PRACTICE_TASKS_ARG_KEYS=("${PRACTICE_TASKS_ESSENTIALS[@]}" "${!PRACTICE_TASKS_DEFAULTS[@]}")
