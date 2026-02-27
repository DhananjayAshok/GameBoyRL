source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }


function get_env_id(){    
    declare -A ARGS
    REQUIRED_ARGS=("max_steps" "env" "init_state" "controller" "game")

    ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}")
    # 3. Parser
    while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            # Extract the name (remove the leading --)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                exit 1
            fi            
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        *)
            exit 1
            usage
            ;;
    esac
    done

    # 4. Strict Validation
    for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        FAILED=true
    fi
    done

    if [ "$FAILED" = true ]; then exit 1; fi
    env_id="poke_worlds-${ARGS["game"]}-${ARGS["env"]}-${ARGS["init_state"]}-${ARGS["controller"]}-${ARGS["max_steps"]}-False"
    echo $env_id
}


function get_exp_name(){    
    declare -A ARGS
    REQUIRED_ARGS=("algorithm" "gamma" "similarity_metric" "observation_embedder" "curiosity_module" "max_steps" "timesteps" "env" "init_state" "replay_buffer_save_folder" "buffer_save_path" "buffer_load_path" "controller" "game")

    ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}")
    # 3. Parser
    while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            # Extract the name (remove the leading --)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                exit 1
            fi            
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        *)
            exit 1
            usage
            ;;
    esac
    done

    # 4. Strict Validation
    for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        FAILED=true
    fi
    done

    if [ "$FAILED" = true ]; then exit 1; fi
    env_id=$(get_env_id --max_steps ${ARGS["max_steps"]} --env ${ARGS["env"]} --init_state ${ARGS["init_state"]} --controller ${ARGS["controller"]} --game ${ARGS["game"]})
    exp_name="${ARGS["algorithm"]}-${ARGS["timesteps"]}-${ARGS["gamma"]}-${ARGS["observation_embedder"]}-${ARGS["curiosity_module"]}-${ARGS["similarity_metric"]}-${env_id}"
    echo $exp_name
}
