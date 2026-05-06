#!/usr/bin/env bash

# Define defaults and required args. 
# These should be specific to this script and not shared across scripts (that is handled below).
declare -A ARGS

REQUIRED_ARGS=("env_dir")

# Handle parsing and input errors below:
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

# Add Required to string
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

# Add Optionals to string
for opt in "${!ARGS[@]}"; do
    # Only list if NOT in required (to avoid double listing)
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

# Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validation
for req in "${REQUIRED_ARGS[@]}"; do
    echo $req : "${ARGS[$req]}"
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

uv --version || { echo "uv is not installed. Please install uv"; exit 1; }

# if llm-utils/setup/.venv already exists, skip the env creation and just install the dependencies.
if [ -d "llm-utils/setup/.venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    # if env_dir already exists, ask if we can delete it and start fresh
    if [ -d "${ARGS[env_dir]}" ]; then
        read -p "Directory ${ARGS[env_dir]} already exists. Do you want to delete it and create a new virtual environment? (y/n) " yn
        case $yn in
            [Yy]* ) rm -rf "${ARGS[env_dir]}";;
            * ) echo "Exiting without creating virtual environment."; exit 1;;
        esac
    fi
    uv venv "${ARGS[env_dir]}" --python 3.12 || { echo "Failed to create virtual environment. Please check the error messages above."; exit 1; }
    ln -s "${ARGS[env_dir]}" llm-utils/setup/.venv || { echo "Failed to create symbolic link for virtual environment. Please check the error messages above."; exit 1; }
fi

cd llm-utils/setup
uv sync || { echo "Failed to sync virtual environment. Please check the error messages above."; exit 1; }
