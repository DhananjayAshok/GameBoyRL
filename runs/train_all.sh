source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

MODEL="google/gemma-4-31b-it"
EXECUTOR="single_visual"
CONTROLLER_VARIANT="low_level"
RUN_PREFIX="practice"
OVERWRITE="false"
PUSH_TO_HUB="true"

if [[ -z "$huggingface_repo_namespace" ]]; then
    echo "Error: huggingface_repo_namespace is unset. Source configs/config.env before running."
    exit 1
fi

TRAIN_GAMES="deja_vu_1 \
             pokemon_red \
             sword_of_hope_1 \
             legend_of_zelda_links_awakening \
             bomberman_quest"

declare -A TRAIN_FILES
declare -A VAL_FILES
MISSING=""

for GAME in $TRAIN_GAMES; do
    attempts=$(path_of attempts_dir --game "$GAME" --model_name "$MODEL" \
                 --executor "$EXECUTOR" --controller_variant "$CONTROLLER_VARIANT")
    practice_path="$attempts/practice_${EXECUTOR}"
    train_file="$practice_path/train_dataset.csv"
    val_file="$practice_path/validation_dataset.csv"
    if [[ -f "$train_file" ]]; then
        TRAIN_FILES["$GAME"]="$train_file"
        if [[ -f "$val_file" ]]; then
            VAL_FILES["$GAME"]="$val_file"
        else
            echo "SKIP: $GAME — train_dataset.csv present but no validation_dataset.csv at $val_file"
            unset 'TRAIN_FILES[$GAME]'
            MISSING+="$GAME "
        fi
    else
        echo "SKIP: $GAME — no dataset at $train_file"
        MISSING+="$GAME "
    fi
done

if [[ ${#TRAIN_FILES[@]} -eq 0 ]]; then
    echo "No games have practice datasets. Nothing to do."
    exit 1
fi

echo ""
echo "Will fine-tune one adapter per game for: ${!TRAIN_FILES[@]}"
[[ -n "$MISSING" ]] && echo "Skipping: $MISSING"

FAILED_GAMES=""

for GAME in $TRAIN_GAMES; do
    [[ -z "${TRAIN_FILES[$GAME]}" ]] && continue
    echo ""
    echo "############################################################"
    echo "# $GAME  —  vlm fine-tune"
    echo "############################################################"
    if ! bash scripts/vlm/train_vlm.sh \
            --train_file "${TRAIN_FILES[$GAME]}" \
            --validation_file "${VAL_FILES[$GAME]}" \
            --model_name "$MODEL" \
            --run_name "${RUN_PREFIX}_${GAME}" \
            --overwrite "$OVERWRITE" \
            --push_to_hub "$PUSH_TO_HUB"; then
        echo "FAILED: $GAME | vlm fine-tune"
        FAILED_GAMES+="$GAME "
    fi
done

echo ""
[[ -n "$FAILED_GAMES" ]] && echo "Failed games: $FAILED_GAMES"
echo "DONE ALL"
