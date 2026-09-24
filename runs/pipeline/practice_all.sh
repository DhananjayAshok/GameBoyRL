#!/usr/bin/env bash
# guidance -> practice -> clean -> dataset, for every title with zeroshot attempts to
# practice from. See runs/games.sh PRACTICE_GAMES.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
source configs/config.env || { echo "Could not source configs/config.env"; exit 1; }
source runs/games.sh || { echo "Could not source runs/games.sh"; exit 1; }

MODEL="google/gemma-4-31b-it"
VLM_KIND="vllm"
EXECUTOR="single_visual"
CONTROLLER_VARIANT="low_level"

declare -A STEMS
MISSING=""

for GAME in "${PRACTICE_GAMES[@]}"; do
    base=$(path_of zeroshot_dir --game "$GAME" --model_name "$MODEL")
    stem="$base/zeroshot_tasks_${EXECUTOR}_${CONTROLLER_VARIANT}_attempts/success_trajectories"
    if [[ -f "$stem.json" && -f "$stem.pkl" ]]; then
        STEMS["$GAME"]="$stem"
    else
        echo "SKIP: $GAME — no $EXECUTOR zeroshot attempts at $stem.{json,pkl}"
        MISSING+="$GAME "
    fi
done

if [[ ${#STEMS[@]} -eq 0 ]]; then
    echo "No games have $EXECUTOR zeroshot attempts. Nothing to do."
    exit 1
fi

echo ""
echo "Will run guidance -> practice -> clean -> dataset for: ${!STEMS[@]}"
[[ -n "$MISSING" ]] && echo "Skipping: $MISSING"

bash scripts/core/serve_vllm.sh "$MODEL" || { echo "Could not serve vLLM"; exit 1; }

for GAME in "${PRACTICE_GAMES[@]}"; do
    [[ -z "${STEMS[$GAME]}" ]] && continue
    echo ""
    echo "############################################################"
    echo "# $GAME  —  practice pipeline"
    echo "############################################################"
    if ! bash scripts/pipeline/guidance_and_practice.sh \
            --game "$GAME" \
            --model_name "$MODEL" \
            --vlm_kind "$VLM_KIND" \
            --trajectory_path "${STEMS[$GAME]}" \
            --executor "$EXECUTOR" \
            --controller_variant "$CONTROLLER_VARIANT" \
            --do_clean true; then
        echo "FAILED: $GAME | practice pipeline"
    fi
done

echo "DONE ALL"
