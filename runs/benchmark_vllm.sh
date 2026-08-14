# Full comparison sweep on a vllm-served model: every supervisor x executor combination,
# for each game listed below.
#
# The two axes are independent, which is the point — a difference between two supervisors is
# only attributable if both ran on the same executor, and vice versa. Edit either list to
# narrow the sweep; the run count is printed before anything starts.
#
# The info_subgoal arm goes through benchmark_plan.sh instead, because it needs the document
# flags. Both of its knowledge modes are swept:
#   parametric  needs no artifacts — the document is written from the model's priors
#   retrieval   needs a built info.json, and is skipped for games that have none
source scripts/core/utils.sh
source configs/config.env

games=("deja_vu_2" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_3")
#games=("harvest_moon_1" "harvest_moon_2" "harvest_moon_3")
#games=("survival_kids_1")
#games=("pokemon_red" "legend_of_zelda_links_awakening" "sword_of_hope_1" "harvest_moon_1" "bomberman_pocket" "deja_vu_1")

if [ -z "$model" ]; then
    echo "ERROR: 'model' variable is not set. Export model=<org/name> before running. Aborting."
    exit 1
fi

# The nine executor arms are <action>_<history>. Listed rather than generated so a sweep can
# be narrowed without editing the registry.
executors=("single_none" "single_actions" "single_visual" \
           "scored_none" "scored_actions" "scored_visual" \
           "sequence_none" "sequence_actions" "sequence_visual")

# Knowledge-free supervisor arms, in increasing order of what they add. info_subgoal is
# handled separately below because its flags differ.
supervisors=("dummy" "revision" "subgoal")

# Knowledge modes for the info arm. Set to () to skip that arm entirely.
knowledge_modes=("parametric" "retrieval")

MAX_STEPS=${MAX_STEPS:-150}

# Documents are built once per series, from the title(s) that declare train states — every
# other title in the series borrows them. Asked of `path_of train_games` (GameBoyWorlds'
# can_train_from_init_state column) rather than hardcoded, same as run.sh. It is a list: a
# series may declare more than one train game, and all of their documents are in scope.
#
# Globbed rather than derived: which vertical exists differs per game, and a literal path
# list would hand the arm a file that is not there.
docs_for() {
    local docs_game hits found=""
    for docs_game in "$@"; do
        hits=$(ls "$storage_dir/proposed_tasks/$docs_game"/*/*/*/info_*/info.json 2>/dev/null \
                   | paste -sd, -)
        [[ -n "$hits" ]] && found+="${found:+,}$hits"
    done
    echo "$found"
}

n_free=$(( ${#games[@]} * ${#executors[@]} * ${#supervisors[@]} ))
n_info=$(( ${#games[@]} * ${#executors[@]} * ${#knowledge_modes[@]} ))
echo "############################################################"
echo "# Sweep: ${#games[@]} game(s) x ${#executors[@]} executor(s)"
echo "#   knowledge-free arms : ${#supervisors[@]} -> $n_free runs"
echo "#   info_subgoal arms   : ${#knowledge_modes[@]} -> up to $n_info runs"
echo "#   TOTAL               : up to $(( n_free + n_info )) runs at --max_steps $MAX_STEPS"
echo "############################################################"

for game in "${games[@]}"; do
    echo ""
    echo "=== game: $game ==="
    # Once per game: path_of shells out, and this command pays a ~3s gameboy_worlds import.
    mapfile -t docs_games < <(path_of train_games --game "$game")
    docs=$(docs_for "${docs_games[@]}")

    for executor in "${executors[@]}"; do
        for supervisor in "${supervisors[@]}"; do
            echo "--- $game | $executor | $supervisor"
            bash scripts/benchmark.sh \
                --game "$game" \
                --supervisor "$supervisor" \
                --executor "$executor" \
                --executor_vlm_model "$model" \
                --executor_vlm_kind vllm \
                --max_steps $MAX_STEPS \
                || echo "FAILED: $game | $executor | $supervisor"
        done

        for knowledge_mode in "${knowledge_modes[@]}"; do
            # retrieval needs a document; parametric writes its own.
            if [[ "$knowledge_mode" == "retrieval" && -z "$docs" ]]; then
                echo "--- $game | $executor | info_subgoal/retrieval  SKIPPED (no info.json)"
                continue
            fi
            echo "--- $game | $executor | info_subgoal/$knowledge_mode"
            bash scripts/benchmark_plan.sh \
                --game "$game" \
                --executor "$executor" \
                --executor_vlm_model "$model" \
                --executor_vlm_kind vllm \
                --knowledge_mode "$knowledge_mode" \
                --info_docs "${docs:-none}" \
                --max_steps $MAX_STEPS \
                || echo "FAILED: $game | $executor | info_subgoal/$knowledge_mode"
        done
    done
done

echo ""
echo "Done. Results in $results_dir/benchmark/<game>/<supervisor>_<executor>_<model>.csv"
