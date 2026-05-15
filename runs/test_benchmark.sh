source scripts/utils.sh
max_steps=2
games=("pokemon_red" "pokemon_crystal" "pokemon_starbeasts" "pokemon_prism" "pokemon_brown" "pokemon_fools_gold" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "bomberman_pocket" "bomberman_quest" "bomberman_max" "harry_potter_philosophers_stone" "harry_potter_chamber_of_secrets")

models=(
    "Qwen/Qwen3-VL-8B-Instruct"
)

executors=("simple")

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        for executor in "${executors[@]}"; do
            echo "    Running benchmark for executor: $executor"
            bash runs/benchmark.sh --game $game --executor $executor --executor_vlm_model "$model" --executor_vlm_kind huggingface --max_steps $max_steps
        done
    done
done