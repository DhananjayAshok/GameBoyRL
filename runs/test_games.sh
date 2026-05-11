
#games=("pokemon_red" "pokemon_crystal" "pokemon_starbeasts" "pokemon_prism" "pokemon_brown" "pokemon_fools_gold" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "bomberman_pocket" "bomberman_quest" "bomberman_max" "harry_potter_philosophers_stone" "harry_potter_chamber_of_secrets")
games=("harry_potter_philosophers_stone" "harry_potter_chamber_of_secrets")

#games=("deja_vu_1" "deja_vu_2")
# disable train logging before you do this
mkdir -p logs/test_games
for game in "${games[@]}"; do
    echo "Testing game: $game"
    bash scripts/default_rl.sh --game "$game" --timesteps 1000 &> logs/test_games/${game}.out
done