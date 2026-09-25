// One random success and one random failure per game for gemini-3.6-flash on
// GameBoyWorlds-Execution (subgoal + single_visual + low_level). Seeded pick;
// videos copied from the recorded benchmark sessions.
window.EXAMPLES_DATA = {
  "model": "Gemini 3.6 Flash",
  "games": [
    {
      "id": "deja_vu_1",
      "name": "Déjà Vu 1",
      "series": "Déjà Vu",
      "success": {
        "task": "check the coat",
        "video": "assets/rollouts/deja_vu_1_success.mp4",
        "steps": 6
      },
      "failure": {
        "task": "open the desk",
        "video": "assets/rollouts/deja_vu_1_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "deja_vu_2",
      "name": "Déjà Vu 2",
      "series": "Déjà Vu",
      "success": {
        "task": "chat with taxi driver",
        "video": "assets/rollouts/deja_vu_2_success.mp4",
        "steps": 21
      },
      "failure": {
        "task": "hit the board  at fire escape",
        "video": "assets/rollouts/deja_vu_2_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "legend_of_zelda_links_awakening",
      "name": "Link's Awakening",
      "series": "Zelda",
      "success": {
        "task": "make another call from the chest shop",
        "video": "assets/rollouts/legend_of_zelda_links_awakening_success.mp4",
        "steps": 3
      },
      "failure": {
        "task": "go left see the blue house and then go up to the fan house",
        "video": "assets/rollouts/legend_of_zelda_links_awakening_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "legend_of_zelda_the_oracle_of_seasons",
      "name": "Oracle of Seasons",
      "series": "Zelda",
      "success": {
        "task": "get off green carpet",
        "video": "assets/rollouts/legend_of_zelda_the_oracle_of_seasons_success.mp4",
        "steps": 6
      },
      "failure": {
        "task": "blue snake talk",
        "video": "assets/rollouts/legend_of_zelda_the_oracle_of_seasons_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "pokemon_red",
      "name": "Pokémon Red",
      "series": "Pokémon",
      "success": {
        "task": "Read the sign below you",
        "video": "assets/rollouts/pokemon_red_success.mp4",
        "steps": 23
      },
      "failure": {
        "task": "Defeat Blue",
        "video": "assets/rollouts/pokemon_red_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "pokemon_crystal",
      "name": "Pokémon Crystal",
      "series": "Pokémon",
      "success": {
        "task": "Make Pidgeot hold a Cleanse Tag",
        "video": "assets/rollouts/pokemon_crystal_success.mp4",
        "steps": 26
      },
      "failure": {
        "task": "Speak to the gym leader",
        "video": "assets/rollouts/pokemon_crystal_failure.mp4",
        "steps": 74
      }
    },
    {
      "id": "sword_of_hope_1",
      "name": "Sword of Hope 1",
      "series": "Sword of Hope",
      "success": {
        "task": "Talk to an NPC or interactable and advance one full dialogue page",
        "video": "assets/rollouts/sword_of_hope_1_success.mp4",
        "steps": 14
      },
      "failure": {
        "task": "View Power stats last page then Teleport to Old Man's House (3-subgoal composite)",
        "video": "assets/rollouts/sword_of_hope_1_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "sword_of_hope_2",
      "name": "Sword of Hope 2",
      "series": "Sword of Hope",
      "success": {
        "task": "Reach the weapons shop BUY item list from the shopkeeper interaction",
        "video": "assets/rollouts/sword_of_hope_2_success.mp4",
        "steps": 8
      },
      "failure": {
        "task": "Finish a battle encounter without a game over",
        "video": "assets/rollouts/sword_of_hope_2_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "bomberman_quest",
      "name": "Bomberman Quest",
      "series": "Bomberman",
      "success": {
        "task": "Enter the House",
        "video": "assets/rollouts/bomberman_quest_success.mp4",
        "steps": 64
      },
      "failure": {
        "task": "Jump Off the Cliff and Read the Sign",
        "video": "assets/rollouts/bomberman_quest_failure.mp4",
        "steps": 75
      }
    },
    {
      "id": "bomberman_pocket",
      "name": "Bomberman Pocket",
      "series": "Bomberman",
      "success": {
        "task": "Wind Area 1 Navigate to Find and Take the Exit",
        "video": "assets/rollouts/bomberman_pocket_success.mp4",
        "steps": 17
      },
      "failure": {
        "task": "Ocean Area 1 Take the Exit",
        "video": "assets/rollouts/bomberman_pocket_failure.mp4",
        "steps": 76
      }
    }
  ]
};
