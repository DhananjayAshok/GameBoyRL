// One success and one failure per game for gemini-3.6-flash on GameBoyWorlds-Execution
// (subgoal + single_visual + low_level), chosen for a clean task statement and a
// legible clip. Videos re-encoded to H.264 from the recorded benchmark sessions.
window.EXAMPLES_DATA = {
  "model": "Gemini 3.6 Flash",
  "games": [
    {
      "id": "deja_vu_1",
      "name": "Déjà Vu 1",
      "series": "Déjà Vu",
      "seriesId": "deja_vu",
      "success": {
        "task": "Take coat from the front door",
        "video": "assets/rollouts/deja_vu_1_success.mp4",
        "steps": 37
      },
      "failure": {
        "task": "Take gun from the front door",
        "video": "assets/rollouts/deja_vu_1_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "deja_vu_2",
      "name": "Déjà Vu 2",
      "series": "Déjà Vu",
      "seriesId": "deja_vu",
      "success": {
        "task": "Open the box at the alleway behind joe",
        "video": "assets/rollouts/deja_vu_2_success.mp4",
        "steps": 47
      },
      "failure": {
        "task": "Enter the Joe's place",
        "video": "assets/rollouts/deja_vu_2_failure.mp4",
        "steps": 74
      }
    },
    {
      "id": "legend_of_zelda_links_awakening",
      "name": "Link's Awakening",
      "series": "Zelda",
      "seriesId": "legend_of_zelda",
      "success": {
        "task": "Go in the water",
        "video": "assets/rollouts/legend_of_zelda_links_awakening_success.mp4",
        "steps": 60
      },
      "failure": {
        "task": "Talk to Tarin inside the spawn house to get the shield",
        "video": "assets/rollouts/legend_of_zelda_links_awakening_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "legend_of_zelda_the_oracle_of_seasons",
      "name": "Oracle of Seasons",
      "series": "Zelda",
      "seriesId": "legend_of_zelda",
      "success": {
        "task": "Talk to the shop person",
        "video": "assets/rollouts/legend_of_zelda_the_oracle_of_seasons_success.mp4",
        "steps": 39
      },
      "failure": {
        "task": "Go inside the library and talk to the parrot",
        "video": "assets/rollouts/legend_of_zelda_the_oracle_of_seasons_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "pokemon_red",
      "name": "Pokémon Red",
      "series": "Pokémon",
      "seriesId": "pokemon",
      "success": {
        "task": "Read the letter",
        "video": "assets/rollouts/pokemon_red_success.mp4",
        "steps": 60
      },
      "failure": {
        "task": "Exit the building",
        "video": "assets/rollouts/pokemon_red_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "pokemon_crystal",
      "name": "Pokémon Crystal",
      "series": "Pokémon",
      "seriesId": "pokemon",
      "success": {
        "task": "Teach Pidgeot Toxic",
        "video": "assets/rollouts/pokemon_crystal_success.mp4",
        "steps": 55
      },
      "failure": {
        "task": "Look into the mirror",
        "video": "assets/rollouts/pokemon_crystal_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "sword_of_hope_1",
      "name": "Sword of Hope 1",
      "series": "Sword of Hope",
      "seriesId": "sword_of_hope",
      "success": {
        "task": "Look at surroundings to reveal a previously hidden path",
        "video": "assets/rollouts/sword_of_hope_1_success.mp4",
        "steps": 54
      },
      "failure": {
        "task": "Defeat a boss using offensive magic spells",
        "video": "assets/rollouts/sword_of_hope_1_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "sword_of_hope_2",
      "name": "Sword of Hope 2",
      "series": "Sword of Hope",
      "seriesId": "sword_of_hope",
      "success": {
        "task": "Buy CPR Sword from Weapons Shop via Look-shopkeeper-Buy chain",
        "video": "assets/rollouts/sword_of_hope_2_success.mp4",
        "steps": 13
      },
      "failure": {
        "task": "Hit a tree at Riccar Woods until Wheat is received",
        "video": "assets/rollouts/sword_of_hope_2_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "bomberman_quest",
      "name": "Bomberman Quest",
      "series": "Bomberman",
      "seriesId": "bomberman",
      "success": {
        "task": "Read the Ruins Sign",
        "video": "assets/rollouts/bomberman_quest_success.mp4",
        "steps": 54
      },
      "failure": {
        "task": "Talk to the NPC",
        "video": "assets/rollouts/bomberman_quest_failure.mp4",
        "steps": 76
      }
    },
    {
      "id": "bomberman_pocket",
      "name": "Bomberman Pocket",
      "series": "Bomberman",
      "seriesId": "bomberman",
      "success": {
        "task": "Area 2 Pick up a Bomb Up",
        "video": "assets/rollouts/bomberman_pocket_success.mp4",
        "steps": 38
      },
      "failure": {
        "task": "Kill the Forest Stage Boss and Take the Exit",
        "video": "assets/rollouts/bomberman_pocket_failure.mp4",
        "steps": 76
      }
    }
  ]
};
