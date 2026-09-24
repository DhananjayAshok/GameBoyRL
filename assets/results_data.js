// Task success rates (%) on GameBoyWorlds-Execution, subgoal supervisor +
// single_visual executor, low_level controller. Generated from
// results/benchmark/<game>/subgoal_single_visual_low_level_<model>.csv
window.RESULTS_DATA = {
  "models": [
    {
      "id": "gemini-3.6-flash",
      "name": "Gemini 3.6 Flash",
      "logo": "gemini",
      "color": "#1a73e8"
    },
    {
      "id": "claude-haiku-4.5",
      "name": "Claude Haiku 4.5",
      "logo": "anthropic",
      "color": "#d97757"
    },
    {
      "id": "gpt-5-mini",
      "name": "GPT-5 mini",
      "logo": "openai",
      "color": "#10a37f"
    },
    {
      "id": "gemma-4-31b-it",
      "name": "Gemma 4 31B",
      "logo": "google",
      "color": "#8ab4f8"
    },
    {
      "id": "qwen3-vl-32b-instruct",
      "name": "Qwen3-VL 32B",
      "logo": "qwen",
      "color": "#615ced"
    }
  ],
  "series": [
    {
      "id": "deja_vu",
      "name": "Déjà Vu",
      "games": [
        {
          "id": "deja_vu_1",
          "name": "Déjà Vu 1"
        },
        {
          "id": "deja_vu_2",
          "name": "Déjà Vu 2"
        }
      ],
      "rates": {
        "deja_vu_1": {
          "gemini-3.6-flash": 20.0,
          "claude-haiku-4.5": 6.0,
          "gpt-5-mini": 8.0,
          "gemma-4-31b-it": 14.0,
          "qwen3-vl-32b-instruct": 8.0
        },
        "deja_vu_2": {
          "gemini-3.6-flash": 22.0,
          "claude-haiku-4.5": 2.0,
          "gpt-5-mini": 2.0,
          "gemma-4-31b-it": 8.0,
          "qwen3-vl-32b-instruct": 2.0
        }
      }
    },
    {
      "id": "legend_of_zelda",
      "name": "Zelda",
      "games": [
        {
          "id": "legend_of_zelda_links_awakening",
          "name": "Link's Awakening"
        },
        {
          "id": "legend_of_zelda_the_oracle_of_seasons",
          "name": "Oracle of Seasons"
        }
      ],
      "rates": {
        "legend_of_zelda_links_awakening": {
          "gemini-3.6-flash": 54.0,
          "claude-haiku-4.5": 36.0,
          "gpt-5-mini": 30.0,
          "gemma-4-31b-it": 38.0,
          "qwen3-vl-32b-instruct": 24.0
        },
        "legend_of_zelda_the_oracle_of_seasons": {
          "gemini-3.6-flash": 46.0,
          "claude-haiku-4.5": 14.0,
          "gpt-5-mini": 22.0,
          "gemma-4-31b-it": 40.0,
          "qwen3-vl-32b-instruct": 18.0
        }
      }
    },
    {
      "id": "pokemon",
      "name": "Pokémon",
      "games": [
        {
          "id": "pokemon_red",
          "name": "Pokémon Red"
        },
        {
          "id": "pokemon_crystal",
          "name": "Pokémon Crystal"
        }
      ],
      "rates": {
        "pokemon_red": {
          "gemini-3.6-flash": 61.2,
          "claude-haiku-4.5": 2.0,
          "gpt-5-mini": 18.4,
          "gemma-4-31b-it": 38.8,
          "qwen3-vl-32b-instruct": 22.4
        },
        "pokemon_crystal": {
          "gemini-3.6-flash": 56.0,
          "claude-haiku-4.5": 4.0,
          "gpt-5-mini": 12.0,
          "gemma-4-31b-it": 36.0,
          "qwen3-vl-32b-instruct": 10.0
        }
      }
    },
    {
      "id": "sword_of_hope",
      "name": "Sword of Hope",
      "games": [
        {
          "id": "sword_of_hope_1",
          "name": "Sword of Hope 1"
        },
        {
          "id": "sword_of_hope_2",
          "name": "Sword of Hope 2"
        }
      ],
      "rates": {
        "sword_of_hope_1": {
          "gemini-3.6-flash": 56.0,
          "claude-haiku-4.5": 32.0,
          "gpt-5-mini": 40.0,
          "gemma-4-31b-it": 56.0,
          "qwen3-vl-32b-instruct": 38.0
        },
        "sword_of_hope_2": {
          "gemini-3.6-flash": 58.0,
          "claude-haiku-4.5": 40.0,
          "gpt-5-mini": 34.0,
          "gemma-4-31b-it": 66.0,
          "qwen3-vl-32b-instruct": 54.0
        }
      }
    },
    {
      "id": "bomberman",
      "name": "Bomberman",
      "games": [
        {
          "id": "bomberman_quest",
          "name": "Bomberman Quest"
        },
        {
          "id": "bomberman_pocket",
          "name": "Bomberman Pocket"
        }
      ],
      "rates": {
        "bomberman_quest": {
          "gemini-3.6-flash": 38.0,
          "claude-haiku-4.5": 20.0,
          "gpt-5-mini": 20.0,
          "gemma-4-31b-it": 38.0,
          "qwen3-vl-32b-instruct": 20.0
        },
        "bomberman_pocket": {
          "gemini-3.6-flash": 82.0,
          "claude-haiku-4.5": 52.0,
          "gpt-5-mini": 86.0,
          "gemma-4-31b-it": 72.0,
          "qwen3-vl-32b-instruct": 70.0
        }
      }
    }
  ],
  "averages": {
    "gemini-3.6-flash": 49.3,
    "claude-haiku-4.5": 20.8,
    "gpt-5-mini": 27.2,
    "gemma-4-31b-it": 40.7,
    "qwen3-vl-32b-instruct": 26.6
  }
};
