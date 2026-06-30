// Placeholder task success-rate data for the GameBoyWorlds leaderboard chart.
// Replace with real numbers once evaluation results are available.
window.RESULTS_DATA = {
  "models": [
    {
      "id": "gemini-3-1-pro",
      "name": "Gemini-3.1-Pro",
      "provider": "Google"
    },
    {
      "id": "claude-4-opus",
      "name": "Claude-4-Opus",
      "provider": "Anthropic"
    },
    {
      "id": "gpt-5",
      "name": "GPT-5",
      "provider": "OpenAI"
    },
    {
      "id": "qwen3-vl-32b",
      "name": "Qwen3-VL-32B",
      "provider": "Qwen"
    },
    {
      "id": "gemma-4-31b",
      "name": "Gemma-4-31B",
      "provider": "Google"
    }
  ],
  "providerColors": {
    "Google": "#4285F4",
    "Anthropic": "#D97757",
    "OpenAI": "#10A37F",
    "Qwen": "#7C3AED"
  },
  "series": [
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
        },
        {
          "id": "bomberman_max",
          "name": "Bomberman Max"
        }
      ],
      "rates": {
        "bomberman_quest": {
          "gemini-3-1-pro": 68.9,
          "claude-4-opus": 29.6,
          "gpt-5": 45.6,
          "qwen3-vl-32b": 42.3,
          "gemma-4-31b": 75.1
        },
        "bomberman_pocket": {
          "gemini-3-1-pro": 71.3,
          "claude-4-opus": 85.1,
          "gpt-5": 33.6,
          "qwen3-vl-32b": 55.0,
          "gemma-4-31b": 29.9
        },
        "bomberman_max": {
          "gemini-3-1-pro": 42.0,
          "claude-4-opus": 60.3,
          "gpt-5": 29.7,
          "qwen3-vl-32b": 40.7,
          "gemma-4-31b": 69.6
        }
      }
    },
    {
      "id": "deja_vu",
      "name": "Deja Vu",
      "games": [
        {
          "id": "deja_vu_1",
          "name": "Deja Vu 1"
        },
        {
          "id": "deja_vu_2",
          "name": "Deja Vu 2"
        }
      ],
      "rates": {
        "deja_vu_1": {
          "gemini-3-1-pro": 62.9,
          "claude-4-opus": 42.1,
          "gpt-5": 65.7,
          "qwen3-vl-32b": 79.8,
          "gemma-4-31b": 28.4
        },
        "deja_vu_2": {
          "gemini-3-1-pro": 79.6,
          "claude-4-opus": 72.7,
          "gpt-5": 49.8,
          "qwen3-vl-32b": 38.0,
          "gemma-4-31b": 89.3
        }
      }
    },
    {
      "id": "harry_potter",
      "name": "Harry Potter",
      "games": [
        {
          "id": "harry_potter_philosophers_stone",
          "name": "Harry Potter Philosophers Stone"
        },
        {
          "id": "harry_potter_chamber_of_secrets",
          "name": "Harry Potter Chamber Of Secrets"
        }
      ],
      "rates": {
        "harry_potter_philosophers_stone": {
          "gemini-3-1-pro": 49.5,
          "claude-4-opus": 33.9,
          "gpt-5": 34.2,
          "qwen3-vl-32b": 82.2,
          "gemma-4-31b": 66.6
        },
        "harry_potter_chamber_of_secrets": {
          "gemini-3-1-pro": 79.7,
          "claude-4-opus": 74.7,
          "gpt-5": 62.3,
          "qwen3-vl-32b": 90.3,
          "gemma-4-31b": 52.2
        }
      }
    },
    {
      "id": "harvest_moon",
      "name": "Harvest Moon",
      "games": [
        {
          "id": "harvest_moon_1",
          "name": "Harvest Moon 1"
        },
        {
          "id": "harvest_moon_2",
          "name": "Harvest Moon 2"
        },
        {
          "id": "harvest_moon_3",
          "name": "Harvest Moon 3"
        }
      ],
      "rates": {
        "harvest_moon_1": {
          "gemini-3-1-pro": 63.3,
          "claude-4-opus": 81.1,
          "gpt-5": 67.6,
          "qwen3-vl-32b": 83.1,
          "gemma-4-31b": 65.0
        },
        "harvest_moon_2": {
          "gemini-3-1-pro": 73.1,
          "claude-4-opus": 30.9,
          "gpt-5": 42.6,
          "qwen3-vl-32b": 46.5,
          "gemma-4-31b": 33.1
        },
        "harvest_moon_3": {
          "gemini-3-1-pro": 42.9,
          "claude-4-opus": 34.5,
          "gpt-5": 45.8,
          "qwen3-vl-32b": 68.7,
          "gemma-4-31b": 51.3
        }
      }
    },
    {
      "id": "legend_of_zelda",
      "name": "Legend Of Zelda",
      "games": [
        {
          "id": "legend_of_zelda_links_awakening",
          "name": "Legend Of Zelda Links Awakening"
        },
        {
          "id": "legend_of_zelda_the_oracle_of_seasons",
          "name": "Legend Of Zelda The Oracle Of Seasons"
        }
      ],
      "rates": {
        "legend_of_zelda_links_awakening": {
          "gemini-3-1-pro": 51.7,
          "claude-4-opus": 41.4,
          "gpt-5": 45.1,
          "qwen3-vl-32b": 87.9,
          "gemma-4-31b": 69.5
        },
        "legend_of_zelda_the_oracle_of_seasons": {
          "gemini-3-1-pro": 67.0,
          "claude-4-opus": 39.0,
          "gpt-5": 74.7,
          "qwen3-vl-32b": 38.5,
          "gemma-4-31b": 52.3
        }
      }
    },
    {
      "id": "pokemon",
      "name": "Pokemon",
      "games": [
        {
          "id": "pokemon_red",
          "name": "Pokemon Red"
        }
      ],
      "rates": {
        "pokemon_red": {
          "gemini-3-1-pro": 91.3,
          "claude-4-opus": 69.0,
          "gpt-5": 63.6,
          "qwen3-vl-32b": 71.8,
          "gemma-4-31b": 81.9
        }
      }
    },
    {
      "id": "runes_of_virtue",
      "name": "Runes Of Virtue",
      "games": [
        {
          "id": "runes_of_virtue_1",
          "name": "Runes Of Virtue 1"
        },
        {
          "id": "runes_of_virtue_2",
          "name": "Runes Of Virtue 2"
        }
      ],
      "rates": {
        "runes_of_virtue_1": {
          "gemini-3-1-pro": 77.7,
          "claude-4-opus": 42.7,
          "gpt-5": 30.1,
          "qwen3-vl-32b": 48.2,
          "gemma-4-31b": 45.1
        },
        "runes_of_virtue_2": {
          "gemini-3-1-pro": 41.5,
          "claude-4-opus": 88.3,
          "gpt-5": 84.1,
          "qwen3-vl-32b": 48.1,
          "gemma-4-31b": 69.9
        }
      }
    },
    {
      "id": "survival_kids",
      "name": "Survival Kids",
      "games": [
        {
          "id": "survival_kids_1",
          "name": "Survival Kids 1"
        },
        {
          "id": "survival_kids_2",
          "name": "Survival Kids 2"
        }
      ],
      "rates": {
        "survival_kids_1": {
          "gemini-3-1-pro": 53.3,
          "claude-4-opus": 86.5,
          "gpt-5": 57.4,
          "qwen3-vl-32b": 45.0,
          "gemma-4-31b": 43.8
        },
        "survival_kids_2": {
          "gemini-3-1-pro": 63.9,
          "claude-4-opus": 44.8,
          "gpt-5": 65.4,
          "qwen3-vl-32b": 85.5,
          "gemma-4-31b": 53.6
        }
      }
    },
    {
      "id": "sword_of_hope",
      "name": "Sword Of Hope",
      "games": [
        {
          "id": "sword_of_hope_1",
          "name": "Sword Of Hope 1"
        },
        {
          "id": "sword_of_hope_2",
          "name": "Sword Of Hope 2"
        }
      ],
      "rates": {
        "sword_of_hope_1": {
          "gemini-3-1-pro": 42.0,
          "claude-4-opus": 91.8,
          "gpt-5": 60.6,
          "qwen3-vl-32b": 33.8,
          "gemma-4-31b": 31.0
        },
        "sword_of_hope_2": {
          "gemini-3-1-pro": 35.0,
          "claude-4-opus": 68.2,
          "gpt-5": 78.7,
          "qwen3-vl-32b": 55.0,
          "gemma-4-31b": 32.1
        }
      }
    }
  ]
};
