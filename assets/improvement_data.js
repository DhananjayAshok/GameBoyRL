// Self-improvement results on GameBoyWorlds-Execution, all applied to Gemma 4 31B.
// Mirrors GAIN_RATES in plots.py.
window.IMPROVEMENT_DATA = {
  "methods": [
    {
      "id": "base",
      "name": "Gemma 4 31B (base)",
      "color": "#94a3b8"
    },
    {
      "id": "wm",
      "name": "World Model",
      "color": "#2E6FD9"
    },
    {
      "id": "pae",
      "name": "PAE",
      "color": "#7D4CC4"
    },
    {
      "id": "guide",
      "name": "Curiosity Guides",
      "color": "#0F9D76"
    }
  ],
  "series": [
    {
      "id": "bomberman",
      "name": "Bomberman",
      "games": [
        {
          "id": "bomberman_quest",
          "name": "Bomberman Quest",
          "base": 38.0,
          "wm": 36.0,
          "pae": 16.0,
          "guide": 36.0
        },
        {
          "id": "bomberman_pocket",
          "name": "Bomberman Pocket",
          "base": 72.0,
          "wm": 74.0,
          "pae": 76.0,
          "guide": 88.0
        }
      ]
    },
    {
      "id": "deja_vu",
      "name": "Déjà Vu",
      "games": [
        {
          "id": "deja_vu_1",
          "name": "Déjà Vu 1",
          "base": 14.0,
          "wm": 12.0,
          "pae": 10.0,
          "guide": 10.0
        },
        {
          "id": "deja_vu_2",
          "name": "Déjà Vu 2",
          "base": 8.0,
          "wm": 6.0,
          "pae": 2.0,
          "guide": 6.0
        }
      ]
    },
    {
      "id": "legend_of_zelda",
      "name": "Zelda",
      "games": [
        {
          "id": "legend_of_zelda_links_awakening",
          "name": "Link's Awakening",
          "base": 38.0,
          "wm": 40.0,
          "pae": 28.0,
          "guide": 40.0
        },
        {
          "id": "legend_of_zelda_the_oracle_of_seasons",
          "name": "Oracle of Seasons",
          "base": 40.0,
          "wm": 38.0,
          "pae": 14.0,
          "guide": 44.0
        }
      ]
    },
    {
      "id": "pokemon",
      "name": "Pokémon",
      "games": [
        {
          "id": "pokemon_red",
          "name": "Pokémon Red",
          "base": 38.8,
          "wm": 39.0,
          "pae": 8.2,
          "guide": 36.7
        },
        {
          "id": "pokemon_crystal",
          "name": "Pokémon Crystal",
          "base": 36.0,
          "wm": 34.0,
          "pae": 4.0,
          "guide": 30.0
        }
      ]
    },
    {
      "id": "sword_of_hope",
      "name": "Sword of Hope",
      "games": [
        {
          "id": "sword_of_hope_1",
          "name": "Sword of Hope 1",
          "base": 56.0,
          "wm": 48.0,
          "pae": 36.0,
          "guide": 52.0
        },
        {
          "id": "sword_of_hope_2",
          "name": "Sword of Hope 2",
          "base": 66.0,
          "wm": 66.0,
          "pae": 32.0,
          "guide": 72.0
        }
      ]
    }
  ],
  "averages": {
    "base": 40.7,
    "wm": 39.3,
    "pae": 22.6,
    "guide": 41.5
  }
};
