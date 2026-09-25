// Self-improvement on GameBoyWorlds-Execution, all methods applied to Gemma 4 31B.
// Mirrors GAIN_RATES / GAIN_METHODS in plots.py. `gains` are percentage points
// relative to the base model, which is what the dotplot shows.
window.IMPROVEMENT_DATA = {
  "baseModel": "Gemma 4 31B",
  "methods": [
    {
      "id": "wm",
      "name": "WM",
      "color": "#2E6FD9",
      "badge": "assets/logos/method_wm.png"
    },
    {
      "id": "pae",
      "name": "PAE",
      "color": "#7D4CC4",
      "badge": "assets/logos/method_pae.png"
    },
    {
      "id": "guide",
      "name": "Guide",
      "color": "#0F9D76",
      "badge": "assets/logos/method_guide.png"
    }
  ],
  "series": [
    {
      "id": "deja_vu",
      "name": "Déjà Vu",
      "badge": "assets/logos/series_deja_vu.png",
      "games": [
        "deja_vu_1",
        "deja_vu_2"
      ]
    },
    {
      "id": "legend_of_zelda",
      "name": "Zelda",
      "badge": "assets/logos/series_legend_of_zelda.png",
      "games": [
        "legend_of_zelda_links_awakening",
        "legend_of_zelda_the_oracle_of_seasons"
      ]
    },
    {
      "id": "pokemon",
      "name": "Pokémon",
      "badge": "assets/logos/series_pokemon.png",
      "games": [
        "pokemon_red",
        "pokemon_crystal"
      ]
    },
    {
      "id": "sword_of_hope",
      "name": "Sword of Hope",
      "badge": "assets/logos/series_sword_of_hope.png",
      "games": [
        "sword_of_hope_1",
        "sword_of_hope_2"
      ]
    },
    {
      "id": "bomberman",
      "name": "Bomberman",
      "badge": "assets/logos/series_bomberman.png",
      "games": [
        "bomberman_quest",
        "bomberman_pocket"
      ]
    }
  ],
  "games": [
    {
      "id": "deja_vu_1",
      "name": "Déjà Vu 1",
      "short": "1",
      "series": "deja_vu",
      "seriesName": "Déjà Vu",
      "base": 14.0,
      "rates": {
        "wm": 12.0,
        "pae": 10.0,
        "guide": 10.0
      },
      "gains": {
        "wm": -2.0,
        "pae": -4.0,
        "guide": -4.0
      }
    },
    {
      "id": "deja_vu_2",
      "name": "Déjà Vu 2",
      "short": "2",
      "series": "deja_vu",
      "seriesName": "Déjà Vu",
      "base": 8.0,
      "rates": {
        "wm": 6.0,
        "pae": 2.0,
        "guide": 6.0
      },
      "gains": {
        "wm": -2.0,
        "pae": -6.0,
        "guide": -2.0
      }
    },
    {
      "id": "legend_of_zelda_links_awakening",
      "name": "Link's Awakening",
      "short": "LA",
      "series": "legend_of_zelda",
      "seriesName": "Zelda",
      "base": 38.0,
      "rates": {
        "wm": 40.0,
        "pae": 28.0,
        "guide": 40.0
      },
      "gains": {
        "wm": 2.0,
        "pae": -10.0,
        "guide": 2.0
      }
    },
    {
      "id": "legend_of_zelda_the_oracle_of_seasons",
      "name": "Oracle of Seasons",
      "short": "OS",
      "series": "legend_of_zelda",
      "seriesName": "Zelda",
      "base": 40.0,
      "rates": {
        "wm": 38.0,
        "pae": 14.0,
        "guide": 44.0
      },
      "gains": {
        "wm": -2.0,
        "pae": -26.0,
        "guide": 4.0
      }
    },
    {
      "id": "pokemon_red",
      "name": "Pokémon Red",
      "short": "R",
      "series": "pokemon",
      "seriesName": "Pokémon",
      "base": 38.8,
      "rates": {
        "wm": 39.0,
        "pae": 8.2,
        "guide": 36.7
      },
      "gains": {
        "wm": 0.2,
        "pae": -30.6,
        "guide": -2.1
      }
    },
    {
      "id": "pokemon_crystal",
      "name": "Pokémon Crystal",
      "short": "C",
      "series": "pokemon",
      "seriesName": "Pokémon",
      "base": 36.0,
      "rates": {
        "wm": 34.0,
        "pae": 4.0,
        "guide": 30.0
      },
      "gains": {
        "wm": -2.0,
        "pae": -32.0,
        "guide": -6.0
      }
    },
    {
      "id": "sword_of_hope_1",
      "name": "Sword of Hope 1",
      "short": "1",
      "series": "sword_of_hope",
      "seriesName": "Sword of Hope",
      "base": 56.0,
      "rates": {
        "wm": 48.0,
        "pae": 36.0,
        "guide": 52.0
      },
      "gains": {
        "wm": -8.0,
        "pae": -20.0,
        "guide": -4.0
      }
    },
    {
      "id": "sword_of_hope_2",
      "name": "Sword of Hope 2",
      "short": "2",
      "series": "sword_of_hope",
      "seriesName": "Sword of Hope",
      "base": 66.0,
      "rates": {
        "wm": 66.0,
        "pae": 32.0,
        "guide": 72.0
      },
      "gains": {
        "wm": 0.0,
        "pae": -34.0,
        "guide": 6.0
      }
    },
    {
      "id": "bomberman_quest",
      "name": "Bomberman Quest",
      "short": "Qu",
      "series": "bomberman",
      "seriesName": "Bomberman",
      "base": 38.0,
      "rates": {
        "wm": 36.0,
        "pae": 16.0,
        "guide": 36.0
      },
      "gains": {
        "wm": -2.0,
        "pae": -22.0,
        "guide": -2.0
      }
    },
    {
      "id": "bomberman_pocket",
      "name": "Bomberman Pocket",
      "short": "Po",
      "series": "bomberman",
      "seriesName": "Bomberman",
      "base": 72.0,
      "rates": {
        "wm": 74.0,
        "pae": 76.0,
        "guide": 88.0
      },
      "gains": {
        "wm": 2.0,
        "pae": 4.0,
        "guide": 16.0
      }
    }
  ],
  "averages": {
    "base": 40.7,
    "wm": 39.3,
    "pae": 22.6,
    "guide": 41.5
  },
  "averageGains": {
    "wm": -1.4,
    "pae": -18.1,
    "guide": 0.8
  }
};
