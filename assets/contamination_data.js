// Progression-step question accuracy (%) per model per game.
// Mirrors contamination_accuracy() / CONTAMINATION_* in plots.py.
window.CONTAMINATION_DATA = {
  "models": [
    {
      "id": "gemini-3.5-flash-lite",
      "name": "Gemini 3.5 Flash-Lite",
      "color": "#1a73e8",
      "badge": "assets/logos/contam_model_gemini.png"
    },
    {
      "id": "chatgpt_gpt-5.6-sol",
      "name": "GPT-5.6-sol",
      "color": "#10a37f",
      "badge": "assets/logos/contam_model_openai.png"
    },
    {
      "id": "Claude-Code-Best-No-Internet",
      "name": "Claude Opus 5",
      "color": "#d97757",
      "badge": "assets/logos/contam_model_anthropic.png"
    }
  ],
  "games": [
    {
      "id": "Pokémon Red",
      "name": "Pokémon Red",
      "letter": "R",
      "short": "Red",
      "color": "#c8161d",
      "badge": "assets/logos/contam_game_red.png"
    },
    {
      "id": "Pokémon Crystal",
      "name": "Pokémon Crystal",
      "letter": "C",
      "short": "Crystal",
      "color": "#1b5fa8",
      "badge": "assets/logos/contam_game_crystal.png"
    },
    {
      "id": "Pokémon Brown",
      "name": "Pokémon Brown",
      "letter": "B",
      "short": "Brown",
      "color": "#7a4a22",
      "badge": "assets/logos/contam_game_brown.png"
    },
    {
      "id": "Pokémon Prism",
      "name": "Pokémon Prism",
      "letter": "P",
      "short": "Prism",
      "color": "#56209e",
      "badge": "assets/logos/contam_game_prism.png"
    }
  ],
  "groups": [
    {
      "name": "Classic",
      "games": [
        "Pokémon Red",
        "Pokémon Crystal"
      ]
    },
    {
      "name": "GameBoyWorlds-Playthrough",
      "games": [
        "Pokémon Brown",
        "Pokémon Prism"
      ]
    }
  ],
  "accuracy": {
    "gemini-3.5-flash-lite": {
      "Pokémon Red": 92.0,
      "Pokémon Crystal": 94.0,
      "Pokémon Brown": 13.0,
      "Pokémon Prism": 15.0
    },
    "chatgpt_gpt-5.6-sol": {
      "Pokémon Red": 100.0,
      "Pokémon Crystal": 100.0,
      "Pokémon Brown": 29.0,
      "Pokémon Prism": 29.0
    },
    "Claude-Code-Best-No-Internet": {
      "Pokémon Red": 97.0,
      "Pokémon Crystal": 93.0,
      "Pokémon Brown": 8.0,
      "Pokémon Prism": 16.0
    }
  }
};
