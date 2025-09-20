# Connect Four Interactive Game

Play Connect Four in your terminal!

## Usage

Run the game with poetry:

```bash
poetry run python scripts/play_human.py
```

Or directly if you have the environment activated:

```bash
python scripts/play_human.py
```

## Game Modes

1. **Human vs Human**: Two players take turns
2. **Human vs AI**: Play against a random AI opponent

## Controls

- Enter column number (0-6) to drop your piece
- Type 'q' or 'quit' to exit the game
- Press Ctrl+C to force quit

## Game Display

- `X` = Player 1 (Human in vs AI mode)
- `O` = Player 2 (AI in vs AI mode)
- Empty spaces are shown as blank

## Features

- Clear board visualization
- Move validation
- Win detection (horizontal, vertical, diagonal)
- Draw detection
- Play again option
- Color indicators for game status