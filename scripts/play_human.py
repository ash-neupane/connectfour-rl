#!/usr/bin/env python3
"""Interactive Connect Four game for human vs human or human vs random AI play."""

import sys

import numpy as np

from src.game import ConnectFour

ROWS = 11
COLS = 27
WIN_LENGTH = 7

class InteractiveGame:
    """Interactive Connect Four game with terminal interface."""

    def __init__(self, vs_ai: bool = False):
        """Initialize the game.

        Args:
            vs_ai: If True, player 2 will be a random AI.
        """
        self.game = ConnectFour(rows=ROWS, cols=COLS, win_length=WIN_LENGTH, batch_size=1)
        self.vs_ai = vs_ai
        self.player_symbols = {1: "X", -1: "O"}

    def display_board(self) -> None:
        """Display the current board state using the game's __repr__."""
        print("\n" + repr(self.game))

    def get_human_move(self, player: int) -> int:
        """Get a move from a human player.

        Args:
            player: Player number (1 or 2).

        Returns:
            Column number for the move.
        """
        valid_moves = self.game.get_valid_moves()[0]
        valid_cols = [i for i in range(self.game.cols) if valid_moves[i]]

        while True:
            try:
                symbol = self.player_symbols[self.game.current_player[0]]
                move = input(
                    f"Player {player} ({symbol}) - Enter column (0-{self.game.cols-1})"
                    f" [{', '.join(map(str, valid_cols))}]: "
                )

                if move.lower() in ["q", "quit", "exit"]:
                    print("\nThanks for playing! Goodbye!")
                    sys.exit(0)

                col = int(move)
                if col in valid_cols:
                    return col
                else:
                    print(f"Invalid move! Column {col} is not available.")
                    print(f"Available columns: {valid_cols}")
            except ValueError:
                print(f"Please enter a valid number between 0 and {self.game.cols-1} (or 'q' to quit).")
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Goodbye!")
                sys.exit(0)

    def get_ai_move(self) -> int:
        """Get a random move from the AI.

        Returns:
            Column number for the move.
        """
        valid_moves = self.game.get_valid_moves()[0]
        valid_cols = [i for i in range(self.game.cols) if valid_moves[i]]
        return np.random.choice(valid_cols)

    def play(self) -> None:
        """Main game loop."""
        print("\n🎮 Welcome to Connect Four! 🎮")
        print("=" * 29)
        if self.vs_ai:
            print("Mode: Human vs AI")
            print("You are X, AI is O")
        else:
            print("Mode: Human vs Human")
            print("Player 1 is X, Player 2 is O")
        print("Type 'q' to quit at any time")
        print("=" * 29)

        move_count = 0

        while not self.game.game_over[0]:
            self.display_board()

            # Determine current player
            current_player = 1 if self.game.current_player[0] == 1 else 2

            # Get move
            if self.vs_ai and current_player == 2:
                print("AI is thinking...")
                col = self.get_ai_move()
                print(f"AI plays column {col}")
            else:
                col = self.get_human_move(current_player)

            # Make move
            _, done, _ = self.game.make_move(np.array([col]))
            move_count += 1

            if done[0]:
                self.display_board()
                print("\n" + "🎉" * 15)
                if self.game.winner[0] == 1:
                    if self.vs_ai:
                        print("🏆 Congratulations! You won! 🏆")
                    else:
                        print("🏆 Player 1 (X) wins! 🏆")
                elif self.game.winner[0] == -1:
                    if self.vs_ai:
                        print("🤖 AI wins! Better luck next time! 🤖")
                    else:
                        print("🏆 Player 2 (O) wins! 🏆")
                else:
                    print("🤝 It's a draw! Well played! 🤝")
                print("🎉" * 15)
                print(f"\nGame ended in {move_count} moves.")
                break

        # Ask to play again
        self.play_again()

    def play_again(self) -> None:
        """Ask if the player wants to play again."""
        while True:
            response = input("\nPlay again? (y/n): ").lower()
            if response in ["y", "yes"]:
                self.__init__(vs_ai=self.vs_ai)
                self.play()
                break
            elif response in ["n", "no", "q", "quit"]:
                print("\nThanks for playing! Goodbye!")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")


def main():
    """Main entry point for the interactive game."""
    print("\n" + "=" * 40)
    print(" " * 10 + "CONNECT FOUR")
    print("=" * 40)
    print("\nSelect game mode:")
    print("1. Human vs Human")
    print("2. Human vs AI (random)")
    print("Q. Quit")

    while True:
        choice = input("\nEnter your choice (1/2/Q): ").strip().lower()

        if choice == "1":
            game = InteractiveGame(vs_ai=False)
            game.play()
            break
        elif choice == "2":
            game = InteractiveGame(vs_ai=True)
            game.play()
            break
        elif choice in ["q", "quit"]:
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter 1, 2, or Q.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)
