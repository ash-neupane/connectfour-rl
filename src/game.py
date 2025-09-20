import numba
import numpy as np


class ConnectFour:
    """Connect Four game with configurable board size and vectorized operations."""

    def __init__(
        self,
        batch_size: int = 1,
        rows: int = 6,
        cols: int = 7,
        win_length: int = 4,
    ):
        """Initialize the game.

        Args:
            batch_size: Number of games to play in parallel.
            rows: Number of rows in the board.
            cols: Number of columns in the board.
            win_length: Number of pieces in a row needed to win.
        """
        self.batch_size = batch_size
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.reset()

    def reset(self) -> None:
        """Reset all games to initial state."""
        # Board state: 0 = empty, 1 = player 1, -1 = player 2
        self.boards = np.zeros((self.batch_size, self.rows, self.cols), dtype=np.int8)
        self.current_player = np.ones(self.batch_size, dtype=np.int8)
        self.game_over = np.zeros(self.batch_size, dtype=bool)
        self.winner = np.zeros(self.batch_size, dtype=np.int8)
        self.move_count = np.zeros(self.batch_size, dtype=np.int16)

    def get_valid_moves(self) -> np.ndarray:
        """Get valid moves for all games.

        Returns:
            Boolean array of shape (batch_size, cols) indicating valid moves.
        """
        # A column is valid if the top row is empty
        valid = self.boards[:, 0, :] == 0
        # No moves are valid for finished games
        valid[self.game_over] = False
        return valid

    def make_move(self, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make a move in the specified columns.

        Args:
            cols: Array of column indices for each game.

        Returns:
            Tuple of (rewards, game_over, valid_move) arrays.
        """
        batch_indices = np.arange(self.batch_size)
        valid_moves = self.get_valid_moves()
        is_valid = valid_moves[batch_indices, cols]

        rewards = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            if not is_valid[i]:
                continue

            # Find the lowest empty row in the column
            col = cols[i]
            for row in range(self.rows - 1, -1, -1):
                if self.boards[i, row, col] == 0:
                    self.boards[i, row, col] = self.current_player[i]
                    self.move_count[i] += 1

                    # Check for win
                    if self._check_win(i, row, col):
                        self.game_over[i] = True
                        self.winner[i] = self.current_player[i]
                        rewards[i] = 1.0 if self.current_player[i] == 1 else -1.0
                    # Check for draw
                    elif self.move_count[i] == self.rows * self.cols:
                        self.game_over[i] = True
                        self.winner[i] = 0
                    # Switch player
                    else:
                        self.current_player[i] *= -1

                    break

        return rewards, self.game_over.copy(), is_valid

    def _check_win(self, batch_idx: int, row: int, col: int) -> bool:
        """Check if the last move resulted in a win.

        Args:
            batch_idx: Index of the game in the batch.
            row: Row of the last move.
            col: Column of the last move.

        Returns:
            True if the move resulted in a win.
        """
        player = self.boards[batch_idx, row, col]
        if player == 0:
            return False

        # Check horizontal
        count = 1
        # Check left
        for c in range(col - 1, max(-1, col - self.win_length), -1):
            if self.boards[batch_idx, row, c] == player:
                count += 1
            else:
                break
        # Check right
        for c in range(col + 1, min(self.cols, col + self.win_length)):
            if self.boards[batch_idx, row, c] == player:
                count += 1
            else:
                break
        if count >= self.win_length:
            return True

        # Check vertical
        count = 1
        # Check up
        for r in range(row - 1, max(-1, row - self.win_length), -1):
            if self.boards[batch_idx, r, col] == player:
                count += 1
            else:
                break
        # Check down
        for r in range(row + 1, min(self.rows, row + self.win_length)):
            if self.boards[batch_idx, r, col] == player:
                count += 1
            else:
                break
        if count >= self.win_length:
            return True

        # Check diagonal (top-left to bottom-right)
        count = 1
        # Check up-left
        for i in range(1, self.win_length):
            r, c = row - i, col - i
            if r < 0 or c < 0:
                break
            if self.boards[batch_idx, r, c] == player:
                count += 1
            else:
                break
        # Check down-right
        for i in range(1, self.win_length):
            r, c = row + i, col + i
            if r >= self.rows or c >= self.cols:
                break
            if self.boards[batch_idx, r, c] == player:
                count += 1
            else:
                break
        if count >= self.win_length:
            return True

        # Check anti-diagonal (top-right to bottom-left)
        count = 1
        # Check up-right
        for i in range(1, self.win_length):
            r, c = row - i, col + i
            if r < 0 or c >= self.cols:
                break
            if self.boards[batch_idx, r, c] == player:
                count += 1
            else:
                break
        # Check down-left
        for i in range(1, self.win_length):
            r, c = row + i, col - i
            if r >= self.rows or c < 0:
                break
            if self.boards[batch_idx, r, c] == player:
                count += 1
            else:
                break
        return count >= self.win_length

    def get_board_state(self) -> np.ndarray:
        """Get the board state for neural network input.

        Returns:
            Array of shape (batch_size, 2, rows, cols) with separate channels
            for each player.
        """
        state = np.zeros((self.batch_size, 2, self.rows, self.cols), dtype=np.float32)
        state[:, 0, :, :] = (self.boards == 1).astype(np.float32)
        state[:, 1, :, :] = (self.boards == -1).astype(np.float32)
        return state

    def clone(self) -> "ConnectFour":
        """Create a deep copy of the game state.

        Returns:
            A new ConnectFour instance with the same state.
        """
        new_game = ConnectFour(
            batch_size=self.batch_size,
            rows=self.rows,
            cols=self.cols,
            win_length=self.win_length,
        )
        new_game.boards = self.boards.copy()
        new_game.current_player = self.current_player.copy()
        new_game.game_over = self.game_over.copy()
        new_game.winner = self.winner.copy()
        new_game.move_count = self.move_count.copy()
        return new_game

    def __repr__(self) -> str:
        """String representation of the game state."""
        boards_repr = []
        for batch_idx in range(self.batch_size):
            lines = []
            if self.batch_size > 1:
                lines.append(f"Game {batch_idx}:")

            # Determine the maximum width needed for column numbers
            max_col_width = len(str(self.cols - 1))

            # Display the board with proper spacing
            for row in range(self.rows):
                row_str = ""
                for col in range(self.cols):
                    if self.boards[batch_idx, row, col] == 1:
                        symbol = "X"
                    elif self.boards[batch_idx, row, col] == -1:
                        symbol = "O"
                    else:
                        symbol = "-"
                    # Right-align symbol to match column width
                    row_str += symbol.rjust(max_col_width) + " "
                lines.append(row_str.rstrip())
                # Add blank line between rows for better readability
                if row < self.rows - 1:
                    lines.append("")

            # Column numbers with zero-padding for consistency
            col_nums = ""
            for col in range(self.cols):
                col_nums += str(col).zfill(max_col_width) + " "
            lines.append(col_nums.rstrip())

            # Game status
            if self.game_over[batch_idx]:
                if self.winner[batch_idx] == 1:
                    lines.append("Winner: Player 1 (X)")
                elif self.winner[batch_idx] == -1:
                    lines.append("Winner: Player 2 (O)")
                else:
                    lines.append("Draw")
            else:
                player = "1 (X)" if self.current_player[batch_idx] == 1 else "2 (O)"
                lines.append(f"Current player: {player}")

            boards_repr.append("\n".join(lines))

        return "\n\n".join(boards_repr)


@numba.njit
def check_win_fast(
    board: np.ndarray, row: int, col: int, player: int, win_length: int
) -> bool:
    """Fast win checking using numba.

    Args:
        board: 2D game board.
        row: Row of the last move.
        col: Column of the last move.
        player: Player who made the move (1 or -1).
        win_length: Number of pieces needed to win.

    Returns:
        True if the move resulted in a win.
    """
    rows, cols = board.shape

    # Check horizontal
    count = 1
    for c in range(col - 1, max(-1, col - win_length), -1):
        if board[row, c] == player:
            count += 1
        else:
            break
    for c in range(col + 1, min(cols, col + win_length)):
        if board[row, c] == player:
            count += 1
        else:
            break
    if count >= win_length:
        return True

    # Check vertical
    count = 1
    for r in range(row - 1, max(-1, row - win_length), -1):
        if board[r, col] == player:
            count += 1
        else:
            break
    for r in range(row + 1, min(rows, row + win_length)):
        if board[r, col] == player:
            count += 1
        else:
            break
    if count >= win_length:
        return True

    # Check diagonal
    count = 1
    for i in range(1, win_length):
        r, c = row - i, col - i
        if r < 0 or c < 0 or board[r, c] != player:
            break
        count += 1
    for i in range(1, win_length):
        r, c = row + i, col + i
        if r >= rows or c >= cols or board[r, c] != player:
            break
        count += 1
    if count >= win_length:
        return True

    # Check anti-diagonal
    count = 1
    for i in range(1, win_length):
        r, c = row - i, col + i
        if r < 0 or c >= cols or board[r, c] != player:
            break
        count += 1
    for i in range(1, win_length):
        r, c = row + i, col - i
        if r >= rows or c < 0 or board[r, c] != player:
            break
        count += 1
    return count >= win_length
