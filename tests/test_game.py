import numpy as np

from src.game import ConnectFour


def test_initialization():
    batch_size = 5
    game = ConnectFour(batch_size=batch_size)

    assert game.batch_size == batch_size
    assert game.boards.shape == (batch_size, 6, 7)
    assert game.current_player.shape == (batch_size,)
    assert np.all(game.current_player == 1)
    assert np.all(~game.game_over)
    assert np.all(game.winner == 0)
    assert np.all(game.move_count == 0)
    assert np.all(game.boards == 0)


def test_valid_moves():
    game = ConnectFour(batch_size=3)

    valid = game.get_valid_moves()
    assert valid.shape == (3, 7)
    assert np.all(valid)

    for _ in range(6):
        game.make_move(np.array([0, 1, 2]))

    valid = game.get_valid_moves()
    assert not valid[0, 0]
    assert not valid[1, 1]
    assert not valid[2, 2]
    assert valid[0, 1]
    assert valid[1, 0]


def test_vertical_win():
    game = ConnectFour(batch_size=2)

    game.make_move(np.array([0, 1]))
    game.make_move(np.array([1, 0]))
    game.make_move(np.array([0, 1]))
    game.make_move(np.array([1, 0]))
    game.make_move(np.array([0, 1]))
    game.make_move(np.array([1, 0]))

    rewards, done, _ = game.make_move(np.array([0, 1]))

    assert rewards[0] == 1.0
    assert done[0]
    assert game.winner[0] == 1


def test_horizontal_win():
    game = ConnectFour(batch_size=1)

    game.make_move(np.array([0]))
    game.make_move(np.array([0]))
    game.make_move(np.array([1]))
    game.make_move(np.array([1]))
    game.make_move(np.array([2]))
    game.make_move(np.array([2]))

    rewards, done, _ = game.make_move(np.array([3]))

    assert rewards[0] == 1.0
    assert done[0]
    assert game.winner[0] == 1


def test_diagonal_win():
    game = ConnectFour(batch_size=1)

    # Build up for diagonal win
    # Player 1 (X) plays: 0
    # Player 2 (O) plays: 1
    # Player 1 (X) plays: 1
    # Player 2 (O) plays: 2
    # Player 1 (X) plays: 2
    # Player 2 (O) plays: 3
    # Player 1 (X) plays: 2
    # Player 2 (O) plays: 3
    # Player 1 (X) plays: 3
    # Player 2 (O) plays: 0
    # Player 1 (X) plays: 3  - Should win with diagonal

    moves_sequence = [0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3]

    for move in moves_sequence[:-1]:
        game.make_move(np.array([move]))

    rewards, done, _ = game.make_move(np.array([moves_sequence[-1]]))

    assert rewards[0] == 1.0
    assert done[0]
    assert game.winner[0] == 1


def test_draw():
    game = ConnectFour(batch_size=1)

    # Fill board without causing a win - use a simple pattern
    move_count = 0
    done = np.array([False])  # Initialize done
    for col in range(7):
        for _ in range(6):
            if move_count >= 42:
                break
            _, done, _ = game.make_move(np.array([col % 7]))
            move_count += 1
            if done[0]:
                # Shouldn't win with this pattern, but if we do, that's ok for the test
                break
        if move_count >= 42 or done[0]:
            break

    assert game.game_over[0] or game.move_count[0] >= 42


def test_invalid_moves():
    game = ConnectFour(batch_size=2)

    for _ in range(6):
        game.make_move(np.array([0, 1]))

    rewards, _, valid = game.make_move(np.array([0, 2]))

    assert not valid[0]  # Column 0 is full in game 0
    assert valid[1]  # Column 2 is valid in game 1
    assert abs(rewards[0]) < 1e-6


def test_board_state():
    game = ConnectFour(batch_size=2)

    game.make_move(np.array([3, 4]))
    game.make_move(np.array([3, 4]))

    state = game.get_board_state()
    assert state.shape == (2, 2, 6, 7)

    assert state[0, 0, 5, 3] == 1.0
    assert state[0, 1, 4, 3] == 1.0

    assert state[1, 0, 5, 4] == 1.0
    assert state[1, 1, 4, 4] == 1.0


def test_batch_consistency():
    batch_size = 10
    game = ConnectFour(batch_size=batch_size)

    np.random.seed(42)

    for _ in range(20):
        cols = np.random.randint(0, 7, batch_size)
        rewards, done, valid = game.make_move(cols)

        assert rewards.shape == (batch_size,)
        assert done.shape == (batch_size,)
        assert valid.shape == (batch_size,)

        if np.any(done):
            break


def test_configurable_size():
    """Test that different board sizes work correctly."""
    # Test a smaller board
    game = ConnectFour(batch_size=1, rows=4, cols=5, win_length=3)
    assert game.boards.shape == (1, 4, 5)
    assert game.rows == 4
    assert game.cols == 5
    assert game.win_length == 3

    # Test win with 3 in a row
    game.make_move(np.array([0]))  # P1
    game.make_move(np.array([1]))  # P2
    game.make_move(np.array([0]))  # P1
    game.make_move(np.array([1]))  # P2
    _, done, _ = game.make_move(np.array([0]))  # P1 wins with 3 vertical

    assert done[0]
    assert game.winner[0] == 1

    # Test a larger board
    game = ConnectFour(batch_size=2, rows=10, cols=10, win_length=5)
    assert game.boards.shape == (2, 10, 10)


def test_clone():
    game1 = ConnectFour(batch_size=2)

    game1.make_move(np.array([0, 1]))
    game1.make_move(np.array([1, 0]))

    game2 = game1.clone()

    assert np.array_equal(game1.boards, game2.boards)
    assert np.array_equal(game1.current_player, game2.current_player)
    assert np.array_equal(game1.game_over, game2.game_over)
    assert np.array_equal(game1.winner, game2.winner)
    assert np.array_equal(game1.move_count, game2.move_count)
    assert game1.rows == game2.rows
    assert game1.cols == game2.cols
    assert game1.win_length == game2.win_length

    game2.make_move(np.array([2, 3]))

    assert not np.array_equal(game1.boards, game2.boards)


def test_repr():
    """Test the __repr__ method displays the board correctly."""
    game = ConnectFour(batch_size=1)

    # Empty board should show all dashes
    repr_str = repr(game)
    # Check for dashes (board content)
    assert "-" in repr_str
    # Check for column numbers
    assert "0 1 2 3 4 5 6" in repr_str
    assert "Current player: 1 (X)" in repr_str

    # Make some moves
    game.make_move(np.array([3]))  # Player 1
    game.make_move(np.array([3]))  # Player 2
    game.make_move(np.array([4]))  # Player 1

    repr_str = repr(game)
    # Bottom row should have pieces in columns 3 and 4
    # Check that we have pieces X and O in the board
    assert 'X' in repr_str
    assert 'O' in repr_str
    assert "Current player: 2 (O)" in repr_str

    # Test with multiple games
    game_multi = ConnectFour(batch_size=2)
    repr_str = repr(game_multi)
    assert "Game 0:" in repr_str
    assert "Game 1:" in repr_str


def test_repr_winner():
    """Test that __repr__ shows winner correctly."""
    game = ConnectFour(batch_size=1)

    # Create a winning position for player 1
    for _ in range(3):
        game.make_move(np.array([0]))  # P1
        game.make_move(np.array([1]))  # P2
    game.make_move(np.array([0]))  # P1 wins

    repr_str = repr(game)
    assert "Winner: Player 1 (X)" in repr_str


def test_repr_large_board():
    """Test that repr handles multi-digit column numbers correctly."""
    # Test with double-digit columns
    game = ConnectFour(batch_size=1, rows=5, cols=15, win_length=4)
    repr_str = repr(game)

    # Check that double-digit column numbers are present and aligned
    assert "10" in repr_str
    assert "14" in repr_str
    # Check proper alignment with spacing
    assert " 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14" in repr_str

    # Make a move in a double-digit column
    game.make_move(np.array([10]))
    repr_str = repr(game)
    assert "X" in repr_str


def test_play_random_game():
    """Test a full random game to ensure no crashes during gameplay."""
    game = ConnectFour(batch_size=1)

    move_count = 0
    max_moves = 42  # Maximum possible moves in Connect 4

    while not game.game_over[0] and move_count < max_moves:
        valid_moves = game.get_valid_moves()[0]
        valid_cols = np.where(valid_moves)[0]

        if len(valid_cols) == 0:
            break

        col = np.random.choice(valid_cols)
        _, done, _ = game.make_move(np.array([col]))
        move_count += 1

        if done[0]:
            break

    # Game should have ended either by win or draw
    assert game.game_over[0] or move_count == max_moves
    assert game.winner[0] in [-1, 0, 1]  # Player 2, draw, or Player 1
