from classes import (
    Board,
    OccupiedFieldEror,
    WrongSignError,
    Player,
    InvalidPlayerData,
    PcRandomPlayer,
    FieldNotOnBoardError,
    WrongFieldDataError,
    BoardCombinations,
    Game,
    Direction
)
import pytest
from Boards import (
    empty_board,
    full_board_x,
    board_win_1,
    board_win_2,
    board_win_3,
    board_win_4,
    board_win_5,
    board_win_6,
    board_win_7,
    board_not_win_1,
    board_not_win_2,
    board_not_win_3,
    board_not_win_4,
    board_not_win_5,
    board_1,
    board_2,
    board_3,
    board_4,
    board_5,
    board_6
)


def test_board_state():
    board = Board()
    assert board.state() == empty_board


def test_board_put_cross():
    board = Board()
    board.put('X', 0, 0)
    assert board.state() == [
        ['X', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-']
    ]
    board.put('x', 1, 1)
    assert board.state() == [
        ['X', '-', '-', '-', '-', '-'],
        ['-', 'X', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-']
    ]


def test_board_put_circle():
    board = Board()
    board.put('O', 3, 4)
    assert board.state() == [
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'O', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-']
    ]


def test_board_put_both():
    board = Board()
    board.put('O', 3, 4)
    board.put('X', 3, 5)
    assert board.state() == [
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'O', 'X'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-']
    ]


def test_board_put_upper_lower():
    board = Board()
    board.put('o', 3, 4)
    board.put('O', 3, 5)
    board.put('x', 4, 4)
    board.put('X', 4, 5)
    assert board.state() == [
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', 'O', 'O'],
        ['-', '-', '-', '-', 'X', 'X'],
        ['-', '-', '-', '-', '-', '-']
    ]


def test_board_put_occupied():
    board = Board()
    board.put('o', 3, 4)
    with pytest.raises(OccupiedFieldEror):
        board.put('X', 3, 4)


def test_board_put_wrong_sign():
    board = Board()
    with pytest.raises(WrongSignError):
        board.put('a', '3', '4')


def test_board_put_wrong_sign_int():
    board = Board()
    with pytest.raises(WrongSignError):
        board.put(1, '3', '4')


def test_board_put_wrong_field_not_on_board():
    board = Board()
    with pytest.raises(FieldNotOnBoardError):
        board.put('X', '3', '7')


def test_board_put_wrong_field_data():
    board = Board()
    with pytest.raises(WrongFieldDataError):
        board.put('X', 'k', '7')


def test_player_create():
    player = Player('Jurek Ogorek', 'chaos')
    assert player.name() == 'Jurek Ogorek'
    assert player.goal() == 'chaos'


def test_player_create_no_name():
    with pytest.raises(InvalidPlayerData):
        Player('', 'chaos')


def test_player_create_no_goal():
    with pytest.raises(InvalidPlayerData):
        Player('Jurek Ogorek', '')


def test_pc_random_player_create():
    pc_random_player = PcRandomPlayer('PC', 'chaos')
    assert pc_random_player.name() == 'PC'
    assert pc_random_player.goal() == 'chaos'


def test_pc_random_player_generate_move(monkeypatch):
    pc_random_player = PcRandomPlayer('PC', 'chaos')
    board = Board()

    def const(a, b):
        return 3

    def sign_x(a):
        return 'X'
    monkeypatch.setattr('classes.random.randint', const)
    monkeypatch.setattr('classes.random.choice', sign_x)

    assert pc_random_player.generate_move(board) == (3, 3, 'X')


def test_game_create():
    game = Game()
    assert game.board.state() == empty_board
    assert game.player_1 is None
    assert game.player_2 is None
    assert game.order_player is None
    assert game.chaos_player is None
    assert game.is_game_finished is False
    assert game.is_round_finished is False
    assert game.is_first_move_finished is False
    assert game.is_new_game_set is False


def test_board_combinations_create():
    combinations = BoardCombinations()
    assert combinations.X_counter() == 0
    assert combinations.O_counter() == 0
    assert combinations.state() == [
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-', '-']
    ]


def test_is_field_empty_and_on_board_empty():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_2)
    assert board_combinations.is_field_empty_and_on_board((0, 0)) is True
    assert board_combinations.is_field_empty_and_on_board((0, 3)) is True
    assert board_combinations.is_field_empty_and_on_board((4, 5)) is True
    assert board_combinations.is_field_empty_and_on_board((5, 3)) is True


def test_is_field_empty_and_on_board_occupied():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_2)
    assert board_combinations.is_field_empty_and_on_board((0, 1)) is False
    assert board_combinations.is_field_empty_and_on_board((3, 1)) is False
    assert board_combinations.is_field_empty_and_on_board((2, 5)) is False
    assert board_combinations.is_field_empty_and_on_board((2, 3)) is False


def test_is_field_empty_and_on_board_not_on_board():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_2)
    assert board_combinations.is_field_empty_and_on_board((0, 8)) is False
    assert board_combinations.is_field_empty_and_on_board((9, 1)) is False
    assert board_combinations.is_field_empty_and_on_board((10, 9)) is False
    assert board_combinations.is_field_empty_and_on_board((22, 1)) is False


def test_is_order_win_yes_1():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_1)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_2():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_2)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_3():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_3)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_4():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_4)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_5():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_5)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_6():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_6)
    assert board_combinations.is_win_order() is True


def test_is_order_win_yes_7():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_win_7)
    assert board_combinations.is_win_order() is True


def test_is_order_win_not_1():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_1)
    assert board_combinations.is_win_order() is False


def test_is_order_win_not_2():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_2)
    assert board_combinations.is_win_order() is False


def test_is_order_win_not_3():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_3)
    assert board_combinations.is_win_order() is False


def test_is_order_win_not_4():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_4)
    assert board_combinations.is_win_order() is False


def test_is_order_win_not_5():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_5)
    assert board_combinations.is_win_order() is False


def test_board_combinations_is_full_not():
    board_combinations = BoardCombinations(empty_board)
    assert board_combinations.is_full() is False


def test_board_combinations_is_full_yes():
    board_combinations = BoardCombinations(full_board_x)
    assert board_combinations.is_full() is True


def test_board_find_combinationss_1():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_1)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (1,
         2,
         1,
         'horizontally'),
        (1,
         2,
         1,
         'vertically'),
        (1,
         2,
         1,
         'diagonally_short_down')]


def test_board_find_combinationss_2():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_2)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (2, 2, 2, 'horizontally'),
        (1, 2, 1, 'vertically'),
        (1, 2, 2, 'vertically'),
        (1, 2, 2, 'diagonally_short_up'),
        (1, 2, 1, 'diagonally_short_down'),
        (1, 2, 2, 'diagonally_long_down')
    ]
    assert O_combinations == [
        (1, 2, 4, 'horizontally'),
        (1, 3, 4, 'horizontally'),
        (2, 3, 4, 'vertically'),
        (1, 2, 4, 'diagonally_short_up'),
        (1, 3, 4, 'diagonally_short_down')
    ]


def test_board_find_combinationss_3():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_3)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (2, 2, 2, 'horizontally'),
        (1, 2, 1, 'vertically'),
        (1, 2, 2, 'vertically'),
        (1, 2, 2, 'diagonally_short_up'),
        (1, 2, 1, 'diagonally_short_down'),
        (1, 2, 2, 'diagonally_long_down')
    ]
    assert O_combinations == [
        (1, 2, 4, 'horizontally'),
        (2, 3, 4, 'horizontally'),
        (1, 3, 3, 'vertically'),
        (2, 3, 4, 'vertically'),
        (2, 2, 4, 'diagonally_short_up'),
        (1, 3, 4, 'diagonally_short_down'),
        (1, 3, 3, 'diagonally_long_down')
    ]


def test_board_find_combinationss_4():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_not_win_2)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (4, 2, 5, 'horizontally'),
        (1, 2, 2, 'vertically'),
        (1, 2, 3, 'vertically'),
        (1, 2, 4, 'vertically'),
        (1, 2, 5, 'vertically'),
        (1, 2, 2, 'diagonally_short_up'),
        (1, 2, 4, 'diagonally_short_up'),
        (1, 2, 3, 'diagonally_short_down'),
        (1, 2, 2, 'diagonally_long_down'),
        (1, 2, 3, 'diagonally_long_up')
    ]
    assert O_combinations == [
        (1, 0, 1, 'horizontally'),
        (1, 1, 1, 'horizontally'),
        (1, 2, 1, 'horizontally'),
        (1, 3, 1, 'horizontally'),
        (1, 4, 1, 'horizontally'),
        (1, 5, 1, 'horizontally'),
        (6, 5, 1, 'vertically'),
        (1, 3, 1, 'diagonally_short_up'),
        (1, 5, 1, 'diagonally_short_up'),
        (1, 2, 1, 'diagonally_short_down'),
        (1, 0, 1, 'diagonally_short_down'),
        (1, 1, 1, 'diagonally_long_down'),
        (1, 4, 1, 'diagonally_long_up')
    ]


def test_board_find_combinationss_6():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_5)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (1, 5, 3, 'horizontally'),
        (1, 5, 3, 'vertically')
    ]


def test_board_find_combinations_7():
    board_combinations = BoardCombinations()
    board_combinations.set_board_state(board_6)
    X_combinations, O_combinations = board_combinations.find_combinations()
    assert X_combinations == [
        (1, 5, 4, 'horizontally'),
        (1, 5, 4, 'vertically'),
        (1, 5, 4, 'diagonally_short_down')
    ]
    assert O_combinations == [
        (1, 5, 3, 'horizontally'),
        (1, 5, 3, 'vertically')
    ]


def test_move_with_5_comb_first():
    board_combinations = BoardCombinations()
    chosen_comb = (1, 2, 1, Direction.vertically.value, 'X')
    first_field = 1, 1, 'X'
    second_field = 3, 1, 'X'
    order_facotr = 1
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field == first_field


def test_move_with_5_comb_second():
    board_combinations = BoardCombinations()
    chosen_comb = (1, 2, 1, Direction.vertically.value, 'X')
    first_field = 1, 1, 'X'
    second_field = 3, 1, 'X'
    order_facotr = 0
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field == second_field


def test_move_with_5_comb_none():
    board_combinations = BoardCombinations(board_1)
    chosen_comb = (1, 2, 1, Direction.vertically.value, 'O')
    first_field = 1, 1, 'X'
    second_field = 3, 1, 'X'
    order_facotr = 0
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field is None


def test_move_with_5_comb_must_first():
    board_combinations = BoardCombinations(board_4)
    chosen_comb = (1, 2, 5, Direction.horizontally.value, 'X')
    first_field = 2, 4, 'X'
    second_field = 2, 6, 'X'
    order_facotr = 1
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field == first_field


def test_move_with_5_comb_must_second():
    board_combinations = BoardCombinations()
    chosen_comb = (1, 2, 0, Direction.horizontally.value, 'X')
    first_field = 2, -1, 'X'
    second_field = 2, 1, 'X'
    order_facotr = 0
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field == second_field


def test_move_with_5_comb_blocked():
    board_combinations = BoardCombinations(board_3)
    chosen_comb = (2, 2, 2, Direction.horizontally.value, 'X')
    first_field = 2, 3, 'X'
    second_field = 2, 0, 'X'
    order_facotr = 1
    chosen_field = board_combinations.move_with_5_comb(
        chosen_comb, first_field, second_field, order_facotr)
    assert chosen_field is None
