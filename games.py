from classes import *
from AiPlayer import OrderTrainer, ChaosTrainer

class Game:
    """
    Klasa reprezentująca grę
    """

    def __init__(self):
        """
        Inicjalizuje grę
        """
        self.board = Board()
        self.board_combinations = BoardCombinations()
        self.player_1 = None
        self.player_2 = None
        self.order_player = None
        self.chaos_player = None
        self.is_game_finished = False
        self.is_round_finished = False
        self.is_first_move_finished = False
        self.is_new_game_set = False

    def players_choice(self):
        """
        Funkcja tworząca dwóch graczy, jako atrybuty klasy Game.
        Parametry tworzenia graczy podawane są przez konsolę.
        Pierwszy gracz domyślnie jest graczem klasy Player (człowiek)
        Po podaniu celu gracza pierwszego, cel drugiego gracza
        przypisany jest automatycznie.
        """
        os.system('clear')
        print('Choose the players')
        try:
            player_1_name = input('First player name: ')
            player_1_goal = input('First player goal (chaos/order): ').lower()
            self.player_1 = PcPlayer(player_1_name, player_1_goal)
            player_2_type = input(
                'Choose type of your oponent (Player/PcRandomPlayer/PcPlayer): ').lower()
            if player_2_type not in ('player', 'pcrandomplayer', 'pcplayer'):
                raise InvalidPlayerData(
                    'Oponent has to be either Player or PCRandomPlayer or PcPlayer')
            player_2_name = input('Second player name: ')
            if player_2_name == player_1_name:
                player_2_name += '_2'
            if player_1_goal == 'chaos':
                self.chaos_player = self.player_1
                player_2_goal = 'order'
            else:
                self.order_player = self.player_1
                player_2_goal = 'chaos'
            if player_2_type == 'player':
                self.player_2 = Player(player_2_name, player_2_goal)
            if player_2_type == 'pcrandomplayer':
                self.player_2 = PcRandomPlayer(player_2_name, player_2_goal)
            if player_2_type == 'pcplayer':
                self.player_2 = PcPlayer(player_2_name, player_2_goal)
            if self.player_2.goal() == 'order':
                self.order_player = self.player_2
            else:
                self.chaos_player = self.player_2
        except InvalidPlayerData:
            print('Players data was incorrect, choose players again')
            self.players_choice()

    def play_round(self):
        """
        Funkcja realizuje rozegranie rundy gry.
        Każdą rundę rozpoczyna gracz z przypisanym celem porządek(order)
        Obaj gracze podają kolejno poprzez konsolę, współzędne pola na planszy
        i znak jaki ma zostać na nim postawiony.
        Przed każdym wyborem zostaje wypisany na terminal aktualny stan planszy
        """
        self.is_round_finished = False
        self.is_first_move_finished = False
        while self.is_round_finished is False:
            while self.is_first_move_finished is False:
                try:
                    self.board.board_print()
                    print(f'{self.order_player.name()} is choosing field')
                    row, column, sign = self.order_player.generate_move(
                        self.board_combinations)
                    self.board.put(sign, row, column)
                    self.board_combinations.set_board_state(self.board.state())
                    self.is_first_move_finished = True
                except WrongFieldDataError:
                    new_game_commands = ('new_game', 'q', 'reset', 'quit')
                    if (
                        row.lower() in new_game_commands
                        or column.lower() in new_game_commands
                        or sign.lower() in new_game_commands
                    ):
                        self.set_new_game()
                    else:
                        print('Row and column must be numbers')
                except FieldNotOnBoardError:
                    print('Selected field is off the board')
                except WrongSignError:
                    print('Sign must be either x or X or o or O')
                except OccupiedFieldEror:
                    print('You cant put sign on occupied field')
            if (
                not self.board.is_win_order()
                and not self.board.is_full()
                and not self.is_game_finished
            ):
                try:
                    self.board.board_print()
                    print(f'{self.chaos_player.name()} is choosing field')
                    row_2, column_2, sign_2 = self.chaos_player.generate_move(
                        self.board_combinations)
                    self.board.put(sign_2, row_2, column_2)
                    self.board_combinations.set_board_state(self.board.state())
                    self.is_round_finished = True
                except WrongFieldDataError:
                    new_game_commands = ('new_game', 'q', 'reset', 'quit')
                    if (
                        row_2 in new_game_commands
                        or column_2 in new_game_commands
                        or sign_2 in new_game_commands
                    ):
                        self.set_new_game()
                    else:
                        print('Row and column must be numbers')
                except FieldNotOnBoardError:
                    print('Selected field is off the board')
                except WrongSignError:
                    print('Sign must be either x or X or o or O')
                except OccupiedFieldEror:
                    print('You cant put sign on occupied field')
            else:
                self.is_round_finished = True

    def set_new_game(self):
        """
        Funkcja pozwala graczowi na decyzję poprzez komendę na terminalu,
        na rozegranie gry od nowa lub zakończenie trwającej gry (i programu)
        """
        msg = input('Do you want to play a new game? (y/n/exit): ')
        if msg.lower() in ('y', 'yes', 'Y'):
            self.is_game_finished = True
            self.is_first_move_finished = True
            self.is_round_finished = True
            self.is_new_game_set = True
        if msg.lower() in ('n', 'no', 'N'):
            self.is_game_finished = True
            self.is_first_move_finished = True
            self.is_round_finished = True
            os.system('clear')

    def play_new_game(self):
        """
        Funkcja tworzy nową grę i rozpoczyna rozgrywkę.
        """
        os.system('clear')
        new_game = Game()
        new_game.play_game()

    def play_game(self):
        """
        Funkcja realizuję rozegranie całej gry poprzez rozpoczynanie
        kolejnych rund do momentu
        zwycięztwa któregoś z graczy lub decyzji o rozpoczęciu nowej gry
        albo o zakończeniu rozgrywki.
        """
        self.is_game_finished = False
        self.players_choice()
        os.system('clear')
        print('The game starts. To rerstart a game, type (q/quit/restart/new_game)')
        while (
            not self.board.is_win_order()
            and not self.board.is_full()
            and not self.is_game_finished
        ):
            self.play_round()
        if self.board_combinations.is_win_order():
            self.board.board_print()
            print(trophy)
            print(f'{self.order_player.name().upper()} WON !!!')
        if self.board_combinations.is_full():
            self.board.board_print()
            print(trophy)
            print(f'{self.chaos_player.name().upper()} WON !!!')
        if self.is_new_game_set is True:
            self.play_new_game()

class TrainingGame():
    def __init__(self, validation_mode=False):
        self.validation_mode = validation_mode
        self.is_game_running = False
        self.is_training_running = False
        self.order_llegal_move = False
        self.chaos_illegal_move = False

        self.order_player = OrderTrainer()
        self.chaos_player = ChaosTrainer()

        self.board = Board()
        self.board_combinations = BoardCombinations()
    
        self.states = []

    def play_game(self):
        while self.should_game_run():
            self.play_round()
        

    def check_winner(self):
        if self.board_combinations.is_win_order() or self.chaos_illegal_move:
            return 'order'
        if self.board_combinations.is_full() or self.order_llegal_move:
            return 'chaos'
        else:
            raise ValueError

    def should_game_run(self):
        if (self.board_combinations.is_win_order() or self.board_combinations.is_full() 
            or self.order_llegal_move or self.chaos_illegal_move):
                return False
        else:
            return True

    def play_round(self):
        try:
            row, column, sign = self.order_player.generate_move(
                self.board_combinations)
            self.board.put(sign, row, column)
            self.board_combinations.set_board_state(self.board.state())

            self.order_player.add_move_data(self.board.state(), (row, column, sign))
        except WrongFieldDataError:
            pass
        except OccupiedFieldEror:
            self.order_llegal_move = True
        if (
            not self.board.is_win_order()
            and not self.board.is_full()
            and not self.order_llegal_move
        ):
            try:
                row_2, column_2, sign_2 = self.chaos_player.generate_move(
                    self.board_combinations)
                self.board.put(sign_2, row_2, column_2)
                self.board_combinations.set_board_state(self.board.state())
                self.chaos_player.add_move_data(self.board.state(), (row, column, sign))
            except WrongFieldDataError:
                pass
            except OccupiedFieldEror:
                self.chaos_llegal_move = True