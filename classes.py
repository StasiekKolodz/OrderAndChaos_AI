import random
from enum import Enum
from tabulate import tabulate
from enum import Enum
import os
from text_pictures import trophy
from AiPlayer import OrderTrainer, ChaosTrainer
import numpy as np

class InvalidPlayerData(Exception):
    pass


class FieldNotOnBoardError(Exception):
    def __init__(self):
        super().__init__('Given row and column must be beetwen 0 snd 5')


class WrongFieldDataError(Exception):
    def __init__(self):
        super().__init__('Given row and column must be numbers')


class OccupiedFieldEror(Exception):
    def __init__(self):
        super().__init__('You cannot place a sign on occupied field')


class WrongSignError(Exception):
    def __init__(self):
        super().__init__('Sign must be either cross or circle')


class WrongFieldError(Exception):
    pass


class Direction(Enum):
    diagonally_long_up = 'diagonally_long_up'
    diagonally_long_down = 'diagonally_long_down'
    diagonally_short_up = 'diagonally_short_up'
    diagonally_short_down = 'diagonally_short_down'
    horizontally = 'horizontally'
    vertically = 'vertically'


class BoardCombinations():
    """
    Klasa reprezentująca planszę 6x6 wykorzystywaną w grze
    w celach analizy zawartości planszy w trakcie rozgrywki.
    Puste pola są w niej reprezentowane znakiem '-'
    Domyślnie tworzona jako pusta plansza.
    Atrybuty klasy:
        X_combinations:
            Kombinacje złożone z X na planszy
        O_combinations:
            Kombinacje złożone z O na planszy
        X_counter:
            Licznik długości kombinacji z X
        O_counter:
            Licznik długości kombinacji z O
        state:
            Stan (zawartość) planszy reprezentowany przez macierz (listę)
            Pola reprezentowane są wierszem i kolumną stanu planszy
    """

    def __init__(self, board_state=None):
        self._X_combinations = []
        self._O_combinations = []
        self._X_counter = 0
        self._O_counter = 0
        self._state = board_state if board_state else [
            ['-', '-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-', '-']
        ]

    def state(self):
        return self._state

    def set_board_state(self, board_state):
        self._state = board_state

    def X_counter(self):
        return self._X_counter

    def O_counter(self):
        return self._O_counter

    def is_full(self):
        """Sprawdza czy plansza jest pełna"""
        for i in range(6):
            if '-' in self.state()[i]:
                return False
        return True

    def is_field_empty_and_on_board(self, field):
        """
        Przyjmuje współżędne pola na planszy (wiersz i kolumnę)
        Sprawdza czy pole jest puste i czy jest na planszy
        """
        row = field[0]
        column = field[1]
        if 0 <= row <= 5 and 0 <= column <= 5:
            if self._state[row][column] == '-':
                return True
        return False

    def is_win_order(self):
        """
        Sprawdza czy porządek zwycięrzył
        (Czy jest na planszy kombinacja o długości 5)
        """
        combinations = self.find_combinations()
        for O_X_combination in combinations:
            for combination in O_X_combination:
                if combination[0] == 5:
                    return True
        return False

    def O_combination_update(
            self,
            row_number,
            column_number,
            direction,
            previous_O_count
    ):
        """
        Funkcja dodaje kombinacje znaków O do O_combinations,
        jeśli dana kombinacja skończyła się poprzez wystąpienie
        pustego pola lub przeciwnego znaku na następnym polu.
        Kombinacja znaków jest reprezentowana przez
        długość kombinacji, wiersz i kolumnę zakończenia kombinacji
        oraz jej kierunek.
        """
        if self._O_counter == 0 and previous_O_count >= 1:
            if direction == Direction.horizontally.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.vertically.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number - 1,
                    column_number,
                    direction
                ))

            if direction == Direction.diagonally_short_up.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number + 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_long_up.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number + 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_short_down.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number - 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_long_down.value:
                self._O_combinations.append((
                    previous_O_count,
                    row_number - 1,
                    column_number - 1,
                    direction
                ))

    def O_combination_update_edge(
            self,
            row_number,
            column_number,
            direction,
    ):
        """
        Funkcja dodaje kombinacje znaków O do O_combinations,
        jeśli dana kombinacja dotarła do krawędzi planszy
        (następne pole byłoby poza planszą).
        Kombinacja znaków jest reprezentowana przez
        długość kombinacji, wiersz i kolumnę zakończenia kombinacji
        oraz jej kierunek.
        """
        if self._O_counter != 0:
            if direction == Direction.horizontally.value:
                if column_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))

            if direction == Direction.vertically.value:
                if row_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_short_up.value:
                if row_number == 0 and column_number == 4:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
                if row_number == 1 and column_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_long_up.value:
                if row_number == 0 and column_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_short_down.value:
                if row_number == 5 and column_number == 4:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
                if row_number == 4 and column_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_long_down.value:
                if row_number == 5 and column_number == 5:
                    self._O_combinations.append((
                        self._O_counter,
                        row_number,
                        column_number,
                        direction
                    ))

    def X_combination_update(
            self,
            row_number,
            column_number,
            direction,
            previous_X_count
    ):
        """
        Funkcja dodaje kombinacje znaków X do X_combinations,
        jeśli dana kombinacja skończyła się poprzez wystąpienie
        pustego pola lub przeciwnego znaku na następnym polu.
        Kombinacja znaków jest reprezentowana przez
        długość kombinacji, wiersz i kolumnę zakończenia kombinacji
        oraz jej kierunek.
        """
        if self._X_counter == 0 and previous_X_count >= 1:
            if direction == Direction.horizontally.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.vertically.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number - 1,
                    column_number,
                    direction
                ))

            if direction == Direction.diagonally_short_up.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number + 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_long_up.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number + 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_short_down.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number - 1,
                    column_number - 1,
                    direction
                ))

            if direction == Direction.diagonally_long_down.value:
                self._X_combinations.append((
                    previous_X_count,
                    row_number - 1,
                    column_number - 1,
                    direction
                ))

    def X_combination_update_edge(
            self,
            row_number,
            column_number,
            direction,
    ):
        """
        Funkcja dodaje kombinacje znaków X do X_combinations,
        jeśli dana kombinacja dotarła do krawędzi planszy
        (następne pole byłoby poza planszą).
        Kombinacja znaków jest reprezentowana przez
        długość kombinacji, wiersz i kolumnę zakończenia kombinacji
        oraz jej kierunek.
        """
        if self._X_counter != 0:
            if direction == Direction.horizontally.value:
                if column_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))

            if direction == Direction.vertically.value:
                if row_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_short_up.value:
                if row_number == 0 and column_number == 4:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
                if row_number == 1 and column_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_long_up.value:
                if row_number == 0 and column_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_short_down.value:
                if row_number == 5 and column_number == 4:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
                if row_number == 4 and column_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))
            if direction == Direction.diagonally_long_down.value:
                if row_number == 5 and column_number == 5:
                    self._X_combinations.append((
                        self._X_counter,
                        row_number,
                        column_number,
                        direction
                    ))

    def combinations_counter_update(
        self,
        sign,
        row_number,
        column_number,
        direction
    ):
        """
        Funkcja aktualizuje wartość liczników długości kombinacji
        O_counter i X_counter.
        Jeśli znak na polu to X licznik X_counter ziększa się o 1
        (wydłużenie kombuinacji X o 1),
        a licznik O_counter jest wyzerowany (zakończenie
        lub podtrzymanie braku kombinacji O), analogicznie dla pola o znaku O.
        Gdy pole jest puste oba liczniki są wyzerowane ponieważ,
        oznacza to zakończenie lub brak kombinacji.
        Wartości liczników przekazane są do funkcji aktualizującej kombinacje.
        """
        previous_X_count = self._X_counter
        previous_O_count = self._O_counter
        if sign == 'X':
            self._X_counter += 1
            self._O_counter = 0
        if sign == 'O':
            self._O_counter += 1
            self._X_counter = 0
        if sign == '-':
            self._O_counter = 0
            self._X_counter = 0

        self.O_combination_update(
            row_number,
            column_number,
            direction,
            previous_O_count
        )

        self.O_combination_update_edge(
            row_number,
            column_number,
            direction,
        )

        self.X_combination_update(
            row_number,
            column_number,
            direction,
            previous_X_count
        )

        self.X_combination_update_edge(
            row_number,
            column_number,
            direction,
        )

    def find_combinations(self):
        """
        Funkcja dokonuje skanowania po kolejnych wierszach,
        kolumnach i ukosach planszy wywołując aktualizacje liczników
        kombinacji dla każdego napotkanego pola.
        Skanowanie odbywa się od lewej do prawej
        Skanowanie odbywa się kolejno po:
        Wierszach
        Kolumnach
        Krótkich przekątnych skierowanych do góry (diagonally_short_up):
            Są to dwie przekątne o długości 4 pola,
            pierwsza z początkiem na polu (4,0) i końcem (0,4),
            a druga z  początkiem na (5,1) i końcem (1,5)
        Krótkich przekątnych skierowanych do dołu (diagonally_short_down):
            Są to dwie przekątne o długości 4 pola,
            pierwsza z początkiem na polu (0,1) i końcem (4,5),
            a druga z  początkiem na (1,0) i końcem (5,4)
        Długiej przekątnej skierowanej do góry (diagonally_long_up):
            Przekątna o długości 5 pól
            z początkiem na polu (5,0) i końcem (0,5)
        Długiej przekątnej skierowanej do dołu (diagonally_long_down):
            Przekątna o długości 5 pól
            z początkiem na polu (0,0) i końcem (5,5)
        """
        self._X_combinations = []
        self._O_combinations = []
        row_number = 0
        for row in range(6):
            row = self._state[row]
            column_number = 0
            self._X_counter = 0
            self._O_counter = 0
            for sign in row:
                self.combinations_counter_update(
                    sign,
                    row_number,
                    column_number,
                    Direction.horizontally.value
                )
                column_number += 1
            row_number += 1
        for column in range(6):
            row_number = 0
            self._X_counter = 0
            self._O_counter = 0
            for row in range(6):
                sign = self._state[row][column]
                self.combinations_counter_update(
                    sign,
                    row_number,
                    column,
                    Direction.vertically.value
                )
                row_number += 1

        for i in range(2):
            self._X_counter = 0
            self._O_counter = 0
            for row in range(5):
                row_number = 4 + i - row
                column = row + i
                sign = self._state[row_number][column]
                self.combinations_counter_update(
                    sign,
                    row_number,
                    column,
                    Direction.diagonally_short_up.value
                )

        for i in range(2):
            self._X_counter = 0
            self._O_counter = 0
            for row in range(5):
                row_number = row - i + 1
                column = i + row
                sign = self._state[row_number][column]
                self.combinations_counter_update(
                    sign,
                    row_number,
                    column,
                    Direction.diagonally_short_down.value
                )

        self._X_counter = 0
        self._O_counter = 0
        for row in range(6):
            sign = self._state[row][row]
            self.combinations_counter_update(
                sign,
                row,
                row,
                Direction.diagonally_long_down.value
            )

        self._X_counter = 0
        self._O_counter = 0
        for row in range(6):
            row_number = 5 - row
            column = row
            sign = self._state[row_number][column]
            self.combinations_counter_update(
                sign, row_number, column, Direction.diagonally_long_up.value)
        return self._X_combinations, self._O_combinations

    def is_5_combination_possible(self, chosen_comb, field):
        field_row, field_column = field[0:2]
        comb_sign = chosen_comb[4]
        comb_direction = chosen_comb[3]
        X_combinations, O_combinations = self.find_combinations()
        if comb_direction in (
                Direction.diagonally_short_down.value,
                Direction.diagonally_short_up.value):
            return True
        if comb_sign == 'X':
            if comb_direction == Direction.horizontally.value:
                for combination in O_combinations:
                    if (
                        combination[3] == Direction.horizontally.value
                        and combination[1] == field_row
                    ):
                        if combination[0] > 1:
                            return False
                        if combination[2] not in (0, 5):
                            return False
                return True
            if comb_direction == Direction.vertically.value:
                for combination in O_combinations:
                    if (
                        combination[3] == Direction.vertically.value
                        and combination[2] == field_column
                    ):
                        if combination[0] > 1:
                            return False
                        if combination[1] not in (0, 5):
                            return False
                return True

            if comb_direction == Direction.diagonally_long_up.value:
                for combination in O_combinations:
                    if combination[3] == Direction.diagonally_long_up.value:
                        if combination[0] > 1:
                            return False
                        if combination[1] != 5 and combination[2] != 0:
                            if combination[1] != 0 and combination[2] != 5:
                                return False
                return True

            if comb_direction == Direction.diagonally_long_down.value:
                for combination in O_combinations:
                    if combination[3] == Direction.diagonally_long_down.value:
                        if combination[0] > 1:
                            return False
                        if combination[1] != 5 and combination[2] != 5:
                            if combination[1] != 0 and combination[2] != 0:
                                return False
                return True

        if comb_sign == 'O':
            if comb_direction == Direction.horizontally.value:
                for combination in X_combinations:
                    if (
                        combination[3] == Direction.horizontally.value
                        and combination[1] == field_row
                    ):
                        if combination[0] > 1:
                            return False
                        if combination[2] not in (0, 5):
                            return False
                return True
            if comb_direction == Direction.vertically.value:
                for combination in X_combinations:
                    if (
                        combination[3] == Direction.vertically.value
                        and combination[2] == field_column
                    ):
                        if combination[0] > 1:
                            return False
                        if combination[1] not in (0, 5):
                            return False
                return True

            if comb_direction == Direction.diagonally_long_up.value:
                for combination in X_combinations:
                    if combination[3] == Direction.diagonally_long_up.value:
                        if combination[0] > 1:
                            return False
                        if combination[1] != 5 and combination[2] != 0:
                            if combination[1] != 0 and combination[2] != 5:
                                return False
                return True

            if comb_direction == Direction.diagonally_long_down.value:
                for combination in X_combinations:
                    if combination[3] == Direction.diagonally_long_down.value:
                        if combination[0] > 1:
                            return False
                        if combination[1] != 5 and combination[2] != 5:
                            if combination[1] != 0 and combination[2] != 0:
                                return False
                return True

    def move_with_5_comb(
            self,
            chosen_comb,
            first_field,
            second_field,
            order_factor):
        """
        Funkcja sprawdza czy podane pola są na planszy i czy w lini
        określonej przez wybraną kombinację (chosen_comb)
        możliwe jest stworzenie kombinacji
        o długości 5 znaków (określonych przez znak kombinacji)
        Dodatkowo warunkiem sprawdzenia pierwszego pola
        jest bool(order_factor) == True.
        """
        if self.is_field_empty_and_on_board(first_field) and order_factor:
            if self.is_5_combination_possible(chosen_comb, first_field):
                return first_field
        if self.is_field_empty_and_on_board(second_field):
            if self.is_5_combination_possible(chosen_comb, second_field):
                return second_field


class Board(BoardCombinations):
    """Klasa Board dziedziczy po BoardCombinations.
       reprezentuje ona polanszę 6x6 gry i wykorzystana jest w wizualnym
       interfejsie urzytkownika.
    """

    def put(self, sign, row, column):
        """
        Funkcja przyjmuje współżędne pola oraz jego znak,
        a następnie wstawia ten znak na podane pole.
        Może ona wznośić następujące wyjątki:
            WrongFieldData gdy któraś współżędna pola nie jest liczbą
            FieldNotOnBoard gdy podane pole nie znajduje się na planszy
            OccupiedFieldError gdy podane pole nie jest pustego
            Wrong sign error gdy podany znak nie jest X lub O
        """
        if isinstance(row, str) and isinstance(column, str):
            if not row.isdigit() or not column.isdigit():
                raise WrongFieldDataError
        row = int(row)
        column = int(column)
        if row not in range(6) or column not in range(6):
            raise FieldNotOnBoardError
        if self._state[row][column] != '-':
            raise OccupiedFieldEror
        if sign not in ('X', 'x', 'o', 'O'):
            raise WrongSignError

        if sign in ('X', 'x'):
            self._state[row][column] = 'X'
        else:
            self._state[row][column] = 'O'

    def board_print(self):
        """
        Funkcja wypisuje stan planszy na ekran w formie tabeli.
        Tabela tworzona jest przy pomocy biblioteki tabulate
        w formacie fancy_grid.
        Nagłówek tabeli to kolejne liczby od 0 do 5,
        a indeksowanie wierszy jest włączone (od 0 do 5)
        """
        header = ['0', '1', '2', '3', '4', '5']
        print(
            tabulate(
                self._state,
                headers=header,
                tablefmt='fancy_grid',
                showindex=True))


class Player:
    """
    Klasa Player reprezentuje gracza w grze.
    Ruchy gracza klasy Player sterowane są przez urzytkownika.
    Atrybuty:
        Nazwa
        Cel:
            Określa on czy gracz jest porządkiem czy chaosem,
            (czyli czy jego cel to stworzenie kombinacji
            czy powstrzymanie jej otworzenia)

    """

    def __init__(self, name, player_goal):
        """
        Tworząc instancję klasy Player niezbędne jest podanie
        jego imienia oraz celu gry,
        a cel gry musi być to chaos lub order (chaos lub porządek),
        w przeciwnym wypadku zostaje wzniesiony wyjątek InvalidPlayerData.
        """
        if not name:
            raise InvalidPlayerData('Player needs to have name')
        if player_goal not in ('chaos', 'order'):
            raise InvalidPlayerData('Player has to be either chaos or order')
        self._name = name
        self._goal = player_goal

    def name(self):
        return self._name

    def goal(self):
        return self._goal

    def generate_move(self, board_combinations=None):
        """
        Funkcja zwraca dane niezbędne do wykonania ruchu
        (współżędne pola i jego znak) na podstawie danych
        wpisanych przez użytkownika w terminalu.
        """
        row = input('Row: ')
        column = input('Column: ')
        sign = input('Sign: ')
        return row, column, sign


class PcRandomPlayer(Player):
    """
    Klasa PcRandomPlayer dziedziczy po Player.
    Jest to gracz którego ruchy generowane są w sposób losowy
    """

    def generate_random_move(self, board_combinations):
        """
        Funkcja zwraca współżędnie losowo wybranego pola
        spośród pustych pól na planszy losowy znak ( O lub X)
        """
        row = random.randint(0, 5)
        column = random.randint(0, 5)
        while board_combinations.state()[row][column] != '-':
            row = random.randint(0, 5)
            column = random.randint(0, 5)
        sign_list = ['X', 'O']
        sign = random.choice(sign_list)
        return row, column, sign

    def generate_move(self, board_combinations):
        """
        Funkcja generuje ruch gracza klasy PcRandomPlayer.
        Jest to losowy ruch.
        """
        return self.generate_random_move(board_combinations)


class PcPlayer(PcRandomPlayer):
    """
    Klasa PcPlayer dziedziczy po PcRandomPlayer.
    Ruchy gracza sterowane są przez algorytm tak by były możliwe
    najbardziej skuteczne.
    """

    def choose_sign(self, comb_sign):
        """
        Funkcja wybiera znak jaki zostanie postawiony w danym ruchu
        w zależności czy gracz PcPlayer buduje kombinacje (jest porządkiem)
        czy blokuje powstawanie kombuinacji (jest chaosem)
        """
        block_signs = ('X', 'O')
        build_signs = ('O', 'X')
        signs = build_signs if self._goal == 'order' else block_signs
        return signs[0] if comb_sign == 'O' else signs[1]

    def choose_move(self, chosen_comb, board_combinations):
        """
        Funkcja wybiera ruch (współżędne pola i znak)
        opierający się na dostawieniu znaku przy którymś z dwóch końców
        przekazanej do funkcji kombinacji.
        W celu zmniejszenia przewidywalności wybranego ruchu i
        uniknięcia sytuacji w której do kombinacji która ma wolne pola
        przy obu końcach zawsze dostawiany jest znak z tej samej strony,
        wykorzystany jest czynnik losowej kolejności który orzyjmuje losową
        wartość 0 lub 1 i determinuje kolejność sprawdzenia możliwości
        dostawienia (przykładowo gdy do poziomej kombinacji
        można dołożyć zarówno z prawej i z lewej to wybór ten będzie losowy,
        dzięki temu nie występuje sytuacja w której zawsze dostawiony
        jest znak z lewej lub zawsze z prawej).
        Kombinacja zostaje wjybrana edynie w sytuacji
        w której któreś z pól (lub oba) przy końcach kombinacji jest wolne
        oraz jeśli w lini kominacji da się utworzyć kombinację o długości 5
        ze znaku tej kombinacji. Dzięki temu chaos blokuje jedynie
        te linie w których istnieje zagrożenie stworzenia kombinacji 5,
        a porządek dokłada jedynie w tej sytuacji

        """
        (
            comb_len,
            comb_end_row,
            comb_end_column,
            comb_direction,
            comb_sign
        ) = chosen_comb
        sign = self.choose_sign(comb_sign)
        random_order_factor = random.randint(0, 1)
        while random_order_factor < 2:
            if comb_direction == Direction.horizontally.value:
                first_field = comb_end_row, comb_end_column + 1, sign
                second_field = comb_end_row, comb_end_column - comb_len, sign
                chosen_field = board_combinations.move_with_5_comb(
                    chosen_comb,
                    first_field,
                    second_field,
                    random_order_factor
                )
                if chosen_field:
                    return chosen_field

            if comb_direction == Direction.vertically.value:
                first_field = comb_end_row + 1, comb_end_column, sign
                second_field = comb_end_row - comb_len, comb_end_column, sign
                chosen_field = board_combinations.move_with_5_comb(
                    chosen_comb,
                    first_field,
                    second_field,
                    random_order_factor
                )
                if chosen_field:
                    return chosen_field

            if comb_direction in (
                    Direction.diagonally_short_up.value,
                    Direction.diagonally_long_up.value):
                first_field = comb_end_row - 1, comb_end_column + 1, sign
                second_field = comb_end_row + comb_len, comb_end_column - comb_len, sign
                chosen_field = board_combinations.move_with_5_comb(
                    chosen_comb,
                    first_field,
                    second_field,
                    random_order_factor
                )
                if chosen_field:
                    return chosen_field

            if comb_direction in (
                    Direction.diagonally_short_down.value,
                    Direction.diagonally_long_down.value):
                first_field = comb_end_row + 1, comb_end_column + 1, sign
                second_field = comb_end_row - comb_len, comb_end_column - comb_len, sign
                chosen_field = board_combinations.move_with_5_comb(
                    chosen_comb,
                    first_field,
                    second_field,
                    random_order_factor
                )
                if chosen_field:
                    return chosen_field
            random_order_factor += 1

    def generate_move(self, board_combinations):
        """
        Funkcja generuje ruch gracza komputerowego.
        Spośród najdłuższych kombinacji wybierana jest najdłóższa kombinacja
        do której da się dołożyć znak, a w jej lini da się zbudować kombinację
        o długości 5 ze znaku który ma ta kombinacja. Jeśli kilka kombinacji ma
        taką samą długość to zostaje wybrana losowa.
        Jeśli wedle tego algorytmu nie zostanie wybrana żadna z kombinacji
        zostaje wykonany ruch losowy.

        """
        X_combinations, O_combinations = board_combinations.find_combinations()
        X_combinations_random_order = random.sample(
            X_combinations, k=len(X_combinations))
        O_combinations_random_order = random.sample(
            O_combinations, k=len(O_combinations))
        is_move_generated = False

        def first_value(value):
            return value[0]
        while is_move_generated is False:
            if not X_combinations_random_order:
                X_chosen_comb = (0,)
            if X_combinations_random_order:
                X_chosen_comb = max(
                    X_combinations_random_order, key=first_value)
                X_combinations_random_order.remove(X_chosen_comb)
                X_chosen_comb += ('X',)
            if not O_combinations_random_order:
                O_chosen_comb = (0,)
            if O_combinations_random_order:
                O_chosen_comb = max(
                    O_combinations_random_order, key=first_value)
                O_combinations_random_order.remove(O_chosen_comb)
                O_chosen_comb += ('O',)
            chosen_comb = max(X_chosen_comb, O_chosen_comb, key=first_value)
            if chosen_comb != (0,):
                chosen_move = self.choose_move(chosen_comb, board_combinations)
                if chosen_move:
                    return chosen_move
            if chosen_comb == (0,):
                return self.generate_random_move(board_combinations)


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
        self.order_player = OrderTrainer
        self.chaos_player = ChaosTrainer
    
        self.states = []
        self.
    def 