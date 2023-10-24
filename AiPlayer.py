from classes import PcRandomPlayer, WrongSignError
from nn_model import Network, Trainer
import numpy as np
import torch
import copy


class AiPlayer(PcRandomPlayer):
    """ Pc Player driven by neural network """

    def __init__(self, name, player_goal):
        super().__init__(name, player_goal)
        self.model = Network()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        print(f"Device: {self.device}")
        self.model = self.model.to(self.device)
        self.moves = []
        self.moves_counter = 0


    def decode_sign_board(self, signs):
        for i, sign in enumerate(signs):
            if sign == '-':
                signs[i] = 0
            elif sign == 'X':
                signs[i] = 1
            elif sign == 'O':
                signs[i] = 2
            else:
                print(f"Invalid sign: {sign}")
                raise WrongSignError
        return signs

    def decode_x_o(self, sign):
        if sign == 'X':
            return 1
        elif sign == 'O':
            return 2
        else:
            print(f"Invalid sign: {sign}")
            raise WrongSignError

    def encode_sign(self, sign_num):
        if sign_num == 0:
            return '-'
        if sign_num == 1:
            return 'X'
        if sign_num == 2:
            return 'O'


    def generate_move(self, board_combinations):
        if self.moves_counter >=300000:
            """Generate next move on board based on network prediction"""
            board_state = list(map(self.decode_sign_board, copy.deepcopy(board_combinations.state())))
            board_state = torch.tensor(board_state, dtype=torch.float).flatten()
            prediction = self.model(board_state.to(self.device))
            max_pred = prediction.argmax().item()
            sign = self.encode_sign(max_pred//36 + 1)
            row = (max_pred%36)//6
            column = (max_pred%36)%6
            return row, column, sign
        else:
            self.moves_counter += 1
            if self.moves_counter == 300000 -2:
                print("Last random move")
            return self.generate_random_move(board_combinations)
        
     

class PlayerTrainer(AiPlayer):
    def __init__(self, name, goal):
        super().__init__(name, goal)
        # self.model = Network()

        # self.name = name
        # self.goal = goal
        self.trainer = Trainer(self.model)

        self.board_states = []
        self.moves = []

    def set_trainer_lr(self, lr):
        self.trainer.set_lr(lr)

    def add_move_data(self, board_state, move):
        self.board_states.append(board_state)
        sign = self.decode_x_o(move[2])
        move = list(move)
        move[2] = sign
        self.moves.append(move)

    def reset_moves_data(self):
        self.board_states = []
        self.moves = []
    
    def moves_to_tensor(self, moves):
        row, column, sign = moves
        idx = 6 *row + column
        if sign == 2:
            idx+=36
        tensor = torch.zeros(72)
        tensor[idx] = 1
        return tensor
        
class OrderTrainer(PlayerTrainer):
    def __init__(self):
       super().__init__('Order Trainer', 'order')

    def train(self, game_winner, moves_number, illegal_move):
        
        rewards = torch.zeros(moves_number)
        if illegal_move:
            rewards[-1] = -1000
        elif game_winner == 'chaos':
            rewards[-1] =  -10
        elif game_winner == 'order':
            # TODO: Implement moves_number sensitivity to win faster
            rewards[-1] = 10

            # reward = 8 - moves_number*0.1
        
        for i, board in enumerate(self.board_states):
            move = self.moves[i]
            decoded_board = list(map(self.decode_sign_board, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            # print(f"move: {move}")
            # print(f"x_tensor: {x_tensor.shape}")
            self.trainer.train_step(decoded_tensor.to(self.device), self.moves_to_tensor(move), reward)
        


class ChaosTrainer(PlayerTrainer):
    def __init__(self):
        super().__init__('Chaos Trainer', 'chaos')

    def train(self, game_winner, moves_number, illegal_move):
        # print(f"ww: {game_winner}")

        rewards = torch.zeros(moves_number)

        if illegal_move:
            rewards[-1] = -1000
        elif game_winner == 'order':
            rewards[:] = -10
        elif game_winner == 'chaos':
            rewards[:] = 10

        for i, board in enumerate(self.board_states):
            move = self.moves[i]
            decoded_board = list(map(self.decode_sign_board, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()

            if i < len(self.board_states)-1:
                next_board = self.board_states[i+1]
            else:
                next_board = None
            decoded_next_board = list(map(self.decode_sign_board, copy.deepcopy(board)))
            decoded_next_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            # print(f"move: {move}")
            # print(f"x_tensor: {x_tensor.shape}")
            self.trainer.train_step(decoded_tensor.to(self.device), 
                decoded_next_tensor.to(self.device), 
                self.moves_to_tensor(move).to(self.device), rewards[i])

class RandomTrainer(PlayerTrainer):
    def train(self, game_winner, moves_number, illegal_move):
        pass

    def generate_move(self, board_combinations):
        return self.generate_random_move(board_combinations)