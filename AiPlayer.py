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
        self.moves = []


    def decode_sign(self, signs):
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

    def encode_sign(self, sign_num):
        if sign_num == 0:
            return '-'
        if sign_num == 1:
            return 'X'
        if sign_num == 2:
            return 'O'


    def generate_move(self, board_combinations):
        """Generate next move on board based on network prediction"""
        board_state = list(map(self.decode_sign, copy.deepcopy(board_combinations.state())))
        board_state = torch.tensor(board_state, dtype=torch.float).flatten()
        print(f"shape = {board_state.shape}")
        prediction = self.model(board_state)
        max_pred = prediction.argmax().item()
        sign = self.encode_sign(max_pred//36 + 1)
        row = (max_pred%36)//6
        column = (max_pred%36)%6
        return row, column, sign
     

class PlayerTrainer(AiPlayer):
    def __init__(self, name, goal):
        self.model = Network()

        self.name = name
        self.goal = goal
        self.trainer = Trainer(self.model)

        self.board_states = []
        self.moves = []

    def set_trainer_lr(self, lr):
        self.trainer.set_lr(lr)

    def add_move_data(self, board_state, move):
        self.board_states.append(board_state)
        move = list(move)
        move[2] = self.encode_sign(move[2])
        self.moves.append(move)

    def reset_moves_data(self):
        self.board_states = []
        self.moves = []
    
    def moves_to_tensor(self, moves):
        row, column, sign = moves
        idx = 6 *row + column
        if sign == 'O':
            idx+=36
        tensor = torch.zeros(72)
        tensor[idx] = 1
        return tensor
        
class OrderTrainer(PlayerTrainer):
    def __init__(self):
        super().__init__('Order Trainer', 'order')

    def train(self, game_winner, moves_number):
        
        if game_winner == 'chaos':
            reward = -0.5
        if game_winner == 'order':
            reward = 8 - moves_number*0.1
        
        for board, move in zip(self.board_states, self.moves):
            decoded_board = list(map(self.decode_sign, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            
            # print(f"x_tensor: {x_tensor.shape}")
            self.trainer.train_step(decoded_tensor, self.moves_to_tensor(move), reward)
        


class ChaosTrainer(PlayerTrainer):
    def __init__(self):
        super().__init__('Chaos Trainer', 'chaos')

    def train(self, game_winner, moves_number):
        
        if game_winner == 'order':
            reward = -0.5
        if game_winner == 'chaos':
            reward = 8 - moves_number*0.1
        for board, move in zip(self.board_states, self.moves):
            decoded_board = list(map(self.decode_sign, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            
            # print(f"x_tensor: {x_tensor.shape}")
            self.trainer.train_step(decoded_tensor, self.moves_to_tensor(move), reward)
