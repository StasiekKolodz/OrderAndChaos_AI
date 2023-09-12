from classes import PcRandomPlayer, WrongSignError
from nn_model import Network, Trainer
import numpy as np

class AiPlayer(PcRandomPlayer):
    """ Pc Player driven by neural network """

    def __init__(self, name, player_goal):
        super().__init__(self, name, player_goal)
        self.model = Network()
        self.moves = []


    def decode_sign(self, sign):
        if sign == '-':
            return 0
        if sign == 'x':
            return 1
        if sign == 'o':
            return 2
        else:
            raise WrongSignError

    def encode_sign(self, sign_num):
        if sign_num == 0:
            return '-'
        if sign_num == 1:
            return 'x'
        if sign_num == 2:
            return 'o'
        else:
            raise WrongSignError

    def generate_move(self, board_combinations):
        """Generate next move on board based on network prediction"""
        board_state = list(map(self.decode_sign, board_combinations.state()))
        board_state = torch.tensor(board_state).flatten()
        prediction = self.model(board_state)
        max_pred = prediction.argmax()
        sign = self.encode_sign(max_pred//36 + 1)
        row = (max_pred%36)//6
        column = (max_pred%36)%6
        return row, column, sign
     

class PlayerTrainer(AiPlayer):
    def __init__(self, name, goal):
        self.name = name
        self.goal = goal
        self.trainer = Trainer(self.model)

        self.board_states = []
        self.moves = []

    def set_trainer_lr(self, lr):
        self.trainer.set_lr(lr)

    def add_move_data(self, board_state, move):
        self.board_states.append(board_state)
        self.moves.append(move)

    def reset_moves_data(self):
        self.board_states = []
        self.moves = []

    

class OrderTrainer(AiPlayer):
    def __init__(self):
        super().__init__('Order Trainer', 'order')

    def train(self, game_winner, moves_number):
        
        if game_winner == 'chaos':
            reward = -0.5
        if game_winner == 'order':
            reward = 8 - moves_number*0.1
        
        self.trainer.train_step(self.board_states, self.moves, reward)
        


class ChaosTrainer(AiPlayer):
    def __init__(self):
        super().__init__('Chaos Trainer', 'chaos')

    def train(self, game_winner, moves_number):
        
        if game_winner == 'order':
            reward = -0.5
        if game_winner == 'chaos':
            reward = 8 - moves_number*0.1

        self.trainer.train_step(self.board_states, self.moves, reward)