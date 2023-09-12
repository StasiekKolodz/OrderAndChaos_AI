from classes import PcRandomPlayer, WrongSignError
from nn_model import Network, Trainer
import numpy as np

class AiPlayer(PcRandomPlayer):
    """ Pc Player driven by neural network """

    def __init__(self, name, player_goal):
        super().__init__(self, name, player_goal)
        self.model = Network()

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
     
class OrderTrainer(AiPlayer):
    def __init__(self, name, player_goal):
        self.trainer = Trainer


