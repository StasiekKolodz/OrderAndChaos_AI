from classes import PcRandomPlayer, WrongSignError
from nn_model import Network, Trainer
import numpy as np
import torch
import copy
from numpy import random



class AiPlayer(PcRandomPlayer):
    """ Pc Player driven by neural network """

    def __init__(self, name, player_goal, path=None):
        super().__init__(name, player_goal)
        self.model = Network()

        if path:
            self.model.load_state_dict(torch.load(path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_games_num = 1000
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

    def load_model(self, path):
        print(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path))
        print("Model loaded")

    def generate_move(self, board_combinations):
        rand_val = random.rand()
        if self.moves_counter >= self.random_games_num and rand_val>0.2:
            """Generate next move on board based on network prediction"""
            board_state = list(map(self.decode_sign_board, copy.deepcopy(board_combinations.state())))
            board_state = torch.tensor(board_state, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
            self.model.eval()
            with torch.no_grad():
                
                prediction = self.model(board_state.to(self.device))
                prediction = prediction[0]
                # Look for the best move but possible on board
                top_k = torch.topk(prediction, 6*6*2)
                idx = 0
                max_pred = top_k.indices[idx]
                row = (max_pred%36)//6
                column = (max_pred%36)%6
                while board_combinations.state()[row][column] != '-':
                    idx += 1
                    max_pred = top_k.indices[idx]
                    row = (max_pred%36)//6
                    column = (max_pred%36)%6
        
                # max_pred = prediction.argmax().item()
            self.model.train()
            sign = self.encode_sign(max_pred//36 + 1)

            self.moves_counter += 1
            # if self.moves_counter == self.random_games_num * 2:
            #     self.moves_counter = 0
            #     print("Counter reset")
            return row, column, sign
        else:
            self.moves_counter += 1
            if self.moves_counter == self.random_games_num -2:
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
    
    def save_model(self, model_name="model"):
        print(f"Saving model {model_name}...")
        self.model.save(file_name=model_name+".pth")
        print("Model saved...")

class OrderTrainer(PlayerTrainer):
    def __init__(self):
       super().__init__('Order Trainer', 'order')

    def train(self, game_winner, moves_number, illegal_move):
        # print(f"ww: {game_winner}")

        rewards = torch.zeros(moves_number)

        if illegal_move:
            rewards[:] = 1
            rewards[-1] = -1000
        elif game_winner == 'order':
            rewards[:] = 10
            rewards[-1] = 100
        elif game_winner == 'chaos':
            rewards[:] = -10
        else:
            raise ValueError

        batch_size = len(self.board_states)
        tensor_boards = torch.empty((batch_size,6*6), dtype=torch.float)
        tensor_next_boards = torch.empty((batch_size,6*6), dtype=torch.float)
        tensor_moves = torch.empty((batch_size,6*6*2), dtype=torch.float)

        for i, board in enumerate(self.board_states):
            tensor_move = self.moves_to_tensor(self.moves[i])
            tensor_moves[i][:] = tensor_move
            decoded_board = list(map(self.decode_sign_board, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float).flatten()
            tensor_boards[i] = decoded_tensor
            if i < batch_size-1:
                next_board = self.board_states[i+1]
                decoded_next_board = list(map(self.decode_sign_board, copy.deepcopy(next_board)))
                decoded_next_tensor = torch.tensor(decoded_next_board, dtype=torch.float).flatten()
                tensor_next_boards[i] = decoded_next_tensor
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            # print(f"move: {move}")
            # print(f"x_tensor: {x_tensor.shape}")
        self.trainer.train_step(tensor_boards.to(self.device), 
            tensor_next_boards.to(self.device), 
            tensor_moves.to(self.device), rewards)
        


class ChaosTrainer(PlayerTrainer):
    def __init__(self):
        super().__init__('Chaos Trainer', 'chaos')
        # path = "./model/chaos_model.pth"
        # self.load_model(path)
        # print(f"Chaos model loaded form {path}")

    def train(self, game_winner, moves_number, illegal_move):
        # print(f"ww: {game_winner}")

        rewards = torch.zeros(moves_number)

        if illegal_move:
            rewards[:] = 0.1
            rewards[-1] = -20
        elif game_winner == 'order':
            rewards[:] = 0.1
            rewards[-1] = -15
        elif game_winner == 'chaos':
            rewards[:] = 0.1
            rewards[-1] = 15
        else:
            raise ValueError

        batch_size = len(self.board_states)
        tensor_boards = torch.empty((batch_size,1, 6, 6), dtype=torch.float)
        tensor_next_boards = torch.empty((batch_size,1, 6, 6), dtype=torch.float)
        tensor_moves = torch.empty((batch_size,6*6*2), dtype=torch.float)

        for i, board in enumerate(self.board_states):
            tensor_move = self.moves_to_tensor(self.moves[i])
            tensor_moves[i][:] = tensor_move
            decoded_board = list(map(self.decode_sign_board, copy.deepcopy(board)))
            decoded_tensor = torch.tensor(decoded_board, dtype=torch.float)
            tensor_boards[i][0] = decoded_tensor
            if i < batch_size-1:
                next_board = self.board_states[i+1]
                decoded_next_board = list(map(self.decode_sign_board, copy.deepcopy(next_board)))
                decoded_next_tensor = torch.tensor(decoded_next_board, dtype=torch.float)
                tensor_next_boards[i][0] = decoded_next_tensor
            # torch.cat((x_tensor, decoded_tensor), dim=0)
            # print(f"board: {decoded_tensor.shape}")
            # print(f"move: {move}")
            # print(f"x_tensor: {x_tensor.shape}")
        self.trainer.train_step(tensor_boards.to(self.device), 
            tensor_next_boards.to(self.device), 
            tensor_moves.to(self.device), rewards)
class RandomTrainer(PlayerTrainer):
    def train(self, game_winner, moves_number, illegal_move):
        pass

    def generate_move(self, board_combinations):
        return self.generate_random_move(board_combinations)