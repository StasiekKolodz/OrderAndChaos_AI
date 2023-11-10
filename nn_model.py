import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        """Input of the NN is board content flatened and output is 72 neurons for 
        every field (36) and every sign on that field(36*2=72)

        Firstly all fields and x sign next all fields and o sign"""
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.lin1 = nn.Linear(360, 72)
        self.relu = nn.ReLU()
        # self.b_norm = nn.BatchNorm1d(50)
        self.lin2 = nn.Linear(50, 72)

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        x = torch.flatten(x,1)
        # print(f"SHAPE: {x.shape}")
        x = self.lin1(x)
        # x = self.b_norm(x)
        # x = self.relu(x)
        
        # x = self.lin2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self, model, lr=0.0001, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def set_lr(self, lr):
        self.lr = lr

    def train_step(self, boards, next_boards, actions, rewards):
        # print(f" boards shape: {boards.shape}")
        pred = self.model(boards)
        next_pred = self.model(next_boards)
        target = pred.clone()
        num_samples = len(boards)
        for idx in range(num_samples):

            if idx != num_samples-1:
                Q_new = rewards[idx] + self.gamma * torch.max(next_pred[idx]).item()
            else:
                Q_new = rewards[idx]

    
            target[idx][torch.argmax(actions[idx]).item()] = Q_new      
            # print(f"IDX: {idx}")
            # print(f"pred: {pred[idx]}")
            # # print(f"next_pred: {next_pred[idx]}")
            # # print(f"board: {boards[idx]}")
            # print(f"target: {target[idx]}")
            # # print(f"actions: {actions[idx].argmax()}")
            # # print(f"actions_q: {torch.argmax(pred[idx]).item()}")
            # print(f"Q_new: {Q_new}")
            # print(f"Q: {torch.max(pred[idx]).item()}")
            # print(f"reward: {rewards[idx]}")
        loss = self.criterion(target, pred)
        # print(f"loss: {loss.item()}")
     
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()