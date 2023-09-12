# import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        """Input of the NN is board content flatened and output is 72 neurons for 
        every field (36) and every sign on that field(36*2=72)

        Firstly all fields and x sign next all fields and o sign"""
        self.lin1 = nn.Linear(36, 100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, 72)
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

class Trainer:
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def set_lr(self, lr):
        self.lr = lr
        
    def train_step(self, states, actions, reward):

        self.optimizer.zero_grad()
        pred = self.model(states)
        loss = self.criterion(pred, actions) * reward
        loss.backward()
        self.optimizer.step()