# import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(36, 100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, 72)
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, states, actions, reward):

        self.optimizer.zero_grad()
        pred = self.model(states)
        loss = self.criterion(pred, actions) * reward
        loss.backward()
        self.optimizer.step()