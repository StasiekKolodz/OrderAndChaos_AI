import torch
import os 

class WinMeter:
    def __init__(self):
        self.order_wins = 0
        self.chaos_wins = 0
        self.avg_order_results = []
        self.avg_chaos_results = []

    def add_win(self, winner):
        
        if winner == "chaos":
            self.chaos_wins += 1
        elif winner == "order":
            self.order_wins += 1
        else:
            raise ValueError("Unknown winner")

        if (self.order_wins+self.chaos_wins)%1000 == 0:
            self.print_stats()
            self.avg_order_results.append(self.order_wins/(self.chaos_wins+self.order_wins))
            self.avg_chaos_results.append(self.chaos_wins/(self.chaos_wins+self.order_wins))
            self.reset()

    def print_stats(self):
        if self.order_wins != 0 or self.chaos_wins != 0:
            print("WinMeter stats:")
            print(f"Order wins: {self.order_wins}  ({(self.order_wins/(self.chaos_wins+self.order_wins))*100}%)")
            print(f"Chaos wins: {self.chaos_wins}  ({(self.chaos_wins/(self.chaos_wins+self.order_wins))*100}%)")

    def reset(self):
        self.order_wins = 0
        self.chaos_wins = 0

  
    def save(self, file_name="stats_state.pth"):
        model_folder_path = './stats'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({
            'chaos_stats': self.avg_chaos_results ,
            'order_stats': self.avg_order_results ,
            }, file_name)


