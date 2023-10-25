class WinMeter:
    def __init__(self):
        self.order_wins = 0
        self.chaos_wins = 0

    def add_win(self, winner):
        if winner == "chaos":
            self.chaos_wins += 1
        elif winner == "order":
            self.order_wins += 1
        else:
            raise ValueError("Unknown winner")

    def print_stats(self):
        if self.order_wins != 0 or self.chaos_wins != 0:
            print("WinMeter stats:")
            print(f"Order wins: {self.order_wins}  ({(self.order_wins/(self.chaos_wins+self.order_wins))*100}%)")
            print(f"Chaos wins: {self.chaos_wins}  ({(self.chaos_wins/(self.chaos_wins+self.order_wins))*100}%)")

