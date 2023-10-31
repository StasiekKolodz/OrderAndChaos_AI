import torch
from matplotlib import pyplot as plt

print("A")
checkpoint = torch.load('stats/stats_autosave.pth')
chaos_stats = checkpoint["chaos_stats"]
print("b")
plt.plot(chaos_stats)
plt.show()