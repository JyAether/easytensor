import torch.nn as nn


# 创建对应的PyTorch网络
class TorchMLP(nn.Module):
    def __init__(self):
        super(TorchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)
