import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, temperature=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))
        self.temperature = temperature

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512, temperature)
        self.fc2 = PrunableLinear(512, 256, temperature)
        self.fc3 = PrunableLinear(256, 10, temperature)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
