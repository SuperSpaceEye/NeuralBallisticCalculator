import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

layers = [100, 100, 100]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_fc = nn.Linear(2, layers[0])

        self.linears = nn.ModuleList()

        for width in layers:
            self.linears.append(nn.LazyLinear(width))
            # self.layers.append(nn.BatchNorm1d(8))

        self.fc_end = nn.LazyLinear(1)

    def forward(self, x):
        x = F.tanh(self.input_fc(x))
        for item in self.linears:
            x = F.tanh(item(x))
        x = self.fc_end(x)
        return x


if __name__ == "__main__":
    net = Net()
    summary(net, (64, 2), -1, "cpu")
