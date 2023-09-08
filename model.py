import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

# layers = [100, 100, 100, 100, 100]
layers = [32, 64, 128, 64, 128, 64, 128]
num_inputs = 2
num_outputs = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_fc = nn.Linear(num_inputs, layers[0])

        self.layers = nn.ModuleList()

        for b_width, width in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.BatchNorm1d(b_width, affine=False))
            self.layers.append(nn.LazyLinear(width))
            self.layers.append(nn.SiLU())

        self.fc_end = nn.LazyLinear(num_outputs)

    def forward(self, x):
        x = F.silu(self.input_fc(x))
        for item in self.layers:
            # print(x.shape)
            x = item(x)
        x = self.fc_end(x)
        return x


if __name__ == "__main__":
    net = Net()
    summary(net, (2,), 32, "cpu")
