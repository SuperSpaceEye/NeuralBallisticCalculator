import train_functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import pickle

from model import Net
from prepare_data import prepare_data

epoches = 100
batch_size = 8192
lr = 0.01

net = Net()
summary(net, (2,), batch_size=batch_size, device="cpu")

with open("data", mode="rb") as file:
    rdata = pickle.load(file)

data = prepare_data(rdata)

inputs = data["inputs"]
labels = data["outputs"]

header = data
del header["inputs"]
del header["outputs"]

data_size = int(len(inputs) / batch_size)*batch_size
inputs = torch.from_numpy(inputs[:data_size]).float()
labels = torch.from_numpy(labels[:data_size]).float()

train_dataset   = torch.utils.data.TensorDataset(inputs, labels)
train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
evaluate_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(net.parameters(), lr)
# optimizer = optim.SGD(net.parameters(), lr, momentum)
criterion = nn.MSELoss()


delta_t_weight = 1
tried_pitch_weight = 1
airtime_weight = 0.1
def custom_loss(pred, label):
    delta_t     = criterion(pred[:, 0], label[:, 0]) * delta_t_weight
    tried_pitch = criterion(pred[:, 1], label[:, 1]) * tried_pitch_weight
    airtime     = criterion(pred[:, 2], label[:, 2]) * airtime_weight

    return delta_t + tried_pitch + airtime

mult = header["max_tried_pitch"]

train_functions.train_with_epoches(
    net, epoches, train_loader, optimizer, custom_loss, evaluate_loader,
    torch.device("cpu"), True, "Models/net", 1,
)