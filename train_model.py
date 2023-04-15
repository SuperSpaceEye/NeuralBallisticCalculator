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

net = Net(batch_size)
summary(net, (5,), batch_size=batch_size, device="cpu")

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
labels = torch.from_numpy(labels[:data_size]).float()[:, 1].reshape(-1, 1)

train_dataset   = torch.utils.data.TensorDataset(inputs, labels)
train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
evaluate_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(net.parameters(), lr)
# optimizer = optim.SGD(net.parameters(), lr, momentum)
criterion = nn.MSELoss()

mult = header["max_tried_pitch"]

train_functions.train_with_epoches(
    net, epoches, train_loader, optimizer, criterion, evaluate_loader,
    torch.device("cuda"), True, "Models/net", header["max_tried_pitch"], 1
)