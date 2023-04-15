import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import pickle

from test import Net
from prepare_data import prepare_data

net = Net()
summary(net, (32, 2), batch_size=-1, device="cpu")

epoches = 100
batch_size = 64
lr = 0.0001

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

train_dataset = torch.utils.data.TensorDataset(inputs, labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(net.parameters(), lr)
criterion = nn.MSELoss()

mult = header["max_tried_pitch"]

# for epoch in range(epoches):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs * mult, labels[:, 1].reshape(-1, 1) * mult)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
#             running_loss = 0.0
#
# torch.save(net.state_dict(), "test_model2")

train_dataset = torch.utils.data.TensorDataset(inputs, labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
#
print("Validation")
#
net.load_state_dict(torch.load("test_model2"))

heat_data = []

# # for normal x/y coordinates
# temp = []
# for item in rdata:
#     temp += item
# rdata = temp

# optimizer.zero_grad()
# total_loss = 0.0
# with torch.no_grad():
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         avg_loss = criterion(outputs, labels)
#
#         delta_t_loss = criterion(outputs[0][0] * header["max_delta_t"],
#                                  labels [0][0] * header["max_delta_t"]
#                                  ).item()
#         pitch_loss   = criterion(outputs[0][1] * header["max_tried_pitch"],
#                                  labels [0][1] * header["max_tried_pitch"]
#                                  ).item()
#         airtime_loss = criterion(outputs[0][2] * header["max_airtime"],
#                                  labels [0][2] * header["max_airtime"]
#                                  ).item()
#
#         heat_data.append([
#             rdata[i][0][0], rdata[i][0][1],
#             avg_loss.item(), delta_t_loss, pitch_loss, airtime_loss
#         ])
#
#         # print statistics
#         total_loss += avg_loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{i + 1:5d}] loss: {total_loss / i:.5f}')
#
# print(total_loss / len(train_loader))
#
# with open("test_result", mode="wb") as file:
#     pickle.dump(heat_data, file)
#
# with open("test_result", mode="rb") as file:
#     heat_data = pickle.load(file)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def scatter_plot(x, y, z):
#     plt.scatter(x, y, c=z)
#     plt.colorbar()
#     plt.show()
#
# x = np.array([x[0] for x in heat_data])
# y = np.array([x[1] for x in heat_data])
#
# scatter_plot(x,y,np.array([x[2] for x in heat_data]))
# scatter_plot(x,y,np.array([x[3] for x in heat_data]))
# scatter_plot(x,y,np.array([x[4] for x in heat_data]))
# scatter_plot(x,y,np.array([x[5] for x in heat_data]))
