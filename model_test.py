import math

import torch
import pickle

from prepare_data import prepare_data
from model import Net

with open("data", mode="rb") as file:
    rdata = pickle.load(file)

header = prepare_data(rdata)
del header["inputs"]
del header["outputs"]
del rdata
print(header)

net = Net()
net.load_state_dict(torch.load("test_model"))

x = 200
y = 20 / header["max_height"]
z = 200

distance = math.sqrt(x * x + z * z) / header["max_distance"]

with torch.no_grad():
    result = net(torch.tensor([distance, y], dtype=torch.float))

    n_delta_t = result[0] * header["max_delta_t"]
    n_pitch = result[1] * header["max_tried_pitch"]
    n_airtime = result[2] * header["max_airtime"]

from CannonBallisticFunctions import try_pitch

res = try_pitch((0, 0, 0), (200, 20, 200), 7, 32, 100000)

print(n_delta_t, n_pitch, n_airtime)
print(res)