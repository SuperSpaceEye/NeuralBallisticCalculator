import math

import torch
import pickle

from prepare_data import prepare_data
from model import Net

# with open("data", mode="rb") as file:
#     rdata = pickle.load(file)
#
# header = prepare_data(rdata)
# del header["inputs"]
# del header["outputs"]
# del rdata
# print(header)

header = {'max_distance': 599.0, 'max_height': -255.0, 'max_delta_t': 0.9999949932098389, 'max_tried_pitch': 59.99999237060547, 'max_airtime': 160.0}

net = Net()
net.load_state_dict(torch.load("test_model"))
net.eval()

x = 200
y = 20 / header["max_height"]
z = 200

distance = math.sqrt(x * x + z * z) / header["max_distance"]

with torch.no_grad():
    test_data = torch.tensor([[distance, y]], dtype=torch.float)
    result = net(test_data)[0]

    # n_pitch = result[0] * header["max_tried_pitch"]

    n_delta_t = result[0] * header["max_delta_t"]
    n_pitch   = result[1] * header["max_tried_pitch"]
    n_airtime = result[2] * header["max_airtime"]

from make_data.BallisticFunctions import calculate_pitch

res = calculate_pitch((0, 0, 0), (200, 20, 200), 8, 32)

print(n_delta_t, n_pitch, n_airtime)
print(res)