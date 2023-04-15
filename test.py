import matplotlib.pyplot as plt
import pickle

data = []

with open("data", mode="rb") as file:
    data = pickle.load(file)

x_axis = []
y_axis = []

for thread_result in data:
    for item in thread_result:
        x_axis.append(item[0][0])
        y_axis.append(item[0][1])

plt.scatter(x_axis, y_axis)
plt.show()